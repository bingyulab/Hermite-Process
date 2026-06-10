"""
gamma_views_patch.py
====================

Adds to the gamma (UNet layer-trace) experiment:

  (Q1) a per-layer comparison of FOUR kappa4 reductions
       - mean     : spatial average over (H,W)   -> (N, C)        [CLT-suppressed]
       - center   : central spatial cell         -> (N, C)
       - unit     : every (c,h,w) marginal        -> (N, C*H*W)   [per-coordinate]
       - channels : pool batch and space          -> (N*H*W, C)   [per-channel field]
       sample axis differs between `unit` (batch) and `channels` (batch x space),
       which is exactly why they disagree (see notes at bottom).

  (Q2) an explicit "input" point (raw x0) on the same trace.

  (Q3) per-stage diagnostic logging that exposes when a large kappa4 is driven
       by near-degenerate (near-constant) columns rather than by genuine
       non-Gaussianity.

Integration
-----------
Inside the existing `run_experiment_gamma` loop, after `model, fwd` are
obtained for a given noise_type, call:

    from gamma_views_patch import gamma_views_rows
    view_rows += gamma_views_rows(model, fwd, runner.test_ds, cfg,
                                  mname, ctx.logger,
                                  n_samples=min(cfg.n_samples, 1024))

then once after the noise_type loop:

    from rcd.train.save import save_csv
    save_csv(view_rows, ctx.get_path("metric", "gamma_views.csv"), silent=True)

Plot with:

    from gamma_views_patch import plot_gamma_views
    plot_gamma_views(ctx.get_path("metric", "gamma_views.csv"), ctx.plot_dir)

`gamma_views.csv` is written alongside the existing `gamma.csv`; no existing
column or consumer is touched.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from rcd.evaluation.measurement import (
    capture_activations, get_unet_modules, kappa4_three_views,
)
from rcd.experiments.registry import UNET_LAYER_KEYS, LAYER_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# (Q3) diagnostic logging
# ─────────────────────────────────────────────────────────────────────────────

def log_kappa4_diagnostics(name: str, X: torch.Tensor, logger=None,
                           var_floor: float = 1e-3,
                           max_cols: int = 20000) -> dict:
    """
    Report whether a stage's mean kappa4 is dominated by near-degenerate
    columns. X is (N, D) or (N, C, H, W) (flattened to (N, D) here).
 
    A column whose variance is ~0 (e.g. a border pixel that is almost always
    background) is standardised by a clamped variance (1e-8); a single
    deviation then produces an enormous standardised 4th moment. The mean over
    such columns can reach the tens or hundreds. This makes the value REAL but
    an artifact of sparse marginals, not evidence of a heavy-tailed image law.
 
    kappa4 and variance are computed from the SAME column set so the
    degenerate-column mask always aligns (unlike compute_marginal_cumulants,
    which randomly subsamples columns to <=2048 for speed).
    """
    if X.numel() == 0:
        return {}
    if X.dim() > 2:
        X = X.reshape(X.size(0), -1)
    X = X.float()
    N, D_full = X.shape
 
    # Optional column cap for very wide per-unit layers. Same columns are used
    # for var and k4, so the mask is always consistent.
    if D_full > max_cols:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(D_full, generator=g)[:max_cols]
        X = X[:, idx]
    D = X.shape[1]
 
    Xc = X - X.mean(0)
    var = Xc.var(0)
    std = var.clamp(min=1e-8).sqrt()
    Z = Xc / std
    k4 = ((Z ** 4).mean(0) - 3.0).cpu().numpy()      # (D,)
    var = var.cpu().numpy()                           # (D,)
 
    n_degen = int((var < var_floor).sum())
    keep = var >= var_floor
    mean_nondegen = float(k4[keep].mean()) if keep.any() else float("nan")
    msg = {
        "stage": name,
        "N": int(N), "D": int(D),
        "mean_k4": float(k4.mean()),
        "median_k4": float(np.median(k4)),
        "max_abs_k4": float(np.max(np.abs(k4))),
        "n_cols_var_lt_floor": n_degen,
        "mean_k4_drop_degenerate": mean_nondegen,
    }
    line = (f"[k4-diag] {name:18s} N={msg['N']} D={msg['D']}  "
            f"mean={msg['mean_k4']:+8.3f}  median={msg['median_k4']:+7.3f}  "
            f"max|k4|={msg['max_abs_k4']:8.1f}  "
            f"degen(var<{var_floor})={n_degen}  "
            f"mean_nondegen={mean_nondegen:+8.3f}")
    (logger.info if logger else print)(line)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Raw 4D per-layer capture (reduce="none"), matched-t forward
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def capture_layer_views(model, fwd, test_ds, cfg, n_samples: int = 1024):
    """
    Forward the corrupted test set once, capturing raw (B,C,H,W) activations
    for every UNet layer plus the raw input. t_corrupt = t_eval = 1.0,
    condition = null (matched to the existing gamma settings).

    Returns (input_x0, {layer_key: (N,C,H,W) tensor}).
    """
    device = cfg.device
    model.eval()
    modules = get_unet_modules(model)
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)

    raw_list, n_done = [], 0
    with capture_activations(modules, reduce="none") as stores:
        for x0, y in loader:
            if n_done >= n_samples:
                break
            x0, y = x0.to(device), y.to(device)
            B = x0.size(0)
            null = torch.full_like(y, 10)
            t1 = torch.ones(B, device=device)
            x_T, _, _ = fwd.corrupt(x0, t1, y=y)
            c_in = fwd.c_in(t1).view(-1, 1, 1, 1)
            model(x_T * c_in, t1, null)
            raw_list.append(x0.detach().cpu())
            n_done += B

    acts = {k: s.get()[:n_samples] for k, s in stores.items()}
    inp = torch.cat(raw_list, 0)[:n_samples]          # (N,1,28,28)
    return inp, acts


def gamma_views_rows(model, fwd, test_ds, cfg, model_name: str, logger=None,
                     n_samples: int = 1024) -> list[dict]:
    """
    Build per-layer four-view kappa4 rows (+ an `input` row) for one model.
    Also logs (Q3) diagnostics for the input and bottleneck stages.
    """
    inp, acts = capture_layer_views(model, fwd, test_ds, cfg, n_samples=n_samples)
    rows: list[dict] = []

    # (Q2) input point. Treat the image as a 4D map so all four views are defined.
    log_kappa4_diagnostics("input(raw x0)", inp, logger)
    iv = kappa4_three_views(inp)            # 4D -> mean/center/unit/channels
    rows.append({
        "noise_type": model_name, "depth_index": -1, "layer_label": "input",
        "k4_mean": iv["mean"]["kappa4"], "k4_center": iv["center"]["kappa4"],
        "k4_unit": iv["unit"]["kappa4"], "k4_channels": iv["channels"]["kappa4"],
    })

    for depth, key in enumerate(UNET_LAYER_KEYS):
        a = acts.get(key, torch.empty(0))
        if a.numel() == 0:
            continue
        v = kappa4_three_views(a)
        rows.append({
            "noise_type": model_name, "depth_index": depth,
            "layer_label": LAYER_LABELS.get(key, key),
            "k4_mean": v["mean"]["kappa4"], "k4_center": v["center"]["kappa4"],
            "k4_unit": v["unit"]["kappa4"], "k4_channels": v["channels"]["kappa4"],
        })
        if key == "mid2":
            log_kappa4_diagnostics("bottleneck(mid2)", a, logger)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# (Q1, Q2) plot
# ─────────────────────────────────────────────────────────────────────────────
# input and out are pixel-space stages; their per-unit kappa4 is the sparse-
# pixel artifact (var-clamp on near-constant pixels) and is ~50x the feature-
# space layers. They are annotated off-axis so they do not compress the signal.
PIXEL_DEPTHS = {-1, 16}


def plot_gamma_views(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    R_COLOR, G_COLOR = "#E07B39", "#3A7EBF"
    NT_COLOR = {"Rosenblatt": R_COLOR, "Gaussian": G_COLOR,
                "rosenblatt": R_COLOR, "gaussian": G_COLOR}

    df = pd.read_csv(csv_path)
    df = df.groupby(["noise_type", "depth_index", "layer_label"]).mean(
        numeric_only=True).reset_index().sort_values("depth_index")

    order  = df.drop_duplicates("depth_index").sort_values("depth_index")
    labels = order["layer_label"].tolist()
    depths = order["depth_index"].tolist()

    panels = [("k4_unit",     "per-unit  (N, C·H·W)  — per-coordinate marginal"),
              ("k4_center",   "center cell  (N, C)"),
              ("k4_mean",     "spatial-mean / pooled  (N, C)  — CLT-suppressed"),
              ("k4_channels", "channels  (N·H·W, C)  — spatial non-stationarity")]

    # encoder / bottleneck / decoder bands keyed on depth_index
    bands = [(-0.5, 7.5, "#EFF3FA"), (7.5, 10.5, "#FFF8E1"), (10.5, 16.5, "#FCEEE8")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels):
        for lo, hi, bg in bands:
            ax.axvspan(lo, hi, color=bg, alpha=0.5, zorder=0)
        for nt, sub in df.groupby("noise_type"):
            sub = sub.sort_values("depth_index")
            ax.plot(sub["depth_index"], sub[col], marker="o", ms=4,
                    color=NT_COLOR.get(nt, "gray"), label=nt, zorder=3)
        ax.axhline(0, color="red", lw=0.8, ls="--", zorder=2)

        # autoscale to feature-space stages; annotate off-scale pixel stages
        feat = df[~df["depth_index"].isin(PIXEL_DEPTHS)][col]
        lo, hi = float(feat.min()), float(feat.max())
        pad = 0.12 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)

        # Collect annotations to prevent overlap at the same depth index
        annotations = {}  # key: (d, side), value: list of (v, nt)
        for nt, sub in df.groupby("noise_type"):
            for d in PIXEL_DEPTHS:
                r = sub[sub["depth_index"] == d]
                if r.empty:
                    continue
                v = float(r[col].iloc[0])
                if v > hi + pad:
                    annotations.setdefault((d, "top"), []).append((v, nt))
                elif v < lo - pad:
                    annotations.setdefault((d, "bottom"), []).append((v, nt))

        # Draw annotations with horizontal offsets if multiple exist at the same position
        for (d, side), items in annotations.items():
            yy = (hi + pad) if side == "top" else (lo - pad)
            va = "bottom" if side == "top" else "top"
            items = sorted(items, key=lambda x: x[0], reverse=(side == "bottom"))
            for idx, (v, nt) in enumerate(items):
                y_offset_pts = idx * 10 if side == "top" else -idx * 10
                
                ax.annotate(f"{v:.1f}", xy=(d, yy), 
                            xytext=(0, y_offset_pts), textcoords="offset points",
                            ha="center", va=va, fontsize=7, 
                            color=NT_COLOR.get(nt, "gray"),
                            annotation_clip=False)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("$\\kappa_4$")
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xticks(depths)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)

    handles = [Line2D([0], [0], color=R_COLOR, marker="o", label="Rosenblatt"),
               Line2D([0], [0], color=G_COLOR, marker="o", label="Gaussian")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "gamma_unit_vs_mean.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Notes on unit vs channels (Q1)
# ─────────────────────────────────────────────────────────────────────────────
# `unit`     : columns are individual (c,h,w); sample axis = batch (N).
#              Measures per-COORDINATE marginal non-Gaussianity across the data
#              distribution, averaged over coordinates. This is the quantity
#              relevant to "is the representation Gaussianized".
# `channels` : columns are channels c; sample axis = batch x space (N*H*W).
#              Each channel is treated as a stationary field. If the per-channel
#              field is NOT spatially stationary (location-dependent mean or
#              variance — the norm for conv feature maps), pooling space into
#              the sample axis forms a mixture and INFLATES kappa4 (a scale
#              mixture is leptokurtic). The unit/channels gap therefore measures
#              spatial non-stationarity, not per-coordinate non-Gaussianity.
# Recommendation: report `unit` (and `center`, which avoids border pixels) for
# the Gaussianization claim; treat the channels-minus-unit gap as a separate
# non-stationarity diagnostic; do not read `channels` as "the κ4 of the
# representation". `mean` (pooled) is CLT-suppressed and the least informative.