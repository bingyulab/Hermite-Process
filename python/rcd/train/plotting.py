"""
Generic plot utilities.

This module no longer contains any experiment-specific plot logic.
Experiment-specific plotting (e.g. _plot_beta, _plot_mu, _plot_rho,
plot_optimizer_summary, plot_ablation_summary) lives next to the
experiment in `rcd.experiments._plots`.

Generic primitives provided here:
  * _finalize_and_save          — dual-format figure export
  * _style_axis                 — consistent grid/baseline rendering
  * _load_csv_records           — CSV → list-of-objects parsing
  * plot_kappa4_violins         — distribution-of-cumulants comparison plot
  * plot_layer_profiles         — model-vs-layer-depth curve plot
  * plot_all_sigma_patterns     — visualise Σ(x₀) patterns across factories
  * plot_restoration_grid       — denoising trajectory grid (UNet only)

All experiment-specific symbols (COLORS, MARKERS, …) come from
rcd.experiments.registry — this module never defines its own.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rcd.data.config import Config
from rcd.data.datasets import _NORM_TF, _get_dataset, class_name
from rcd.train.training import generate_samples
from rcd.experiments.registry import (
    COLORS, UNET_LAYER_KEYS, LAYER_LABELS,
)


# =============================================================================
# 1. Save / style primitives
# =============================================================================

def _finalize_and_save(fig, target_path: Path,
                       filename_if_dir: Optional[str] = None,
                       dpi: int = 160) -> None:
    """Tight layout, dual-format (pdf + png) export, canvas cleanup."""
    target_path = Path(target_path)
    plt.tight_layout()
    if target_path.suffix in (".png", ".pdf"):
        base_path = target_path.with_suffix("")
    else:
        target_path.mkdir(parents=True, exist_ok=True)
        if filename_if_dir is None:
            raise ValueError("filename_if_dir required when target_path is a directory")
        base_path = target_path / filename_if_dir

    for ext in ("pdf", "png"):
        fig.savefig(base_path.with_suffix(f".{ext}"),
                    bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _style_axis(ax, title: str, ylabel: str = "",
                hline_at: Optional[float] = 0.0,
                grid_axis: str = "both") -> None:
    if hline_at is not None:
        ax.axhline(hline_at, color="red", lw=1.0, ls="--", alpha=0.6)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(axis=grid_axis, alpha=0.3)


# =============================================================================
# 2. Generic CSV loader
# =============================================================================

def _load_csv_records(save_dir: Path, filename: str,
                       record_cls=None) -> list:
    path = save_dir / filename
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            processed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    processed[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except (ValueError, AttributeError):
                    processed[k] = v
            if record_cls is not None:
                rows.append(record_cls(**processed))
            else:
                rows.append(type("CSVRecord", (), processed)())
    return rows


# =============================================================================
# 3. Generic comparison plots
# =============================================================================

def plot_kappa4_violins(all_k4: dict[str, dict[str, np.ndarray]],
                         save_path: Path) -> None:
    """Per-component κ4 distribution across pipeline stages."""
    models = list(all_k4.keys())
    stages = sorted({s for m in all_k4.values() for s in m})
    n_stages = len(stages)
    pos = np.arange(len(models))

    fig, axes = plt.subplots(1, n_stages, figsize=(max(3 * n_stages, 14), 5))
    axes = np.atleast_1d(axes)

    for ax, stage in zip(axes, stages):
        data = [all_k4[m].get(stage, np.array([])) for m in models]
        valid = [d[np.isfinite(d)] for d in data if len(d[np.isfinite(d)]) > 0]
        if valid:
            vp = ax.violinplot(valid, positions=pos[:len(valid)],
                                showmedians=True, showextrema=False)
            for pc, m in zip(vp["bodies"], models):
                pc.set_facecolor(COLORS.get(m, "#888"))
                pc.set_alpha(0.75)
            vp["cmedians"].set_color("k")
        _style_axis(ax, stage,
                    ylabel=r"$\kappa_4$" if ax is axes[0] else "")
        ax.set_xticks(pos)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)

    fig.suptitle(r"Per-component $\kappa_4$ distribution across stages",
                 fontsize=10)
    _finalize_and_save(fig, save_path, dpi=150)


def plot_layer_profiles(rows: list, plot_dir: Path,
                         metrics: Optional[List[tuple]] = None) -> None:
    """
    Model-vs-layer-depth curves.
    `rows` must expose `.model`, `.layer_key`, and the attributes listed in
    `metrics` (default: mean_k4, pr, whiteness, mardia_b2p_z).
    """
    if metrics is None:
        metrics = [
            ("mean_k4",      r"Mean $\kappa_4$",                  "kappa4_profile"),
            ("pr",           "Participation Ratio (PR)",          "pr_profile"),
            ("whiteness",    "Covariance whiteness",              "whiteness_profile"),
            ("mardia_b2p_z", r"Mardia-$Z$",                       "mardiaZ_profile"),
        ]

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_names = sorted({r.model for r in rows})
    labels = [LAYER_LABELS.get(k, k) for k in UNET_LAYER_KEYS]

    for attr, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=(12, 4))
        for mname in model_names:
            sub = {r.layer_key: getattr(r, attr) for r in rows if r.model == mname}
            vals = [sub.get(k, float("nan")) for k in UNET_LAYER_KEYS]
            ax.plot(range(len(UNET_LAYER_KEYS)), vals, marker="o",
                    color=COLORS.get(mname, "#888"), linewidth=2, label=mname)
        if attr == "mean_k4":
            ax.axhline(0.0, color="red", lw=1.0, ls="--",
                        label=r"$\kappa_4=0$")
        ax.axvline(UNET_LAYER_KEYS.index("mid2"), color="gray",
                    lw=1.0, ls=":", alpha=0.6, label="Bottleneck (mid2)")
        ax.set_xticks(range(len(UNET_LAYER_KEYS)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} vs layer depth", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _finalize_and_save(fig, plot_dir / f"{stem}.png", dpi=150)


# =============================================================================
# 4. Diagnostic plots — Sigma patterns, restoration trajectories
# =============================================================================

@torch.no_grad()
def plot_all_sigma_patterns(sigma_fns: Iterable[Any],
                              save_path: str | Path,
                              dataset_name: str = "FashionMNIST",
                              example_classes: Optional[List[int]] = None) -> None:
    if example_classes is None:
        example_classes = [0]
    ds = _get_dataset(dataset_name, train=False, tf=_NORM_TF)
    sigma_fns = list(sigma_fns)

    found: dict[int, torch.Tensor] = {}
    for i in range(len(ds)):
        lb = ds[i][1]
        if lb in example_classes and lb not in found:
            found[lb] = ds[i][0].unsqueeze(0)
        if len(found) == len(example_classes):
            break

    n_rows, n_cols = len(example_classes), 1 + len(sigma_fns)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cls in enumerate(example_classes):
        x0 = found[cls]
        axes[row, 0].imshow((x0[0, 0].numpy() + 1) / 2,
                             cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(class_name(dataset_name, cls),
                                  fontsize=9, rotation=0, labelpad=42, va="center")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].axis("off")

        for col, sfn in enumerate(sigma_fns):
            y_dummy = torch.tensor([cls], dtype=torch.long)
            S = (sfn(x0, y_dummy)
                  if getattr(sfn, "needs_label", False)
                  else sfn(x0))
            S_np = S[0, 0].numpy()
            ax = axes[row, col + 1]
            im = ax.imshow(S_np, cmap="hot",
                            vmin=float(S_np.min()), vmax=float(S_np.max()))
            if row == 0:
                eg2 = getattr(sfn, "eg2", float((S ** 2).mean()))
                ax.set_title(f"{sfn.label}\n$E[\\Sigma^2]={eg2:.2f}$", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(r"Per-pixel noise coefficient $\Sigma(\mathbf{x}_0)$",
                  fontsize=11, y=1.01)
    _finalize_and_save(fig, save_path, dpi=140)


@torch.no_grad()
def plot_restoration_grid(model, forward, cfg: Config,
                           save_path: str | Path, tag: str = "",
                           bridge: str = "stochastic") -> None:
    model.eval()
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)

    found: dict[int, int] = {}
    for i in range(len(test_ds)):
        lb = test_ds[i][1]
        if lb not in found:
            found[lb] = i
        if len(found) == 10:
            break

    x0  = torch.stack([test_ds[found[c]][0] for c in range(10)]).to(cfg.device)
    lbl = torch.arange(10, device=cfg.device)
    null = torch.full_like(lbl, 10)
    xc, _, _ = forward.corrupt(x0, torch.ones(10, device=cfg.device), y=lbl)

    sched   = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=cfg.device)
    save_at = set([0] + [int(round(cfg.n_steps * i / (cfg.n_display - 1)))
                          for i in range(cfg.n_display)] + [cfg.n_steps - 1])
    x_cur, hist = xc.clone(), {}

    for k in range(cfg.n_steps):
        tc, tn = sched[k].expand(10), sched[k + 1].expand(10)
        c_in = forward.c_in(tc).view(-1, 1, 1, 1)
        x0c = model(x_cur * c_in, tc, lbl).float()
        x0u = model(x_cur * c_in, tc, null).float()
        x0h = (x0u + cfg.cfg_scale * (x0c - x0u)).clamp(-1.0, 1.0)
        if k + 1 in save_at:
            hist[k + 1] = x0h.cpu()
        if k < cfg.n_steps - 1:
            if bridge == "stochastic":
                x_cur = forward.recorrupt_stochastic(x0h, tn, y=lbl)
            elif bridge == "hybrid":
                x_cur = forward.recorrupt_hybrid(x_cur, x0h, tc, tn, y=lbl)
            elif bridge == "deterministic":
                x_cur = forward.recorrupt_deterministic(x_cur, x0h, tc, tn)
            else:
                raise ValueError(f"Unknown bridge: {bridge!r}")
        else:
            x_cur = x0h

    snap_keys = sorted(hist.keys())
    snaps = [hist[k] for k in snap_keys]
    n_cols = 2 + len(snaps)
    fig, axes = plt.subplots(10, n_cols, figsize=(2.0 * n_cols, 14))
    for i in range(10):
        axes[i, 0].imshow((x0[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow((xc[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        for col, snap in enumerate(snaps):
            axes[i, col + 2].imshow((snap[i, 0] + 1) / 2,
                                      cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(class_name(cfg.dataset, i),
                                fontsize=8, rotation=0, labelpad=40, va="center")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 1].set_title("Corrupted\nt=1", fontsize=7)
    for col, k in enumerate(snap_keys):
        axes[0, col + 2].set_title(f"t={1.0 - k / cfg.n_steps:.2f}\nstep {k}",
                                     fontsize=7)
    fig.suptitle(f"Restoration ({cfg.n_steps} steps) — {tag}\n{forward.label}",
                  fontsize=10)
    _finalize_and_save(fig, save_path, dpi=120)


@torch.no_grad()
def plot_input_diversity_grid(
    model: nn.Module,
    forward: Any,
    cfg: Config,
    save_path: str | Path,
    bridge: str = "stochastic",
    ae: Optional[nn.Module] = None,
    target_class: int = 0
) -> None:
    """Generates a scannable grid showing output variations given diverse uncorrupted inputs."""
    import matplotlib.pyplot as plt
    model.eval()
    if ae is not None:
        ae.eval()

    # 1. Fetch 3 distinct clean images from the target class dataset
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = []
    for i in range(len(test_ds)):
        img, lb = test_ds[i]
        if lb == target_class:
            real_imgs.append(img)
            if len(real_imgs) == 3:
                break
    while len(real_imgs) < 3:
        real_imgs.append(torch.zeros((1, 28, 28)))
    real_tensor = torch.stack(real_imgs).to(cfg.device)

    # 2. Generate 3 synthetic geometric shapes (scaled to range [-1.0, 1.0])
    sq_img = torch.full((1, 28, 28), -1.0)
    sq_img[0, 7:21, 7:21] = 1.0

    cross_img = torch.full((1, 28, 28), -1.0)
    cross_img[0, 13:15, :] = 1.0
    cross_img[0, :, 13:15] = 1.0

    circle_img = torch.full((1, 28, 28), -1.0)
    for r in range(28):
        for c in range(28):
            if (r - 14)**2 + (c - 14)**2 <= 49:
                circle_img[0, r, c] = 1.0
    shapes_tensor = torch.stack([sq_img, cross_img, circle_img]).to(cfg.device)

    # Combine into 6 unique context inputs
    x_in_batch = torch.cat([real_tensor, shapes_tensor], dim=0)
    n_inputs = x_in_batch.shape[0]
    labels = torch.full((n_inputs,), target_class, dtype=torch.long, device=cfg.device)

    # Execute generation process
    outputs = generate_samples(
        model=model, fwd=forward, labels=labels, cfg=cfg,
        bridge=bridge, x_in=x_in_batch, ae=ae
    )

    # Plot configuration layout
    fig, axes = plt.subplots(2, n_inputs, figsize=(2.2 * n_inputs, 4.5))
    col_names = ["Real Img 1", "Real Img 2", "Real Img 3", "Square Shape", "Cross Shape", "Circle Shape"]

    for i in range(n_inputs):
        # Row 1: Uncorrupted Input
        axes[0, i].imshow((x_in_batch[i, 0].cpu() + 1.0) / 2.0, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(col_names[i], fontsize=9)
        
        # Row 2: Model output
        axes[1, i].imshow(outputs[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        
        for ax in [axes[0, i], axes[1, i]]:
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("Initial Input (x_in)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Generated Output", fontsize=10, fontweight="bold")
    
    fig.suptitle(f"Input Sensitivity Profile (Class {target_class}) — Bridge: {bridge}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    
    # Finalize visual asset via system utility
    _finalize_and_save(fig, save_path, dpi=120)


def plot_rigidity(results: dict[str, dict[float, float]], save_path: Path, title: str = "") -> None:
    σ_vals = sorted(next(iter(results.values())).keys())
    colors = {
        "clean":      ("gray",    "--", "Clean (no perturb.)"),
        "gaussian":   ("#4C72B0", "-",  "Gaussian A"),
        "laplace":    ("#55A868", "-",  "Laplace B"),
        "rosenblatt": ("#DD8452", "-",  "Rosenblatt C"),
        "student_t3": ("#C44E52", "-",  "Student-t(3) D"),
    }
    
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, (col, ls, lab) in colors.items():
        if key in results:
            ax.plot(σ_vals, [results[key].get(σ, float("nan")) for σ in σ_vals], 
                    color=col, linestyle=ls, marker="o", linewidth=2, label=lab)
        
    ax.set(xlabel=r"Perturbation scale $\sigma$", ylabel="Huber reconstruction loss", 
           title=title or "Latent Perturbation Rigidity Test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_beta_rigidity_grid(rows: list[Any], save_path: Path, sigma_label: str = "0.5", silent: bool = False) -> None:
    panels = [
        ("perturb_gauss_huber", "Gaussian"), ("perturb_laplace_huber", "Laplace"),
        ("perturb_rosenblatt_huber", "Rosenblatt"), ("perturb_t3_huber", "Student-t(3)")
    ]
    styles = {"gaussian": ("#4C72B0", "o"), "rosenblatt": ("#DD8452", "s")}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    
    for ax, (attr, name) in zip(axes.flatten(), panels):
        for model_name, (color, marker) in styles.items():
            sub = sorted([r for r in rows if r.noise_type == model_name], key=lambda r: r.bottleneck_factor)
            if sub:
                ax.plot([r.bottleneck_factor for r in sub], [getattr(r, attr) for r in sub],
                        color=color, marker=marker, linewidth=2, label=model_name.capitalize())

        ax.set_title(f"Perturbation: {name}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0", "3.0"])

    for ax in axes[:, 0]: ax.set_ylabel("Huber loss")
    for ax in axes[1, :]: ax.set_xlabel("Bottleneck factor")

    fig.suptitle(f"Experiment beta rigidity by factor and model (sigma={sigma_label})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
