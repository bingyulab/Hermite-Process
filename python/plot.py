"""
Plots for the sharpened thesis question (difference between Rosenblatt and
Gaussian drivers). Reads the metric CSVs produced by the runners and the
discriminability CSV produced by twosample.py.

Usage:
  python plots.py --results_dir output --out_dir output/figures

It globs recursively for the known CSV filenames, so point --results_dir at
the parent of your run directories (or a single seed dir).
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rcd.experiments.gamma_views_patch import plot_gamma_views
R_COLOR = "#E07B39"   # rosenblatt
G_COLOR = "#3A7EBF"   # gaussian
NT_COLOR = {"rosenblatt": R_COLOR, "gaussian": G_COLOR,
            "Rosenblatt": R_COLOR, "Gaussian": G_COLOR}

PIXEL_STAGES = ["input", "corrupted", "mid_t05", "bottleneck", "x0hat"]
DISC_STAGES  = ["corrupted", "bottleneck_mean", "bottleneck_unit", "output"]


def _read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def _find(results_dir, name):
    hits = glob.glob(os.path.join(results_dir, "**", name), recursive=True)
    return hits[0] if hits else None


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha(path, out_dir):
    df = _read(path)
    df = df[df["cfg_stage"].isin(PIXEL_STAGES)].copy()
    df["ord"] = df["cfg_stage"].map({s: i for i, s in enumerate(PIXEL_STAGES)})
    df = df.sort_values("ord")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col, ax, title, logy in (
        ("dist_k4",       axes[0], "Marginal excess kurtosis  $\\kappa_4$", True),
        ("dist_mardia_z", axes[1], "Mardia $z$ (joint; >2 rejects normality)", True),
        ("dist_whiteness", axes[2], "Whiteness (off-diag corr.)", False),
    ):
        for nt, sub in df.groupby("noise_type"):
            ax.plot(sub["cfg_stage"], sub[col], marker="o",
                    color=NT_COLOR.get(nt, "gray"), label=nt)
        ax.set_title(title)
        ax.set_xticklabels(PIXEL_STAGES, rotation=30, ha="right")
        if logy:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].axhline(2, ls="--", color="k", lw=0.8)
    fig.suptitle("alpha: marginal $\\kappa_4$ collapses by the bottleneck, "
                 "but Mardia/whiteness show the joint law stays non-Gaussian")
    _save(fig, out_dir, "alpha_propagation.png")


def plot_gamma(path, out_dir):
    hits = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "**", "gamma.csv"), recursive=True)
    if not hits:
        hits = [path]
    dfs = []
    for p in hits:
        d = _read(p)
        if "mean_k4_unit" in d.columns:
            dfs.append(d)
    
    if not dfs:
        print("No valid gamma.csv with mean_k4_unit found.")
        return
        
    df = pd.concat(dfs).groupby(["noise_type", "depth_index", "layer_label"]).mean(numeric_only=True).reset_index().sort_values("depth_index")
    
    fig, ax = plt.subplots(figsize=(13, 5))
    for nt, sub in df.groupby("noise_type"):
        c = NT_COLOR.get(nt, "gray")
        ax.plot(sub["depth_index"], sub["mean_k4_unit"], marker="o", color=c,
                label=f"{nt} — per-unit (no pooling)")
        ax.plot(sub["depth_index"], sub["mean_k4"], marker="s", ls="--", color=c,
                alpha=0.6, label=f"{nt} — spatial-mean (pooled)")
    labels = df.drop_duplicates("depth_index").sort_values("depth_index")["layer_label"].tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.axhline(0, color="red", lw=1.0, ls="--", zorder=2)
    
    # Add background shaded regions and labels for network stages
    for s, e, lab, bg in [(0, 6.5, "Encoder", "#EFF3FA"),
                          (6.5, 10.5, "Bottleneck", "#FFF8E1"),
                          (10.5, len(labels) - 1, "Decoder", "#FCEEE8")]:
        ax.axvspan(s, e, color=bg, alpha=0.45, zorder=0)
        ax.text((s+e)/2, ax.get_ylim()[1]*0.7, lab, ha="center",
                fontsize=8, color="dimgray", style="italic")
                
    ax.set_ylabel("Mean excess kurtosis $\\bar{\\kappa}_4$ (symlog)")
    ax.set_title("gamma: per-unit vs spatial-mean $\\kappa_4$. "
                 "Pooling hides early-layer/output non-Gaussianity; both agree (\u22480) only at the bottleneck.")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    _save(fig, out_dir, "gamma_unit_vs_mean.png")


def plot_theta(path, out_dir):
    # Retrieve all matched paths instead of just the first one if we need to
    # but path here is a single string. Let's find all theta.csv that have the columns.
    base_dir = path
    for _ in range(3):
        base_dir = os.path.dirname(base_dir)
    if not base_dir:
        base_dir = "."
    hits = glob.glob(os.path.join(base_dir, "**", "theta.csv"), recursive=True)
    if not hits:
        hits = [path]
    dfs = []
    for p in hits:
        d = _read(p)
        if "kappa4_unit" in d.columns:
            dfs.append(d)
            
    if not dfs:
        print("No valid theta.csv with kappa4_unit found.")
        return
        
    df = pd.concat(dfs).groupby("cfg_t_value").mean(numeric_only=True).reset_index().sort_values("cfg_t_value")
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, mk, lab, color in (("kappa4_unit", "o", "per-unit", "#E07B39"),
                                ("kappa4_center", "^", "center cell", "#3A7EBF"),
                                ("kappa4_mean", "s", "spatial-mean", "#42a861"),
                                ("kappa4_channels", "D", "channels", "#9b59b6"),
                                ):
        if col in df:
            ax.plot(df["cfg_t_value"], df[col], marker=mk, label=lab, color=color)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("corruption level $t$")
    ax.set_ylabel("bottleneck $\\kappa_4$")
    ax.set_title("theta: bottleneck $\\kappa_4$ stays near 0 across $t$ (Rosenblatt)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir, "theta_t.png")


def plot_generation(path, out_dir):
    df = _read(path).sort_values("cfg_n_steps")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, ax, title in (("FID", axes[0], "FID vs sampling steps"),
                           ("fFID", axes[1], "fFID vs sampling steps")):
        for nt, sub in df.groupby("noise_type"):
            ax.plot(sub["cfg_n_steps"], sub[col], marker="o",
                    color=NT_COLOR.get(nt, "gray"), label=nt)
        ax.set_xlabel("n_steps")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("generation: Rosenblatt \u2248 Gaussian (no quality difference)")
    _save(fig, out_dir, "generation_fid.png")


def plot_discriminability(path, out_dir):
    df = _read(path)
    df = df[df["stage"].isin(DISC_STAGES)].copy()
    df["ord"] = df["stage"].map({s: i for i, s in enumerate(DISC_STAGES)})
    df = df.sort_values("ord")
    nets = sorted(df["net"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(nets), 1)
    x = range(len(DISC_STAGES))
    for j, net in enumerate(nets):
        sub = df[df["net"] == net].set_index("stage").reindex(DISC_STAGES)
        xs = [i + j * width for i in x]
        bars = ax.bar(xs, sub["auc"], width=width,
                      color=NT_COLOR.get(net, "gray"), alpha=0.85,
                      label=f"net trained on {net}")
        for xi, (_, row) in zip(xs, sub.iterrows()):
            if pd.notna(row.get("p_value")):
                ax.text(xi, (row["auc"] if pd.notna(row["auc"]) else 0.5) + 0.01,
                        f"p={row['p_value']:.3f}", ha="center", va="bottom",
                        fontsize=7)
    ax.axhline(0.5, ls="--", color="k", lw=1, label="chance (AUC=0.5)")
    ax.set_xticks([i + width * (len(nets) - 1) / 2 for i in x])
    ax.set_xticklabels(DISC_STAGES, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("discriminator AUC (R vs G driver)")
    ax.set_title("Driver discriminability, fixed network. "
                 "AUC\u21920.5 \u21d2 the network does not register the driver.")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "discriminability.png")


def plot_delta(path, out_dir):
    df = _read(path)
    nts = sorted(df["noise_type"].unique())
    fig, axes = plt.subplots(1, len(nts), figsize=(6 * len(nts), 4.5), squeeze=False)
    for ax, nt in zip(axes[0], nts):
        sub = df[df["noise_type"] == nt]
        clean_val = None
        
        # Try to find the clean baseline value
        for pert, s in sub.groupby("perturbation"):
            if pert == "clean":
                s = s.sort_values("sigma")
                if len(s) > 0:
                    clean_val = s["huber_loss"].iloc[0]
                break
                
        for pert, s in sub.groupby("perturbation"):
            s = s.sort_values("sigma")
            ax.plot(s["sigma"], s["huber_loss"], marker=".", label=pert)
            
        # Shade the ±1% band around clean if we found it
        if clean_val is not None:
            ax.axhspan(clean_val * 0.99, clean_val * 1.01, color="lightgreen",
                       alpha=0.25, label="±1% of clean")
                       
        ax.set_title(f"{nt} model")
        ax.set_xlabel("$\\sigma$ (in bottleneck-std units)")
        ax.set_ylabel("Huber reconstruction loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, out_dir, "delta_rigidity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="output")
    ap.add_argument("--out_dir", default="output/figures")
    args = ap.parse_args()

    jobs = [
        ("alpha.csv", plot_alpha),
        ("gamma.csv", plot_gamma),
        ("gamma_views.csv", plot_gamma_views),
        ("theta.csv", plot_theta),
        ("generation_steps_sweep.csv", plot_generation),
        ("discriminability.csv", plot_discriminability),
        ("delta.csv", plot_delta),
    ]
    for name, fn in jobs:
        p = _find(args.results_dir, name)
        if p is None:
            print(f"[skip] {name} not found under {args.results_dir}")
            continue
        try:
            print(f"[plot] {name}  ({p})")
            fn(p, args.out_dir)
        except Exception as e:
            print(f"[error] {name}: {e}")


if __name__ == "__main__":
    main()