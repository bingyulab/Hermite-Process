"""
Experiment-specific plot functions.

These functions live next to the experiments, not in the generic
`rcd.train.plotting` module. They consume `ExperimentRecord`-derived rows
and CSV records produced by experiments, and they call into the generic
primitives (_finalize_and_save, _style_axis) plus the label/colour maps
from `rcd.experiments.registry`.

Each function follows the same signature:
    plot_<x>(rows: list, save_path: Path, **kwargs) -> None

No experiment-specific logic exists inside `rcd.train.plotting` after
this split. No labels, colours, or markers are defined here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np

from rcd.experiments.registry import (
    COLORS, DECODER_KEYS, DECODER_LABELS, LINESTYLES, MARKERS,
    OPT_LABELS, SKIP_VARIANTS,
)
from rcd.train.plotting import _finalize_and_save, _load_csv_records, _style_axis

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ε, ζ, κ — Generic Bar Charts for Ablation Studies
# =============================================================================

def plot_ablation_bars(rows: list, save_dir: Path, experiment_type: str, title: str, filename: str) -> None:
    """Generic bar chart plotter for ε (loss), ζ (norm), and κ (act) ablations."""
    # Filter rows by experiment type
    exp_rows = [r for r in rows if getattr(r, "experiment_type", None) == experiment_type]
    if not exp_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _style_axis(axes[0], f"Bottleneck $\\kappa_4$ across {title}", r"Bottleneck $\bar\kappa_4$")
    _style_axis(axes[1], f"Reconstruction Quality across {title}", r"Validation $L_1$ Loss")

    for nt_idx, noise_type in enumerate(("gaussian", "rosenblatt")):
        sub = [r for r in exp_rows if getattr(r, "noise_type", None) == noise_type]
        if not sub:
            continue
        
        labels = [r.label for r in sub]
        k4_vals = [r.dist.k4 for r in sub]
        l1_vals = [r.loss.l1 for r in sub]
        
        x = np.arange(len(labels))
        width = 0.35
        offset = -width/2 if nt_idx == 0 else width/2
        
        c = COLORS.get(noise_type, "blue")
        axes[0].bar(x + offset, k4_vals, width, label=noise_type.capitalize(), color=c, alpha=0.8)
        axes[1].bar(x + offset, l1_vals, width, label=noise_type.capitalize(), color=c, alpha=0.8)
        
        # Set ticks only once
        if nt_idx == 0 or len(axes[0].get_xticks()) == 0:
            for ax in axes:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    # Highlight Gaussianity threshold on the kurtosis plot
    axes[0].fill_between([-1, len(labels)], -0.1, 0.1, color="lightgreen", alpha=0.2, label=r"$|\kappa_4|<0.1$")
    axes[0].set_xlim(-0.5, len(labels) - 0.5)

    for ax in axes:
        ax.legend(fontsize=8)
        
    fig.suptitle(f"Experiment {experiment_type} — {title} Ablation", fontsize=11)
    _finalize_and_save(fig, save_dir, filename_if_dir=filename)


# =============================================================================
# β — Bottleneck width curves
# =============================================================================

def plot_beta_curves(rows: list, save_path: Path) -> None:
    """Four-panel β figure: κ4(bf), relative κ4, PR, rigidity."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for noise_type in ("gaussian", "rosenblatt"):
        sub = sorted(
            [r for r in rows if getattr(r, "noise_type", None) == noise_type],
            key=lambda r: r.bottleneck_factor,
        )
        if not sub:
            continue
        bfs    = [r.bottleneck_factor for r in sub]
        k4_bn  = [r.mean_k4_bneck for r in sub]
        k4_in  = [r.mean_k4_input for r in sub]
        k4_rel = [(bn / inp if abs(inp) > 1e-6 else float("nan"))
                   for bn, inp in zip(k4_bn, k4_in)]
        c, mk, lbl = COLORS[noise_type], MARKERS[noise_type], noise_type.capitalize()

        axes[0, 0].plot(bfs, k4_bn,  c=c, marker=mk, lw=2, label=lbl)
        axes[0, 1].plot(bfs, k4_rel, c=c, marker=mk, lw=2, label=lbl)
        axes[1, 0].plot(bfs, [r.pr_bneck for r in sub], c=c, marker=mk, lw=2, label=lbl)
        axes[1, 1].plot(bfs, [r.perturb_gauss_huber for r in sub],
                         c=c, marker=mk, lw=2, ls="-",  label=f"{lbl} Gauss")
        axes[1, 1].plot(bfs, [r.perturb_rosenblatt_huber for r in sub],
                         c=c, marker=mk, lw=2, ls="--", label=f"{lbl} Rosen")

    _style_axis(axes[0, 0], r"$\kappa_4$ at bottleneck",       r"$\bar\kappa_4$")
    _style_axis(axes[0, 1], "Relative Gaussianization",
                r"$\kappa_4^{\rm bn}/\kappa_4^{\rm input}$", hline_at=0.0)
    axes[0, 1].axhline(1, color="gray", lw=1, ls=":", label="no change")
    _style_axis(axes[1, 0], "Participation Ratio (PR)", "PR", hline_at=None)
    _style_axis(axes[1, 1], r"Rigidity (Huber, $\sigma=0.5$)", "Huber loss", hline_at=None)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0", "3.0"])
        ax.set_xlabel(r"Bottleneck factor $\alpha_{\rm bf}$", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(r"Experiment $\beta$: Bottleneck width vs Gaussianization", fontsize=11)
    _finalize_and_save(fig, save_path, dpi=150)


# =============================================================================
# μ — Skip-connection decoder κ4 profile
# =============================================================================

def plot_mu_skip(rows: list, save_dir: Path) -> None:
    """Per-noise-type μ figure: decoder κ4 profile + bottleneck/L1 bars."""
    variant_order = list(SKIP_VARIANTS.keys())
    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if getattr(r, "noise_type", None) == noise_type]
        if not sub:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        _style_axis(
            ax,
            f"Decoder $\\kappa_4$ profile ({noise_type})",
            r"$\bar\kappa_4$ at decoder layer",
        )
        for var in variant_order:
            vrows = [r for r in sub if r.config.get("variant") == var]
            if not vrows:
                continue
            vals = [vrows[0].extras.get(f"dec_{k}_k4", float("nan")) for k in DECODER_KEYS]
            ax.plot(range(len(DECODER_KEYS)), vals, marker="o", ms=5,
                     color=COLORS[var], ls=LINESTYLES[var], lw=1.8,
                     label=SKIP_VARIANTS[var])
        ax.fill_between(range(len(DECODER_KEYS)), -0.1, 0.1,
                          color="lightgreen", alpha=0.2, label=r"$|\kappa_4|<0.1$")
        ax.set_xticks(range(len(DECODER_KEYS)))
        ax.set_xticklabels(DECODER_LABELS, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")

        ax2 = axes[1]
        _style_axis(ax2, f"Bottleneck $\\kappa_4$ + recon. loss ({noise_type})",
                     r"Bottleneck $\bar\kappa_4$")
        valid_v = [v for v in variant_order
                    if [r for r in sub if r.config.get("variant") == v]]
        v_rows = [next(r for r in sub if r.config.get("variant") == v) for v in valid_v]
        labels = [SKIP_VARIANTS[v] for v in valid_v]
        x = np.arange(len(labels))
        ax2.bar(x - 0.2, [r.dist.k4 for r in v_rows], 0.35,
                  color=[COLORS[v] for v in valid_v], alpha=0.8)
        ax2_r = ax2.twinx()
        ax2_r.bar(x + 0.2, [r.loss.l1 for r in v_rows], 0.35,
                    color=[COLORS[v] for v in valid_v], alpha=0.4)
        ax2_r.set_ylabel(r"$L_1$ reconstruction", color="gray")
        ax2_r.tick_params(axis="y", labelcolor="gray")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)

        fig.suptitle(f"Experiment μ — {noise_type} model", fontsize=10)
        _finalize_and_save(fig, save_dir, filename_if_dir=f"mu_skip_{noise_type}")


# =============================================================================
# θ — Time-conditional κ4
# =============================================================================

def plot_theta_time(rows: list, save_dir: Path) -> None:
    """θ figure: κ4 and Mardia-Z vs corruption time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t_arr = sorted({r.config.get("t_value") for r in rows
                    if r.config.get("t_value") is not None})
    _style_axis(axes[0], r"$\kappa_4$ at bottleneck vs $t$",
                 r"Bottleneck $\bar\kappa_4$")
    _style_axis(axes[1], "Mardia-$Z$ at bottleneck vs $t$", "Mardia-$Z$")

    for noise_type in ("gaussian", "rosenblatt"):
        sub = sorted(
            [r for r in rows if getattr(r, "noise_type", None) == noise_type],
            key=lambda r: r.config.get("t_value", 0.0),
        )
        if not sub:
            continue
        tv = [r.config["t_value"] for r in sub]
        axes[0].plot(tv, [r.dist.k4 for r in sub],
                       color=COLORS[noise_type], marker=MARKERS[noise_type],
                       lw=2, label=noise_type.capitalize())
        axes[1].plot(tv, [r.dist.mardia_z for r in sub],
                       color=COLORS[noise_type], marker=MARKERS[noise_type],
                       lw=2, label=noise_type.capitalize())

    for ax in axes:
        if t_arr:
            ax.fill_between(t_arr, -0.1, 0.1, color="lightgreen", alpha=0.2)
        ax.set_xlabel("Corruption time $t$")
        ax.legend(fontsize=8)
    fig.suptitle("Experiment θ — Time-conditional bottleneck Gaussianization",
                  fontsize=10)
    _finalize_and_save(fig, save_dir, filename_if_dir="theta_time_kappa4")


# =============================================================================
# π — Gradient noise distribution
# =============================================================================

def plot_pi_grad_noise(rows: list, save_dir: Path) -> None:
    """π figure: bottleneck κ4 and val L1 vs gradient noise σ."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _style_axis(axes[0], "Bottleneck $\\kappa_4$ vs gradient noise σ",
                 r"$\bar\kappa_4$ at bottleneck")
    _style_axis(axes[1], "Reconstruction quality vs gradient noise σ",
                 r"$L_1$ val loss")

    dist_colors = {"clean": "gray", "gaussian": "#4C72B0",
                   "rosenblatt": "#DD8452", "rosenblatt_product": "#C44E52",
                   "laplace": "#55A868"}
    ls_map = {"gaussian": "-", "rosenblatt": "--"}

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if getattr(r, "noise_type", None) == noise_type]
        dists = sorted({r.config.get("noise_dist") for r in sub
                          if r.config.get("noise_dist") is not None})
        for dist in dists:
            dsub = sorted(
                [r for r in sub if r.config.get("noise_dist") == dist],
                key=lambda r: r.config.get("noise_std", 0.0),
            )
            if not dsub:
                continue
            stds = [r.config["noise_std"] for r in dsub]
            col = dist_colors.get(dist, "black")
            ls  = ls_map.get(noise_type, "-")
            axes[0].plot(stds, [r.dist.k4 for r in dsub],
                           marker="o", lw=1.8, color=col, linestyle=ls,
                           label=f"{noise_type}/{dist}")
            axes[1].plot(stds, [r.loss.l1 for r in dsub],
                           marker="o", lw=1.8, color=col, linestyle=ls,
                           label=f"{noise_type}/{dist}")

    for ax in axes:
        ax.set_xlabel("Gradient noise σ")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Experiment π — Gradient noise distribution", fontsize=10)
    _finalize_and_save(fig, save_dir, filename_if_dir="pi_grad_noise")


# =============================================================================
# ρ — Before/after Rosenblatt-SGLD
# =============================================================================

def plot_rho_landscape(rows: list, save_dir: Path) -> None:
    """ρ figure: before→after sharpness and κ4 arrow plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = {
        "none":               "gray",
        "gaussian":           COLORS["gaussian"],
        "rosenblatt_product": COLORS["rosenblatt"],
    }

    for nt_idx, noise_type in enumerate(("gaussian", "rosenblatt")):
        sub = [r for r in rows if getattr(r, "noise_type", None) == noise_type]
        if not sub:
            continue
        y = nt_idx
        for grad_noise, col in palette.items():
            bf = next((r for r in sub
                       if r.config.get("phase") == "before"
                          and r.config.get("grad_noise") == grad_noise), None)
            af = next((r for r in sub
                       if r.config.get("phase") == "after"
                          and r.config.get("grad_noise") == grad_noise), None)
            if bf is None or af is None:
                continue
            for ax_idx, (bv, av) in enumerate([
                (bf.optim.sharpness, af.optim.sharpness),
                (bf.dist.k4,         af.dist.k4),
            ]):
                axes[ax_idx].annotate(
                    "", xy=(av, y), xytext=(bv, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5),
                )
                axes[ax_idx].plot([bv], [y], "o", color=col, ms=6)
                axes[ax_idx].plot([av], [y], "s", color=col, ms=6)

    _style_axis(axes[0], "Sharpness before→after (ρ)", "Sharpness",
                 hline_at=None, grid_axis="x")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Gaussian data", "Rosenblatt data"])
    _style_axis(axes[1], "$\\kappa_4$ before→after (ρ)", r"Bottleneck $\bar\kappa_4$",
                 hline_at=0.0, grid_axis="x")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Gaussian data", "Rosenblatt data"])
    fig.suptitle("Experiment ρ — Rosenblatt-SGLD landscape effects", fontsize=10)
    _finalize_and_save(fig, save_dir, filename_if_dir="rho_landscape")

# =============================================================================
# ο — Optimizer Comparison Scatter Plot
# =============================================================================

def plot_omicron_landscape(rows: list, save_dir: Path) -> None:
    """ο figure: Sharpness vs Bottleneck κ4 for different optimizers."""
    exp_rows = [r for r in rows if getattr(r, "experiment_type", None) == "omicron"]
    if not exp_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    _style_axis(ax, "Optimizer Landscape: Sharpness vs Gaussianization", "Sharpness")
    ax.set_ylabel(r"Bottleneck $\bar\kappa_4$")

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in exp_rows if getattr(r, "noise_type", None) == noise_type]
        if not sub:
            continue
        
        c = COLORS.get(noise_type, "black")
        mk = MARKERS.get(noise_type, "o")
        
        for r in sub:
            # Check if sharpness and k4 are valid floats
            sharpness = getattr(r.optim, "sharpness", float("nan"))
            k4 = getattr(r.dist, "k4", float("nan"))
            
            ax.scatter(sharpness, k4, color=c, marker=mk, s=120, edgecolors='white')
            ax.annotate(r.label, (sharpness, k4), xytext=(6, 6), 
                        textcoords='offset points', fontsize=8, color="#333333")

    # Add a horizontal span for Gaussian geometry
    ax.axhspan(-0.1, 0.1, color="lightgreen", alpha=0.2, zorder=0, label="Gaussian Zone ($|\kappa_4| < 0.1$)")

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=MARKERS.get('gaussian', 'o'), color='w', markerfacecolor=COLORS.get('gaussian', 'blue'), markersize=10, label='Gaussian'),
        Line2D([0], [0], marker=MARKERS.get('rosenblatt', 's'), color='w', markerfacecolor=COLORS.get('rosenblatt', 'orange'), markersize=10, label='Rosenblatt')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    fig.suptitle("Experiment ο — Optimizer Landscape Effects", fontsize=11)
    _finalize_and_save(fig, save_dir, filename_if_dir="omicron_landscape")
    

# =============================================================================
# τ — Gradient κ4 evolution during training
# =============================================================================

def plot_tau_evolution(rows: list, save_dir: Path) -> None:
    """τ figure: gradient κ4 and gradient norm vs training step per optimiser."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    by_opt: dict[str, list[dict]] = {}
    for r in rows:
        d = r if isinstance(r, dict) else dict(r.__dict__)
        by_opt.setdefault(d["opt_name"], []).append(d)

    palette = {"adamw": COLORS["gaussian"], "lion": COLORS["rosenblatt"],
                 "sgd":   COLORS["no_h1"]}

    for opt_name, log in by_opt.items():
        log = sorted(log, key=lambda d: d["step"])
        steps = [d["step"]      for d in log]
        k4s   = [d["kappa4"]    for d in log]
        norms = [d["grad_norm"] for d in log]
        col   = palette.get(opt_name, "black")
        label = OPT_LABELS.get(opt_name, opt_name).split("(")[0].strip()
        axes[0].plot(steps, k4s,   color=col, lw=1.5, label=label)
        axes[1].plot(steps, norms, color=col, lw=1.5, label=label)

    _style_axis(axes[0], r"Gradient $\kappa_4$ evolution",
                 r"Gradient $\kappa_4$ at bottleneck", hline_at=0.0)
    axes[0].axhline(-2, color="gray", lw=0.8, ls=":", alpha=0.5, label=r"$\kappa_4=-2$")
    _style_axis(axes[1], "Gradient norm evolution",
                 "Gradient L2 norm at bottleneck", hline_at=None)
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.legend(fontsize=7.5)
    fig.suptitle("Experiment τ — Gradient κ4 evolution during training", fontsize=10)
    _finalize_and_save(fig, save_dir, filename_if_dir="tau_grad_evolution")


# =============================================================================
# Summary panels (load from streamed CSVs)
# =============================================================================

def plot_ablation_summary(metric_dir: Path, plot_dir: Path) -> None:
    """Combined four-panel summary loading from streamed CSVs."""
    eps = _load_csv_records(metric_dir, "epsilon.csv")
    zet = _load_csv_records(metric_dir, "zeta.csv")
    mu  = _load_csv_records(metric_dir, "mu.csv")
    tht = _load_csv_records(metric_dir, "theta.csv")

    fig = plt.figure(figsize=(18, 10))
    gs  = mgs.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])

    _bar_by_label(ax1, [r for r in eps if r.noise_type == "rosenblatt"],
                  "dist_k4", "(A) Loss-function ablation (ε)",
                  r"$\bar\kappa_4$ at bottleneck")
    _bar_by_label(ax2, [r for r in zet if r.noise_type == "rosenblatt"],
                  "dist_k4", "(B) Normalization ablation (ζ)",
                  r"$\bar\kappa_4$ at bottleneck")
    _bar_by_label(ax3, [r for r in mu  if r.noise_type == "rosenblatt"],
                  "dist_k4", "(C) Skip-connection ablation (μ)",
                  r"$\bar\kappa_4$ at bottleneck")
    _bar_by_label(ax4, [r for r in tht if r.noise_type == "rosenblatt"],
                  "dist_k4", "(D) Time-conditional ablation (θ)",
                  r"$\bar\kappa_4$ at bottleneck")

    fig.suptitle("Ablation summary (Rosenblatt model)", fontsize=11)
    _finalize_and_save(fig, plot_dir, filename_if_dir="ablation_summary")


def plot_optimizer_summary(metric_dir: Path, plot_dir: Path) -> None:
    """Combined four-panel optimizer summary from streamed CSVs."""
    om = _load_csv_records(metric_dir, "omicron.csv")
    pi = _load_csv_records(metric_dir, "pi.csv")
    rho = _load_csv_records(metric_dir, "rho.csv")

    fig = plt.figure(figsize=(16, 9))
    gs  = mgs.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    _bar_by_label(fig.add_subplot(gs[0, 0]),
                   [r for r in om if r.noise_type == "rosenblatt"],
                   "dist_k4", "(A) Optimiser comparison (ο)",
                   r"$\bar\kappa_4$ at bottleneck")
    _bar_by_label(fig.add_subplot(gs[0, 1]),
                   [r for r in om if r.noise_type == "rosenblatt"],
                   "optim_update_w1", "(B) Update whiteness",
                   r"$W_1$ from $\mathcal{N}(0,1)$")
    _bar_by_label(fig.add_subplot(gs[1, 0]),
                   [r for r in rho if r.noise_type == "rosenblatt"],
                   "optim_sharpness", "(C) ρ sharpness (after)",
                   "Sharpness")
    _bar_by_label(fig.add_subplot(gs[1, 1]),
                   [r for r in pi  if r.noise_type == "rosenblatt"],
                   "dist_k4", "(D) Gradient noise dist. (π)",
                   r"$\bar\kappa_4$ at bottleneck")

    fig.suptitle("Optimizer summary (Rosenblatt model)", fontsize=11)
    _finalize_and_save(fig, plot_dir, filename_if_dir="optimizer_summary")


def _bar_by_label(ax, rows: list, attr: str, title: str, ylabel: str) -> None:
    """Internal helper: bar plot of `attr` keyed by row.label, in input order."""
    _style_axis(ax, title, ylabel, grid_axis="y")
    if not rows:
        return
    labels = []
    vals = []
    for r in rows:
        lab = getattr(r, "label", "").split("(")[0].strip()
        if lab not in labels:
            labels.append(lab)
            vals.append(getattr(r, attr, float("nan")))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=COLORS["rosenblatt"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)