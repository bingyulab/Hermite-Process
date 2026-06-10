"""
plot_gaussianization_results.py
================================
Produces all publication-quality figures for Experiments α, β, γ, δ.

Run:  python plot_gaussianization_results.py
Outputs (all in ./output/gaussianization/):
  fig1_alpha_overview.pdf          — central finding: input vs bottleneck vs output
  fig2_gamma_kappa4_profile.pdf    — layer-by-layer κ4 trace
  fig3_gamma_pr_profile.pdf        — layer-by-layer PR & whiteness
  fig4_gamma_mardia_profile.pdf    — layer-by-layer Mardia-Z
  fig5_beta_kappa4_vs_bf.pdf       — κ4 at bottleneck vs bottleneck factor
  fig6_beta_pr_vs_bf.pdf           — PR & effective rank vs bottleneck factor
  fig7_beta_rigidity_vs_bf.pdf     — rigidity (σ=0.5) vs bottleneck factor
  fig8_delta_rigidity_sweep.pdf    — rigidity test: Huber vs σ for both models
  fig9_alpha_stage_heatmap.pdf     — heatmap: all stages × all statistics
  fig10_summary_four_panels.pdf    — thesis-ready summary figure (4 panels)

FIXES APPLIED vs. original:
  1. CSV paths updated to match actual file locations.
  2. alpha.csv column names corrected throughout:
       r["model"]           -> r["model_name"]
       r["stage"]           -> r["label"]
       "mean_k4"            -> "dist_k4"
       "mean_abs_k3"        -> "dist_k3"
       "frac_non_gauss"     -> "dist_frac_nong"
       "mardia_b2p_z"       -> "dist_mardia_z"   (alpha only; gamma keeps mardia_b2p_z)
  3. Indentation errors fixed in every figure function (plt.tight_layout /
     plt.savefig / plt.close were misindented, causing IndentationError).
  4. beta x-axis ticks corrected to match available bottleneck factors
     (0.5, 1.0, 2.0 only — 0.25 and 3.0 are not in the data).
  5. fig9 heatmap: _build_matrix now uses corrected alpha key names.
  6. fig10 _get_alpha helper: uses corrected alpha key names.
"""
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "text.usetex":       False,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

FIGURES = Path("output/gaussianization")
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS  = {"Gaussian": "#3A7EBF", "Rosenblatt": "#E07B39"}
MARKERS = {"Gaussian": "o", "Rosenblatt": "s"}
LIGHT   = {"Gaussian": "#A8C8E8", "Rosenblatt": "#F0C09A"}

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")

# alpha  = _load_csv("output/diffusion/gaussianization/alpha_cumulants.csv")
# beta   = _load_csv("output/diffusion/gaussianization/beta_bottleneck.csv")
# gamma  = _load_csv("output/diffusion/gaussianization/gamma_layer_stats.csv")

alpha  = _load_csv("output/metrics/gaussianity/alpha.csv")
beta   = _load_csv("output/metrics/gaussianity/beta.csv")
gamma  = _load_csv("output/metrics/gaussianity/gamma.csv")

# ── δ data (hard-coded from console output) ───────────────────────────────────
def _load_delta(path="output/metrics/gaussianity/delta.csv"):
    rows = _load_csv(path)
    name_map = {"gaussian": "Gaussian", "rosenblatt": "Rosenblatt"}
    sigmas = sorted({float(r["sigma"]) for r in rows})
    nested = {}
    for r in rows:
        model = name_map.get(r["noise_type"].lower(), r["noise_type"])
        nested.setdefault(model, {}).setdefault(r["perturbation"], {})
        nested[model][r["perturbation"]][float(r["sigma"])] = _f(r, "huber_loss")
    DELTA = {m: {p: [d.get(s, float("nan")) for s in sigmas] for p, d in perts.items()}
             for m, perts in nested.items()}
    return sigmas, DELTA

DELTA_SIGMA, DELTA = _load_delta()

# ── Layer ordering for Experiment γ ──────────────────────────────────────────
LAYER_KEYS = [
    "init_conv","down1_0","down1_1","pool1",
    "down2_0","down2_1","attn2",
    "pool2","mid1","attn_mid","mid2",
    "up_res2_0","up_res2_1","up_attn2",
    "up_res1_0","up_res1_1","out",
]
LAYER_SHORT = {
    "init_conv":  "init", "down1_0": "d1[0]", "down1_1": "d1[1]",
    "pool1":      "p1",   "down2_0": "d2[0]", "down2_1": "d2[1]",
    "attn2":      "a2",   "pool2":   "p2",    "mid1":    "m1",
    "attn_mid":   "am",   "mid2":    "m2★",
    "up_res2_0":  "u2[0]","up_res2_1":"u2[1]","up_attn2":"ua2",
    "up_res1_0":  "u1[0]","up_res1_1":"u1[1]","out":"out",
}
BOTTLENECK_IDX = LAYER_KEYS.index("mid2")   # 10


def _gamma_by_model(model_name: str, key: str) -> list[float]:
    d = {r["layer_key"]: _f(r, key)
         for r in gamma if r["model"] == model_name}
    return [d.get(k, float("nan")) for k in LAYER_KEYS]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: vertical bottleneck line
# ─────────────────────────────────────────────────────────────────────────────

def _vline_bottleneck(ax, idx=BOTTLENECK_IDX, label=True):
    ax.axvline(idx, color="gray", lw=1.2, ls=":", alpha=0.7, zorder=0)
    if label:
        ax.text(idx + 0.15, ax.get_ylim()[1] * 0.97, "bottleneck",
                fontsize=6.5, color="gray", va="top", rotation=90)


def _xticklabels(ax, rotation=55):
    ax.set_xticks(range(len(LAYER_KEYS)))
    ax.set_xticklabels([LAYER_SHORT[k] for k in LAYER_KEYS],
                       rotation=rotation, ha="right", fontsize=7.5)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Experiment α overview — the central finding
# ─────────────────────────────────────────────────────────────────────────────

def fig1_alpha_overview():
    """
    Two-row bar chart showing κ4 at key stages.
    Top row: UNet stages.  Bottom row: AE + MLP stages.
    Side-by-side bars: Gaussian (blue) vs Rosenblatt (orange).

    Central finding: both models converge to κ4 ≈ 0 at the bottleneck
    despite having wildly different corrupted-input κ4.
    """
    unet_stage_keys = [
        "Input $x_0$", "Corrupted $x_T$", "Mid-gen $x_{0.5}$",
        "Bottleneck", "Output $\\hat{x}_0$",
    ]
    latent_stage_keys = [
        "Image $x_0$", "AE latent $z_0$", "Corrupted $z_T$",
        "MLP mid-layer", "Decoded $\\hat{x}_0$",
    ]

    def _get(stage_label, model):
         # average dist_k4 over all seeds (one row per seed in alpha.csv)
        vals = [_f(r, "dist_k4") for r in alpha
                if r["model_name"] == model and r["label"] == stage_label]
        vals = [v for v in vals if v == v]  # drop NaN
        return float(np.mean(vals)) if vals else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, stage_keys, title in zip(
            axes,
            [unet_stage_keys, latent_stage_keys],
            ["UNet image-space stages", "AE + MLP latent-space stages"]):

        x     = np.arange(len(stage_keys))
        width = 0.33

        vals_g = [_get(s, "Gaussian")   for s in stage_keys]
        vals_r = [_get(s, "Rosenblatt") for s in stage_keys]

        # Use symlog scale for readability (κ4 spans ~(-0.1, 62))
        bars_g = ax.bar(x - width/2, vals_g, width, color=COLORS["Gaussian"],
                        label="Gaussian model", alpha=0.85, zorder=3)
        bars_r = ax.bar(x + width/2, vals_r, width, color=COLORS["Rosenblatt"],
                        label="Rosenblatt model", alpha=0.85, zorder=3)

        ax.set_yscale("symlog", linthresh=0.5)
        ax.axhline(0, color="black", lw=0.7, zorder=2)
        ax.axhline(0, color="red", lw=1.2, ls="--", zorder=2, alpha=0.6,
                   label="$\\kappa_4=0$ (Gaussian value)")

        # Annotate bottleneck bar with value
        for i, (vg, vr) in enumerate(zip(vals_g, vals_r)):
            label_g = f"{vg:.2f}" if abs(vg) < 10 else f"{vg:.0f}"
            label_r = f"{vr:.2f}" if abs(vr) < 10 else f"{vr:.0f}"
            ax.annotate(label_g, xy=(x[i] - width/2, vg),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=COLORS["Gaussian"])
            
            ax.annotate(label_r, xy=(x[i] + width/2, vr),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=COLORS["Rosenblatt"])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("$", "").replace("\\hat{x}_0", "x̂₀")
              .replace("x_{0.5}", "x₀.₅")
              .replace("z_T", "z_T")
              .replace("z_0", "z₀")
              .replace("\\", "")
             for s in stage_keys],
            rotation=15, ha="right")
        ax.set_ylabel("Mean excess kurtosis $\\bar{\\kappa}_4$")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # Highlight bottleneck
        bn_idx = stage_keys.index("Bottleneck") if "Bottleneck" in stage_keys else \
                 stage_keys.index("MLP mid-layer")
        ax.axvspan(bn_idx - 0.5, bn_idx + 0.5, color="gold", alpha=0.18,
                   zorder=1, label="_nolegend_")
        ax.text(bn_idx, ax.get_ylim()[0] * 0.7, "◀ bottleneck",
                ha="center", fontsize=7, color="goldenrod", style="italic")

    plt.tight_layout()
    path = FIGURES / "fig1_alpha_overview.pdf"
    plt.savefig("imgs/fig1_alpha_overview.pdf", bbox_inches="tight")
    plt.savefig(path, bbox_inches="tight")
    # plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Experiment γ — κ4 layer profile  (THE KEY PLOT)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_gamma_kappa4_profile():
    """Per-unit (no pooling) vs spatial-mean κ4 by layer. Pooling can look
    near-Gaussian by averaging; the per-unit curve is the honest marginal."""
    x = np.arange(len(LAYER_KEYS))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for model, col, mk in [("Gaussian", COLORS["Gaussian"], "o"),
                           ("Rosenblatt", COLORS["Rosenblatt"], "s")]:
        unit = np.array(_gamma_by_model(model, "mean_k4_unit"), dtype=float)
        mean = np.array(_gamma_by_model(model, "mean_k4"), dtype=float)
        vu, vm = ~np.isnan(unit), ~np.isnan(mean)
        ax.plot(x[vu], unit[vu], marker=mk, color=col, lw=2,
                label=f"{model} — per-unit (no pooling)", zorder=4)
        ax.plot(x[vm], mean[vm], marker=mk, ms=4, color=col, lw=1.4, ls="--",
                alpha=0.55, label=f"{model} — spatial-mean (pooled)", zorder=3)
    ax.axhline(0, color="red", lw=1.0, ls="--", zorder=2)
    ax.set_yscale("symlog", linthresh=0.1)
    for s, e, lab, bg in [(0,6.5,"Encoder","#EFF3FA"),
                          (6.5,10.5,"Bottleneck","#FFF8E1"),
                          (10.5,16,"Decoder","#FCEEE8")]:
        ax.axvspan(s, e, color=bg, alpha=0.45, zorder=0)
        ax.text((s+e)/2, ax.get_ylim()[1]*0.7, lab, ha="center",
                fontsize=8, color="dimgray", style="italic")
    _xticklabels(ax)
    ax.set_ylabel("Mean excess kurtosis $\\bar{\\kappa}_4$ (symlog)")
    ax.set_title("Experiment γ: per-unit vs pooled $\\kappa_4$. Pooling hides "
                 "early-layer and output non-Gaussianity; the two agree near 0 "
                 "only at the bottleneck.")
    ax.legend(loc="lower center", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = FIGURES / "fig2_gamma_kappa4_profile.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig("imgs/fig2_gamma_kappa4_profile.pdf", bbox_inches="tight")
    # plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: γ — PR & effective rank profile
# ─────────────────────────────────────────────────────────────────────────────

def fig3_gamma_pr_profile():
    pr_g = _gamma_by_model("Gaussian",   "pr")
    pr_r = _gamma_by_model("Rosenblatt", "pr")
    er_g = _gamma_by_model("Gaussian",   "effective_rank")
    er_r = _gamma_by_model("Rosenblatt", "effective_rank")
    x    = np.arange(len(LAYER_KEYS))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, vals_g, vals_r, ylabel, title_suffix in [
        (axes[0], pr_g, pr_r,
            "Participation Ratio (PR)",
            "Participation Ratio (PR) Profile"),
        (axes[1], er_g, er_r,
            "Effective Rank (exp entropy)",
            "Effective Rank Profile"),
    ]:
        for model, vals, col, mk in [
                ("Gaussian",   vals_g, COLORS["Gaussian"],   "o"),
                ("Rosenblatt", vals_r, COLORS["Rosenblatt"], "s")]:
            v = np.array(vals, dtype=float)
            valid = ~np.isnan(v)
            ax.plot(x[valid], v[valid], marker=mk, color=col, lw=2, label=model)

        # Shade regions
        ymax = max(np.nanmax(np.array(vals_g, dtype=float)),
                   np.nanmax(np.array(vals_r, dtype=float))) * 1.15
        for start, end, bg in [(0, 6.5,"#EFF3FA"),(6.5,10.5,"#FFF8E1"),(10.5,16,"#FCEEE8")]:
            ax.axvspan(start, end, color=bg, alpha=0.45, zorder=0)
        ax.axvline(BOTTLENECK_IDX, color="gray", lw=1, ls=":", alpha=0.6)
        _xticklabels(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Experiment γ: {title_suffix}", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = FIGURES / "fig3_gamma_pr_profile.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: γ — Mardia-Z profile
# ─────────────────────────────────────────────────────────────────────────────

def fig4_gamma_mardia_profile():
    mz_g = _gamma_by_model("Gaussian",   "mardia_b2p_z")
    mz_r = _gamma_by_model("Rosenblatt", "mardia_b2p_z")
    x    = np.arange(len(LAYER_KEYS))

    fig, ax = plt.subplots(figsize=(11, 4))
    for model, mz, col, mk in [
            ("Gaussian",   mz_g, COLORS["Gaussian"],   "o"),
            ("Rosenblatt", mz_r, COLORS["Rosenblatt"], "s")]:
        v = np.array(mz, dtype=float)
        # Remove init_conv which is an outlier
        if len(v) > 0: v[0] = np.nan
        valid = ~np.isnan(v)
        ax.plot(x[valid], v[valid], marker=mk, color=col, lw=2, label=model, zorder=4)

    ax.axhline(0, color="red", lw=1.2, ls="--",
               label="$Z=0$: consistent with $\\mathcal{N}_p$", zorder=2)
    ax.fill_between(x, -2, 2, color="lightgreen", alpha=0.18, zorder=1,
                    label="|Z| < 2: accept $H_0$")

    for start, end, bg in [(0,6.5,"#EFF3FA"),(6.5,10.5,"#FFF8E1"),(10.5,16,"#FCEEE8")]:
        ax.axvspan(start, end, color=bg, alpha=0.35, zorder=0)
    for start, end, name in [(0,6.5,"Encoder"),(6.5,10.5,"Bottleneck"),(10.5,16,"Decoder")]:
        ylim = ax.get_ylim()
        ax.text((start+end)/2, ylim[0] if not np.isnan(ylim[0]) else -500,
                name, ha="center", fontsize=8, color="dimgray", style="italic")

    # Annotate large outlier: attn2 for Rosenblatt
    attn2_idx = LAYER_KEYS.index("attn2")
    attn2_z = next((_f(r, "mardia_b2p_z") for r in gamma
                    if r["model"] == "Rosenblatt" and r["layer_key"] == "attn2"),
                   float("nan"))
    if not np.isnan(attn2_z):
        ax.annotate(f"Rosenblatt attn2, Z={attn2_z:.0f}",
                    xy=(attn2_idx, attn2_z), xytext=(attn2_idx + 1.5, attn2_z * 0.6),
                    fontsize=7, color="purple",
                    arrowprops=dict(arrowstyle="->", color="purple"))
        
    _xticklabels(ax)
    ax.set_ylabel("Mardia kurtosis $Z$-score")
    ax.set_title("Mardia-Z profile")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = FIGURES / "fig4_gamma_mardia_profile.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: β — κ4 at bottleneck vs bottleneck factor
# ─────────────────────────────────────────────────────────────────────────────

def fig5_beta_kappa4_vs_bf():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, noise_type, mname in [(axes[0],"gaussian","Gaussian"),
                                  (axes[1],"rosenblatt","Rosenblatt")]:
        rows = sorted([r for r in beta if r.get("noise_type") == noise_type],
                      key=lambda r: float(r["bottleneck_factor"]))
        if not rows:
            ax.set_title(f"{mname}: No data")
            continue
        bfs     = [float(r["bottleneck_factor"]) for r in rows]
        k4_bn   = [_f(r,"mean_k4_bneck")      for r in rows]
        k4_unit = [_f(r,"mean_k4_bneck_unit") for r in rows]
        std_bn  = [_f(r,"std_k4_bneck")       for r in rows]
        k4_in   = [_f(r,"mean_k4_input")      for r in rows]
        k4_ou   = [_f(r,"mean_k4_x0hat")      for r in rows]
        col = COLORS[mname]
        ax.axhline(0, color="red", lw=1.2, ls="--", label="$\\kappa_4=0$", alpha=0.7)
        ax.fill_between(bfs,
                        np.array(k4_unit)-0.5*np.array(std_bn),
                        np.array(k4_unit)+0.5*np.array(std_bn),
                        color=col, alpha=0.18)
        ax.plot(bfs, k4_unit, "o-",  color=col, lw=2.5, label="Bottleneck per-unit", zorder=5)
        ax.plot(bfs, k4_bn,  "s--", color=col, lw=1.4, alpha=0.6, label="Bottleneck pooled")
        ax.set_xscale("log", base=2); ax.set_xticks([0.5,1.0,2.0])
        ax.set_xticklabels(["0.5","1.0","2.0"])
        ax.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$")
        ax.set_ylabel("$\\bar{\\kappa}_4$ at bottleneck")
        ou_mean = np.nanmean(k4_ou) if k4_ou else np.nan
        ax.set_title(f"{mname}: bottleneck $\\kappa_4$ vs width\n"
                     f"(input $\\kappa_4={k4_in[0]:.0f}$, output $\\approx{ou_mean:.0f}$, off-axis)")
        ax.legend(loc="upper left", fontsize=7.5); ax.grid(alpha=0.3)
        lo = min(0.0, np.nanmin(k4_unit), np.nanmin(k4_bn)) - 0.1
        hi = max(np.nanmax(k4_unit), np.nanmax(k4_bn)) + 0.15
        if not np.isnan(lo) and not np.isnan(hi):
            ax.set_ylim(lo, hi)
    plt.tight_layout()
    path = FIGURES / "fig5_beta_kappa4_vs_bf.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: β — PR & effective rank vs bottleneck factor
# ─────────────────────────────────────────────────────────────────────────────

def fig6_beta_pr_vs_bf():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, noise_type, mname in [
            (axes[0], "gaussian",   "Gaussian"),
            (axes[1], "rosenblatt", "Rosenblatt")]:
        rows = sorted([r for r in beta if r["noise_type"] == noise_type],
                      key=lambda r: float(r["bottleneck_factor"]))
        bfs  = [float(r["bottleneck_factor"]) for r in rows]
        bnch = [int(r["bneck_ch"])            for r in rows]
        pr   = [_f(r, "pr_bneck")             for r in rows]
        er   = [_f(r, "effective_rank_bneck") for r in rows]

        col = COLORS[mname]
        ax.plot(bfs, pr, "o-", color=col, lw=2.5, label="PR", zorder=4)
        ax.plot(bfs, er, "s--", color=col, lw=1.8, alpha=0.7, label="Effective rank")

        # Annotate channel count
        for bf, pr_v, c in zip(bfs, pr, bnch):
            ax.text(bf, pr_v + 0.2, f"C={c}", ha="center", fontsize=6.5, color="gray")

        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.5", "1.0", "2.0"])
        ax.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$")
        ax.set_ylabel("Effective dimensionality")
        ax.set_title(f"{mname}: PR & effective rank at bottleneck")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 16)

    plt.tight_layout()
    path = FIGURES / "fig6_beta_pr_vs_bf.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: β — rigidity test across bottleneck factors
# ─────────────────────────────────────────────────────────────────────────────

def fig7_beta_rigidity_vs_bf():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    PERT_COLS = {
        "perturb_gauss_huber":      ("#3A7EBF", "-",  "Gaussian"),
        "perturb_laplace_huber":    ("#55A868", "--", "Laplace"),
        "perturb_rosenblatt_huber": ("#E07B39", "-.", "Rosenblatt"),
        "perturb_t3_huber":         ("#C44E52", ":",  "Student-t(3)"),
    }

    for ax, noise_type, mname in [
            (axes[0], "gaussian",   "Gaussian"),
            (axes[1], "rosenblatt", "Rosenblatt")]:
        rows = sorted([r for r in beta if r["noise_type"] == noise_type],
                      key=lambda r: float(r["bottleneck_factor"]))
        bfs   = [float(r["bottleneck_factor"]) for r in rows]
        huber = [_f(r, "offline_loss_huber")   for r in rows]

        ax.plot(bfs, huber, "k-", lw=1.5, alpha=0.4, label="No perturbation (baseline)")

        for key, (col, ls, lab) in PERT_COLS.items():
            vals = [_f(r, key) for r in rows]
            ax.plot(bfs, vals, color=col, ls=ls, lw=2, marker="o",
                    markersize=5, label=lab)

        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.5", "1.0", "2.0"])
        ax.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$")
        ax.set_ylabel("Huber reconstruction loss (σ = 0.5)")
        ax.set_title(f"{mname}: perturbation loss ≈ baseline\n"
                     "→ decoder is insensitive to bottleneck noise type")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

        ax.text(1.0, ax.get_ylim()[1] * 0.85,
                "All four noise types\ngive identical loss\n→ No Gaussian rigidity",
                fontsize=7.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gold", alpha=0.9))

    plt.tight_layout()
    path = FIGURES / "fig7_beta_rigidity_vs_bf.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: δ — rigidity sweep over σ
# ─────────────────────────────────────────────────────────────────────────────

def fig8_delta_rigidity_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    σ_arr = np.array(DELTA_SIGMA)

    NOISE_STYLES = {
        "clean":      ("black",    ":",  1.0, "Clean baseline"),
        "gaussian":   ("#3A7EBF", "-",  2.0, "A: Gaussian $\\mathcal{N}(0,\\sigma^2)$"),
        "laplace":    ("#55A868", "--", 2.0, "B: Laplace (equal variance)"),
        "rosenblatt": ("#E07B39", "-.", 2.0, "C: Rosenblatt (unit-var scaled)"),
        "student_t3": ("#C44E52", ":",  2.0, "D: Student-$t$(3)"),
    }

    for ax, mname in [(axes[0], "Gaussian"), (axes[1], "Rosenblatt")]:
        data = DELTA[mname]
        clean_val = data["clean"][0]

        for key, (col, ls, lw, lab) in NOISE_STYLES.items():
            vals = np.array(data[key])
            ax.plot(σ_arr, vals, color=col, ls=ls, lw=lw, marker="o",
                    markersize=4, label=lab)

        # Shade the ±1% band around clean
        ax.axhspan(clean_val * 0.99, clean_val * 1.01, color="lightgreen",
                   alpha=0.25, label="±1% of clean")

        ax.set_xlabel("Perturbation scale $\\sigma$")
        ax.set_ylabel("Huber reconstruction loss")
        ax.set_title(f"{mname} model (bf=1.0)\n"
                     "All noise types track the clean baseline across all σ")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES / "fig8_delta_rigidity_sweep.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig("imgs/fig8_delta_rigidity_sweep.pdf", bbox_inches="tight")
    # plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: α heatmap — full stage × metric matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig9_alpha_heatmap():
    """
    Heat-map with rows = stages, columns = (|κ3|, κ4, frac>0.5, Mardia-Z)
    Two halves: Gaussian model (top half) and Rosenblatt model (bottom half).
    """
    stage_order = [
        "Input $x_0$", "Corrupted $x_T$", "Mid-gen $x_{0.5}$",
        "Bottleneck", "Output $\\hat{x}_0$",
        "AE latent $z_0$", "Corrupted $z_T$", "MLP mid-layer",
    ]
    metrics = ["dist_k3", "dist_k4", "dist_frac_nong", "dist_mardia_z"]
    metric_labels = [
        "$\\overline{|\\kappa_3|}$", "$\\overline{\\kappa_4}$",
        "Frac $|\\kappa_4|>0.5$", "Mardia-$Z$"
    ]

    def _build_matrix(model_name):
        M = []
        for s in stage_order:
            row = []
            for m in metrics:
                v = float("nan")
                for r in alpha:
                    if r["model_name"] == model_name and r["label"] == s:
                        v = _f(r, m)
                        break
                row.append(v)
            M.append(row)
        return np.array(M, dtype=float)

    M_g = _build_matrix("Gaussian")
    M_r = _build_matrix("Rosenblatt")

    # Apply symlog transform for display (handles large κ4 values)
    def _symlog(x, lin=1.0):
        return np.where(np.abs(x) <= lin, x,
                        np.sign(x) * (lin + np.log10(np.abs(x / lin) + 1)))

    M_g_d = _symlog(M_g)
    M_r_d = _symlog(M_r)
    M_full = np.vstack([M_g_d, np.full((1, len(metrics)), np.nan), M_r_d])

    stage_labels_short = [
        s.replace("$","").replace("\\hat{x}_0","x̂₀").replace("x_{0.5}","x₀.₅")
         .replace("\\","").replace("z_0","z₀").replace("z_T","z_T")
        for s in stage_order
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    n_stages = len(stage_order)
    full_labels = (
        [f"G: {s}" for s in stage_labels_short] +
        ["── separator ──"] +
        [f"R: {s}" for s in stage_labels_short]
    )

    im = ax.imshow(M_full, cmap="RdYlGn_r", aspect="auto",
                   vmin=-2, vmax=4)
    plt.colorbar(im, ax=ax, label="symlog-scaled metric value",
                 fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(full_labels)))
    ax.set_yticklabels(full_labels, fontsize=8)

    # Annotate cells with actual values
    for i in range(M_full.shape[0]):
        for j in range(M_full.shape[1]):
            v = M_full[i, j]
            if np.isnan(v):
                continue
            # Get original value for display
            row_idx = i if i < n_stages else i - n_stages - 1
            orig_M  = M_g if i < n_stages else M_r
            if 0 <= row_idx < n_stages:
                orig_v = orig_M[row_idx, j]
                txt = f"{orig_v:.2f}" if abs(orig_v) < 10 else f"{orig_v:.0f}"
            else:
                txt = ""
            col = "white" if abs(v) > 2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color=col)

    # Draw separator line
    ax.axhline(n_stages + 0.5, color="white", lw=2)
    ax.axhline(n_stages - 0.5, color="white", lw=2)

    ax.set_title("Stage × metric heatmap",
                 fontsize=9)
    plt.tight_layout()
    path = FIGURES / "fig9_alpha_stage_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Summary — 4-panel thesis figure
# ─────────────────────────────────────────────────────────────────────────────

def fig10_summary_four_panels():
    """
    Four-panel thesis-ready summary figure:
    (A) Stage-wise κ4 bars
    (B) Bottleneck κ4 vs bf
    (C) Layer κ4 profile
    (D) Rigidity sweep
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.45)
    axA = fig.add_subplot(gs[0, :2])
    axB = fig.add_subplot(gs[0, 2:])
    axC = fig.add_subplot(gs[1, :2])
    axD = fig.add_subplot(gs[1, 2:])

    # ── Panel A: κ4 at corrupted input ──────────────────────────────────────
    def _get_alpha(stage, model, metric="dist_k4"):
        for r in alpha:
            if r["model_name"] == model and r["label"] == stage:
                return _f(r, metric)
        return float("nan")

    stages_A = ["Input $x_0$", "Corrupted $x_T$", "Bottleneck", "Output $\\hat{x}_0$"]
    x_A = np.arange(len(stages_A))
    w   = 0.35
    vg  = [_get_alpha(s, "Gaussian")   for s in stages_A]
    vr  = [_get_alpha(s, "Rosenblatt") for s in stages_A]

    axA.bar(x_A - w/2, vg, w, color=COLORS["Gaussian"],   label="Gaussian",   alpha=0.85)
    axA.bar(x_A + w/2, vr, w, color=COLORS["Rosenblatt"], label="Rosenblatt", alpha=0.85)
    axA.set_yscale("symlog", linthresh=0.5)
    axA.axhline(0, color="red", lw=1.0, ls="--")
    axA.set_xticks(x_A)
    axA.set_xticklabels(["Input", "Corrupted $x_T$", "Bottleneck", "Output"], fontsize=8)
    axA.set_ylabel("$\\bar{\\kappa}_4$ (symlog)")
    axA.set_title("(A)  Stage-wise $\\bar{\\kappa}_4$", fontweight="bold")
    axA.legend(fontsize=7.5)
    axA.grid(axis="y", alpha=0.25)

    # Annotation: Rosenblatt corrupted is non-Gaussian but bottleneck is Gaussian
    axA.annotate("", xy=(1.35, 8.84), xytext=(1.5, 2.0),
                 arrowprops=dict(arrowstyle="->", color=COLORS["Rosenblatt"], lw=1.5))
    axA.text(1.55, 2.5, "non-Gaussian\ncorruption →\nmarginal $\\kappa_4$\nattenuated",
             fontsize=6.5, color=COLORS["Rosenblatt"])

    # ── Panel B: bottleneck κ4 vs bf ─────────────────────────────────────────
    for noise_type, mname in [("gaussian","Gaussian"),("rosenblatt","Rosenblatt")]:
        rows = sorted([r for r in beta if r["noise_type"]==noise_type],
                      key=lambda r: float(r["bottleneck_factor"]))
        bfs  = [float(r["bottleneck_factor"]) for r in rows]
        k4bn = [_f(r,"mean_k4_bneck") for r in rows]
        axB.plot(bfs, k4bn, "o-", color=COLORS[mname], lw=2, label=mname, zorder=4)

    axB.axhline(0, color="red", lw=1.2, ls="--", label="$\\kappa_4=0$")
    axB.set_xscale("log", base=2)
    axB.set_xticks([0.5, 1.0, 2.0])
    axB.set_xticklabels(["0.5", "1.0", "2.0"], fontsize=8)
    axB.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$")
    axB.set_ylabel("$\\bar{\\kappa}_4$ at bottleneck")
    axB.set_title("(B)  $\\bar{\\kappa}_4$ vs bottleneck width", fontweight="bold")
    axB.legend(fontsize=7.5)
    axB.grid(alpha=0.3)
    axB.set_ylim(-0.5, 0.8)

    # ── Panel C: Layer κ4 profile ─────────────────────────────────────────────
    x_C = np.arange(len(LAYER_KEYS))
    for model, col, mk in [("Gaussian", COLORS["Gaussian"], "o"),
                           ("Rosenblatt", COLORS["Rosenblatt"], "s")]:
        unit = np.array(_gamma_by_model(model, "mean_k4_unit"), dtype=float)
        mean = np.array(_gamma_by_model(model, "mean_k4"), dtype=float)
        vu, vm = ~np.isnan(unit), ~np.isnan(mean)
        axC.plot(x_C[vu], unit[vu], marker=mk, ms=4, color=col, lw=1.8,
                 label=f"{model} (per-unit)")
        axC.plot(x_C[vm], mean[vm], marker=mk, ms=3, color=col, lw=1.2,
                 ls="--", alpha=0.5, label=f"{model} (pooled)")

    axC.axhline(0, color="red", lw=1.0, ls="--")
    axC.fill_between(x_C, -0.15, 0.15, color="lightgreen", alpha=0.20,
                     label="$|\\kappa_4|<0.15$")
    axC.set_yscale("symlog", linthresh=0.1)

    for start, end, bg in [(0,6.5,"#EFF3FA"),(6.5,10.5,"#FFF8E1"),(10.5,16,"#FCEEE8")]:
        axC.axvspan(start, end, color=bg, alpha=0.4, zorder=0)

    axC.set_xticks(x_C)
    axC.set_xticklabels([LAYER_SHORT[k] for k in LAYER_KEYS],
                        rotation=60, ha="right", fontsize=6.5)
    axC.set_ylabel("$\\bar{\\kappa}_4$")
    axC.set_title("(C)  Layer-by-layer $\\kappa_4$ trace (Exp. γ)", fontweight="bold")
    axC.legend(fontsize=7.5)
    axC.grid(axis="y", alpha=0.25)

    # ── Panel D: δ rigidity ───────────────────────────────────────────────────
    σ_arr  = np.array(DELTA_SIGMA)
    NSTYLES = {
        "clean":      ("black",    ":",  1.0),
        "gaussian":   ("#3A7EBF", "-",  1.8),
        "laplace":    ("#55A868", "--", 1.8),
        "rosenblatt": ("#E07B39", "-.", 1.8),
        "student_t3": ("#C44E52", ":",  1.8),
    }
    NLABELS = {
        "clean":"Clean","gaussian":"Gaussian","laplace":"Laplace",
        "rosenblatt":"Rosenblatt","student_t3":"Student-t(3)"
    }
    # Show Rosenblatt model (more interesting)
    for key, (col, ls, lw) in NSTYLES.items():
        vals = np.array(DELTA["Rosenblatt"][key])
        axD.plot(σ_arr, vals, color=col, ls=ls, lw=lw, marker="o",
                 ms=3, label=NLABELS[key])

    axD.set_xlabel("Perturbation scale $\\sigma$")
    axD.set_ylabel("Huber loss (Rosenblatt model)")
    axD.set_title("(D)  Latent rigidity — Rosenblatt model (Exp. δ)", fontweight="bold")
    axD.legend(fontsize=7, loc="upper left", ncol=2)
    axD.grid(alpha=0.3)
    clean_v = DELTA["Rosenblatt"]["clean"][0]
    axD.axhspan(clean_v*0.99, clean_v*1.01, color="lightgreen", alpha=0.3)

    path = FIGURES / "fig10_summary_four_panels.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all figures …\n")
    fig1_alpha_overview()
    fig2_gamma_kappa4_profile()
    fig3_gamma_pr_profile()
    fig4_gamma_mardia_profile()
    fig5_beta_kappa4_vs_bf()
    fig6_beta_pr_vs_bf()
    fig7_beta_rigidity_vs_bf()
    fig8_delta_rigidity_sweep()
    fig9_alpha_heatmap()
    fig10_summary_four_panels()
    print(f"\nAll figures saved to ./{FIGURES}/")