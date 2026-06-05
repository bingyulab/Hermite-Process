"""
Rosenblatt Process — Thesis Experiments
========================================

Five experiments that directly illustrate the theoretical results of the
thesis. All simulation code is imported from the two existing modules:

    density_simulation.py  —  RosenblattDensityLP, eigenvalues_LP
    path_simulation.py     —  WaveletRosenblatt

Run from the project root:
    python experiments.py

Outputs saved to output/experiments/.

Experiment 1 — Eigenvalue decay & normalisation identity
    λ_{D,n} ~ C_D n^{D-1};  Σ λ_{D,n}² → 1/2
    Connects to: chi-square representation (density chapter).

Experiment 2 — Density regularity as D changes
    p_D ∈ S(R): smooth, right-skewed, superexponential tails.
    Connects to: Schwartz-class smoothness result.

Experiment 3 — Non-Gaussianity vs Gaussian / chi-squared
    Rosenblatt ≠ N(0,1); qualitatively distinct from chi-squared.
    Connects to: Introduction (generative modelling motivation).

Experiment 4 — Variance of the Rosenblatt-driven SDE
    Var(X_t) ~ σ² t^{2H} for b(x) = -αx.
    Connects to: SDE well-posedness chapter.

Experiment 5 — Variogram and Hölder exponent
    E[|Z_{t+h}-Z_t|^2] ~ h^{2H}; fit vs theoretical 2H.
    Connects to: self-similarity / path regularity chapter.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, gaussian_kde, skew

# ── project root on path ──────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
from density_simulation import RosenblattDensityLP, eigenvalues_LP
from path_simulation import WaveletRosenblatt

# ── output & logging ──────────────────────────────────────────────────────
print(f"Experiment outputs will be saved to {ROOT_DIR}/../output/experiments/")
os.makedirs(f"{ROOT_DIR}/../output/experiments/", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{ROOT_DIR}/../output/experiments/experiments.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

_trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ============================================================
#  Experiment 1 — Eigenvalue decay & normalisation identity
# ============================================================

def experiment_1_eigenvalue_decay():
    log.info("=" * 65)
    log.info("Experiment 1: Eigenvalue decay and normalisation identity")
    log.info("=" * 65)

    D_vals = [0.10, 0.25, 0.40]
    K_plot = 300
    K_sum  = 800
    colors = ["#2980b9", "#e67e22", "#27ae60"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): log-log eigenvalue curves
    ax = axes[0]
    for D, cc in zip(D_vals, colors):
        lam = eigenvalues_LP(D, K_plot)       # <── density_simulation.py
        ns  = np.arange(1, K_plot + 1)
        ax.loglog(ns, lam, color=cc, lw=2.0, label=f"$D = {D}$")
        # theoretical slope anchored at n=10
        sx = np.array([10, K_plot])
        ax.loglog(sx, lam[9] * (sx / 10) ** (D - 1),
                  color=cc, lw=1.5, ls="--", alpha=0.9)
        for n in [10, 50, 100]:
            log.info(f"  D={D:.2f}  n={n:3d}  lambda={lam[n-1]:.6f}")

    ax.set_xlabel("$n$", fontsize=12)
    ax.set_ylabel("$\\lambda_{D,n}$", fontsize=12)
    ax.set_title("(a) Eigenvalue decay: $\\lambda_{D,n} \\sim C_D n^{D-1}$",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.20, which="both")
    ax.text(0.60, 0.30, "Dashed = slope $D-1$",
            transform=ax.transAxes, fontsize=8.5, color="#555", style="italic")

    # Panel (b): partial sums → 1/2
    ax = axes[1]
    log.info("\nPartial sums Sigma lambda^2:")
    for D, cc in zip(D_vals, colors):
        lam     = eigenvalues_LP(D, K_sum)    # <── density_simulation.py
        partial = np.cumsum(lam ** 2)
        ns      = np.arange(1, K_sum + 1)
        ax.semilogx(ns, partial, color=cc, lw=2.0, label=f"$D = {D}$")
        for N in [10, 50, 100, 500, 800]:
            log.info("  D=%.2f  N=%4d  sum_lam2=%.6f", D, N, partial[N - 1])

    ax.axhline(0.5, color="#c0392b", lw=1.8, ls="--", label="$1/2$ (theory)")
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel("$\\sum_{n=1}^N \\lambda_{D,n}^2$", fontsize=12)
    ax.set_title(
        "(b) Normalisation identity: $\\sum_{n=1}^{\\infty}\\lambda_{D,n}^2 = 1/2$",
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.20)
    ax.set_ylim(0, 0.56)

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp1_eigenvalue_decay.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ============================================================
#  Experiment 2 — Density regularity as D changes
# ============================================================

def experiment_2_density_regularity():
    log.info("=" * 65)
    log.info("Experiment 2: Density regularity as D changes")
    log.info("=" * 65)

    D_list = [0.05, 0.15, 0.25, 0.35, 0.45]
    cmap   = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_list)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    log.info("\n  %-5s  %-5s  %-8s  %-8s  %-8s  %-10s",
             "D", "H", "mean", "std", "skew", "exkurt")

    for D, cc in zip(D_list, cmap):
        lp   = RosenblattDensityLP(a=D, K=400)           # <── density_simulation.py
        x, d = lp.density_fft(x_min=-3.0, x_max=10.0)

        # moments from density
        m1 = _trap(x * d, x)
        m2 = _trap((x - m1) ** 2 * d, x)
        m3 = _trap((x - m1) ** 3 * d, x)
        m4 = _trap((x - m1) ** 4 * d, x)
        skew = m3 / m2 ** 1.5
        kurt = m4 / m2 ** 2 - 3.0
        log.info("  %-5.2f  %-5.2f  %-8.4f  %-8.4f  %-8.4f  %-10.4f",
                 D, 1 - D, m1, np.sqrt(m2), skew, kurt)

        axes[0].plot(x, d, color=cc, lw=2.0, label=f"$D={D}$")
        mask = d > 1e-6
        axes[1].semilogy(x[mask], d[mask], color=cc, lw=1.8, label=f"$D={D}$")

    xr = np.linspace(2, 9, 200)
    axes[1].semilogy(xr, 0.25 * np.exp(-0.8 * xr),
                     "k--", lw=1.3, label="$e^{-cx}$ (ref.)")

    for ax, title, ylabel in zip(
        axes,
        ["(a) Density variation with $D$", "(b) Right-tail decay (log scale)"],
        ["$p_D(x)$", "$p_D(x)$  (log scale)"]
    ):
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.20)

    axes[1].set_xlim(0, 9)
    axes[1].set_ylim(1e-6, 1)

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp2_density_regularity.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ============================================================
#  Experiment 3 — Non-Gaussianity: Rosenblatt vs N(0,1) vs chi-squared
# ============================================================

def experiment_3_non_gaussianity():
    log.info("=" * 65)
    log.info("Experiment 3: Non-Gaussianity")
    log.info("=" * 65)

    D   = 0.25
    lp  = RosenblattDensityLP(a=D, K=400)               # <── density_simulation.py
    x_lp, d_lp = lp.density_fft(x_min=-4.0, x_max=9.0)

    # standardised chi2(4)
    k_df  = 4
    x_chi = np.linspace(-4, 9, 800)
    y_chi = x_chi * np.sqrt(2 * k_df) + k_df
    d_chi = chi2.pdf(y_chi, k_df) * np.sqrt(2 * k_df)
    d_chi[y_chi <= 0] = 0.0
    xg = np.linspace(-4, 9, 800)

    m1 = _trap(x_lp * d_lp, x_lp)
    m2 = _trap((x_lp - m1) ** 2 * d_lp, x_lp)
    m3 = _trap((x_lp - m1) ** 3 * d_lp, x_lp)
    m4 = _trap((x_lp - m1) ** 4 * d_lp, x_lp)
    log.info("Rosenblatt D=%.2f: skew=%.4f  exkurt=%.4f",
             D, m3 / m2 ** 1.5, m4 / m2 ** 2 - 3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax in axes:
        ax.plot(x_lp, d_lp, "#c0392b", lw=2.5,
                label=f"Rosenblatt $D={D}$, $H={1-D}$")
        ax.plot(xg, norm.pdf(xg, 0, 1), "#2980b9", lw=2.0, ls="--",
                label="$\\mathcal{N}(0,1)$")
        mk = y_chi > 0
        ax.plot(x_chi[mk], d_chi[mk], "#27ae60", lw=1.8, ls=":",
                label="$\\chi^2_4$ (standardised)")
        ax.set_xlabel("$x$", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.20)

    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("(a) Distribution comparison", fontsize=11)
    axes[0].set_xlim(-4, 9)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Density (log scale)", fontsize=12)
    axes[1].set_title("(b) Tail comparison (log scale)", fontsize=11)
    axes[1].set_xlim(-1, 8)
    axes[1].set_ylim(1e-5, 1)

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp3_non_gaussianity.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ============================================================
#  Experiment 4 — Variance of the Rosenblatt-driven SDE
# ============================================================

def experiment_4_additive_sde_variance():
    log.info("=" * 65)
    log.info("Experiment 4: Variance of Rosenblatt-driven SDE")
    log.info("=" * 65)

    H       = 0.7
    alpha   = 0.5    # b(x) = -alpha * x
    sigma   = 1.0
    T       = 1.0
    n_pts   = 300
    n_paths = 300

    log.info("H=%.1f  alpha=%.2f  sigma=%.2f  T=%.1f  n_paths=%d",
             H, alpha, sigma, T, n_paths)

    # wavelet paths                               <── path_simulation.py
    wav = WaveletRosenblatt(H=H, J=7)
    t0_sim = time.time()
    ts, Z_mat = wav.simulate_paths_batch(T=T, n_points=n_pts, n_paths=n_paths)
    log.info("Wavelet paths in %.1f s", time.time() - t0_sim)

    dt = ts[1] - ts[0]

    # Euler-Maruyama for X_t = ∫ b(X_s) ds + σ Z_t
    X_mat = np.zeros((n_paths, n_pts))
    for p in range(n_paths):
        X = 0.0
        for j in range(n_pts - 1):
            dZ = Z_mat[p, j + 1] - Z_mat[p, j]
            X  = X + (-alpha * X) * dt + sigma * dZ
            X_mat[p, j + 1] = X

    log.info("\n  %-6s  %-14s  %-12s  %-8s",
             "t", "Var(X_t) emp.", "sigma^2 t^{2H}", "ratio")
    for t_check in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx   = np.argmin(np.abs(ts - t_check))
        v_emp = np.var(X_mat[:, idx])
        v_th  = sigma ** 2 * t_check ** (2 * H)
        log.info("  %-6.2f  %-14.6f  %-12.6f  %-8.3f",
                 t_check, v_emp, v_th, v_emp / v_th)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i in range(min(12, n_paths)):
        ax.plot(ts, X_mat[i], "#8e44ad", alpha=0.25, lw=0.7)
    ax.plot(ts, np.mean(X_mat, axis=0), "#8e44ad", lw=2.4, label="Mean $X_t$")
    ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$X_t$", fontsize=12)
    ax.set_title(
        f"(a) SDE paths: $b(x)=-{alpha}x$, $\\sigma={sigma}$, $H={H}$",
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.20)

    ax = axes[1]
    var_X  = np.var(X_mat, axis=0)
    t_plot = ts[1:]
    ax.loglog(t_plot, var_X[1:], "#8e44ad", lw=2.2,
              label="$\\mathrm{Var}(X_t)$ empirical")
    ax.loglog(t_plot, sigma ** 2 * t_plot ** (2 * H), "#c0392b", lw=1.8,
              ls="--", label=f"$\\sigma^2 t^{{2H}}$, $H={H}$")
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$\\mathrm{Var}(X_t)$", fontsize=12)
    ax.set_title("(b) Variance scaling: $\\mathrm{Var}(X_t) \\approx \\sigma^2 t^{2H}$",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.20, which="both")

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp4a_additive_sde_variance.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.savefig(f"{ROOT_DIR}/../imgs/exp4a_additive_sde_variance.png", dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def experiment_4_multiplicative_sde_density():
    """
    Validate smooth density for the multiplicative Rosenblatt SDE:
        dX_t = -alpha * X_t dt  +  g(X_t) * dZ_t^{H,2}
        g(x) = (1 + |x|) / 2

    Doss-Sussmann (Theorem 4.3) guarantees X_t has a C-inf density for all t > 0.
    We verify this empirically:
      1. Kernel density estimate of X_t at t = 0.5 and t = 1.0 is smooth
      2. Compare with the additive SDE (same H, alpha) to show g(x)!=const matters
      3. Log-scale density plot confirms no atoms and smooth tails
    """
    H, alpha, T = 0.7, 0.5, 1.0
    n_pts, n_paths = 300, 2000   # more paths for density estimate
    t_checks = [0.25, 0.5, 1.0]

    wav = WaveletRosenblatt(H=H, J=10, L=0, N_vanishing=2)
    ts, Z_mat = wav.simulate_paths_batch(T=T, n_points=n_pts, n_paths=n_paths)
    dt = ts[1] - ts[0]

    X_mult = np.zeros((n_paths, n_pts))
    for p in range(n_paths):
        X = 0.0
        dZ = np.diff(Z_mat[p])          # (n_pts-1,)
        xs = np.zeros(n_pts)
        for j in range(n_pts - 1):
            xs[j] = X
            X = X + (-alpha * X) * dt + (1.0 + abs(X)) / 2.0 * dZ[j]
        xs[-1] = X
        X_mult[p] = xs

    # ── Euler-Maruyama: additive sigma=1 (baseline) ───────────────────
    X_add = np.zeros((n_paths, n_pts))
    for p in range(n_paths):
        X = 0.0
        for j in range(n_pts - 1):
            dZ  = Z_mat[p, j + 1] - Z_mat[p, j]
            X   = X + (-alpha * X) * dt + 1.0 * dZ
            X_add[p, j + 1] = X

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): sample paths of multiplicative SDE
    ax = axes[0]
    for i in range(min(20, n_paths)):
        ax.plot(ts, X_mult[i], "#e67e22", alpha=0.18, lw=0.7)
    ax.plot(ts, np.mean(X_mult, axis=0), "#e67e22", lw=2.4,
            label=r"Mean $X_t$")
    ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    ax.set(xlabel="$t$", ylabel="$X_t$",
           title=fr"(a) Multiplicative SDE paths ($H={H}$)")
    ax.legend(fontsize=9);  ax.grid(alpha=0.20)

    # Panel (b): smooth KDE density at t = 0.5 and t = 1.0
    ax = axes[1]
    colors = ["#e74c3c", "#8e44ad", "#2980b9"]
    for t_c, col in zip(t_checks, colors):
        idx    = np.argmin(np.abs(ts - t_c))
        sample = X_mult[:, idx]
        kde    = gaussian_kde(sample, bw_method="silverman")
        xgrid  = np.linspace(sample.min() - 0.5, sample.max() + 0.5, 400)
        ax.plot(xgrid, kde(xgrid), color=col, lw=2.0,
                label=fr"Multiplicative $t={t_c}$")
        # add additive comparison at t=1 only
    idx1 = np.argmin(np.abs(ts - 1.0))
    kde_add = gaussian_kde(X_add[:, idx1], bw_method="silverman")
    xgrid2  = np.linspace(X_add[:,idx1].min()-0.5, X_add[:,idx1].max()+0.5, 400)
    ax.plot(xgrid2, kde_add(xgrid2), "k--", lw=1.5,
            label="Additive $t=1$ (reference)")
    ax.set(xlabel="$x$", ylabel="density",
           title="(b) Smooth marginal densities (KDE)")
    ax.legend(fontsize=8);  ax.grid(alpha=0.20)

    # Panel (c): log-scale density at t=1 — confirms no atoms, smooth tails
    ax = axes[2]
    idx1   = np.argmin(np.abs(ts - 1.0))
    sample = X_mult[:, idx1]
    kde    = gaussian_kde(sample, bw_method="silverman")
    xgrid  = np.linspace(sample.min() - 1, sample.max() + 1, 500)
    dens   = kde(xgrid)
    ax.semilogy(xgrid, np.maximum(dens, 1e-10), "#e74c3c", lw=2.0,
                label=r"Multiplicative $t=1$ (log scale)")
    ax.set(xlabel="$x$", ylabel="log density",
           title="(c) Log-scale density: smooth tails, no atoms")
    ax.legend(fontsize=9);  ax.grid(alpha=0.20)
    ax.text(0.05, 0.10,
            "Doss-Sussmann guarantees\n$C^\\infty$ density for all $t>0$",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp4b_multiplicative_sde_density.png"
    plt.savefig(out, dpi=140, bbox_inches="tight");  plt.close()
    print(f"Saved {out}")

    # ── numerical summary ─────────────────────────────────────────────
    print(f"\n{'t':>6}  {'mean(X)':>10}  {'std(X)':>10}  {'skew(X)':>10}")
    for t_c in t_checks:
        idx = np.argmin(np.abs(ts - t_c))
        s   = X_mult[:, idx]
        print(f"  {t_c:>4.2f}  {s.mean():>10.4f}  {s.std():>10.4f}  {skew(s):>10.4f}")

    return ts, X_mult

# ============================================================
#  Experiment 5 — Variogram and Hölder exponent
# ============================================================

def experiment_5_variogram():
    log.info("=" * 65)
    log.info("Experiment 5: Variogram and Holder exponent")
    log.info("=" * 65)

    H_vals  = [0.6, 0.7, 0.8]
    colors  = ["#2980b9", "#e67e22", "#27ae60"]
    n_paths = 350
    n_pts   = 512
    T       = 1.0
    h_steps = np.array([1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 130])
    h_steps = h_steps[h_steps < n_pts // 3]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    slopes_emp = []

    log.info("\n  %-4s  %-10s  %-10s  %-10s", "H", "fitted 2H", "theory 2H", "rel err %")

    for H, cc in zip(H_vals, colors):
        log.info("Simulating %d wavelet paths (H=%.1f) ...", n_paths, H)
        wav = WaveletRosenblatt(H=H, J=7)              # <── path_simulation.py
        t0_sim = time.time()
        Z_mat  = np.zeros((n_paths, n_pts))
        for p in range(n_paths):
            ts, Z = wav.simulate_path(T=T, n_points=n_pts)
            Z_mat[p] = Z
        log.info("  done in %.1f s", time.time() - t0_sim)

        dt    = T / (n_pts - 1)
        h_arr = h_steps * dt
        V_arr = np.array([
            np.mean((Z_mat[:, s:] - Z_mat[:, :-s]) ** 2)
            for s in h_steps
        ])

        # empirical variogram (solid circles + line)
        ax.loglog(h_arr, V_arr, "o-", color=cc, lw=1.8, ms=5,
                  label=f"$H={H}$  (empirical)")
 
        # theoretical reference h^{2H} (dashed, same colour)
        h_th = np.logspace(np.log10(h_arr[0]), np.log10(h_arr[-1]), 120)
        ax.loglog(h_th, h_th ** (2 * H), "--", color=cc, lw=1.2, alpha=0.55,
                  label=f"$h^{{2H}}={h_th[0]**(2*H):.0e}\\cdots$"
                        if False else f"$h^{{{2*H}}}$ (theory)")
 
        # fit and annotate slope
        slope = np.polyfit(np.log(h_arr), np.log(V_arr), 1)[0]
        slopes_emp.append(slope)
        rel   = (slope - 2 * H) / (2 * H) * 100
        sign  = "+" if rel >= 0 else ""
        # place annotation near the last point of the empirical curve
        ax.annotate(
            f"$\\hat{{s}}={slope:.2f}$  ({sign}{rel:.1f}\\%)",
            xy=(h_arr[-1], V_arr[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8.5, color=cc, va="center",
        )
        log.info("  %-4.1f  %-10.3f  %-10.3f  %-10.1f",
                 H, slope, 2 * H, rel)
 
    ax.set_xlabel("$h$", fontsize=12)
    ax.set_ylabel("$\\mathbb{E}[|Z_{t+h}-Z_t|^2]$", fontsize=12)
    ax.set_title(
        "Variogram: $\\mathbb{E}[|Z_{t+h}-Z_t|^2] \\sim h^{2H}$  (log-log)\n"
        "Solid = empirical, dashed = theoretical $h^{2H}$, "
        "annotation = fitted slope",
        fontsize=11)
    # Build a clean legend: one entry per H (empirical line only)
    handles, labels = ax.get_legend_handles_labels()
    # keep only every other entry (the empirical ones, not the theory dashes)
    ax.legend(handles[::2], labels[::2], fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.20, which="both")

    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/exp5_holder_variogram.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def merge_exp2_exp3():
    """
    Merges the left panel of Exp2 (Density Regularity) and the left panel 
    of Exp3 (Non-Gaussianity) into a single 1x2 figure.
    """
    log.info("Generating merged plot for Exp2 (left) and Exp3 (left)...")
    
    # Create a 1x2 figure layout once
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ─── LEFT PANEL: Density Regularity (from exp2) ────────────────────────
    ax1 = axes[0]
    D_list = [0.05, 0.15, 0.25, 0.35, 0.45]
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_list)))

    log.info("\n  %-5s  %-5s  %-8s  %-8s  %-8s  %-10s",
             "D", "H", "mean", "std", "skew", "exkurt")

    for D, cc in zip(D_list, cmap):
        lp = RosenblattDensityLP(a=D, K=400)
        x, d = lp.density_fft(x_min=-3.0, x_max=10.0)

        # moments from density
        m1 = _trap(x * d, x)
        m2 = _trap((x - m1) ** 2 * d, x)
        m3 = _trap((x - m1) ** 3 * d, x)
        m4 = _trap((x - m1) ** 4 * d, x)
        skew = m3 / m2 ** 1.5
        kurt = m4 / m2 ** 2 - 3.0
        log.info("  %-5.2f  %-5.2f  %-8.4f  %-8.4f  %-8.4f  %-10.4f",
                 D, 1 - D, m1, np.sqrt(m2), skew, kurt)

        ax1.plot(x, d, color=cc, lw=2.0, label=f"$D={D}$")
    
    ax1.set_xlim(-4, 6)
    ax1.set_title("Density Regularity (Varying $D$)", fontsize=11)
    ax1.set_xlabel("$x$", fontsize=10)
    ax1.set_ylabel("Density $p_D(x)$", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ─── RIGHT PANEL: Non-Gaussianity Comparison (from exp3) ───────────────
    ax2 = axes[1]
    D   = 0.25
    lp  = RosenblattDensityLP(a=D, K=400)               # <── density_simulation.py
    x_lp, d_lp = lp.density_fft(x_min=-4.0, x_max=9.0)

    # standardised chi2(4)
    k_df  = 4
    x_chi = np.linspace(-4, 9, 800)
    y_chi = x_chi * np.sqrt(2 * k_df) + k_df
    d_chi = chi2.pdf(y_chi, k_df) * np.sqrt(2 * k_df)
    d_chi[y_chi <= 0] = 0.0
    xg = np.linspace(-4, 9, 800)

    m1 = _trap(x_lp * d_lp, x_lp)
    m2 = _trap((x_lp - m1) ** 2 * d_lp, x_lp)
    m3 = _trap((x_lp - m1) ** 3 * d_lp, x_lp)
    m4 = _trap((x_lp - m1) ** 4 * d_lp, x_lp)
    log.info("Rosenblatt D=%.2f: skew=%.4f  exkurt=%.4f",
             D, m3 / m2 ** 1.5, m4 / m2 ** 2 - 3)

    ax2.plot(x_lp, d_lp, "#c0392b", lw=2.5,
            label=f"Rosenblatt $D={D}$, $H={1-D}$")
    ax2.plot(xg, norm.pdf(xg, 0, 1), "#2980b9", lw=2.0, ls="--",
            label="$\\mathcal{N}(0,1)$")
    mk = y_chi > 0
    ax2.plot(x_chi[mk], d_chi[mk], "#27ae60", lw=1.8, ls=":",
            label="$\\chi^2_4$ (standardised)")
    ax2.set_xlabel("$x$", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.20)
    
    ax2.set_xlim(-4, 6)
    ax2.set_title("Non-Gaussianity Comparison", fontsize=11)
    ax2.set_xlabel("$x$", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Formatting and saving
    plt.tight_layout()
    out = f"{ROOT_DIR}/../output/experiments/merged_exp2_exp3.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.savefig(f"imgs/merged_exp2_exp3.png", dpi=140, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(2025)
    t_total = time.time()
    log.info("Rosenblatt Thesis Experiments")
    log.info("=" * 65)

    experiment_1_eigenvalue_decay()
    experiment_2_density_regularity()
    experiment_3_non_gaussianity()
    merge_exp2_exp3()
    experiment_4_additive_sde_variance()
    experiment_4_multiplicative_sde_density()
    experiment_5_variogram()
    merge_exp2_exp3()

    log.info("=" * 65)
    log.info("All experiments completed in %.1f s", time.time() - t_total)
    log.info(f"Figures saved to {ROOT_DIR}/../output/experiments/")