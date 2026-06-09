"""
Rosenblatt Process — Comprehensive Marginal Validation
=======================================================
Prof. Ivan's check implemented at scale:
    - Three Hurst values: H = 0.6, 0.7, 0.8
    - Four time points:   t = 0.25, 0.5, 0.75, 1.0
    - Methods shown here: Wavelet, Donsker, LP density

Outputs (saved to output/marginal/)
-------------------------------------
  fig_A_multi_H_t1.png      Main figure: paths + marginal at t=1, three H values
  fig_B_time_evolution.png  How marginal evolves over t for H=0.7
  fig_C_non_gaussianity.png Skewness and kurtosis bar chart vs theory
  fig_D_overlay_H.png       All three H values on one marginal plot (t=1)
"""

from path_simulation import (
    WaveletRosenblatt as SharedWaveletRosenblatt,
    DonskerRosenblatt as SharedDonskerRosenblatt,
)
from density_simulation import RosenblattDensityLP
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
from scipy.stats import gaussian_kde, norm
from scipy.special import gamma as gamma_fn
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")


os.makedirs("output/marginal/", exist_ok=True)
np.random.seed(42)

PAL = dict(wavelet="#3a86c8", donsker="#e07b39", markov="#42a861",
           lp="#c0392b", gauss="#999999")
H_COLS = {0.6: "#9b59b6", 0.7: "#2980b9", 0.8: "#27ae60"}

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ═══════════════════════════════════════════════════════════════
#  LP exact density
# ═══════════════════════════════════════════════════════════════

def lp_density(a, x_min=-5.0, x_max=9.0, K=200, N=2**16, z_max=40.0):
    lp = RosenblattDensityLP(a=a, K=K)
    return lp.density_fft(x_min=x_min, x_max=x_max, N=N, z_max=z_max)


# Use the shared Donsker implementation from path_simulation.py.
DonskerRosenblatt = SharedDonskerRosenblatt


# ═══════════════════════════════════════════════════════════════
#  Algorithm 3: Markovian OU (corrected)
# ═══════════════════════════════════════════════════════════════

class MarkovianRosenblatt:
    def __init__(self, H=0.7, n_modes=50, r=2.0, T=1.0, M=400):
        assert 0.5 < H < 1.0
        self.H = H
        self.Ht = H-0.5
        self.n = n_modes
        self.r = r
        self._T = T
        self._M = M
        self._build(T, M)

    def _build(self, T, M):
        Ht = self.Ht
        n = self.n
        r = self.r
        dt = T/M
        gp = 0.5-Ht
        dp = Ht
        xmin = max(n**(-r/gp), 1e-10)
        xmax = min(n**(r/dp), 0.1/dt)
        if xmax <= xmin:
            xmin, xmax = 1e-4, max(10/T, 1e-3)
        lx = np.linspace(np.log(xmin), np.log(xmax), n)
        xk = np.exp(lx)
        dlg = (lx[-1]-lx[0])/(n-1) if n > 1 else 1.0
        nc = gamma_fn(Ht+0.5)*gamma_fn(0.5-Ht)
        wk = xk**(0.5-Ht)*dlg/nc
        self.xk = xk
        self.wk = wk
        self._wo = np.outer(wk, wk)
        self._sx = xk[:, None]+xk[None, :]

    def _va(self, ts):
        va = np.zeros(len(ts))
        for i, t in enumerate(ts):
            if t > 0:
                va[i] = np.sum(self._wo*(1-np.exp(-self._sx*t))/self._sx)
        return va

    def _cov(self, s, t):
        if s > t:
            s, t = t, s
        return float(np.sum(self._wo*(1-np.exp(-self._sx*s))/self._sx
                            * np.exp(-self.xk*(t-s))[None, :]))

    def _c(self, T):
        nc = 60
        dc = T/nc
        tc = np.arange(1, nc+1)*dc
        EV = np.zeros((nc, nc))
        for i in range(nc):
            EV[i, i] = np.sum(self._wo*(1-np.exp(-self._sx*tc[i]))/self._sx)
            for j in range(i+1, nc):
                v = self._cov(tc[i], tc[j])
                EV[i, j] = v
                EV[j, i] = v
        intg = 2*np.sum(EV**2)*dc**2
        return np.sqrt(T**(2*self.H)/intg) if intg > 1e-15 else 1.0

    def simulate_paths_batch(self, T=1.0, M=400, n_paths=100):
        if T != self._T or M != self._M:
            self._T = T
            self._M = M
            self._build(T, M)
        dt = T/M
        ts = np.linspace(0, T, M+1)
        va = self._va(ts)
        c = self._c(T)
        dc = np.exp(-self.xk*dt)
        Z = np.zeros((n_paths, M+1))
        for p in range(n_paths):
            U = np.zeros(self.n)
            for j in range(M):
                dW = np.random.randn()*np.sqrt(dt)
                U = U*dc+dW
                V = np.dot(self.wk, U)
                Z[p, j+1] = Z[p, j]+c*(V**2-va[j+1])*dt
        return ts, Z


# Use the shared Wavelet implementation from path_simulation.py.
WaveletRosenblatt = SharedWaveletRosenblatt


# ═══════════════════════════════════════════════════════════════
#  Simulation for one H value
# ═══════════════════════════════════════════════════════════════

def run_H(H, T=1.0, n_wav=250, n_don=45, n_mou=800, N_don=15, M=400):
    print(f"  H={H}", end=" ", flush=True)

    t0 = time.time()
    wav = WaveletRosenblatt(H=H, J=7)
    tw, Zw = wav.simulate_paths_batch(T=T, n_points=M, n_paths=n_wav)
    print(f"wav:{time.time()-t0:.0f}s", end=" ", flush=True)

    t0 = time.time()
    don = DonskerRosenblatt(H=H)
    td, Zd = don.simulate_paths_batch(T=T, N=N_don, n_paths=n_don)
    print(f"don:{time.time()-t0:.0f}s", end=" ", flush=True)

    t0 = time.time()
    xd, yd = lp_density(1-H, x_min=-5, x_max=9)
    print(f"lp:{time.time()-t0:.0f}s")

    return (tw, Zw), (td, Zd), (xd, yd)


# ═══════════════════════════════════════════════════════════════
#  Helper: extract samples and density at time t
# ═══════════════════════════════════════════════════════════════

def samples_at(Z, tg, t_fix):
    return Z[:, np.argmin(np.abs(tg-t_fix))]


def scaled_density(x_lp, d_lp, H, t_fix):
    """Scale Z(1) density to Z(t_fix)=^d t_fix^H Z(1)."""
    sc = t_fix**H
    return x_lp*sc, d_lp/sc


# ═══════════════════════════════════════════════════════════════
#  Figure A — Paths + marginal, 3 rows (H), 3 cols (method)
# ═══════════════════════════════════════════════════════════════

def fig_A(all_data, H_vals, T=1.0, n_show=15, show_donsker=True):
    """
    Grid where each row represents a single method-H combination.
    Each cell: paths (left 3/4) + horizontal marginal at t=T (right 1/4).
    
    Parameters:
    -----------
    show_donsker : bool
        If True, both Wavelet and Donsker methods are rendered in separate rows.
        If False, only Wavelet rows are plotted.
    """
    # Dynamic setup based on whether we display Donsker
    methods = ["wavelet", "donsker"] if show_donsker else ["wavelet"]
    
    # Each H gets a row for every active method
    nrow = len(methods)
    ncol = len(H_vals) 
    print(f"  Creating figure with {nrow} rows and {ncol} columns (show_donsker={show_donsker})")
    fig = plt.figure(figsize=(10, 7.5 * nrow))
    outer = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.25, wspace=0.15)

    method_metadata = {
        "wavelet": {
            "label": "Wavelet  (Abry & Pipiras 2000)",
            "color": PAL["wavelet"],
            "data_idx": 0  # index mapping inside datasets collection
        },
        "donsker": {
            "label": "Donsker  (Torres & Tudor 2007, N=15)",
            "color": PAL["donsker"],
            "data_idx": 1
        }
    }

    for h_idx, H in enumerate(H_vals):
        (tw, Zw), (td, Zd), (xd, yd) = all_data[H]
        # Store both, though we might only query index 0 (wavelet)
        datasets = [(tw, Zw), (td, Zd)]

        for m_idx, method_name in enumerate(methods):
            meta = method_metadata[method_name]
            cc = meta["color"]
            tg, Zp = datasets[meta["data_idx"]]

            # Split the outer slot into Path (3) and Marginal (1.1) subplots
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[m_idx, h_idx],
                width_ratios=[3, 1.1], wspace=0.05
            )
            ax_p = fig.add_subplot(inner[0])
            ax_m = fig.add_subplot(inner[1], sharey=ax_p)

            # ── paths ──────────────────────────────────
            ns = min(n_show, len(Zp))
            for i in range(ns):
                ax_p.plot(tg, Zp[i], color=cc, alpha=0.25, lw=0.6)
            ax_p.plot(tg, np.mean(Zp, axis=0),
                      color=cc, lw=2.5, alpha=0.95, label="Mean")
            ax_p.axhline(0, color="#cccccc", lw=0.7, ls="--")
            ax_p.axvline(T, color="#444444", lw=1.3, ls=":", label=f"$t={T}$")
            ax_p.set_xlabel("Time $t$", fontsize=9)
            ax_p.set_ylabel("$Z_t$", fontsize=9)
            if h_idx == 0:
                ax_p.annotate(meta["label"], xy=(-0.18, 0.5), xycoords="axes fraction", 
                              rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")

            ax_p.legend(fontsize=7.5, loc="upper right")
            ax_p.grid(True, alpha=0.15)
            
            ax_p.text(0.02, 0.97, f"$H={H}$", transform=ax_p.transAxes,
                      fontsize=9, va="top",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            # ── marginal ───────────────────────────────
            s = samples_at(Zp, tg, T)
            xs, ds = scaled_density(xd, yd, H, T)

            s_min_robust = np.percentile(s, 0.5)
            s_max_robust = np.percentile(s, 99.5)
            
            if np.any(ds > 1e-3):
                ylo = min(s_min_robust, xs[ds > 1e-3][0]) - 0.2
                yhi = max(s_max_robust, xs[ds > 1e-3][-1]) + 0.2
            else:
                ylo, yhi = s_min_robust - 0.2, s_max_robust + 0.2

            # Enforce robust limits
            ax_p.set_ylim(ylo, yhi)

            # Binning configs
            bins = np.linspace(s.min(), s.max(), 60)

            ax_m.hist(s, bins=bins, orientation="horizontal",
                      density=True, color=cc, alpha=0.35)
            try:
                kde = gaussian_kde(s, bw_method="scott")
                yk = np.linspace(s.min(), s.max(), 400)
                ax_m.plot(kde(yk), yk, color=cc, lw=2.0, label="KDE")
            except Exception:
                pass

            ax_m.plot(ds, xs, color=PAL["lp"], lw=2.3,
                      ls="-", label="LP exact", zorder=6)
            
            yg = np.linspace(s.min(), s.max(), 300)
            ax_m.plot(norm.pdf(yg, 0, T**H), yg,
                      color=PAL["gauss"], lw=1.2, ls="--", alpha=0.7, label="Gauss")

            sk = sp_skew(s)
            ku = sp_kurtosis(s)
            ok_s = "ok" if sk > 0.05 else "??"
            ok_k = "ok" if ku > 0.05 else "??"
            ax_m.text(0.96, 0.97,
                      f"$n={len(s)}$\nskew$={sk:.2f}$ ({ok_s})\nexkurt$={ku:.2f}$ ({ok_k})",
                      transform=ax_m.transAxes, fontsize=7, va="top", ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

            ax_m.set_xlabel("Density", fontsize=8)
            ax_m.tick_params(labelleft=False)
            ax_m.xaxis.set_major_locator(MaxNLocator(3))
            ax_m.grid(True, alpha=0.15)
            
            # Show legend on the very first active marginal block
            if m_idx == 0 and h_idx == 0:
                ax_m.legend(fontsize=7, loc="lower right")

    out = "output/marginal/figA_paths_and_marginal.png"
    plt.savefig(out, dpi=135, bbox_inches="tight")
    plt.savefig("imgs/figA_paths_and_marginal.png", dpi=135, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════
#  Figure B — Time evolution of marginal (H=0.7, 4 time points)
# ═══════════════════════════════════════════════════════════════

def fig_B(all_data, H=0.7, t_vals=None):
    if t_vals is None:
        t_vals = [0.25, 0.5, 0.75, 1.0]
    (tw, Zw), (td, Zd), (xd, yd) = all_data[H]
    T = 1.0

    fig, axes = plt.subplots(1, len(t_vals), figsize=(16, 5), sharey=False)

    for ax, t_fix in zip(axes, t_vals):
        xs, ds = scaled_density(xd, yd, H, t_fix)
        ax.plot(xs, ds, color=PAL["lp"], lw=2.8, label="LP exact", zorder=8)

        yg = np.linspace(xs[0], xs[-1], 300)
        ax.plot(yg, norm.pdf(yg, 0, t_fix**H), color=PAL["gauss"],
                lw=1.3, ls="--", alpha=0.65, label="Gaussian")

        def add_kde(Zp, tg, cc, lab, lw=1.8, ls="-"):
            s = samples_at(Zp, tg, t_fix)
            try:
                kde = gaussian_kde(s, bw_method="scott")
                y = np.linspace(xs[0], xs[-1], 400)
                ax.plot(y, kde(y), color=cc, lw=lw, ls=ls, label=lab)
            except Exception:
                pass

        add_kde(Zw, tw, PAL["wavelet"], "Wavelet")
        add_kde(Zd, td, PAL["donsker"], "Donsker", lw=1.5, ls=":")

        ax.set_xlabel(f"$Z_{{t}}$, $t={t_fix}$", fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_title(
            f"$t = {t_fix}$\n$\\sigma_{{th}} = {t_fix**H:.3f}$", fontsize=11)
        ax.grid(True, alpha=0.18)
        if t_fix == t_vals[0]:
            ax.legend(fontsize=8.5)

    fig.suptitle(
        f"Marginal Distribution at Four Time Points   ($H={H}$)\n"
        "KDEs from path simulation vs LP exact density — all should be non-Gaussian",
        fontsize=13, fontweight="bold")
    out = "output/marginal/figB_time_evolution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=135, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════
#  Figure C — Skewness and kurtosis bar chart
# ═══════════════════════════════════════════════════════════════

def fig_C(all_data, H_vals):
    """Bar chart: skewness and excess kurtosis at t=1 for all methods × H."""
    T = 1.0
    method_names = ["Wavelet", "Donsker"]
    method_cols = [PAL["wavelet"], PAL["donsker"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (stat_fn, stat_name, theo_key) in enumerate([
        (sp_skew, "Skewness", "skew"),
        (sp_kurtosis, "Excess Kurtosis", "kurt"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(H_vals))
        w = 0.22
        offsets = [-w, w]

        for mi, (mname, mc, offs) in enumerate(zip(method_names, method_cols, offsets)):
            vals = []
            for H in H_vals:
                dat = all_data[H]
                datasets = [(dat[0][0], dat[0][1]), (dat[1][0], dat[1][1])]
                tg, Zp = datasets[mi]
                s = samples_at(Zp, tg, T)
                vals.append(stat_fn(s))
            ax.bar(x+offs, vals, width=w*0.9, color=mc, alpha=0.8,
                   label=mname, zorder=3)

        # Theoretical values from LP density at t=1
        for hi, H in enumerate(H_vals):
            xd, yd = all_data[H][2]
            xs, ds = scaled_density(xd, yd, H, T)
            _trap = np.trapz
            m1 = _trap(xs*ds, xs); cx = xs - m1
            m2 = _trap(cx**2*ds, xs)
            m3 = _trap(cx**3*ds, xs)
            m4 = _trap(cx**4*ds, xs)
            if stat_name == "Skewness":
                th = m3/m2**1.5
            else:
                th = m4/m2**2-3
            ax.scatter([x[hi]], [th], marker="*", s=200,
                       color=PAL["lp"], zorder=5,
                       label="LP theory" if hi == 0 else "")

        ax.axhline(0, color="#444444", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$H={H}$" for H in H_vals], fontsize=11)
        ax.set_ylabel(stat_name, fontsize=11)
        ax.set_title(f"{stat_name} at $t=1$", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Non-Gaussianity Check: Skewness > 0 and Excess Kurtosis > 0\n"
        "All methods should have positive values  (Rosenblatt $\\in$ 2nd Wiener chaos)\n"
        "Star = LP exact theory value",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "output/marginal/figC_non_gaussianity.png"
    plt.savefig(out, dpi=135, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════
#  Figure D — All H values overlaid on one marginal plot (t=1)
# ═══════════════════════════════════════════════════════════════

def fig_D(all_data, H_vals, T=1.0):
    """One plot per method, all H values overlaid."""
    method_labels = ["Wavelet (Alg 1)", "Donsker (Alg 2)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, mi, mlabel in zip(axes, range(2), method_labels):
        for H in H_vals:
            cc = H_COLS[H]
            dat = all_data[H]
            tg, Zp = dat[mi][:2]
            s = samples_at(Zp, tg, T)
            xd, yd = dat[2]
            xs, ds = scaled_density(xd, yd, H, T)

            # KDE from simulation
            try:
                kde = gaussian_kde(s, bw_method="scott")
                y = np.linspace(xs[0], xs[-1], 400)
                ax.plot(y, kde(y), color=cc, lw=2.0, ls="-.",
                        alpha=0.8, label=f"$H={H}$ KDE")
            except Exception:
                pass

            # Exact LP density (solid, same colour but darker)
            darker = cc  # same colour; distinguish by linestyle
            ax.plot(xs, ds, color=cc, lw=2.5, ls="-",
                    alpha=0.95, label=f"$H={H}$ LP exact")

        ax.set_xlabel("$Z_1$", fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, ncol=2)
        ax.grid(True, alpha=0.18)

    fig.suptitle(
        "Marginal at $t=1$ for Three Hurst Values\n"
        "Solid = LP exact density, Dash-dot = path simulation KDE\n"
        "Good methods: KDE closely follows LP exact curve",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "output/marginal/figD_H_overlay.png"
    plt.savefig(out, dpi=135, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════
#  Summary statistics table
# ═══════════════════════════════════════════════════════════════

def print_table(all_data, H_vals, T=1.0):
    _trap = np.trapz
    method_names = ["Wavelet", "Donsker"]

    print("\n" + "="*86)
    print(f"  MARGINAL STATISTICS AT t=T={T}")
    print("="*86)
    hdr = (f"  {'H':>4}  {'Method':<12}  {'n':>5}  {'Mean':>7}  "
           f"{'Var/th':>7}  {'Skew':>7}  {'ExKurt':>7}  {'OK?':<6}")
    print(hdr)
    print("  "+"-"*82)

    for H in H_vals:
        th_var = T**(2*H)
        dat = all_data[H]
        xd, yd = dat[2]
        xs, ds = scaled_density(xd, yd, H, T)
        m2 = _trap(xs**2*ds, xs)
        m3 = _trap(xs**3*ds, xs)
        m4 = _trap(xs**4*ds, xs)
        sk_th = m3/m2**1.5
        ku_th = m4/m2**2-3

        for mi, mname in enumerate(method_names):
            tg, Zp = dat[mi][:2]
            s = samples_at(Zp, tg, T)
            mu = np.mean(s)
            v = np.var(s)
            sk = sp_skew(s)
            ku = sp_kurtosis(s)
            ok = "PASS" if (0.5 < v/th_var < 1.5 and sk >
                            0.05 and ku > 0.05) else "WARN"
            print(f"  {H:>4.1f}  {mname:<12}  {len(s):>5}  {mu:>7.3f}  "
                  f"{v/th_var:>7.3f}  {sk:>7.3f}  {ku:>7.3f}  {ok}")

        print(f"  {H:>4.1f}  {'LP theory':<12}  {'–':>5}  {'0':>7}  "
              f"{'1.000':>7}  {sk_th:>7.3f}  {ku_th:>7.3f}  EXACT")
        print("  " + "-"*82)
    print()


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_all = time.time()
    H_vals = [0.6, 0.7, 0.8]

    print("="*60)
    print("Rosenblatt — Comprehensive Marginal Validation")
    print("="*60)

    all_data = {}
    for H in H_vals:
        print(f"\nH = {H}:")
        all_data[H] = run_H(H, T=1.0, n_wav=250, n_don=45,
                            n_mou=800, N_don=15, M=400)

    print_table(all_data, H_vals)

    print("Generating figures …")
    fig_A(all_data, [0.6, 0.7], n_show=14, show_donsker=False)
    fig_B(all_data, H=0.7, t_vals=[0.25, 0.5, 0.75, 1.0])
    fig_C(all_data, H_vals)
    fig_D(all_data, H_vals)

    print(f"\nTotal: {time.time()-t_all:.1f}s")
    print("Figures in /output/marginal/")
