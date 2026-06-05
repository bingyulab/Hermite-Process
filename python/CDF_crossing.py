import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from density_simulation import RosenblattDensityLP

# ── Characteristic function inversion → density → CDF ──────────────────────

def rosenblatt_pdf_cdf(D: float, M: int = 400, N_grid: int = 6000,
                       x_range: tuple = (-5, 8)):
    lp = RosenblattDensityLP(a=D, K=M)
    x_fft, pdf_fft = lp.density_fft(x_min=x_range[0], x_max=x_range[1],
                                     N=2**17, z_max=80.0)
    x_grid   = np.linspace(x_range[0], x_range[1], N_grid)
    pdf_grid = np.maximum(np.interp(x_grid, x_fft, pdf_fft), 0.0)
    cdf_grid = cumulative_trapezoid(pdf_grid, x_grid, initial=0)
    cdf_grid /= cdf_grid[-1]
    return x_grid, pdf_grid, cdf_grid
 
# ── Compute CDFs ─────────────────────────────────────────────────────────────
 
D_vals = [0.10, 0.20, 0.30, 0.40, 0.45]
colors = ["#1a237e", "#1565c0", "#0288d1", "#26a69a", "#ef6c00"]
labels = [f"$D={D}$  ($H={1-D}$)" for D in D_vals]
 
print("Computing CDFs …")
results = {}
for D in D_vals:
    print(f"  D={D}…", flush=True)
    results[D] = rosenblatt_pdf_cdf(D, M=300)
 
# ── Find crossing points (Bug 1 fix) ─────────────────────────────────────────
# Strategy: compare only the EXTREME pair (D_min vs D_max).
# This is the most separated pair and gives the cleanest signal.
# Restrict to CDF ∈ (0.01, 0.99) to avoid numerical noise deep in the tails.
# Use threshold = 0 to separate left (negative x) from right (positive x).
 
x_dense = np.linspace(-4.5, 7.0, 6000)   # high-resolution evaluation grid
 
c_lo = np.interp(x_dense, results[D_vals[ 0]][0], results[D_vals[ 0]][2])  # D=0.10
c_hi = np.interp(x_dense, results[D_vals[-1]][0], results[D_vals[-1]][2])  # D=0.45
 
diff_extremes = c_lo - c_hi
# Mask: only look for crossings where both CDFs are well away from 0 or 1
valid = (np.minimum(c_lo, c_hi) > 0.01) & (np.maximum(c_lo, c_hi) < 0.99)
diff_valid = np.where(valid, diff_extremes, 0.0)
 
crossing_xs = []
signs = np.sign(diff_valid)
for j in range(len(signs) - 1):
    if signs[j] != 0 and signs[j+1] != 0 and signs[j] != signs[j+1]:
        # Linear interpolation for sub-grid precision
        x1, x2 = x_dense[j], x_dense[j+1]
        d1, d2 = diff_valid[j], diff_valid[j+1]
        x_cross = x1 - d1 * (x2 - x1) / (d2 - d1)
        crossing_xs.append(x_cross)
 
crossing_xs = np.array(crossing_xs)
 
# Threshold = 0: left crossing is at negative x, right crossing at positive x
neg = crossing_xs[crossing_xs < 0]
pos = crossing_xs[crossing_xs > 0]
if len(neg) == 0 or len(pos) == 0:
    raise RuntimeError("CDF crossings not found in (0.01, 0.99); "
                       "check the density inversion before annotating.")
left_cross  = float(neg.mean())
right_cross = float(pos.mean())

print(f"Left crossing  ≈ {left_cross:.3f}")
print(f"Right crossing ≈ {right_cross:.3f}")
 
# ── y-values at crossings: MEAN across all D (Bug 2 fix) ─────────────────────
# At the true crossing x, all CDFs should agree; averaging removes interpolation
# rounding noise and, crucially, avoids using a single (possibly very different)
# curve as the centre of the inset y-axis.
 
def mean_cdf_at(x_val: float) -> float:
    return float(np.mean([np.interp(x_val, results[D][0], results[D][2])
                          for D in D_vals]))
 
y_left  = mean_cdf_at(left_cross)
y_right = mean_cdf_at(right_cross)
 
# ── Dynamic inset y-limits helper (Bug 3 fix) ─────────────────────────────────
# Compute the actual data range of ALL five curves inside the visible x-window,
# then add 20 % padding. This is robust regardless of where the curves sit.
 
def inset_ylim(x_centre: float, half_width: float, pad: float = 0.20):
    mask = (x_dense >= x_centre - half_width) & (x_dense <= x_centre + half_width)
    y_all = np.concatenate([
        np.interp(x_dense[mask], results[D][0], results[D][2])
        for D in D_vals
    ])
    lo, hi = y_all.min(), y_all.max()
    margin = max((hi - lo) * pad, 0.005)   # at least 0.005 to avoid degenerate range
    return lo - margin, hi + margin
 
# ── Figure layout ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "serif",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.labelsize"     : 12,
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "legend.fontsize"    : 10,
})
 
fig     = plt.figure(figsize=(14, 5.5))
ax_main = fig.add_axes([0.06, 0.12, 0.54, 0.80])   # full CDF
ax_left = fig.add_axes([0.64, 0.52, 0.16, 0.38])   # left crossing zoom
ax_right= fig.add_axes([0.82, 0.52, 0.16, 0.38])   # right crossing zoom
ax_diff = fig.add_axes([0.64, 0.12, 0.34, 0.32])   # CDF difference panel
 
# ── Main CDF panel ────────────────────────────────────────────────────────────
for D, col, lab in zip(D_vals, colors, labels):
    c_dense = np.interp(x_dense, results[D][0], results[D][2])
    ax_main.plot(x_dense, c_dense, color=col, lw=1.8, label=lab)
 
ax_main.axvspan(left_cross  - 0.15, left_cross  + 0.15, alpha=0.10, color="#c0392b", zorder=0)
ax_main.axvspan(right_cross - 0.15, right_cross + 0.15, alpha=0.10, color="#c0392b", zorder=0)
 
ax_main.annotate(
    "crossing 1\n$x\\approx{:.2f}$".format(left_cross),
    xy=(left_cross, y_left),
    xytext=(left_cross - 1.4, y_left + 0.14),
    arrowprops=dict(arrowstyle="->", color="#c0392b"),
    color="#c0392b", fontsize=9.5,
)
ax_main.annotate(
    "crossing 2\n$x\\approx{:.2f}$".format(right_cross),
    xy=(right_cross, y_right),
    xytext=(right_cross + 0.7, y_right - 0.12),
    arrowprops=dict(arrowstyle="->", color="#c0392b"),
    color="#c0392b", fontsize=9.5,
)
 
ax_main.set_xlabel("$x$", fontsize=13)
ax_main.set_ylabel("$F_D(x)$", fontsize=13)
ax_main.set_title("CDF of the standardised Rosenblatt distribution for varying $D$",
                  fontsize=12)
ax_main.legend(loc="upper left", framealpha=0.6)
ax_main.set_xlim(-4.5, 7.0)
ax_main.set_ylim(-0.02, 1.02)
ax_main.grid(True, alpha=0.18)
 
# ── Left crossing inset ───────────────────────────────────────────────────────
INSET_HALF = 0.40   # ± x-window half-width for both insets
 
for D, col in zip(D_vals, colors):
    c_dense = np.interp(x_dense, results[D][0], results[D][2])
    ax_left.plot(x_dense, c_dense, color=col, lw=1.5)
 
ax_left.set_xlim(left_cross - INSET_HALF, left_cross + INSET_HALF)
ax_left.set_ylim(*inset_ylim(left_cross, INSET_HALF))
ax_left.axvline(left_cross, color="#c0392b", lw=0.9, ls="--", alpha=0.7)
ax_left.set_title("Left crossing", fontsize=9.5)
ax_left.set_xlabel("$x$", fontsize=9)
ax_left.tick_params(labelsize=8)
ax_left.grid(True, alpha=0.2)
 
# ── Right crossing inset ──────────────────────────────────────────────────────
for D, col in zip(D_vals, colors):
    c_dense = np.interp(x_dense, results[D][0], results[D][2])
    ax_right.plot(x_dense, c_dense, color=col, lw=1.5)
 
ax_right.set_xlim(right_cross - INSET_HALF, right_cross + INSET_HALF)
ax_right.set_ylim(*inset_ylim(right_cross, INSET_HALF))   # Bug 2+3 fix
ax_right.axvline(right_cross, color="#c0392b", lw=0.9, ls="--", alpha=0.7)
ax_right.set_title("Right crossing", fontsize=9.5)
ax_right.set_xlabel("$x$", fontsize=9)
ax_right.tick_params(labelsize=8)
ax_right.grid(True, alpha=0.2)
 
# ── CDF difference panel ──────────────────────────────────────────────────────
x_base, _, cdf_base = results[D_vals[0]]
 
ax_diff.axhline(0, color="#333", lw=0.9)
for D, col in zip(D_vals[1:], colors[1:]):
    diff = (np.interp(x_dense, results[D][0], results[D][2])
            - np.interp(x_dense, x_base, cdf_base))
    ax_diff.plot(x_dense, diff, color=col, lw=1.5, label=f"$D={D}$")
 
ax_diff.axvspan(left_cross  - 0.15, left_cross  + 0.15, alpha=0.10, color="#c0392b", zorder=0)
ax_diff.axvspan(right_cross - 0.15, right_cross + 0.15, alpha=0.10, color="#c0392b", zorder=0)
ax_diff.set_xlim(-4.5, 7.0)
ax_diff.set_xlabel("$x$", fontsize=11)
ax_diff.set_ylabel("$F_D - F_{D=0.10}$", fontsize=10)
ax_diff.set_title("CDF difference relative to $D=0.10$", fontsize=9.5)
ax_diff.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.5)
ax_diff.grid(True, alpha=0.18)
 
plt.savefig("output/experiments/exp2_cdf_crossing.png", dpi=160, bbox_inches="tight")
print("Saved exp2_cdf_crossing.png")