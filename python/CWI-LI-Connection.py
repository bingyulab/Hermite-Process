import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Global Plot Settings for Academic Quality
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False, # Set to True if you have a full LaTeX engine installed
})

# ==========================================
# Figure 1: 1D Łojasiewicz vs Non-Analytic
# ==========================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

t = np.linspace(0.001, 1, 1000)
# Polynomial P(t) = t^2 (Łojasiewicz exponent 2)
P_t = t**2 
# Smooth non-analytic f(t) = exp(-c/t^2)
# FIX: Increased coefficient to 0.5 so the curve stays strictly BELOW P(t) without crossing
f_t = np.exp(-0.5 / t**2) 

# Left Panel: Linear Scale
ax1.plot(t, P_t, 'b-', lw=2, label=r'Polynomial $P(t) = t^2$')
ax1.plot(t, f_t, 'r-', lw=2, label=r'Smooth non-analytic $f(t) = e^{-0.5/t^2}$')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r'Distance to zero set, $t$')
ax1.set_ylabel(r'Function value')
ax1.set_title('Linear Scale: Behavior near zero')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right Panel: Log-Log Scale
t_log = np.logspace(-1.5, 0, 500)
P_t_log = t_log**2
f_t_log = np.exp(-0.5 / t_log**2)

ax2.loglog(t_log, P_t_log, 'b-', lw=2, label=r'Constant slope (Exponent $\theta=2$)')
ax2.loglog(t_log, f_t_log, 'r-', lw=2, label=r'Slope $\to \infty$ (No finite $\theta$)')
ax2.set_xlim(10**-1.5, 1)
ax2.set_ylim(10**-10, 1)
ax2.set_xlabel(r'Distance to zero set, $t$ (log scale)')
ax2.set_ylabel(r'Function value (log scale)')
# FIX: Replaced \L{} with unicode Ł to prevent missing glyph "cross" boxes
ax2.set_title('Log-Log Scale: The Łojasiewicz Exponent')
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig1.savefig('./output/lojasiewicz_vs_analytic.pdf', dpi=300, bbox_inches='tight')
print("Saved 1D plot to ./output/lojasiewicz_vs_analytic.pdf")

# ==========================================
# Figure 2: 2D Geometry (Tube, Zero Set, Gaussian)
# ==========================================
fig2, ax = plt.subplots(figsize=(7, 6))

# Grid setup
x = np.linspace(-2.5, 2.5, 400)
y = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x, y)

# Polynomial P(x,y) = x^2 - y (Zero set is a parabola)
P = X**2 - Y
lam = 0.5 # The lambda threshold for the tube

# Gaussian Density
Gaussian = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)

# 1. Plot the Gaussian "blue haze" background
contour_bg = ax.contourf(X, Y, Gaussian, levels=50, cmap='Blues', alpha=0.6)

# 2. Plot the orange tube |P| <= lambda
# FIX: explicitly cast the boolean array to float to prevent contourf warnings/crashes
tube = (np.abs(P) <= lam).astype(float)
ax.contourf(X, Y, tube, levels=[0.5, 1.5], colors=['orange'], alpha=0.3)

# 3. Plot the zero set P = 0 (Red line)
ax.contour(X, Y, P, levels=[0], colors=['red'], linewidths=2.5)

# 4. Annotations
# Add a specific point x and show distance to Z
x0, y0 = 1.2, -0.5 # A point inside the tube
ax.plot(x0, y0, 'ko', markersize=6) # The point X
ax.text(x0 + 0.1, y0 - 0.1, r'$x$', fontsize=14, fontweight='bold')

# Shortest distance to Z (approximate projection onto parabola)
x_proj, y_proj = 0.77, 0.77**2 
ax.plot([x0, x_proj], [y0, y_proj], 'k--', lw=1.5)
ax.text(0.9, 0.0, r'$\mathrm{dist}(x, \mathcal{Z})$', fontsize=12, rotation=55)

# Labels for the sets
ax.text(-1.8, 2.0, r'$\mathcal{Z} = \{P = 0\}$', color='red', fontsize=14, fontweight='bold')
ax.text(1.3, 2.2, r'$\{|P| \leq \lambda\}$', color='darkorange', fontsize=14, fontweight='bold')
ax.text(-2.3, -2.0, r'Gaussian density $\propto e^{-|x|^2/2}$', color='navy', fontsize=12)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
# FIX: Replaced \L{} with unicode Ł
ax.set_title(r'Geometric connection: Łojasiewicz and Carbery--Wright')

plt.tight_layout()
fig2.savefig('./output/carbery_wright_geometry.pdf', dpi=300, bbox_inches='tight')
print("Saved 2D plot to     ./output/carbery_wright_geometry.pdf")