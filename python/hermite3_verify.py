import numpy as np
import matplotlib.pyplot as plt

def simulate_X3_inverse_moments(H=0.7, q=3, N=60, M=5000):
    """
    Numerically simulates the Malliavin variance X_3 of the Hermite process (q=3).
    Evaluates the inverse moments E[X_3^{-p}] to observe convergence limits.
    
    Parameters:
      H : Hurst parameter (0.5 < H < 1)
      N : Number of time grid points for discretization
      M : Number of Monte Carlo sample paths
    """
    dt = 1.0 / N
    t_grid = np.linspace(dt, 1.0, N)
    
    # Exponent for the fractional kernel
    H0 = 1.0 + (H - 1.0) / q
    alpha = H0 - 1.5
    
    # Precompute the kernel matrix g_r(xi_1, xi_2)
    # g_r(xi_1, xi_2) = int_{max(r, xi_1, xi_2)}^1 (s-r)^alpha (s-xi_1)^alpha (s-xi_2)^alpha ds
    g = np.zeros((N, N, N))
    
    print("Precomputing 3D fractional kernel...")
    for i, r in enumerate(t_grid):
        for j, xi1 in enumerate(t_grid):
            for k, xi2 in enumerate(t_grid):
                start_idx = max(i, j, k)
                if start_idx < N - 1:
                    s_vals = t_grid[start_idx+1:]
                    integrand = (s_vals - r)**alpha * (s_vals - xi1)**alpha * (s_vals - xi2)**alpha
                    g[i, j, k] = np.sum(integrand) * dt
                    
    # Ensure off-diagonal evaluation for the multiple Ito integral
    np.fill_diagonal(g[0, :, :], 0)
    for i in range(N):
        np.fill_diagonal(g[i, :, :], 0)

    print(f"Simulating {M} sample paths...")
    X3_samples = np.zeros(M)
    
    for m in range(M):
        # Generate Brownian increments
        dB = np.random.normal(0, np.sqrt(dt), N)
        
        # Compute I_2(g_r) for each r
        # I_2(g_r) = sum_{j != k} g_r(xi_j, xi_k) dB_j dB_k
        I2_gr = np.zeros(N)
        for i in range(N):
            # Matrix multiplication for the quadratic form
            I2_gr[i] = np.dot(dB, np.dot(g[i, :, :], dB))
            
        # Integrate over r to compute the full Malliavin variance X_3
        # X_3 = 9 * int_0^1 [I_2(g_r)]^2 dr
        X3_samples[m] = 9.0 * np.sum(I2_gr**2) * dt

    # Evaluate inverse moments E[min(X3, cutoff)^{-p}]
    cutoffs = np.logspace(-4, 0, 20)
    p_values = [0.25, 0.5, 1.0, 2.0]
    
    plt.figure(figsize=(10, 6))
    for p in p_values:
        moments = [np.mean(np.minimum(X3_samples, c)**(-p)) for c in cutoffs]
        plt.plot(cutoffs, moments, marker='o', label=f'p={p}')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Regularization Cutoff (log scale)')
    plt.ylabel('E[X_3^{-p}] (log scale)')
    plt.title(f'Inverse Moments of Malliavin Variance (q=3, H={H})\nDivergence indicates infinite inverse moment')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./output/experiments/hermite3_inverse_moments.png",
                dpi=150, bbox_inches="tight")

    return X3_samples

if __name__ == "__main__":
    simulate_X3_inverse_moments()