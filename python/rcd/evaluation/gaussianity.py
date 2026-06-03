import math
import numpy as np
import torch


def compute_spectrum_stats(
        X: torch.Tensor,
        q: int = 512,
) -> dict[str, float]:
    """
    Compute spectral geometry of the activation matrix X ∈ ℝ^{N × D}.
 
    Returns
    -------
    pr              : Participation Ratio  = (Σλ)² / Σλ²
                      Measures effective dimensionality (1 ≤ PR ≤ min(N,D)).
    effective_rank  : exp( H(λ/Σλ) )  — Entropy-based effective rank.
    top10_var_frac  : Σ_{i≤10} λ_i / Σ λ_i  — Dominance of top 10 modes.
    spectral_gap    : λ_1 / λ_2  — Ratio of top two eigenvalues.
    """
    X = X.float()
    N, D = X.shape
    q = min(q, N - 1, D)
    if q < 2:
        nan = float("nan")
        return {"pr": nan, "effective_rank": nan,
                "top10_var_frac": nan, "spectral_gap": nan}
 
    Xc = X - X.mean(0)
    # pca_lowrank returns (U, S, Vh); S are sqrt(eigenvalues * N)
    _, S, _ = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
    lam = (S ** 2).cpu()                    # eigenvalues (unnormalised)
    lam = lam.clamp(min=0.0)
 
    lam_sum = lam.sum().item()
    if lam_sum < 1e-12:
        nan = float("nan")
        return {"pr": nan, "effective_rank": nan,
                "top10_var_frac": nan, "spectral_gap": nan}
 
    # Participation Ratio
    pr = float((lam_sum ** 2) / (lam ** 2).sum().item())
 
    # Effective rank via normalised entropy
    p = lam / lam_sum
    p = p.clamp(min=1e-12)
    H = float(-(p * p.log()).sum().item())
    eff_rank = float(math.exp(H))
 
    # Top-10 variance fraction
    top10 = float(lam[:10].sum().item() / lam_sum)
 
    # Spectral gap
    gap = float(lam[0].item() / lam[1].item()) if lam[1].item() > 1e-12 else float("nan")
 
    return {
        "pr":             pr,
        "effective_rank": eff_rank,
        "top10_var_frac": top10,
        "spectral_gap":   gap,
    }
 

def compute_marginal_cumulants(
        X: torch.Tensor,
        max_components: int = 2048,
) -> dict[str, float | np.ndarray]:
    """
    Compute per-component standardised cumulants κ3 (skewness) and κ4
    (excess kurtosis) for a data matrix X ∈ ℝ^{N × D}.

    Only the first *max_components* dimensions are used for speed; all
    summary statistics (mean |κ3|, mean κ4, …) are over those components.

    Returns
    -------
    dict with keys:
        kappa3          : np.ndarray (D',)   — per-component skewness
        kappa4          : np.ndarray (D',)   — per-component excess kurtosis
        mean_abs_kappa3 : float
        std_kappa3      : float
        mean_kappa4     : float
        std_kappa4      : float
        frac_non_gauss  : float  — fraction of components with |κ4| > 0.5
        N               : int    — sample size used
        D               : int    — number of components analysed
    """
    X = X.float()
    N, D_full = X.shape
    D = min(D_full, max_components)
    if D_full > D: # unbiased column cap
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(D_full, generator=g)[:D]
        X = X[:, idx]
    else:
        X = X[:, :D]

    mu = X.mean(0)            # (D,)
    Xc = X - mu              # centred,  (N, D)
    var = Xc.var(0).clamp(min=1e-8)
    std = var.sqrt()          # (D,)

    Xs = Xc / std            # standardised, (N, D)

    k3 = (Xs ** 3).mean(0).cpu().numpy()         # (D,)
    k4 = ((Xs ** 4).mean(0) - 3.0).cpu().numpy() # (D,)  excess kurtosis

    return {
        "kappa3":           k3,
        "kappa4":           k4,
        "mean_abs_kappa3":  float(np.abs(k3).mean()),
        "std_kappa3":       float(np.std(k3)),
        "mean_kappa4":      float(k4.mean()),
        "std_kappa4":       float(np.std(k4)),
        "max_kappa4":       float(np.max(np.abs(k4))),
        "frac_non_gauss":   float((np.abs(k4) > 0.5).mean()),
        "N":                N,
        "D":                D,
    }


def covariance_whiteness(X: torch.Tensor) -> float:
    """
    Frobenius off-diagonal ratio:  ||C - diag(C)||_F / ||C||_F.
 
    = 0  → channels fully uncorrelated (white).
    → 1  → strong inter-channel correlations.
    """
    X = X.float()
    N = X.size(0)
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / N                      # (D, D) sample covariance
    D_vec = torch.diag(torch.diag(C))      # (D, D) diagonal part
    off = C - D_vec
    denom = C.norm().item()
    if denom < 1e-12:
        return float("nan")
    return float(off.norm().item() / denom)
 
 
def js_divergence_from_gaussian(
        X: torch.Tensor,
        n_bins: int = 100,
) -> float:
    """
    Mean per-component Jensen-Shannon divergence from the best-fit Gaussian.
 
    JS(p || q) = ½ KL(p || m) + ½ KL(q || m),  m = (p+q)/2.
    Approximated via histogram with n_bins bins per component.
    Returns mean JS over all D components.
    """
    X = X.float()
    N, D = X.shape
    mu = X.mean(0)
    std = X.std(0).clamp(min=1e-6)
 
    js_vals = []
    for d in range(min(D, 256)):        # cap at 256 for speed
        x_d = ((X[:, d] - mu[d]) / std[d]).cpu().numpy()
        lo, hi = float(np.percentile(x_d, 1)), float(np.percentile(x_d, 99))
        if hi <= lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p_hist, _ = np.histogram(x_d, bins=bins, density=True)
        bin_c = 0.5 * (bins[:-1] + bins[1:])
        q_gauss = (1.0 / (math.sqrt(2 * math.pi))) * np.exp(-0.5 * bin_c ** 2)
        
        # Normalise both to sum-to-one over bins
        dw = bins[1] - bins[0]
        p_hist = p_hist * dw + 1e-12
        q_gauss = q_gauss * dw + 1e-12
        p_hist /= p_hist.sum()
        q_gauss /= q_gauss.sum()
        
        m = 0.5 * (p_hist + q_gauss)
        js = 0.5 * (p_hist * np.log(p_hist / m)).sum() + \
             0.5 * (q_gauss * np.log(q_gauss / m)).sum()
        js_vals.append(max(0.0, float(js)))
 
    return float(np.mean(js_vals)) if js_vals else float("nan")
 

def _compute_mardia_Z(Z: torch.Tensor, p: int, N: int, n_sub: int) -> dict[str, float]:
    """Helper function to calculate Mardia statistics for a specific projection matrix."""
    Zc = Z - Z.mean(0)
    S = Zc.T @ Zc / N                                          # (p, p)

    # Regularised inverse
    eps_reg = 1e-5 * S.diagonal().mean().item()
    try:
        S_inv = torch.linalg.inv(S + eps_reg * torch.eye(p, device=S.device))
    except torch.linalg.LinAlgError:
        return {"b1p": float("nan"), "b2p": float("nan"),
                "b2p_exp": float("nan"), "b2p_z": float("nan"), "p_dim": p}

    # Squared Mahalanobis distances d_ii = (z_i - z̄)ᵀ S⁻¹ (z_i - z̄)
    g_diag = (Zc @ S_inv * Zc).sum(1)                          # (N,)

    # Mardia kurtosis b₂,p = (1/N) Σ d_ii²
    b2p = float(g_diag.pow(2).mean().item())
    b2p_exp = p * (p + 2)
    se_b2p = math.sqrt(8.0 * p * (p + 2) / N) if N > 1 else 1.0
    b2p_z = (b2p - b2p_exp) / se_b2p

    # Mardia skewness b₁,p = (1/n²) Σ_{i,j} d_ij³  — O(n²), use subsample
    idx = torch.randperm(N)[:min(N, n_sub)]
    Zs = Zc[idx]                                              # (n_sub, p)
    G = Zs @ S_inv @ Zs.T                                    # (n_sub, n_sub)
    b1p = float(G.pow(3).mean().item())

    return {
        "b1p":     b1p,
        "b2p":     b2p,
        "b2p_exp": b2p_exp,
        "b2p_z":   b2p_z,
        "p_dim":   p,
        "d_ii":    g_diag,
    }


def mardia_statistics(
        X: torch.Tensor,
        d_proj: int = 32,
        use_pca: bool = True,
        seed: int = 42,
        n_random_seeds: int = 10,
        n_sub: int = 600,
) -> dict[str, float]:
    """
    Mardia's multivariate normality statistics (1970) computed on a
    random *d_proj*-dimensional projection of X to make the test tractable
    for high-dimensional data.

    Parameters
    ----------
    X      : (N, D) raw activations
    d_proj : target projection dimension  (D if D < d_proj)
    seed   : RNG seed for the projection matrix
    n_sub  : subsample size used for the O(n²) skewness term

    Returns
    -------
    dict with keys:
        b1p      : Mardia skewness  (→ 0 under H₀: MVN)
        b2p      : Mardia kurtosis  (→ p(p+2) under H₀)
        b2p_exp  : expected value p(p+2)
        b2p_z    : z-score  (b2p - b2p_exp) / se(b2p)   under H₀
        p_dim    : effective projection dimension used
    """
    X = X.float()
    N, D = X.shape
    p = min(D, d_proj)

    if p < 2 or N < p + 2:
        return {"b1p": float("nan"), "b2p": float("nan"),
                "b2p_exp": float("nan"), "b2p_z": float("nan"), "p_dim": p}

    Xc = X - X.mean(0)

    if use_pca:
        U, S, V = torch.pca_lowrank(Xc, q=p, center=False)
        Z = Xc @ V
        return _compute_mardia_Z(Z, p, N, n_sub)
        
    # Average over n_random_seeds
    res_list = []
    for s in range(n_random_seeds):
        gen = torch.Generator()
        gen.manual_seed(seed + s)
        Q, _ = torch.linalg.qr(
            torch.randn(D, p, generator=gen, dtype=torch.float32))  # (D, p)
        Z = Xc @ Q.to(X.device)                       # (N, p)
        res_list.append(_compute_mardia_Z(Z, p, N, n_sub))

    return {
        "b1p": sum(r["b1p"] for r in res_list) / n_random_seeds,
        "b2p": sum(r["b2p"] for r in res_list) / n_random_seeds,
        "b2p_exp": res_list[0]["b2p_exp"],
        "b2p_z": sum(r["b2p_z"] for r in res_list) / n_random_seeds,
        "p_dim": p
    }