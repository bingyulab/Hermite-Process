
import math

import torch


def compute_spectrum_stats(
        X:     torch.Tensor,
        q:     int = 512,
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
    q    = min(q, N - 1, D)
    if q < 2:
        nan = float("nan")
        return {"pr": nan, "effective_rank": nan,
                "top10_var_frac": nan, "spectral_gap": nan}
 
    Xc = X - X.mean(0)
    # pca_lowrank returns (U, S, Vh); S are sqrt(eigenvalues * N)
    _, S, _ = torch.pca_lowrank(Xc, q=q, center=False, niter=4)
    lam = (S ** 2).cpu()                    # eigenvalues (unnormalised)
    lam = lam.clamp(min=0.0)
 
    lam_sum  = lam.sum().item()
    if lam_sum < 1e-12:
        nan = float("nan")
        return {"pr": nan, "effective_rank": nan,
                "top10_var_frac": nan, "spectral_gap": nan}
 
    # Participation Ratio
    pr = float((lam_sum ** 2) / (lam ** 2).sum().item())
 
    # Effective rank via normalised entropy
    p   = lam / lam_sum
    p   = p.clamp(min=1e-12)
    H   = float(-(p * p.log()).sum().item())
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
 
