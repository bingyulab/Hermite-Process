"""
Experiment α — Cumulant Gaussianization Probe
Experiment β — Bottleneck Width vs. Gaussianization
Experiment γ — Full Layer-by-Layer Kurtosis Trace
Experiment δ — Latent Perturbation "Rigidity" Test
=====================================================

Experiment α:
    Extracts activations at three semantic stages:
        1. "input"       — raw test images x0  (pixel distribution)
        2. "corrupted"   — x_T = x0 + σ(1)·Σ(x0)·ε  (noise-type signal)
        3. "mid"         — UNet bottleneck output spatially averaged (B, C)
        4. "latent_ae"   — AE encoder output z0 ∈ ℝ^64
        5. "x0hat"       — final x0_hat reconstruction (UNet output)
    Computes per-component cumulants κ3, κ4 and Mardia's multivariate
    kurtosis statistic at each stage, for both Gaussian and Rosenblatt models.
    Outputs: console table + CSV + LaTeX fragment + violin-plot figure.

Experiment β:
    Trains ConditionalUNetFlexible with four bottleneck widths:
        {0.25, 0.5, 1.0, 2.0} × (4 × base_ch)
    Measures κ4 at the bottleneck after training.
    Hypothesis: κ4 ↓ as bottleneck narrows → bottleneck-driven Gaussianization.
    Alternative: κ4 independent of width → L² objective drives Gaussianization.
    Outputs: table + scatter plot κ4 vs bottleneck_factor.

Experiment γ: 
    full encoder→decoder kurtosis trace at all named layers,
    using simultaneous hooks (single forward pass, N samples).
    Produces κ4 profile + PR profile + whiteness profile.
 
Experiment δ: 
    latent rigidity test — inject four noise types into the
    bottleneck and measure Huber reconstruction degradation.
    Noise types: Gaussian, Laplace, Student-t(ν=3), Rosenblatt.
    Two regimes: low σ (test alignment) and high σ (test tolerance).
 
Additional analyses added
─────────────────────────
  compute_spectrum_stats:
      Participation Ratio (PR)     = (Σλ_i)² / Σλ_i²
      Effective Rank                = exp(-Σ p_i log p_i), p_i = λ_i/Σλ
      Top-10 variance fraction      = Σ_{i≤10} λ_i / Σ λ_i
  covariance_whiteness:
      ||C - diag(C)||_F / ||C||_F  (0 → fully uncorrelated channels)
  js_divergence_from_gaussian:
      Per-component Jensen-Shannon  JS(p_component || N(μ,σ²))
 
Usage
─────
  python Experiment_Gaussianity.py --mode all  --save_dir ./output/diffusion
  python Experiment_Gaussianity.py --mode alpha
  python Experiment_Gaussianity.py --mode beta  --bf_list 0.25 0.5 1.0 2.0
  python Experiment_Gaussianity.py --mode gamma          # layer trace only
  python Experiment_Gaussianity.py --mode delta          # rigidity test only
  
"""

from __future__ import annotations

import argparse
import csv
import os
import math
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributions as tdist

# ── local imports from the main codebase ──────────────────────────────────────
from Rosenblatt_cold_diffusion_unified import (
    Config,
    ConvAutoencoder,
    ConditionalUNet,
    EMA,
    LatentMLPDenoiser,
    RosenblattForward,
    ResBlockAdaGN,
    SelfAttention,
    SinusoidalTimeEmbed,
    _NORM_TF,
    _get_dataset,
    sample_noise,
    sigma_multiplicative,
    train,
    train_autoencoder,
    train_latent,
    OUT_ROOT,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Constants
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES_ALPHA: int = 2000          # samples for cumulant estimation (Exp α/γ)
GAUSS_OUT = Path("./output/diffusion/gaussianization")
GAUSS_OUT.mkdir(parents=True, exist_ok=True)

# Named UNet layers to probe (key → (module_attr_path, spatial_pool))
# All hooked simultaneously in a single forward pass.
UNET_LAYER_KEYS = [
    "init_conv",    # (B, base_ch,    28, 28)  after first conv
    "down1_0",      # (B, base_ch,    28, 28)  after ResBlock 1
    "down1_1",      # (B, base_ch,    28, 28)  after ResBlock 2  ← h1
    "pool1",        # (B, 2·base_ch,  14, 14)  after stride-2 conv
    "down2_0",      # (B, 2·base_ch,  14, 14)  after ResBlock 3
    "down2_1",      # (B, 2·base_ch,  14, 14)  after ResBlock 4
    "attn2",        # (B, 2·base_ch,  14, 14)  after attention  ← h2
    "pool2",        # (B, bneck_ch,    7,  7)  after stride-2 conv
    "mid1",         # (B, bneck_ch,    7,  7)  after mid ResBlock 1
    "attn_mid",     # (B, bneck_ch,    7,  7)  after bottleneck attention
    "mid2",         # (B, bneck_ch,    7,  7)  bottleneck output  ← h3
    "up_res2_0",    # (B, 2·base_ch,  14, 14)  after skip-cat ResBlock
    "up_res2_1",    # (B, 2·base_ch,  14, 14)  after second up ResBlock
    "up_attn2",     # (B, 2·base_ch,  14, 14)  after up attention  ← h_up2
    "up_res1_0",    # (B, base_ch,    28, 28)  after skip-cat ResBlock
    "up_res1_1",    # (B, base_ch,    28, 28)  after second up ResBlock  ← h_up1
    "out",          # (B, 1,          28, 28)  final output pixel map
]

_LAYER_LABELS = {
    "init_conv":  "init\\_conv",
    "down1_0":    "down1[0]",
    "down1_1":    "down1[1] h1",
    "pool1":      "pool1",
    "down2_0":    "down2[0]",
    "down2_1":    "down2[1]",
    "attn2":      "attn2 h2",
    "pool2":      "pool2",
    "mid1":       "mid1",
    "attn_mid":   "attn\\_mid",
    "mid2":       "mid2 h3",
    "up_res2_0":  "up\\_res2[0]",
    "up_res2_1":  "up\\_res2[1]",
    "up_attn2":   "up\\_attn2 h\\_up2",
    "up_res1_0":  "up\\_res1[0]",
    "up_res1_1":  "up\\_res1[1] h\\_up1",
    "out":        "out",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hook-based activation capture
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Accumulates batches of hook-captured activations into a single tensor."""

    def __init__(self, spatial_pool: bool = True) -> None:
        """
        spatial_pool: if True, apply global average pooling over spatial dims
                      before storing.  Converts (B, C, H, W) → (B, C).
        """
        self._parts: list[torch.Tensor] = []
        self._spatial_pool = spatial_pool

    def hook_fn(self, module: nn.Module,
                input: Any, output: torch.Tensor) -> None:
        x = output.detach().float().cpu()
        if self._spatial_pool and x.dim() == 4:          # (B, C, H, W)
            x = x.mean(dim=(-2, -1))                     # → (B, C)
        elif x.dim() > 2:                                # flatten all but batch
            x = x.flatten(1)
        self._parts.append(x)

    def get(self) -> torch.Tensor:
        """Returns (N, D) tensor of all captured activations."""
        if not self._parts:
            return torch.empty(0)
        return torch.cat(self._parts, dim=0)

    def clear(self) -> None:
        self._parts.clear()


@contextmanager
def capture_layer(module: nn.Module, spatial_pool: bool = True):
    """
    Context manager that registers a forward hook on *module* and yields an
    ActivationStore.  The hook is removed on exit.

    Example
    -------
    with capture_layer(model.mid2) as store:
        for x, t, y in batches:
            model(x, t, y)
    activations = store.get()   # (N, C)
    """
    store = ActivationStore(spatial_pool=spatial_pool)
    handle = module.register_forward_hook(store.hook_fn)
    try:
        yield store
    finally:
        handle.remove()

def _get_unet_modules(model: nn.Module) -> dict[str, nn.Module]:
    """
    Return the module objects for all UNET_LAYER_KEYS, working for both
    ConditionalUNet and ConditionalUNetFlexible.
    """
    return {
        "init_conv":  model.init_conv,
        "down1_0":    model.down1[0],
        "down1_1":    model.down1[1],
        "pool1":      model.pool1,
        "down2_0":    model.down2[0],
        "down2_1":    model.down2[1],
        "attn2":      model.attn2,
        "pool2":      model.pool2,
        "mid1":       model.mid1,
        "attn_mid":   model.attn_mid,
        "mid2":       model.mid2,
        "up_res2_0":  model.up_res2[0],
        "up_res2_1":  model.up_res2[1],
        "up_attn2":   model.up_attn2,
        "up_res1_0":  model.up_res1[0],
        "up_res1_1":  model.up_res1[1],
        "out":        model.out,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cumulant & geometry statistics
# ─────────────────────────────────────────────────────────────────────────────

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
    X = X[:, :D]

    mu  = X.mean(0)           # (D,)
    Xc  = X - mu              # centred,  (N, D)
    var = Xc.var(0).clamp(min=1e-8)
    std = var.sqrt()          # (D,)

    Xs  = Xc / std            # standardised, (N, D)

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
 
 
def covariance_whiteness(X: torch.Tensor) -> float:
    """
    Frobenius off-diagonal ratio:  ||C - diag(C)||_F / ||C||_F.
 
    = 0  → channels fully uncorrelated (white).
    → 1  → strong inter-channel correlations.
    """
    X  = X.float()
    N  = X.size(0)
    Xc = X - X.mean(0)
    C  = Xc.T @ Xc / N                      # (D, D) sample covariance
    D_vec  = torch.diag(torch.diag(C))      # (D, D) diagonal part
    off    = C - D_vec
    denom  = C.norm().item()
    if denom < 1e-12:
        return float("nan")
    return float(off.norm().item() / denom)
 
 
def js_divergence_from_gaussian(
        X:       torch.Tensor,
        n_bins:  int = 100,
) -> float:
    """
    Mean per-component Jensen-Shannon divergence from the best-fit Gaussian.
 
    JS(p || q) = ½ KL(p || m) + ½ KL(q || m),  m = (p+q)/2.
    Approximated via histogram with n_bins bins per component.
    Returns mean JS over all D components.
    """
    X  = X.float()
    N, D = X.shape
    mu  = X.mean(0)
    std = X.std(0).clamp(min=1e-6)
 
    js_vals = []
    for d in range(min(D, 256)):        # cap at 256 for speed
        x_d = ((X[:, d] - mu[d]) / std[d]).cpu().numpy()
        lo, hi = float(np.percentile(x_d, 1)), float(np.percentile(x_d, 99))
        if hi <= lo:
            continue
        bins   = np.linspace(lo, hi, n_bins + 1)
        p_hist, _  = np.histogram(x_d, bins=bins, density=True)
        bin_c  = 0.5 * (bins[:-1] + bins[1:])
        q_gauss= (1.0 / (math.sqrt(2 * math.pi))) * np.exp(-0.5 * bin_c ** 2)
        # Normalise both to sum-to-one over bins
        dw     = bins[1] - bins[0]
        p_hist = p_hist * dw + 1e-12
        q_gauss= q_gauss * dw + 1e-12
        p_hist /= p_hist.sum();  q_gauss /= q_gauss.sum()
        m  = 0.5 * (p_hist + q_gauss)
        js = 0.5 * (p_hist * np.log(p_hist / m)).sum() + \
             0.5 * (q_gauss * np.log(q_gauss / m)).sum()
        js_vals.append(max(0.0, float(js)))
 
    return float(np.mean(js_vals)) if js_vals else float("nan")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Mardia multivariate normality
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mardia_Z(Z: torch.Tensor, p: int, N: int, n_sub: int) -> dict[str, float]:
    Zc = Z - Z.mean(0)
    S  = Zc.T @ Zc / N                                          # (p, p)

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
    se_b2p  = math.sqrt(8.0 * p * (p + 2) / N) if N > 1 else 1.0
    b2p_z   = (b2p - b2p_exp) / se_b2p

    # Mardia skewness b₁,p = (1/n²) Σ_{i,j} d_ij³  — O(n²), use subsample
    idx  = torch.randperm(N)[:min(N, n_sub)]
    Zs   = Zc[idx]                                              # (n_sub, p)
    G    = Zs @ S_inv @ Zs.T                                    # (n_sub, n_sub)
    b1p  = float(G.pow(3).mean().item())

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
        seed:   int = 42,
        n_random_seeds: int = 10,
        n_sub:  int = 600,
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full layer-trace extraction 
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def extract_full_layer_trace(
        model:      nn.Module,
        forward:    RosenblattForward,
        test_ds,
        cfg:        Config,
        n_samples:  int = 2000,
        t_eval:     float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Register forward hooks on ALL named UNet layers simultaneously, run
    n_samples images through at a single time step t_eval, and return
    a dict {layer_key: (n_samples, D) activation tensor}.
 
    t_eval defaults to cfg.T_MIN (clean reconstruction regime).
    Uses spatial average-pooling for all conv feature maps.
 
    This replaces the fragile manual layer-by-layer forward code in the
    original run_experiment_beta (which used only 128 samples).
    """
    model.eval()
    device  = cfg.device
    t_val   = cfg.T_MIN if t_eval is None else t_eval
    loader  = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                         shuffle=False, num_workers=2)
 
    mods    = _get_unet_modules(model)
    stores  = {k: ActivationStore(spatial_pool=True) for k in UNET_LAYER_KEYS}
    handles = [mods[k].register_forward_hook(stores[k].hook_fn)
               for k in UNET_LAYER_KEYS]
 
    n_col = 0
    try:
        for x0, y in loader:
            if n_col >= n_samples:
                break
            B   = x0.size(0)
            x0  = x0.to(device);  y = y.to(device)
            t_T = torch.ones(B, device=device)
            x_T, _, _ = forward.corrupt(x0, t_T, y=y)
            t   = torch.full((B,), t_val, device=device)
            c_in = forward.c_in(t).view(-1, 1, 1, 1)
            null = torch.full_like(y, 10)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(x_T * c_in, t, null)
            else:
                _ = model(x_T * c_in, t, null)
            n_col += B
    finally:
        for h in handles:
            h.remove()
 
    return {k: stores[k].get()[:n_samples] for k in UNET_LAYER_KEYS}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Stage-wise extraction for Experiment α
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_stages(
        model:      ConditionalUNet,
        forward:    RosenblattForward,
        test_ds,
        cfg:        Config,
        n_samples:  int = 2000,
) -> dict[str, torch.Tensor]:
    """
    Extract (N, D) activation tensors at five semantic stages for a trained
    image-space UNet model.

    Stages
    ------
    "input"      — raw x0 pixel vectors, (N, 784)
    "corrupted"  — x_T = x0 + σ(1)·Σ(x0)·ε at t=1, (N, 784)
    "mid_t05"    — x0_hat prediction at t=0.5 during generation, (N, 784)
    "bottleneck" — mid2 output spatially averaged, (N, 4·base_ch)
    "x0hat"      — final x0_hat (UNet output at t→0), (N, 784)
    """
    device = cfg.device
    model.eval()
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)

    # ── raw inputs and corrupted versions ────────────────────────────────────
    raw_imgs:    list[torch.Tensor] = []
    corrupted_T: list[torch.Tensor] = []
    x0hat_out:   list[torch.Tensor] = []

    # Attach hook on mid2 (bottleneck)
    bn_store = ActivationStore(spatial_pool=True)
    bn_handle = model.mid2.register_forward_hook(bn_store.hook_fn)

    n_collected = 0
    for x0, y in loader:
        if n_collected >= n_samples:
            break
        B = x0.size(0)
        x0  = x0.to(device)
        y   = y.to(device)

        # Stage 1: raw input (flatten)
        raw_imgs.append(x0.view(B, -1).cpu())

        # Stage 2: corrupted at t=1
        t_one = torch.ones(B, device=device)
        x_T, _, _ = forward.corrupt(x0, t_one, y=y)
        corrupted_T.append(x_T.view(B, -1).cpu())

        # Stage 4+5: run UNet forward at t→0 (t=T_MIN)
        # Also triggers the mid2 hook → bottleneck activations captured
        t_min = torch.full((B,), cfg.T_MIN, device=device)
        null  = torch.full_like(y, 10)
        c_in  = forward.c_in(t_min).view(-1, 1, 1, 1)
        if cfg.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0_c = model(x_T * c_in, t_min, y).float()
                x0_u = model(x_T * c_in, t_min, null).float()
        else:
            x0_c = model(x_T * c_in, t_min, y)
            x0_u = model(x_T * c_in, t_min, null)
        x0h = (x0_u + cfg.cfg_scale * (x0_c - x0_u)).clamp(-1., 1.)
        x0hat_out.append(x0h.view(B, -1).cpu())

        n_collected += B

    bn_handle.remove()

    # ── mid-generation: run generation to t=0.5 for a clean set ─────────────
    mid_t05: list[torch.Tensor] = []
    labels_gen = torch.arange(10, device=device).repeat(
        math.ceil(n_samples / 10))[:n_samples]

    n_steps_half = cfg.n_steps // 2           # go from t=1 to t≈0.5
    t_full  = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=device)
    t_half  = t_full[:n_steps_half + 1]       # [1.0, ..., ~0.5]

    chunk = min(256, n_samples)
    collected = 0
    while collected < n_samples:
        n_now = min(chunk, n_samples - collected)
        lbl   = labels_gen[collected: collected + n_now]
        null  = torch.full_like(lbl, 10)

        eps = sample_noise(forward.noise_type, (n_now, 1, 28, 28),
                           forward.lam_t, forward.M_eig, device)
        x   = eps * forward.sigma_max

        for k in range(len(t_half) - 1):
            tc   = t_half[k].expand(n_now)
            tn   = t_half[k + 1].expand(n_now)
            c_in = forward.c_in(tc).view(-1, 1, 1, 1)
            if cfg.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    xc = model(x * c_in, tc, lbl).float()
                    xu = model(x * c_in, tc, null).float()
            else:
                xc = model(x * c_in, tc, lbl)
                xu = model(x * c_in, tc, null)
            x0h = (xu + cfg.cfg_scale * (xc - xu)).clamp(-1., 1.)
            if k < len(t_half) - 2:
                x = forward.recorrupt_stochastic(x0h, tn, y=lbl)
            else:
                x = x0h

        mid_t05.append(x.view(n_now, -1).cpu())
        collected += n_now

    def _trim(lst: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(lst, 0)[:n_samples]

    return {
        "input":       _trim(raw_imgs),
        "corrupted":   _trim(corrupted_T),
        "mid_t05":     _trim(mid_t05),
        "bottleneck":  bn_store.get()[:n_samples],
        "x0hat":       _trim(x0hat_out),
    }


@torch.no_grad()
def extract_latent_stages(
        ae:         ConvAutoencoder,
        mlp_model:  LatentMLPDenoiser,
        forward:    RosenblattForward,
        test_ds,
        cfg:        Config,
        n_samples:  int = 2000,
) -> dict[str, torch.Tensor]:
    """
    Analogous extraction for the autoencoder + latent MLP denoiser.

    Stages
    ------
    "image_input"  — raw x0 pixel vectors (N, 784)
    "latent_z0"    — AE encoder output z0 (N, 64)
    "latent_corr"  — z_T = z0 + σ(1)·ε  (N, 64)
    "mlp_mid"      — activations at middle MLP layer (N, hidden_dim)
    "latent_x0hat" — z0_hat after full denoising, mapped to pixel space (N, 784)
    """
    device = cfg.device
    D = ConvAutoencoder.LATENT_DIM
    ae.eval();  mlp_model.eval()

    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)

    img_list:     list[torch.Tensor] = []
    z0_list:      list[torch.Tensor] = []
    z_corr_list:  list[torch.Tensor] = []

    # Hook the middle MLP layer (layer index 3 of 6)
    mlp_store = ActivationStore(spatial_pool=False)
    mlp_handle = mlp_model.layers[3].register_forward_hook(mlp_store.hook_fn)

    n_collected = 0
    for x0, y in loader:
        if n_collected >= n_samples:
            break
        B  = x0.size(0)
        x0 = x0.to(device);  y = y.to(device)
        z0 = ae.encode(x0)                           # (B, 64)

        img_list.append(x0.view(B, -1).cpu())
        z0_list.append(z0.cpu())

        # Corrupt latent at t=1
        sig   = forward.sigma_t(torch.ones(B, device=device)).unsqueeze(1)
        eps   = sample_noise(forward.noise_type, (B, D),
                             forward.lam_t, forward.M_eig, device)
        z_T   = z0 + sig * eps
        z_corr_list.append(z_T.cpu())

        # Trigger MLP forward → captures mlp_mid via hook
        t_min = torch.full((B,), cfg.T_MIN, device=device)
        cin   = (1.0 + forward.sigma_t(t_min).unsqueeze(1) ** 2).pow(-0.5)
        null  = torch.full_like(y, 10)
        _     = mlp_model(z_T * cin, t_min, null)   # hook fires here

        n_collected += B

    mlp_handle.remove()

    # Full denoising: latent_x0hat
    labels_gen = torch.arange(10, device=device).repeat(
        math.ceil(n_samples / 10))[:n_samples]
    z0hat_list: list[torch.Tensor] = []

    t_sched = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=device)
    chunk   = 256
    col     = 0
    while col < n_samples:
        n_now = min(chunk, n_samples - col)
        lbl   = labels_gen[col: col + n_now]
        null2 = torch.full_like(lbl, 10)

        z = sample_noise(forward.noise_type, (n_now, D),
                         forward.lam_t, forward.M_eig, device) * forward.sigma_max

        for k in range(cfg.n_steps):
            tc  = t_sched[k].expand(n_now)
            tn  = t_sched[k + 1].expand(n_now)
            sig = forward.sigma_t(tc).unsqueeze(1)
            cin = (1.0 + sig ** 2).pow(-0.5)
            z0c = mlp_model(z * cin, tc, lbl)
            z0u = mlp_model(z * cin, tc, null2)
            z0h = z0u + cfg.cfg_scale * (z0c - z0u)
            if k < cfg.n_steps - 1:
                sn  = forward.sigma_t(tn).unsqueeze(1)
                z   = z0h + sn * sample_noise(forward.noise_type, (n_now, D),
                                               forward.lam_t, forward.M_eig, device)
            else:
                z = z0h

        decoded = ((ae.decode(z) + 1.0) / 2.0).clamp(0., 1.)
        z0hat_list.append(decoded.view(n_now, -1).cpu())
        col += n_now

    def _trim(lst: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(lst, 0)[:n_samples]

    return {
        "image_input":  _trim(img_list),
        "latent_z0":    _trim(z0_list),
        "latent_corr":  _trim(z_corr_list),
        "mlp_mid":      mlp_store.get()[:n_samples],
        "latent_x0hat": _trim(z0hat_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rigidity test  
# ─────────────────────────────────────────────────────────────────────────────

def rigidity_test(
        model:    "ConditionalUNetFlexible",
        forward:  RosenblattForward,
        test_ds,
        cfg:      Config,
        sigma_levels: list[float] | None = None,
    n_batch:  int = 64,
) -> dict[str, dict[float, float]]:
    """
    Latent Perturbation "Rigidity" Test.

    Injects additive noise into the bottleneck activations (mid2 output)
    right before the decoder, then measures Huber reconstruction loss.

    Four noise types are tested at each σ level:
        A: Gaussian    N(0, σ²I)
        B: Laplace     Laplace(0, σ/√2)   — same variance as Gaussian
        C: Rosenblatt  unit-variance Rosenblatt × σ
        D: Student-t   t(ν=3) scaled to variance σ²  (heavier tails)

    The key question: does the decoder degrade MORE when perturbed with
    heavy-tailed (non-Gaussian) noise than with matched-variance Gaussian?
    If yes → the bottleneck has learned a Gaussian geometry that breaks
    under non-Gaussian perturbation (rigidity).
    If no  → the decoder is geometry-agnostic (robust).

    Parameters
    ----------
    sigma_levels : list of σ values to test.  Defaults to [0.1, 0.3, 0.5, 1.0].

    Returns
    -------
    dict of {noise_name → {sigma → huber_loss}}
    """
    if sigma_levels is None:
        sigma_levels = [0.1, 0.3, 0.5, 1.0]

    model.eval()
    device = cfg.device

    # One representative batch from the test set.
    # Keep this moderate to avoid OOM during Rosenblatt perturbation synthesis.
    x0_batch, y_batch = next(iter(
        DataLoader(test_ds, batch_size=n_batch, shuffle=True)))
    x0_batch = x0_batch.to(device)
    y_batch  = y_batch.to(device)
    B        = x0_batch.size(0)

    t_min = torch.full((B,), cfg.T_MIN, device=device)
    t_one = torch.ones(B, device=device)
    x_T, _, _ = forward.corrupt(x0_batch, t_one, y=y_batch)
    c_in  = forward.c_in(t_min).view(-1, 1, 1, 1)
    t_emb = model.time_mlp(model.t_embed(t_min)) + model.label_emb(y_batch)

    # Encode once; keep h3 (bottleneck), h2, h1 (skip connections)
    with torch.no_grad():
        h3, h2, h1 = model.encode(x_T * c_in, t_emb)

    # Per-channel std of the bottleneck (shape (1, C, 1, 1))
    bneck_std = h3.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Rosenblatt eigenvalues (needed for Test C)
    bneck_numel = h3.numel() // B             # C * H * W per sample
    lam_t = forward.lam_t                     # None for Gaussian model

    results: dict[str, dict[float, float]] = {
        "gaussian":    {},
        "laplace":     {},
        "rosenblatt":  {},
        "student_t3":  {},
        "clean":       {},
    }

    # Baseline: no perturbation
    with torch.no_grad():
        x0h_clean = model.decode(h3, h2, h1, t_emb)
        huber_clean = F.smooth_l1_loss(x0h_clean, x0_batch.float()).item()

    for σ in sigma_levels:
        results["clean"][σ] = huber_clean      # same regardless of σ

        # Test A — Gaussian N(0, σ²)
        noise_g = torch.randn_like(h3) * bneck_std * σ

        # Test B — Laplace  (same variance: scale = σ·std/√2)
        scale_l   = bneck_std * σ / math.sqrt(2.0)
        noise_l   = tdist.Laplace(torch.zeros_like(h3), scale_l).sample()

        # Test C — Rosenblatt (unit-variance, scaled by σ·std)
        if lam_t is not None:
            # Generate Rosenblatt noise in small chunks and on CPU to avoid
            # large temporary CUDA allocations inside sample_noise.
            ros_device = torch.device("cpu") if device.type == "cuda" else device
            lam_t_cpu = lam_t.to(ros_device)
            ros_chunks: list[torch.Tensor] = []
            ros_chunk_bs = min(16, B)
            for i in range(0, B, ros_chunk_bs):
                n_i = min(ros_chunk_bs, B - i)
                ros_i = sample_noise(
                    "rosenblatt",
                    (n_i, bneck_numel),
                    lam_t_cpu,
                    forward.M_eig,
                    ros_device,
                )
                ros_chunks.append(ros_i)
            ros_flat = torch.cat(ros_chunks, dim=0).to(device)
        else:
            # Gaussian model fallback: use Gaussian for Test C too
            ros_flat = torch.randn(B, bneck_numel, device=device)
        ros_flat  = ros_flat.view_as(h3)
        # Normalise to unit std, then scale to match variance
        ros_std   = ros_flat.std(dim=0, keepdim=True).clamp(min=1e-6)
        noise_r   = (ros_flat / ros_std) * bneck_std * σ

        # Test D — Student-t(ν=3)  E[X²] = ν/(ν-2)=3 → scale by √(1/3) for unit variance
        ν = 3.0
        t3_flat   = tdist.StudentT(df=ν).sample(h3.shape).to(device)
        noise_t3  = t3_flat * bneck_std * σ * math.sqrt((ν - 2.0) / ν)

        for name, noise in [("gaussian",   noise_g),
                             ("laplace",    noise_l),
                             ("rosenblatt", noise_r),
                             ("student_t3", noise_t3)]:
            h3_pert = h3 + noise.detach()
            with torch.no_grad():
                x0h_pert = model.decode(h3_pert, h2, h1, t_emb)
                loss = F.smooth_l1_loss(x0h_pert, x0_batch.float()).item()
            results[name][σ] = loss

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def plot_rigidity(
        results: dict[str, dict[float, float]],
        save_path: Path,
        title: str = "",
) -> None:
    """Plot rigidity test: Huber loss vs σ for each noise type."""
    σ_vals = sorted(next(iter(results.values())).keys())
    colors = {
        "clean":      ("gray",    "--", "Clean (no perturb.)"),
        "gaussian":   ("#4C72B0", "-",  "Gaussian A"),
        "laplace":    ("#55A868", "-",  "Laplace B"),
        "rosenblatt": ("#DD8452", "-",  "Rosenblatt C"),
        "student_t3": ("#C44E52", "-",  "Student-t(3) D"),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, (col, ls, lab) in colors.items():
        if key not in results:
            continue
        vals = [results[key].get(σ, float("nan")) for σ in σ_vals]
        ax.plot(σ_vals, vals, color=col, linestyle=ls, marker="o",
                linewidth=2, label=lab)
    ax.set_xlabel("Perturbation scale $\\sigma$", fontsize=11)
    ax.set_ylabel("Huber reconstruction loss", fontsize=11)
    ax.set_title(title or "Latent Perturbation Rigidity Test", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Rigidity plot saved to {save_path}")


def plot_beta_rigidity_grid(
        rows: list[BetaResult],
        save_path: Path,
        sigma_label: str = "0.5",
        silent: bool = False,
) -> None:
    """
    One big comparison figure for Experiment beta rigidity metrics:
    x-axis = bottleneck factor, subplot = perturbation noise type,
    line color = model family (Gaussian vs Rosenblatt).

    This uses the rigidity metrics stored in BetaResult (reported at sigma_label).
    """
    noise_panels = [
        ("perturb_gauss_huber", "Perturbation: Gaussian"),
        ("perturb_laplace_huber", "Perturbation: Laplace"),
        ("perturb_rosenblatt_huber", "Perturbation: Rosenblatt"),
        ("perturb_t3_huber", "Perturbation: Student-t(3)"),
    ]
    colors = {"gaussian": "#4C72B0", "rosenblatt": "#DD8452"}
    markers = {"gaussian": "o", "rosenblatt": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes_f = axes.flatten()

    for ax, (attr, title) in zip(axes_f, noise_panels):
        for model_name in ("gaussian", "rosenblatt"):
            sub = sorted(
                [r for r in rows if r.noise_type == model_name],
                key=lambda r: r.bottleneck_factor,
            )
            if not sub:
                continue
            x = [r.bottleneck_factor for r in sub]
            y = [getattr(r, attr) for r in sub]
            ax.plot(
                x,
                y,
                color=colors[model_name],
                marker=markers[model_name],
                linewidth=2,
                label=model_name.capitalize(),
            )

        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0", "3.0"])

    axes[0, 0].set_ylabel("Huber loss")
    axes[1, 0].set_ylabel("Huber loss")
    axes[1, 0].set_xlabel("Bottleneck factor")
    axes[1, 1].set_xlabel("Bottleneck factor")

    fig.suptitle(
        f"Experiment beta rigidity by factor and model (sigma={sigma_label})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if not silent:
        print(f"  → Plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    model:          str    # "Gaussian" or "Rosenblatt"
    stage:          str    # stage label
    N:              int
    D:              int
    mean_abs_k3:    float
    std_k3:         float
    mean_k4:        float
    std_k4:         float
    max_k4:         float
    frac_nong:      float
    mardia_b2p:     float
    mardia_b2p_exp: float
    mardia_b2p_z:   float
    pr:             float = float("nan")
    effective_rank: float = float("nan")
    whiteness:      float = float("nan")
    js_gauss:       float = float("nan")


@dataclass
class LayerStats:
    """Per-layer statistics for the full layer trace (Experiment γ)."""
    model:          str
    noise_type:     str
    layer_key:      str
    layer_label:    str
    depth_index:    int
    mean_k4:        float
    std_k4:         float
    frac_nong:      float
    pr:             float
    effective_rank: float
    whiteness:      float
    mardia_b2p_z:   float


@dataclass
class BetaResult:
    noise_type:              str
    bottleneck_factor:       float
    bneck_ch:                int
    mean_k4_input:           float
    mean_k4_bneck:           float
    std_k4_bneck:            float
    max_k4_bneck:            float
    frac_nong_bneck:         float
    mean_k4_x0hat:           float
    mardia_b2p_z:            float    # PCA projection
    mardia_b2p_z_avg:        float    # random projection average
    mardia_b2p_z_x0hat:      float
    mardia_b2p_z_x0hat_avg:  float
    offline_loss_mse:        float
    offline_loss_mae:        float
    offline_loss_huber:      float
    offline_loss_quantile:   float
    k4_grad_mse:             float
    k4_grad_mae:             float
    mardia_mse_corr:         float
    ablated_mse:             float
    ablated_mae:             float
    pr_bneck:                float
    effective_rank_bneck:    float
    whiteness_bneck:         float
    js_gauss_bneck:          float
    perturb_gauss_huber:     float    # σ=0.5 rigidity test
    perturb_laplace_huber:   float
    perturb_rosenblatt_huber:float    
    perturb_t3_huber:        float
    

def _fmt(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:+.{decimals}f}" if v < 0 else f"{v:.{decimals}f}"


def print_cumulant_table(rows: list[StageResult]) -> None:
    hdr = (f"{'Model':12s}  {'Stage':22s}  {'N':>5}  {'D':>5}  "
           f"{'|κ3|':>7}  {'κ4':>8}  {'%>0.5':>6}  "
           f"{'PR':>8}  {'ER':>6}  {'White':>6}  "
           f"{'b₂p':>7}  {'b₂p*':>7}  {'Z':>7}")
    sep = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in rows:
        print(f"{r.model:12s}  {r.stage:22s}  {r.N:5d}  {r.D:5d}  "
              f"{_fmt(r.mean_abs_k3):>7}  {_fmt(r.mean_k4):>8}  "
              f"{r.frac_nong*100:5.1f}%  "
              f"{_fmt(r.pr,1):>8}  {_fmt(r.effective_rank,1):>6}  "
              f"{_fmt(r.whiteness,3):>6}  "
              f"{_fmt(r.mardia_b2p,1):>7}  {_fmt(r.mardia_b2p_exp,1):>7}  "
              f"{_fmt(r.mardia_b2p_z,2):>7}")
    print(sep)


def save_cumulant_csv(rows: list[StageResult], path: Path) -> None:
    fields = ["model", "stage", "N", "D",
              "mean_abs_k3", "std_k3", "mean_k4", "std_k4", "max_k4",
              "frac_non_gauss", "pr", "effective_rank", "whiteness", "js_gauss",
              "mardia_b2p", "mardia_b2p_exp", "mardia_b2p_z"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(getattr(r, k.replace("mean_abs_k3","mean_abs_k3")
                                           .replace("frac_non_gauss","frac_nong"), float("nan")), 4)
                        if isinstance(getattr(r, k.replace("frac_non_gauss","frac_nong"),
                                              getattr(r, k, float("nan"))), float)
                        else getattr(r, k.replace("frac_non_gauss","frac_nong"), "")
                        for k in fields})
    # Simple write without the above gymnastics
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r.model, r.stage, r.N, r.D,
                        round(r.mean_abs_k3, 4), round(r.std_k3, 4),
                        round(r.mean_k4, 4), round(r.std_k4, 4), round(r.max_k4, 4),
                        round(r.frac_nong, 4),
                        round(r.pr, 3), round(r.effective_rank, 2),
                        round(r.whiteness, 4), round(r.js_gauss, 4),
                        round(r.mardia_b2p, 2), round(r.mardia_b2p_exp, 2),
                        round(r.mardia_b2p_z, 3)])
    print(f"  → CSV saved to {path}")


def save_latex_table(rows: list[StageResult], path: Path) -> None:
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Cumulants $\kappa_3$, $\kappa_4$, Participation Ratio (PR), "
        r"effective rank (ER), covariance whiteness, and Mardia kurtosis "
        r"at each activation stage. Under $\mathcal{N}_p$: $\bar\kappa_4\to 0$, "
        r"$b_{2,p}\to p(p+2)$.}",
        r"\label{tab:cumulants}",
        r"\scriptsize",
        r"\begin{tabular}{ll r r r r r r r r}",
        r"\toprule",
        r"Model & Stage & $\overline{|\kappa_3|}$ & $\overline{\kappa_4}$ & "
        r"$\%|\kappa_4|{>}0.5$ & PR & ER & White & $b_{2,p}$ & $b_{2,p}^*$ \\",
        r"\midrule",
    ]
    prev_m = None
    for r in rows:
        if prev_m is not None and r.model != prev_m:
            lines.append(r"\midrule")
        prev_m = r.model
        b  = f"{r.mardia_b2p:.1f}"   if not math.isnan(r.mardia_b2p)     else "—"
        bs = f"{r.mardia_b2p_exp:.1f}" if not math.isnan(r.mardia_b2p_exp) else "—"
        pr = f"{r.pr:.1f}"           if not math.isnan(r.pr)              else "—"
        er = f"{r.effective_rank:.1f}" if not math.isnan(r.effective_rank) else "—"
        wh = f"{r.whiteness:.3f}"    if not math.isnan(r.whiteness)       else "—"
        lines.append(
            f"{r.model} & {r.stage} & {r.mean_abs_k3:.3f} & {r.mean_k4:+.3f} & "
            f"{r.frac_nong*100:.1f}\\% & {pr} & {er} & {wh} & {b} & {bs} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{10}{l}{\footnotesize ER = effective rank; "
              r"White = $\|C-\mathrm{diag}(C)\|_F/\|C\|_F$.}",
              r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    print(f"  → LaTeX saved to {path}")


def plot_kappa4_violins(
        all_k4:   dict[str, dict[str, np.ndarray]],
        save_path: Path,
) -> None:
    models   = list(all_k4.keys())
    stages   = sorted({s for m in all_k4.values() for s in m})
    n_stages = len(stages)
    colors   = {"Gaussian": "#4C72B0", "Rosenblatt": "#DD8452"}
    pos      = np.arange(len(models))

    fig, axes = plt.subplots(1, n_stages,
                             figsize=(max(3 * n_stages, 14), 5),
                             sharey=False)
    if n_stages == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        data = [all_k4[m].get(stage, np.array([])) for m in models]
        data = [d[np.isfinite(d)] for d in data]
        valid = [d for d in data if len(d) > 0]
        if valid:
            vp = ax.violinplot(valid, positions=pos[:len(valid)],
                               showmedians=True, showextrema=False)
            for pc, m in zip(vp["bodies"], models):
                pc.set_facecolor(colors.get(m, "#888"))
                pc.set_alpha(0.75)
            vp["cmedians"].set_color("k")
        ax.axhline(0.0, color="red", lw=1.0, ls="--")
        ax.set_xticks(pos);  ax.set_xticklabels(models, rotation=20,
                                                 ha="right", fontsize=7)
        ax.set_title(stage, fontsize=8)
        ax.set_ylabel("$\\kappa_4$" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-component $\\kappa_4$ distribution across stages "
                 "($\\kappa_4\\to 0$: Gaussian)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Violin plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_train_unet(
        noise_type: str,
        cfg: Config,
        save_dir: Path,  # Directory to fetch models from
) -> tuple[ConditionalUNet, RosenblattForward]:
    """Load a trained ConditionalUNet from checkpoint or train from scratch."""
    sfn   = sigma_multiplicative()
    tag   = f"{noise_type}_{sfn.__name__}_H{cfg.H}"
    ckpt  = save_dir / "multiplicative" / f"{tag}_final.pt"

    forward = RosenblattForward(sfn, noise_type=noise_type,
                                H=cfg.H, device=cfg.device,
                                sigma_max=cfg.sigma_max)

    model = ConditionalUNet(num_classes=10, base_ch=cfg.base_ch).to(cfg.device)
    if ckpt.exists():
        print(f"  Loading UNet checkpoint: {ckpt}")
        model.load_state_dict(
            torch.load(ckpt, map_location=cfg.device, weights_only=True))
    else:
        print(f"  Checkpoint not found; training {tag} ...")
        model, forward = train(sfn, cfg, noise_type=noise_type,
                               H=cfg.H, save_dir=str(save_dir))
    model.eval()
    return model, forward


def _load_or_train_latent(
        noise_type: str,
        cfg: Config,
        save_dir: Path,
) -> tuple[ConvAutoencoder, LatentMLPDenoiser, RosenblattForward]:
    """Load trained AE + latent MLP or train them."""
    lat_dir  = save_dir / "latent"
    lat_dir.mkdir(parents=True, exist_ok=True)
    ae_ckpt  = lat_dir / "ae_final.pt"

    ae = ConvAutoencoder().to(cfg.device)
    if ae_ckpt.exists():
        print(f"  Loading AE: {ae_ckpt}")
        ae.load_state_dict(
            torch.load(ae_ckpt, map_location=cfg.device, weights_only=True))
    else:
        print("  Training autoencoder …")
        ae = train_autoencoder(cfg)
    ae.eval()

    mlp_model, fwd = train_latent(ae, cfg,
                                      sigma_max=cfg.sigma_max,
                                      noise_type=noise_type)
    mlp_model.eval()
    return ae, mlp_model, fwd


# ─────────────────────────────────────────────────────────────────────────────
# 9. ConditionalUNetFlexible  (encode + decode methods already present)
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalUNetFlexible(nn.Module):
    """
    ConditionalUNet with variable bottleneck channel count.

    bottleneck_factor:
        0.25 → base_ch       (tight compression)
        0.5  → 2 × base_ch
        1.0  → 4 × base_ch   (identical to standard ConditionalUNet)
        2.0  → 8 × base_ch   (over-complete)
        3.0  → 12 × base_ch

    encode() and decode() are exposed as public methods to support:
        - gradient sensitivity analysis (retain_grad on h3)
        - ablation of non-Gaussian channels
        - rigidity test perturbations
    """

    def __init__(self, t_dim: int = 256, num_classes: int = 10,
                 base_ch: int = 128, in_channels: int = 1,
                 bottleneck_factor: float = 1.0) -> None:
        super().__init__()
        bneck_ch = max(base_ch, int(round(4 * base_ch * bottleneck_factor)))
        enc2_ch  = 2 * base_ch
        self.bottleneck_factor = bottleneck_factor
        self.bneck_ch          = bneck_ch

        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1     = nn.Sequential(ResBlockAdaGN(base_ch,  base_ch,  t_dim),
                                       ResBlockAdaGN(base_ch,  base_ch,  t_dim))
        self.pool1     = nn.Conv2d(base_ch,   enc2_ch,  3, stride=2, padding=1)
        self.down2     = nn.Sequential(ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim),
                                       ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim))
        self.attn2     = SelfAttention(enc2_ch, spatial_size=14)
        self.pool2     = nn.Conv2d(enc2_ch,   bneck_ch, 3, stride=2, padding=1)

        self.mid1      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)
        self.attn_mid  = SelfAttention(bneck_ch, spatial_size=7)
        self.mid2      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)

        self.up2       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(bneck_ch, enc2_ch, 3, padding=1))
        self.up_res2   = nn.ModuleList([
            ResBlockAdaGN(enc2_ch * 2, enc2_ch, t_dim),
            ResBlockAdaGN(enc2_ch,     enc2_ch, t_dim)])
        self.up_attn2  = SelfAttention(enc2_ch, spatial_size=14)
        self.up1       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc2_ch, base_ch, 3, padding=1))
        self.up_res1   = nn.ModuleList([
            ResBlockAdaGN(base_ch * 2, base_ch, t_dim),
            ResBlockAdaGN(base_ch,     base_ch, t_dim)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor,
                t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb      = self.time_mlp(self.t_embed(t)) + self.label_emb(y)
        h3, h2, h1 = self.encode(x, t_emb)
        return self.decode(h3, h2, h1, t_emb)

    def encode(self, x: torch.Tensor,
               t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (h3 bottleneck, h2 skip, h1 skip)."""
        x  = self.init_conv(x)
        h1 = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2 = self.attn2(self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb))
        h3 = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)
        return h3, h2, h1

    def decode(self, h3: torch.Tensor, h2: torch.Tensor,
               h1: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.up_attn2(
                self.up_res2[1](
                    self.up_res2[0](torch.cat([self.up2(h3), h2], 1), t_emb),
                    t_emb))
        h = self.up_res1[1](
                self.up_res1[0](torch.cat([self.up1(h), h1], 1), t_emb),
                t_emb)
        return self.out(h)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Training flexible UNet
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalUNetFlexible(nn.Module):
    """
    Identical to ConditionalUNet but with a variable bottleneck channel count.

    bottleneck_factor : float
        Scales the bottleneck channels relative to the standard 4 × base_ch.
            0.25 → base_ch      (tight compression: bottleneck = encoder entry width)
            0.5  → 2 × base_ch  (moderate compression)
            1.0  → 4 × base_ch  (standard, identical to ConditionalUNet)
            2.0  → 8 × base_ch  (over-complete bottleneck)

    All other architecture choices (encoder/decoder, attention, AdaGN) are
    identical to ConditionalUNet to ensure fair comparison.
    """

    def __init__(
            self,
            t_dim:             int   = 256,
            num_classes:       int   = 10,
            base_ch:           int   = 128,
            in_channels:       int   = 1,
            bottleneck_factor: float = 1.0,
    ) -> None:
        super().__init__()
        bneck_ch = max(base_ch, int(round(4 * base_ch * bottleneck_factor)))
        enc2_ch  = 2 * base_ch    # width of the second encoder level

        self.bottleneck_factor = bottleneck_factor
        self.bneck_ch          = bneck_ch

        # ── Time/class conditioning ──────────────────────────────────────────
        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        # ── Encoder ──────────────────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1  = nn.Sequential(
            ResBlockAdaGN(base_ch,  base_ch,  t_dim),
            ResBlockAdaGN(base_ch,  base_ch,  t_dim))
        self.pool1  = nn.Conv2d(base_ch, enc2_ch, 3, stride=2, padding=1)

        self.down2  = nn.Sequential(
            ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim),
            ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim))
        self.attn2  = SelfAttention(enc2_ch, spatial_size=14)
        self.pool2  = nn.Conv2d(enc2_ch, bneck_ch, 3, stride=2, padding=1)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid1      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)
        self.attn_mid  = SelfAttention(bneck_ch, spatial_size=7)
        self.mid2      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(bneck_ch, enc2_ch, 3, padding=1))
        # skip from enc: enc2_ch channels from h2
        self.up_res2  = nn.ModuleList([
            ResBlockAdaGN(enc2_ch * 2, enc2_ch, t_dim),
            ResBlockAdaGN(enc2_ch,     enc2_ch, t_dim)])
        self.up_attn2 = SelfAttention(enc2_ch, spatial_size=14)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc2_ch, base_ch, 3, padding=1))
        # skip from enc: base_ch channels from h1
        self.up_res1  = nn.ModuleList([
            ResBlockAdaGN(base_ch * 2, base_ch, t_dim),
            ResBlockAdaGN(base_ch,     base_ch, t_dim)])

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor,
                t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.t_embed(t)) + self.label_emb(y)

        # Encode
        h3, h2, h1 = self.encode(x, t_emb)

        # Decode
        return self.decode(h3, h2, h1, t_emb)

    def decode(self, h3: torch.Tensor, h2: torch.Tensor, h1: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h  = self.up_attn2(
                self.up_res2[1](
                    self.up_res2[0](
                        torch.cat([self.up2(h3), h2], dim=1), t_emb),
                    t_emb))
        h  = self.up_res1[1](
                self.up_res1[0](
                    torch.cat([self.up1(h), h1], dim=1), t_emb),
                t_emb)
        return self.out(h)

    def encode(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.init_conv(x)
        h1 = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2 = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2 = self.attn2(h2)

        # Bottleneck
        h3 = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)
        return h3, h2, h1


def train_flexible_unet(
        bottleneck_factor: float,
        noise_type:        str,
        cfg:               Config,
        save_dir:          Path,
) -> tuple[ConditionalUNetFlexible, RosenblattForward]:
    """
    Train (or load) a ConditionalUNetFlexible for Experiment β.

    Saves checkpoint to save_dir / f"beta_{noise_type}_bf{bottleneck_factor}_final.pt".
    """
    sfn  = sigma_multiplicative()
    bf_s = f"{bottleneck_factor:.2f}".replace(".", "p")
    tag  = f"beta_{noise_type}_bf{bf_s}"
    ckpt = save_dir / f"{tag}_final.pt"
    save_dir.mkdir(parents=True, exist_ok=True)

    forward = RosenblattForward(sfn, noise_type=noise_type,
                                H=cfg.H, device=cfg.device,
                                sigma_max=cfg.sigma_max)

    model = ConditionalUNetFlexible(
        num_classes=10, base_ch=cfg.base_ch,
        bottleneck_factor=bottleneck_factor).to(cfg.device)

    if ckpt.exists():
        print(f"  Loading flexible UNet: {ckpt}")
        model.load_state_dict(
            torch.load(ckpt, map_location=cfg.device, weights_only=True))
        model.eval()
        return model, forward

    print(f"  Training {tag}  (bneck_ch={model.bneck_ch}) …")

    train_ds = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    val_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    start_ep = 0
    latest_ckpt = None
    for i in range(1, cfg.epochs):
        p = save_dir / f"{tag}_ep{i}.pt"
        if p.exists():
            start_ep = i
            latest_ckpt = p

    if latest_ckpt is not None:
        print(f"  Resuming from intermediate: {latest_ckpt}")
        model.load_state_dict(
            torch.load(latest_ckpt, map_location=cfg.device, weights_only=True))

    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, last_epoch=start_ep - 1)

    eg2 = float(getattr(sfn, "eg2", 1.0))
    forward.set_eg2(eg2)

    for ep in range(start_ep, cfg.epochs):
        t0 = time.time();  model.train();  el = 0
        for x0, lbl in train_dl:
            x0, lbl = (x0.to(cfg.device, non_blocking=True),
                       lbl.to(cfg.device, non_blocking=True))
            B = x0.size(0)
            # Classifier-free guidance dropout
            cf     = torch.rand(B, device=cfg.device) < 0.1
            lbl2   = lbl.clone();  lbl2[cf] = 10
            t      = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
            x_t, _, sig = forward.corrupt(x0, t, y=lbl2)
            c_in   = forward.c_in(t).view(-1, 1, 1, 1)
            opt.zero_grad(set_to_none=True)
            if cfg.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(x_t * c_in, t, lbl2)
                    loss = F.smooth_l1_loss(pred, x0)
            else:
                pred = model(x_t * c_in, t, lbl2)
                loss = F.smooth_l1_loss(pred, x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step();  ema.update()
            el += loss.item() * B

        el /= len(train_dl.dataset)
        model.eval();  ema.apply_shadow();  vl = 0
        with torch.no_grad():
            for x0, lbl in val_dl:
                x0, lbl = (x0.to(cfg.device, non_blocking=True),
                           lbl.to(cfg.device, non_blocking=True))
                B = x0.size(0)
                t      = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
                x_t, _, _ = forward.corrupt(x0, t, y=lbl)
                c_in   = forward.c_in(t).view(-1, 1, 1, 1)
                pred   = model(x_t * c_in, t, lbl)
                vl    += F.smooth_l1_loss(pred, x0).item() * B
        vl /= len(val_dl.dataset)
        ema.restore();  sch.step()
        print(f"  [{tag}] ep {ep+1:2d}/{cfg.epochs}  "
              f"tr={el:.5f}  va={vl:.5f}  {time.time()-t0:.1f}s")

        if (ep + 1) % 5 == 0 and (ep + 1) != cfg.epochs:
            ckpt_ep = save_dir / f"{tag}_ep{ep+1}.pt"
            ema.apply_shadow()
            torch.save(model.state_dict(), ckpt_ep)
            ema.restore()
            print(f"  Saved intermediate → {ckpt_ep}")

    ema.apply_shadow()
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved → {ckpt}")
    model.eval()
    return model, forward


# ─────────────────────────────────────────────────────────────────────────────
# 11. Experiment α — main function
# ─────────────────────────────────────────────────────────────────────────────

STAGE_LABELS_UNET = {
    "input":       "Input $x_0$",
    "corrupted":   "Corrupted $x_T$",
    "mid_t05":     "Mid-gen $x_{0.5}$",
    "bottleneck":  "Bottleneck",
    "x0hat":       "Output $\\hat{x}_0$",
}

STAGE_LABELS_LATENT = {
    "image_input":  "Image $x_0$",
    "latent_z0":    "AE latent $z_0$",
    "latent_corr":  "Corrupted $z_T$",
    "mlp_mid":      "MLP mid-layer",
    "latent_x0hat": "Decoded $\\hat{x}_0$",
}

N_SAMPLES_ALPHA = 2000   # number of samples for cumulant estimation


def _analyse_stage(
        acts:       torch.Tensor,
        model_name: str,
        stage_key:  str,
        stage_label:str,
) -> tuple[StageResult, np.ndarray]:
    """Compute cumulants + Mardia stats for one stage; return (StageResult, k4 array)."""
    cum  = compute_marginal_cumulants(acts)
    mard = mardia_statistics(acts, use_pca=True)
    result = StageResult(
        model          = model_name,
        stage          = stage_label,
        N              = cum["N"],
        D              = cum["D"],
        mean_abs_k3    = cum["mean_abs_kappa3"],
        std_k3         = cum["std_kappa3"],
        mean_k4        = cum["mean_kappa4"],
        std_k4         = cum["std_kappa4"],
        max_k4         = cum["max_kappa4"],
        frac_nong      = cum["frac_non_gauss"],
        mardia_b2p     = mard["b2p"],
        mardia_b2p_exp = mard["b2p_exp"],
        mardia_b2p_z   = mard["b2p_z"],
    )
    return result, cum["kappa4"]


def run_experiment_alpha(cfg: Config, save_dir: Path) -> list[StageResult]:
    """
    Full Experiment α pipeline.

    For each of {Gaussian, Rosenblatt} models:
        1. Load (or train) the UNet and the latent AE+MLP.
        2. Extract activations at all stages.
        3. Compute cumulants and Mardia statistics.

    Returns list of StageResult records (used for table output).
    """
    print("\n" + "═" * 70)
    print("Experiment α — Cumulant Gaussianization Probe")
    print("═" * 70)

    cfg.save_dir = save_dir
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)

    all_rows:    list[StageResult]          = []
    all_kappa4:  dict[str, dict[str, np.ndarray]] = {}

    for noise_type in ("gaussian", "rosenblatt"):
        model_name = "Gaussian" if noise_type == "gaussian" else "Rosenblatt"
        print(f"\n── Model: {model_name} ──────────────────────────────────────────")
        all_kappa4[model_name] = {}

        # ── 6a. UNet stages ──────────────────────────────────────────────────
        model, forward = _load_or_train_unet(noise_type, cfg, save_dir)

        print("  Extracting UNet stage activations …")
        t0 = time.time()
        unet_acts = extract_all_stages(
            model, forward, test_ds, cfg, n_samples=N_SAMPLES_ALPHA)
        print(f"  Done in {time.time()-t0:.1f}s")

        for stage_key, label in STAGE_LABELS_UNET.items():
            acts = unet_acts[stage_key]
            row, k4 = _analyse_stage(acts, model_name, stage_key, label)
            all_rows.append(row)
            all_kappa4[model_name][label] = k4
            print(f"  {label:22s}  N={row.N}  D={row.D}  "
                  f"κ4={row.mean_k4:+.3f}  Z={row.mardia_b2p_z:+.2f}")

        # ── 6b. Latent stages ────────────────────────────────────────────────
        ae, mlp_model, fwd_lat = _load_or_train_latent(noise_type, cfg, save_dir)

        print("  Extracting latent stage activations …")
        t0 = time.time()
        lat_acts = extract_latent_stages(
            ae, mlp_model, fwd_lat, test_ds, cfg, n_samples=N_SAMPLES_ALPHA)
        print(f"  Done in {time.time()-t0:.1f}s")

        for stage_key, label in STAGE_LABELS_LATENT.items():
            acts = lat_acts[stage_key]
            row, k4 = _analyse_stage(acts, model_name, stage_key, label)
            all_rows.append(row)
            all_kappa4[model_name][label] = k4
            print(f"  {label:22s}  N={row.N}  D={row.D}  "
                  f"κ4={row.mean_k4:+.3f}  Z={row.mardia_b2p_z:+.2f}")

    # ── 6c. Output ───────────────────────────────────────────────────────────
    print_cumulant_table(all_rows)
    save_cumulant_csv(all_rows,  save_dir / "alpha_cumulants.csv")
    save_latex_table(all_rows,   save_dir / "alpha_cumulants.tex")
    plot_kappa4_violins(all_kappa4, save_dir / "alpha_kappa4_violins.png")

    return all_rows

# ─────────────────────────────────────────────────────────────────────────────
# 13. Experiment β
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_beta(
        cfg:               Config,
        save_dir:          Path,
        bottleneck_factors: list[float] | None = None,
) -> list[BetaResult]:
    if bottleneck_factors is None:
        bottleneck_factors = [0.25, 0.5, 1.0, 2.0, 3.0]
 
    print("\n" + "═" * 72)
    print("Experiment β — Bottleneck Width vs. Gaussianization")
    print("═" * 72)
    print(f"  Factors: {bottleneck_factors}")
 
    cfg.save_dir = save_dir
    test_ds  = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    log_dir  = save_dir if os.access(str(save_dir), os.W_OK) else Path(".").resolve()
    beta_rows: list[BetaResult] = []
 
    for noise_type in ("gaussian", "rosenblatt"):
        for bf in bottleneck_factors:
            print(f"\n── noise={noise_type}  bf={bf} "
                  f"──────────────────────────────────")
            model, fwd = train_flexible_unet(bf, noise_type, cfg, save_dir)
 
            # ── Collect bottleneck, input, output activations ─────────────────
            bn_store   = ActivationStore(spatial_pool=True)
            bn_handle  = model.mid2.register_forward_hook(bn_store.hook_fn)
            raw_list:  list[torch.Tensor] = []
            x0h_list:  list[torch.Tensor] = []
 
            loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                                shuffle=False, num_workers=2)
            n_col  = 0
            with torch.no_grad():
                for x0, y in loader:
                    if n_col >= N_SAMPLES_ALPHA:
                        break
                    B   = x0.size(0)
                    x0  = x0.to(cfg.device);  y = y.to(cfg.device)
                    raw_list.append(x0.view(B, -1).cpu())
                    t_one = torch.ones(B, device=cfg.device)
                    x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
                    t_min = torch.full((B,), cfg.T_MIN, device=cfg.device)
                    null  = torch.full_like(y, 10)
                    c_in  = fwd.c_in(t_min).view(-1, 1, 1, 1)
                    if cfg.device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            x0c = model(x_T * c_in, t_min, y).float()
                            x0u = model(x_T * c_in, t_min, null).float()
                    else:
                        x0c = model(x_T * c_in, t_min, y)
                        x0u = model(x_T * c_in, t_min, null)
                    x0h = (x0u + cfg.cfg_scale * (x0c - x0u)).clamp(-1., 1.)
                    x0h_list.append(x0h.view(B, -1).cpu())
                    n_col += B
 
            bn_handle.remove()
 
            bneck_acts = bn_store.get()[:N_SAMPLES_ALPHA]
            raw_acts   = torch.cat(raw_list, 0)[:N_SAMPLES_ALPHA]
            x0h_acts   = torch.cat(x0h_list, 0)[:N_SAMPLES_ALPHA]
 
            # ── Cumulant analysis ─────────────────────────────────────────────
            cum_bn  = compute_marginal_cumulants(bneck_acts, max_components=model.bneck_ch)
            cum_raw = compute_marginal_cumulants(raw_acts)
            cum_x0h = compute_marginal_cumulants(x0h_acts)
 
            mard_bn      = mardia_statistics(bneck_acts, use_pca=True)
            mard_bn_avg  = mardia_statistics(bneck_acts, use_pca=False)
            mard_x0h     = mardia_statistics(x0h_acts,  use_pca=True)
            mard_x0h_avg = mardia_statistics(x0h_acts,  use_pca=False)
 
            spec_bn = compute_spectrum_stats(bneck_acts)
            wh_bn   = covariance_whiteness(bneck_acts)
            js_bn   = js_divergence_from_gaussian(bneck_acts)
 
            # ── Mardia d_ii vs per-sample MSE correlation ─────────────────────
            d_ii = mard_bn.get("d_ii")
            if d_ii is not None:
                sample_mse  = F.mse_loss(x0h_acts, raw_acts, reduction="none").mean(1)
                c_dii  = d_ii.cpu() - d_ii.cpu().mean()
                c_smse = sample_mse - sample_mse.mean()
                denom  = (c_dii**2).sum().sqrt() * (c_smse**2).sum().sqrt() + 1e-8
                mard_mse_corr = float((c_dii * c_smse).sum() / denom)
            else:
                mard_mse_corr = float("nan")
 
            # ── SVD spectrum & PR ─────────────────────────────────────────────
            pr_bn   = spec_bn["pr"]
            er_bn   = spec_bn["effective_rank"]
 
            _Xc = bneck_acts - bneck_acts.mean(0)
            q   = min(N_SAMPLES_ALPHA - 1, model.bneck_ch, 512)
            _, S_svd, _ = torch.pca_lowrank(_Xc, q=q, center=False, niter=4)
 
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(S_svd.cpu().numpy(), ".", markersize=3)
            ax.set_title(f"SVD spectrum — bf={bf}, {noise_type}  PR={pr_bn:.1f}")
            ax.set_xlabel("Singular value index");  ax.set_ylabel("$s_i$")
            ax.set_yscale("log");  ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(log_dir / f"beta_svd_bf{bf}_{noise_type}.png", dpi=130)
            plt.close()
 
            # ── Gradient sensitivity ──────────────────────────────────────────
            x0_b, y_b = next(iter(DataLoader(test_ds, batch_size=128,
                                             shuffle=True)))
            x0_b = x0_b.to(cfg.device);  y_b = y_b.to(cfg.device)
            B_b  = x0_b.size(0)
            t_mn = torch.full((B_b,), cfg.T_MIN, device=cfg.device)
            t_on = torch.ones(B_b, device=cfg.device)
            xT_b, _, _ = fwd.corrupt(x0_b, t_on, y=y_b)
            c_in_b = fwd.c_in(t_mn).view(-1, 1, 1, 1)
            x_in_b = xT_b * c_in_b
            t_emb_b = model.time_mlp(model.t_embed(t_mn)) + model.label_emb(y_b)
 
            with torch.enable_grad():
                h3_b, h2_b, h1_b = model.encode(x_in_b, t_emb_b)
                h3_b.retain_grad()
                x0h_b = model.decode(h3_b, h2_b, h1_b, t_emb_b).float()
                F.mse_loss(x0h_b, x0_b.float()).backward(retain_graph=True)
                grad_mse = h3_b.grad.clone();  h3_b.grad.zero_()
                F.l1_loss(x0h_b, x0_b.float()).backward()
                grad_mae = h3_b.grad.clone()
 
            k4_grad_mse = compute_marginal_cumulants(
                grad_mse.mean((-2,-1)).cpu())["mean_kappa4"]
            k4_grad_mae = compute_marginal_cumulants(
                grad_mae.mean((-2,-1)).cpu())["mean_kappa4"]
 
            # ── Channel ablation ───────────
            k4_all = compute_marginal_cumulants(bneck_acts,
                                                max_components=model.bneck_ch)
            mask_ng = torch.tensor(np.abs(k4_all["kappa4"]) > 0.5,
                                   device=cfg.device)               # (bneck_ch,)
            h3_abl  = h3_b.detach().clone()
            if mask_ng.shape[0] == h3_abl.shape[1]:
                h3_abl[:, mask_ng, :, :] = 0.0
            with torch.no_grad():
                x0h_abl = model.decode(h3_abl, h2_b, h1_b, t_emb_b).float()
                ab_mse  = F.mse_loss(x0h_abl, x0_b.float()).item()
                ab_mae  = F.l1_loss( x0h_abl, x0_b.float()).item()
 
            # ── Offline losses ────────────────────────────────────────────────
            off_mse    = F.mse_loss(x0h_acts, raw_acts).item()
            off_mae    = F.l1_loss( x0h_acts, raw_acts).item()
            off_huber  = F.smooth_l1_loss(x0h_acts, raw_acts).item()
            off_q09    = torch.where(
                x0h_acts - raw_acts > 0,
                0.9 * (x0h_acts - raw_acts),
                0.1 * -(x0h_acts - raw_acts)).mean().item()
 
            # ── Full layer trace (γ) for this bf ─────────────────────────────
            trace = extract_full_layer_trace(model, fwd, test_ds, cfg,
                                             n_samples=N_SAMPLES_ALPHA)
            stage_k4s = [];  stage_mz = []
            for key in UNET_LAYER_KEYS:
                c = compute_marginal_cumulants(trace[key])
                m = mardia_statistics(trace[key], use_pca=True)
                stage_k4s.append(c["mean_kappa4"])
                stage_mz.append(m["b2p_z"])
 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(range(len(UNET_LAYER_KEYS)), stage_k4s, "o-", color="C0")
            ax1.axhline(0, color="red", ls="--", lw=1)
            ax1.axvline(UNET_LAYER_KEYS.index("mid2"), color="gray", ls=":", alpha=0.5)
            ax1.set_xticks(range(len(UNET_LAYER_KEYS)))
            ax1.set_xticklabels([_LAYER_LABELS[k] for k in UNET_LAYER_KEYS],
                                 rotation=55, ha="right", fontsize=7)
            ax1.set_title(f"$\\kappa_4$ trace — bf={bf}, {noise_type}")
            ax1.grid(alpha=0.3)
 
            ax2.plot(range(len(UNET_LAYER_KEYS)), stage_mz, "s-", color="C1")
            ax2.axhline(0, color="red", ls="--", lw=1)
            ax2.axvline(UNET_LAYER_KEYS.index("mid2"), color="gray", ls=":", alpha=0.5)
            ax2.set_xticks(range(len(UNET_LAYER_KEYS)))
            ax2.set_xticklabels([_LAYER_LABELS[k] for k in UNET_LAYER_KEYS],
                                 rotation=55, ha="right", fontsize=7)
            ax2.set_title(f"Mardia-$Z$ trace — bf={bf}, {noise_type}")
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(log_dir / f"beta_trace_bf{bf}_{noise_type}.png", dpi=130)
            plt.close()
 
            # ── Rigidity test (Test C Rosenblatt added) ────────────
            rig = rigidity_test(model, fwd, test_ds, cfg,
                                sigma_levels=[0.3, 0.5, 1.0])
            σ_key = 0.5      # report at σ=0.5
            perturb_g   = rig["gaussian"].get(σ_key, float("nan"))
            perturb_l   = rig["laplace"].get(σ_key, float("nan"))
            perturb_r   = rig["rosenblatt"].get(σ_key, float("nan"))
            perturb_t3  = rig["student_t3"].get(σ_key, float("nan"))
 
            plot_rigidity(rig, log_dir / f"beta_rigidity_bf{bf}_{noise_type}.png",
                          title=f"Rigidity — bf={bf}, {noise_type}")
 
            # ── Assemble result ───────────────────────────────────────────────
            row = BetaResult(
                noise_type              = noise_type,
                bottleneck_factor       = bf,
                bneck_ch                = model.bneck_ch,
                mean_k4_input           = cum_raw["mean_kappa4"],
                mean_k4_bneck           = cum_bn["mean_kappa4"],
                std_k4_bneck            = cum_bn["std_kappa4"],
                max_k4_bneck            = cum_bn["max_kappa4"],
                frac_nong_bneck         = cum_bn["frac_non_gauss"],
                mean_k4_x0hat           = cum_x0h["mean_kappa4"],
                mardia_b2p_z            = mard_bn["b2p_z"],
                mardia_b2p_z_avg        = mard_bn_avg["b2p_z"],
                mardia_b2p_z_x0hat      = mard_x0h["b2p_z"],
                mardia_b2p_z_x0hat_avg  = mard_x0h_avg["b2p_z"],
                offline_loss_mse        = off_mse,
                offline_loss_mae        = off_mae,
                offline_loss_huber      = off_huber,
                offline_loss_quantile   = off_q09,
                k4_grad_mse             = k4_grad_mse,
                k4_grad_mae             = k4_grad_mae,
                mardia_mse_corr         = mard_mse_corr,
                ablated_mse             = ab_mse,
                ablated_mae             = ab_mae,
                pr_bneck                = pr_bn,
                effective_rank_bneck    = er_bn,
                whiteness_bneck         = wh_bn,
                js_gauss_bneck          = js_bn,
                perturb_gauss_huber     = perturb_g,
                perturb_laplace_huber   = perturb_l,
                perturb_rosenblatt_huber= perturb_r,
                perturb_t3_huber        = perturb_t3,
            )
            beta_rows.append(row)
 
            print(f"  bneck_ch={model.bneck_ch:5d}  κ4_bneck={row.mean_k4_bneck:+.3f}"
                  f"  PR={pr_bn:.1f}  ER={er_bn:.1f}  White={wh_bn:.3f}"
                  f"  Z={row.mardia_b2p_z:+.2f}")
            print(f"  Rigidity (σ=0.5): G={perturb_g:.4f}  L={perturb_l:.4f}"
                  f"  R={perturb_r:.4f}  t3={perturb_t3:.4f}")
 
            # Incremental save
            _save_beta_csv(beta_rows,   log_dir / "beta_bottleneck.csv",   silent=True)
            _save_beta_latex(beta_rows, log_dir / "beta_bottleneck.tex",   silent=True)
            _plot_beta(beta_rows,       log_dir / "beta_kappa4_vs_bottleneck.png",
                       silent=True)
            plot_beta_rigidity_grid(
                beta_rows,
                log_dir / "beta_rigidity_grid_sigma0p5.png",
                sigma_label="0.5",
                silent=True,
            )
 
    _print_beta_table(beta_rows)
    _save_beta_csv(beta_rows,   log_dir / "beta_bottleneck.csv")
    _save_beta_latex(beta_rows, log_dir / "beta_bottleneck.tex")
    _plot_beta(beta_rows,       log_dir / "beta_kappa4_vs_bottleneck.png")
    plot_beta_rigidity_grid(
        beta_rows,
        log_dir / "beta_rigidity_grid_sigma0p5.png",
        sigma_label="0.5",
    )
    return beta_rows


# ─────────────────────────────────────────────────────────────────────────────
# 13. Experiment γ — full layer-by-layer kurtosis trace
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_gamma(
        cfg:      Config,
        save_dir: Path,
) -> list[LayerStats]:
    """
    Measure κ4, PR, whiteness, and Mardia-Z at EVERY named UNet layer for
    both Gaussian and Rosenblatt models.

    Produces:
      gamma_layer_stats.csv
      gamma_kappa4_profile.png  — κ4 vs layer depth
      gamma_pr_profile.png      — PR vs layer depth
      gamma_whiteness_profile.png
    """
    print("\n" + "═" * 72)
    print("Experiment γ — Full Layer-by-Layer Kurtosis Trace")
    print("═" * 72)

    cfg.save_dir = save_dir
    test_ds  = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    all_rows: list[LayerStats] = []

    for noise_type in ("gaussian", "rosenblatt"):
        mname = "Gaussian" if noise_type == "gaussian" else "Rosenblatt"
        print(f"\n── {mname} ──")
        model, fwd = _load_or_train_unet(noise_type, cfg, save_dir)

        print(f"  Running full trace ({N_SAMPLES_ALPHA} samples) …")
        t0    = time.time()
        trace = extract_full_layer_trace(model, fwd, test_ds, cfg,
                                         n_samples=N_SAMPLES_ALPHA)
        print(f"  Done in {time.time()-t0:.1f}s")

        for depth, key in enumerate(UNET_LAYER_KEYS):
            acts = trace[key]
            if acts.numel() == 0:
                continue
            cum  = compute_marginal_cumulants(acts)
            spec = compute_spectrum_stats(acts)
            wh   = covariance_whiteness(acts)
            mard = mardia_statistics(acts, use_pca=True)

            row = LayerStats(
                model         = mname,
                noise_type    = noise_type,
                layer_key     = key,
                layer_label   = _LAYER_LABELS[key],
                depth_index   = depth,
                mean_k4       = cum["mean_kappa4"],
                std_k4        = cum["std_kappa4"],
                frac_nong     = cum["frac_non_gauss"],
                pr            = spec["pr"],
                effective_rank = spec["effective_rank"],
                whiteness     = wh,
                mardia_b2p_z  = mard["b2p_z"],
            )
            all_rows.append(row)
            print(f"  {key:14s}  D={cum['D']:4d}  κ4={cum['mean_kappa4']:+.3f}"
                  f"  PR={spec['pr']:.1f}  Z={mard['b2p_z']:+.2f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = save_dir / "gamma_layer_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "noise_type", "layer_key", "layer_label",
                    "depth_index", "mean_k4", "std_k4", "frac_nong",
                    "pr", "effective_rank", "whiteness", "mardia_b2p_z"])
        for r in all_rows:
            w.writerow([r.model, r.noise_type, r.layer_key, r.layer_label,
                        r.depth_index,
                        round(r.mean_k4, 4), round(r.std_k4, 4),
                        round(r.frac_nong, 4), round(r.pr, 2),
                        round(r.effective_rank, 2), round(r.whiteness, 4),
                        round(r.mardia_b2p_z, 3)])
    print(f"  → CSV saved to {csv_path}")

    # ── Plot κ4 profile ───────────────────────────────────────────────────────
    _plot_layer_profiles(all_rows, save_dir)
    return all_rows


def _plot_layer_profiles(rows: list[LayerStats], save_dir: Path) -> None:
    """Three-panel figure: κ4, PR, whiteness vs layer depth."""
    colors = {"Gaussian": "#4C72B0", "Rosenblatt": "#DD8452"}
    models = ["Gaussian", "Rosenblatt"]

    metrics = [
        ("mean_k4",       "Mean $\\kappa_4$",   "gamma_kappa4_profile.png"),
        ("pr",            "Participation Ratio (PR)", "gamma_pr_profile.png"),
        ("whiteness",     "Covariance whiteness", "gamma_whiteness_profile.png"),
        ("mardia_b2p_z",  "Mardia-$Z$ (deviation from $\\mathcal{N}_p$)",
                          "gamma_mardiaZ_profile.png"),
    ]
    layers = UNET_LAYER_KEYS
    labels = [_LAYER_LABELS[k] for k in layers]

    for attr, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(12, 4))
        for mname in models:
            sub = {r.layer_key: getattr(r, attr)
                   for r in rows if r.model == mname}
            vals = [sub.get(k, float("nan")) for k in layers]
            ax.plot(range(len(layers)), vals, marker="o",
                    color=colors[mname], linewidth=2, label=mname)
        if attr == "mean_k4":
            ax.axhline(0.0, color="red", lw=1.0, ls="--",
                       label="$\\kappa_4=0$ (Gaussian)")
        ax.axvline(layers.index("mid2"), color="gray", lw=1.0, ls=":",
                   alpha=0.6, label="Bottleneck (mid2)")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Experiment γ: {ylabel} vs layer depth", fontsize=10)
        ax.legend(fontsize=8);  ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → Plot saved to {save_dir / fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Experiment δ — rigidity test for the bf=1.0 model only
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_delta(cfg: Config, save_dir: Path) -> None:
    """
    Focused rigidity test on the standard (bf=1.0) model only,
    sweeping σ over a fine grid.  Produces one clean figure per noise type.
    """
    print("\n" + "═" * 72)
    print("Experiment δ — Latent Perturbation Rigidity Test (bf=1.0)")
    print("═" * 72)

    cfg.save_dir = save_dir
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    σ_grid  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    for noise_type in ("gaussian", "rosenblatt"):
        mname = "Gaussian" if noise_type == "gaussian" else "Rosenblatt"
        print(f"\n── {mname} (bf=1.0) ──")
        model, fwd = train_flexible_unet(1.0, noise_type, cfg, save_dir)
        rig = rigidity_test(model, fwd, test_ds, cfg, sigma_levels=σ_grid)
        plot_rigidity(rig,
                      save_dir / f"delta_rigidity_{noise_type}.png",
                      title=f"Rigidity test — {mname} model (bf=1.0)")

        # Print table
        print(f"  {'σ':>5}  {'clean':>8}  {'gauss':>8}  {'laplace':>8}"
              f"  {'rosen':>8}  {'t3':>8}")
        for σ in σ_grid:
            print(f"  {σ:5.2f}  {rig['clean'][σ]:8.4f}"
                  f"  {rig['gaussian'][σ]:8.4f}"
                  f"  {rig['laplace'][σ]:8.4f}"
                  f"  {rig['rosenblatt'][σ]:8.4f}"
                  f"  {rig['student_t3'][σ]:8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Beta printing/saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_beta_table(rows: list[BetaResult], silent: bool = False) -> None:
    if silent:
        return
    print("\n── Experiment β Summary ────────────────────────────────────────────")
    hdr = (f"{'Noise':12s}  {'bf':>5}  {'C_bnk':>5}  "
           f"{'κ4_in':>7}  {'κ4_bn':>7}  {'κ4_out':>7}  "
           f"{'PR':>6}  {'White':>6}  "
           f"{'Z(PCA)':>7}  {'Huber':>7}  "
           f"{'Rig-G':>7}  {'Rig-R':>7}")
    print(hdr);  print("─" * len(hdr))
    prev_n = None
    for r in rows:
        if prev_n is not None and r.noise_type != prev_n:
            print()
        prev_n = r.noise_type
        print(f"{r.noise_type:12s}  {r.bottleneck_factor:5.2f}  {r.bneck_ch:5d}  "
              f"{r.mean_k4_input:+7.3f}  {r.mean_k4_bneck:+7.3f}  {r.mean_k4_x0hat:+7.3f}  "
              f"{r.pr_bneck:6.1f}  {r.whiteness_bneck:6.3f}  "
              f"{r.mardia_b2p_z:+7.2f}  {r.offline_loss_huber:7.4f}  "
              f"{r.perturb_gauss_huber:7.4f}  {r.perturb_rosenblatt_huber:7.4f}")
    print("─" * len(hdr))
    print(textwrap.dedent("""
    Rig-G/R = Huber reconstruction loss after Gaussian/Rosenblatt bottleneck
    perturbation at σ=0.5.  If Rig-R >> Rig-G: geometry is Gaussian-rigid.
    If κ4_bneck ≈ const across bf: Gaussianization is L²-objective-driven.
    """))


def _save_beta_csv(rows: list[BetaResult], path: Path, silent: bool = False) -> None:
    if not rows:
        return
    fields = list(rows[0].__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([round(v, 4) if isinstance(v, float) else v
                        for v in asdict(r).values()])
    if not silent:
        print(f"  → CSV saved to {path}")


def _save_beta_latex(rows: list[BetaResult], path: Path,
                     silent: bool = False) -> None:
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Experiment $\beta$: bottleneck width vs Gaussianization. "
        r"PR = Participation Ratio; "
        r"Rig = Huber loss after perturbation at $\sigma=0.5$.}",
        r"\label{tab:beta}", r"\scriptsize",
        r"\begin{tabular}{ll r r r r r r r r r}",
        r"\toprule",
        r"Noise & $\alpha_{\rm bf}$ & $C$ & $\kappa_4^{\rm in}$ & "
        r"$\kappa_4^{\rm bn}$ & $\kappa_4^{\rm out}$ & PR & White & "
        r"Mardia-$Z$ & Rig-$\mathcal{N}$ & Rig-Ros \\",
        r"\midrule",
    ]
    prev_n = None
    for r in rows:
        if prev_n is not None and r.noise_type != prev_n:
            lines.append(r"\midrule")
        prev_n = r.noise_type
        lines.append(
            f"{r.noise_type} & {r.bottleneck_factor:.2f} & {r.bneck_ch} & "
            f"{r.mean_k4_input:+.3f} & {r.mean_k4_bneck:+.3f} & "
            f"{r.mean_k4_x0hat:+.3f} & {r.pr_bneck:.1f} & "
            f"{r.whiteness_bneck:.3f} & {r.mardia_b2p_z:+.2f} & "
            f"{r.perturb_gauss_huber:.4f} & {r.perturb_rosenblatt_huber:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    if not silent:
        print(f"  → LaTeX saved to {path}")


def _plot_beta(rows: list[BetaResult], save_path: Path,
               silent: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors  = {"gaussian": "#4C72B0", "rosenblatt": "#DD8452"}
    markers = {"gaussian": "o", "rosenblatt": "s"}

    for noise_type in ("gaussian", "rosenblatt"):
        sub   = [r for r in rows if r.noise_type == noise_type]
        if not sub:
            continue
        bfs   = [r.bottleneck_factor for r in sub]
        k4_bn = [r.mean_k4_bneck     for r in sub]
        k4_in = [r.mean_k4_input      for r in sub]
        pr    = [r.pr_bneck            for r in sub]
        wh    = [r.whiteness_bneck     for r in sub]
        rig_g = [r.perturb_gauss_huber for r in sub]
        rig_r = [r.perturb_rosenblatt_huber for r in sub]
        k4_rel = [(bn/inp if abs(inp) > 1e-6 else float("nan"))
                  for bn, inp in zip(k4_bn, k4_in)]

        c = colors[noise_type];  mk = markers[noise_type]
        lbl = noise_type.capitalize()
        axes[0,0].plot(bfs, k4_bn,  c=c, marker=mk, lw=2, label=lbl)
        axes[0,1].plot(bfs, k4_rel, c=c, marker=mk, lw=2, label=lbl)
        axes[1,0].plot(bfs, pr,     c=c, marker=mk, lw=2, label=lbl)
        axes[1,1].plot(bfs, rig_g,  c=c, marker=mk, lw=2, ls="-",
                       label=f"{lbl} Gauss")
        axes[1,1].plot(bfs, rig_r,  c=c, marker=mk, lw=2, ls="--",
                       label=f"{lbl} Rosen")

    axes[0,0].axhline(0, color="red", lw=1, ls="--", label="$\\kappa_4=0$")
    axes[0,1].axhline(1, color="gray", lw=1, ls=":", label="no change")
    axes[0,1].axhline(0, color="red",  lw=1, ls="--", label="full Gauss.")

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0.25","0.5","1.0","2.0","3.0"])
        ax.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$", fontsize=9)
        ax.legend(fontsize=7);  ax.grid(alpha=0.3)

    axes[0,0].set_title("$\\kappa_4$ at bottleneck");  axes[0,0].set_ylabel("$\\bar\\kappa_4$")
    axes[0,1].set_title("Relative Gaussianization")
    axes[0,1].set_ylabel("$\\kappa_4^{\\rm bn}/\\kappa_4^{\\rm input}$")
    axes[1,0].set_title("Participation Ratio (PR)");  axes[1,0].set_ylabel("PR")
    axes[1,1].set_title("Rigidity (Huber, $\\sigma=0.5$)")
    axes[1,1].set_ylabel("Huber loss")

    fig.suptitle("Experiment $\\beta$: Bottleneck width vs Gaussianization structure",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if not silent:
        print(f"  → Plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Config helper and CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_cfg_for_experiment() -> Config:
    """Build a Config suitable for the experiment (fast, evaluation-focused)."""
    cfg = Config()
    cfg.epochs      = 30
    cfg.ae_epochs   = 20
    cfg.batch_size  = 128
    cfg.n_steps     = 50
    cfg.n_fid       = 2000
    cfg.n_ssim      = 200
    cfg.no_evaluate = True    # skip full FID during training for speed
    cfg.no_plot     = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments α and β: Gaussianization cumulant probes")
    parser.add_argument("--mode",      default="all",
                        choices=["alpha", "beta", "gamma", "delta", "all"],
                        help="Which experiment(s) to run")
    parser.add_argument("--save_dir",  default=str(OUT_ROOT),
                        help="Root output directory (same as main script)")
    parser.add_argument("--epochs",    type=int,   default=None)
    parser.add_argument("--dataset",   default="FashionMNIST",
                        choices=["FashionMNIST", "MNIST"])
    parser.add_argument("--H",         type=float, default=None)
    parser.add_argument("--n_samples", type=int,   default=2000,
                        help="Samples per stage for cumulant estimation (Exp α)")
    parser.add_argument("--bf_list",   nargs="+",  type=float,
                        default=[0.25, 0.5, 1.0, 2.0, 3.0],
                        help="Bottleneck factors for Experiment β")
    args = parser.parse_args()

    cfg          = get_cfg_for_experiment()
    cfg.dataset  = args.dataset
    if args.epochs is not None:
        cfg.epochs    = args.epochs
        cfg.ae_epochs = max(10, args.epochs // 2)
    if args.H is not None:
        cfg.H = args.H

    global N_SAMPLES_ALPHA
    N_SAMPLES_ALPHA = args.n_samples

    save_dir = Path(args.save_dir)
    t_total = time.time()

    if args.mode in ("alpha", "all"):
        run_experiment_alpha(cfg, save_dir=save_dir)

    beta_out = save_dir / "gaussianization"

    if os.access(save_dir, os.W_OK) or not save_dir.exists():
        beta_out.mkdir(parents=True, exist_ok=True)
    
    if args.mode in ("beta", "all"):
        run_experiment_beta(cfg, save_dir=beta_out,
                            bottleneck_factors=args.bf_list)

    if args.mode in ("gamma", "all"):
        run_experiment_gamma(cfg, save_dir=save_dir)

    if args.mode in ("delta", "all"):
        run_experiment_delta(cfg, save_dir=save_dir)
    print(f"\nAll experiments complete in {time.time()-t_total:.1f}s")
    print(f"Outputs written to: {save_dir}")


if __name__ == "__main__":
    main()