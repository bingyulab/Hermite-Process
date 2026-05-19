"""
Experiment α — Cumulant Gaussianization Probe
Experiment β — Bottleneck Width vs. Gaussianization
=====================================================

Experiment α (mandatory):
    Extracts activations at three semantic stages:
        1. "input"       — raw test images x0  (pixel distribution)
        2. "corrupted"   — x_T = x0 + σ(1)·Σ(x0)·ε  (noise-type signal)
        3. "mid"         — UNet bottleneck output spatially averaged (B, C)
        4. "latent_ae"   — AE encoder output z0 ∈ ℝ^64
        5. "x0hat"       — final x0_hat reconstruction (UNet output)
    Computes per-component cumulants κ3, κ4 and Mardia's multivariate
    kurtosis statistic at each stage, for both Gaussian and Rosenblatt models.
    Outputs: console table + CSV + LaTeX fragment + violin-plot figure.

Experiment β (recommended):
    Trains ConditionalUNetFlexible with four bottleneck widths:
        {0.25, 0.5, 1.0, 2.0} × (4 × base_ch)
    Measures κ4 at the bottleneck after training.
    Hypothesis: κ4 ↓ as bottleneck narrows → bottleneck-driven Gaussianization.
    Alternative: κ4 independent of width → L² objective drives Gaussianization.
    Outputs: table + scatter plot κ4 vs bottleneck_factor.

Usage:
    # Experiment α only (uses existing checkpoints, no retraining):
    python experiment_gaussianization.py --mode alpha --save_dir ./output/diffusion

    # Experiment β only (trains new models):
    python experiment_gaussianization.py --mode beta  --save_dir ./output/diffusion

    # Both:
    python experiment_gaussianization.py --mode both  --save_dir ./output/diffusion
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
# 0. Output directory
# ─────────────────────────────────────────────────────────────────────────────

GAUSS_OUT = Path("./output/gaussianization")
GAUSS_OUT.mkdir(parents=True, exist_ok=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cumulant statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_marginal_cumulants(
        X: torch.Tensor,
        max_components: int = 512,
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
        "frac_non_gauss":   float((np.abs(k4) > 0.5).mean()),
        "N":                N,
        "D":                D,
    }


def mardia_statistics(
        X: torch.Tensor,
        d_proj: int = 32,
        seed:   int = 42,
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

    # Random orthonormal projection: D → p
    gen = torch.Generator()
    gen.manual_seed(seed)
    Q, _ = torch.linalg.qr(
        torch.randn(D, p, generator=gen, dtype=torch.float32))  # (D, p)
    Z = (X - X.mean(0)) @ Q.to(X.device)                       # (N, p)

    # Sample covariance
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
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stage-wise activation extraction
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
# 4. Results formatting
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
    frac_nong:      float
    mardia_b2p:     float
    mardia_b2p_exp: float
    mardia_b2p_z:   float


def _fmt(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:+.{decimals}f}" if v < 0 else f"{v:.{decimals}f}"


def print_cumulant_table(rows: list[StageResult]) -> None:
    """Pretty-print the cumulant table to stdout."""
    header = (f"{'Model':12s}  {'Stage':16s}  {'N':>5s}  {'D':>4s}  "
              f"{'|κ3|':>7s}  {'std(κ3)':>8s}  "
              f"{'κ4':>8s}  {'std(κ4)':>8s}  "
              f"{'%|κ4|>0.5':>9s}  {'b₂,p':>8s}  {'b₂,p*':>8s}  {'Z':>7s}")
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r.model:12s}  {r.stage:16s}  {r.N:5d}  {r.D:4d}  "
            f"{_fmt(r.mean_abs_k3):>7s}  {_fmt(r.std_k3):>8s}  "
            f"{_fmt(r.mean_k4):>8s}  {_fmt(r.std_k4):>8s}  "
            f"{r.frac_nong*100:8.1f}%  "
            f"{_fmt(r.mardia_b2p, 1):>8s}  {_fmt(r.mardia_b2p_exp, 1):>8s}  "
            f"{_fmt(r.mardia_b2p_z, 2):>7s}"
        )
    print(sep)
    print("b₂,p* = expected Mardia kurtosis under H₀ (multivariate normal)")


def save_cumulant_csv(rows: list[StageResult], path: Path) -> None:
    fields = [
        "model", "stage", "N", "D",
        "mean_abs_k3", "std_k3", "mean_k4", "std_k4",
        "frac_non_gauss", "mardia_b2p", "mardia_b2p_exp", "mardia_b2p_z",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "model":         r.model,
                "stage":         r.stage,
                "N":             r.N,
                "D":             r.D,
                "mean_abs_k3":   round(r.mean_abs_k3, 4),
                "std_k3":        round(r.std_k3, 4),
                "mean_k4":       round(r.mean_k4, 4),
                "std_k4":        round(r.std_k4, 4),
                "frac_non_gauss":round(r.frac_nong, 4),
                "mardia_b2p":    round(r.mardia_b2p, 3),
                "mardia_b2p_exp":round(r.mardia_b2p_exp, 3),
                "mardia_b2p_z":  round(r.mardia_b2p_z, 3),
            })
    print(f"  → CSV saved to {path}")


def save_latex_table(rows: list[StageResult], path: Path) -> None:
    """Write a LaTeX booktabs table fragment."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Marginal cumulants $\kappa_3$, $\kappa_4$ and Mardia kurtosis "
        r"at each stage of the Gaussian and Rosenblatt cold diffusion models. "
        r"Under multivariate normality, $\bar{\kappa}_4 \to 0$ and $b_{2,p} \to p(p+2)$.}",
        r"\label{tab:cumulants}",
        r"\small",
        r"\begin{tabular}{ll r r r r r r r}",
        r"\toprule",
        r"Model & Stage & $N$ & $D$ & $\overline{|\kappa_3|}$ & "
        r"$\overline{\kappa_4}$ & \% $|\kappa_4|{>}0.5$ & "
        r"$b_{2,p}$ & $b_{2,p}^*$ \\",
        r"\midrule",
    ]
    prev_model = None
    for r in rows:
        if prev_model is not None and r.model != prev_model:
            lines.append(r"\midrule")
        prev_model = r.model
        b2p_str  = f"{r.mardia_b2p:.1f}" if not math.isnan(r.mardia_b2p) else "—"
        b2p_star = f"{r.mardia_b2p_exp:.1f}" if not math.isnan(r.mardia_b2p_exp) else "—"
        lines.append(
            f"{r.model} & {r.stage} & {r.N} & {r.D} & "
            f"{r.mean_abs_k3:.3f} & {r.mean_k4:+.3f} & "
            f"{r.frac_nong*100:.1f}\\% & {b2p_str} & {b2p_star} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize $b_{2,p}^*$ = expected Mardia kurtosis under $\mathcal{N}_p$.}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))
    print(f"  → LaTeX saved to {path}")


def plot_kappa4_violins(
        all_kappa4: dict[str, dict[str, np.ndarray]],
        save_path:  Path,
) -> None:
    """
    Violin plot of per-component κ4 distributions across stages and models.

    all_kappa4[model_name][stage_name] = np.ndarray of per-component κ4 values
    """
    models = list(all_kappa4.keys())
    all_stages = sorted({s for m in all_kappa4.values() for s in m.keys()})
    n_stages = len(all_stages)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_stages,
                             figsize=(max(3 * n_stages, 12), 5),
                             sharey=False)
    if n_stages == 1:
        axes = [axes]

    colors = {"Gaussian": "#4C72B0", "Rosenblatt": "#DD8452"}
    positions = np.arange(n_models)

    for ax, stage in zip(axes, all_stages):
        data = [all_kappa4[m].get(stage, np.array([])) for m in models]
        data_clean = [d[np.isfinite(d)] for d in data]

        vp = ax.violinplot(
            [d for d in data_clean if len(d) > 0],
            positions=positions[:sum(len(d) > 0 for d in data_clean)],
            showmedians=True, showextrema=False,
        )
        for pc, m in zip(vp["bodies"], models):
            pc.set_facecolor(colors.get(m, "#888888"))
            pc.set_alpha(0.75)
        vp["cmedians"].set_color("k")

        ax.axhline(0.0, color="red", linewidth=1.0, linestyle="--",
                   label="Gaussian ($\\kappa_4=0$)")
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(stage, fontsize=9)
        ax.set_ylabel("per-component $\\kappa_4$" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Distribution of component-wise excess kurtosis $\\kappa_4$ across stages\n"
        "($\\kappa_4 \\to 0$: Gaussian; $\\kappa_4 > 0$: heavy-tailed Rosenblatt)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Violin plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_train_unet(
        noise_type: str,
        cfg: Config,
        save_dir: Path,
) -> tuple[ConditionalUNet, RosenblattForward]:
    """Load a trained ConditionalUNet from checkpoint or train from scratch."""
    sfn   = sigma_multiplicative()
    tag   = f"{noise_type}_{sfn.__name__}_H{cfg.H}"
    ckpt  = save_dir / f"{tag}_final.pt"

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
# 6. Experiment α — main function
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
    mard = mardia_statistics(acts)
    result = StageResult(
        model          = model_name,
        stage          = stage_label,
        N              = cum["N"],
        D              = cum["D"],
        mean_abs_k3    = cum["mean_abs_kappa3"],
        std_k3         = cum["std_kappa3"],
        mean_k4        = cum["mean_kappa4"],
        std_k4         = cum["std_kappa4"],
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
# 7. Experiment β — variable bottleneck UNet
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalUNetFlexible(nn.Module):
    """
    Identical to ConditionalUNet but with a variable bottleneck channel count.

    bottleneck_factor : float
        Scales the bottleneck channels relative to the standard 4 × base_ch.
            0.25 → base_ch      (no compression: bottleneck = encoder entry width)
            0.5  → 2 × base_ch  (mild compression)
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
        x  = self.init_conv(x)
        h1 = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2 = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2 = self.attn2(h2)

        # Bottleneck
        h3 = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)

        # Decode
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

    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    eg2 = float(getattr(sfn, "eg2", 1.0))
    forward.set_eg2(eg2)

    for ep in range(cfg.epochs):
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

    ema.apply_shadow()
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved → {ckpt}")
    model.eval()
    return model, forward


@dataclass
class BetaResult:
    noise_type:        str
    bottleneck_factor: float
    bneck_ch:          int
    mean_k4_bneck:     float
    std_k4_bneck:      float
    mean_k4_input:     float
    mean_k4_x0hat:     float
    mardia_b2p_z:      float


def run_experiment_beta(
        cfg:               Config,
        save_dir:          Path,
        bottleneck_factors: list[float] | None = None,
) -> list[BetaResult]:
    """
    Full Experiment β pipeline.

    Trains ConditionalUNetFlexible for each (noise_type, bottleneck_factor)
    combination, extracts bottleneck activations, computes κ4.

    Default bottleneck_factors: [0.25, 0.5, 1.0, 2.0]
    """
    if bottleneck_factors is None:
        bottleneck_factors = [0.25, 0.5, 1.0, 2.0]

    print("\n" + "═" * 70)
    print("Experiment β — Bottleneck Width vs. Gaussianization")
    print("═" * 70)
    print(f"  Bottleneck factors: {bottleneck_factors}")
    print(f"  Noise types: gaussian, rosenblatt")

    cfg.save_dir = save_dir
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)

    beta_rows: list[BetaResult] = []

    for noise_type in ("gaussian", "rosenblatt"):
        for bf in bottleneck_factors:
            print(f"\n── noise={noise_type}  bf={bf} ──────────────────────")

            model, forward = train_flexible_unet(bf, noise_type, cfg, save_dir)

            # Extract: input, bottleneck, x0hat
            bn_store = ActivationStore(spatial_pool=True)
            bn_handle = model.mid2.register_forward_hook(bn_store.hook_fn)

            raw_list:  list[torch.Tensor] = []
            x0h_list:  list[torch.Tensor] = []
            loader = DataLoader(test_ds,
                                batch_size=min(cfg.batch_size, 128),
                                shuffle=False, num_workers=2)
            n_col = 0
            with torch.no_grad():
                for x0, y in loader:
                    if n_col >= N_SAMPLES_ALPHA:
                        break
                    B = x0.size(0)
                    x0 = x0.to(cfg.device);  y = y.to(cfg.device)
                    raw_list.append(x0.view(B, -1).cpu())

                    t_one = torch.ones(B, device=cfg.device)
                    x_T, _, _ = forward.corrupt(x0, t_one, y=y)
                    t_min = torch.full((B,), cfg.T_MIN, device=cfg.device)
                    null  = torch.full_like(y, 10)
                    c_in  = forward.c_in(t_min).view(-1, 1, 1, 1)
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

            cum_bn   = compute_marginal_cumulants(bneck_acts)
            cum_raw  = compute_marginal_cumulants(raw_acts)
            cum_x0h  = compute_marginal_cumulants(x0h_acts)
            mard_bn  = mardia_statistics(bneck_acts)

            row = BetaResult(
                noise_type        = noise_type,
                bottleneck_factor = bf,
                bneck_ch          = model.bneck_ch,
                mean_k4_bneck     = cum_bn["mean_kappa4"],
                std_k4_bneck      = cum_bn["std_kappa4"],
                mean_k4_input     = cum_raw["mean_kappa4"],
                mean_k4_x0hat     = cum_x0h["mean_kappa4"],
                mardia_b2p_z      = mard_bn["b2p_z"],
            )
            beta_rows.append(row)
            print(f"  bneck_ch={model.bneck_ch:4d}  "
                  f"κ4_input={row.mean_k4_input:+.3f}  "
                  f"κ4_bneck={row.mean_k4_bneck:+.3f}  "
                  f"κ4_x0hat={row.mean_k4_x0hat:+.3f}  "
                  f"Mardia-Z={row.mardia_b2p_z:+.2f}")

    _print_beta_table(beta_rows)
    _save_beta_csv(beta_rows, save_dir / "beta_bottleneck.csv")
    _save_beta_latex(beta_rows, save_dir / "beta_bottleneck.tex")
    _plot_beta(beta_rows, save_dir / "beta_kappa4_vs_bottleneck.png")

    return beta_rows


def _print_beta_table(rows: list[BetaResult]) -> None:
    print("\n── Experiment β Summary ────────────────────────────────────────────")
    header = (f"{'Noise':11s}  {'bf':>5s}  {'bneck_ch':>9s}  "
              f"{'κ4 input':>9s}  {'κ4 bneck':>9s}  {'κ4 x0hat':>9s}  "
              f"{'Mardia-Z':>9s}")
    print(header)
    print("─" * len(header))
    prev_n = None
    for r in rows:
        if prev_n is not None and r.noise_type != prev_n:
            print()
        prev_n = r.noise_type
        print(f"{r.noise_type:11s}  {r.bottleneck_factor:5.2f}  "
              f"{r.bneck_ch:9d}  "
              f"{r.mean_k4_input:+9.3f}  {r.mean_k4_bneck:+9.3f}  "
              f"{r.mean_k4_x0hat:+9.3f}  {r.mardia_b2p_z:+9.2f}")
    print("─" * len(header))
    print(textwrap.dedent("""
    Interpretation:
      If κ4_bneck ↓ monotonically as bf ↓  → bottleneck-driven Gaussianization.
      If κ4_bneck ≈ const across bf        → L²-objective-driven Gaussianization.
    """))


def _save_beta_csv(rows: list[BetaResult], path: Path) -> None:
    fields = ["noise_type", "bottleneck_factor", "bneck_ch",
              "mean_k4_input", "mean_k4_bneck", "std_k4_bneck",
              "mean_k4_x0hat", "mardia_b2p_z"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "noise_type":        r.noise_type,
                "bottleneck_factor": r.bottleneck_factor,
                "bneck_ch":          r.bneck_ch,
                "mean_k4_input":     round(r.mean_k4_input,  4),
                "mean_k4_bneck":     round(r.mean_k4_bneck,  4),
                "std_k4_bneck":      round(r.std_k4_bneck,   4),
                "mean_k4_x0hat":     round(r.mean_k4_x0hat,  4),
                "mardia_b2p_z":      round(r.mardia_b2p_z,   3),
            })
    print(f"  → CSV saved to {path}")


def _save_beta_latex(rows: list[BetaResult], path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Experiment~$\beta$: mean excess kurtosis $\kappa_4$ at the",
        r"bottleneck for varying bottleneck width and noise type.",
        r"If $\kappa_4^{\mathrm{bneck}}$ decreases monotonically with",
        r"$\alpha_{\mathrm{bf}}$, Gaussianization is bottleneck-driven.",
        r"If $\kappa_4^{\mathrm{bneck}} \approx \kappa_4^{\mathrm{input}}$",
        r"independent of width, Gaussianization is $L^2$-objective-driven.}",
        r"\label{tab:beta}",
        r"\small",
        r"\begin{tabular}{ll r r r r r}",
        r"\toprule",
        r"Noise & $\alpha_{\rm bf}$ & $C_{\rm bneck}$ & "
        r"$\kappa_4^{\rm input}$ & $\kappa_4^{\rm bneck}$ & "
        r"$\kappa_4^{\hat{x}_0}$ & Mardia-$Z$ \\",
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
            f"{r.mean_k4_x0hat:+.3f} & {r.mardia_b2p_z:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    print(f"  → LaTeX saved to {path}")


def _plot_beta(rows: list[BetaResult], save_path: Path) -> None:
    """
    Two-panel figure:
    Left:  κ4_bneck  vs bottleneck_factor, coloured by noise_type.
    Right: κ4_bneck / κ4_input  (normalised, shows relative Gaussianization).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"gaussian": "#4C72B0", "rosenblatt": "#DD8452"}
    markers = {"gaussian": "o", "rosenblatt": "s"}

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if r.noise_type == noise_type]
        bfs   = [r.bottleneck_factor for r in sub]
        k4_bn = [r.mean_k4_bneck     for r in sub]
        k4_in = [r.mean_k4_input      for r in sub]
        k4_norm = [
            (bn / inp) if abs(inp) > 1e-6 else float("nan")
            for bn, inp in zip(k4_bn, k4_in)
        ]

        label = noise_type.capitalize()
        c = colors[noise_type];  m = markers[noise_type]
        axes[0].plot(bfs, k4_bn,   color=c, marker=m, label=label, linewidth=2)
        axes[1].plot(bfs, k4_norm, color=c, marker=m, label=label, linewidth=2)

    # Reference lines
    axes[0].axhline(0.0, color="red", linewidth=1.0, linestyle="--",
                    label="$\\kappa_4=0$ (Gaussian)")
    axes[1].axhline(1.0, color="gray", linewidth=1.0, linestyle=":",
                    label="no change")
    axes[1].axhline(0.0, color="red",  linewidth=1.0, linestyle="--",
                    label="full Gaussianization")

    for ax in axes:
        ax.set_xlabel("Bottleneck factor $\\alpha_{\\rm bf}$", fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("$\\overline{\\kappa}_4$ at bottleneck", fontsize=11)
    axes[0].set_title("Absolute bottleneck $\\kappa_4$ vs. width", fontsize=10)

    axes[1].set_ylabel("$\\kappa_4^{\\rm bneck} \\,/\\, \\kappa_4^{\\rm input}$",
                       fontsize=11)
    axes[1].set_title("Relative Gaussianization  (0 = fully Gaussian)", fontsize=10)

    fig.suptitle(
        "Experiment $\\beta$: Does bottleneck width control Gaussianization?\n"
        "If curves are flat → $L^2$ objective drives Gaussianization, not bottleneck.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_cfg_for_experiment() -> Config:
    """Build a Config suitable for the experiment (fast, evaluation-focused)."""
    cfg = Config()
    cfg.epochs     = 30
    cfg.ae_epochs  = 20
    cfg.batch_size = 128
    cfg.n_steps    = 50
    cfg.n_fid      = 2000
    cfg.n_ssim     = 200
    cfg.no_evaluate = True    # skip full FID during training for speed
    cfg.no_plot     = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments α and β: Gaussianization cumulant probes")
    parser.add_argument("--mode",      default="both",
                        choices=["alpha", "beta", "both"],
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
                        default=[0.25, 0.5, 1.0, 2.0],
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
    out_dir  = save_dir / "gaussianization"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    if args.mode in ("alpha", "both"):
        run_experiment_alpha(cfg, save_dir=save_dir)

    if args.mode in ("beta", "both"):
        run_experiment_beta(cfg, save_dir=out_dir,
                            bottleneck_factors=args.bf_list)

    print(f"\nAll experiments complete in {time.time()-t_total:.1f}s")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()