"""
Noise samplers.

Naming convention (deliberately strict to avoid the H10 confusion in the
existing Experiment_Optimizer.py):

    gaussian            standard normal
    rosenblatt          TRUE Rosenblatt marginal via LP eigenvalue expansion
                        Z = sum_n lambda_n (xi_n^2 - 1), xi_n iid N(0,1)
    rosenblatt_product  Product of two iid N(0,1).  kappa4 = 6, unit variance,
                        FINITE higher moments. A surrogate; not Rosenblatt.
    laplace             Laplace, variance-matched
    uniform             Uniform on [-sqrt(3) std, sqrt(3) std]

`sample_grad_noise` accepts `dist='rosenblatt'` as a synonym for
`rosenblatt_product` (with a one-time warning) because the existing
optimiser experiments use that string. New code should use the explicit
`rosenblatt_product` name.
"""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Marginal samplers
# ─────────────────────────────────────────────────────────────────────────────

def sample_gaussian(shape: Tuple[int, ...], device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.float32)


def sample_rosenblatt(shape: Tuple[int, ...],
                      lam_t: torch.Tensor | None,
                      M:     int,
                      device) -> torch.Tensor:
    """True Rosenblatt. `lam_t` must already be normalised so 2*sum(lam^2)=1
    (this is what `build_eigenvalues(H, M)` in the existing codebase returns)."""
    if lam_t is None:
        raise ValueError("sample_rosenblatt requires precomputed lam_t.")
    # Defensive fix for potential device mismatch
    lam_t = lam_t.to(device=device, dtype=torch.float32)
    total = int(np.prod(shape))
    xi    = torch.randn(total, M, device=device, dtype=torch.float32)
    z     = (xi ** 2 - 1.0) @ lam_t
    return z.reshape(shape)


def sample_rosenblatt_product(shape: Tuple[int, ...],
                               device,
                               std: float = 1.0) -> torch.Tensor:
    """Product of two iid N(0,1). Mean 0, variance 1, skewness 0, excess
    kurtosis 6.  This is NOT the Rosenblatt distribution — its tails decay
    like exp(-|x|), not like a power law — but it shares the lowest-order
    cumulants and is cheap to sample at every Adam step."""
    g1 = torch.randn(shape, device=device, dtype=torch.float32)
    g2 = torch.randn(shape, device=device, dtype=torch.float32)
    return (g1 * g2) * std


def sample_rosenblatt_proxy(shape: Tuple[int, ...],
                            device,
                            std: float = 1.0) -> torch.Tensor:
    """
    Best cheap proxy for Rosenblatt. This is mathematically a 1-term 
    Wiener Chaos expansion: (xi^2 - 1) / sqrt(2).
    Mean 0, Variance 1, Skewness ~2.82 (Highly asymmetric).
    """
    xi = torch.randn(shape, device=device, dtype=torch.float32)
    # xi^2 - 1 has variance 2. Divide by sqrt(2) to make variance 1.
    z_unit_var = (xi ** 2 - 1.0) / 1.4142135623730951
    
    return z_unit_var * std


def sample_laplace(shape, device, std: float = 1.0) -> torch.Tensor:
    """Laplace(0, std/sqrt(2)) has variance std^2.
    Implemented robustly via difference of two independent Exponentials."""
    b = std / math.sqrt(2.0)
    # 1.0 - torch.rand ensures values are in (0, 1] avoiding log(0) -> -inf
    e1 = -torch.log(1.0 - torch.rand(shape, device=device, dtype=torch.float32))
    e2 = -torch.log(1.0 - torch.rand(shape, device=device, dtype=torch.float32))
    return b * (e1 - e2)


def sample_uniform(shape: Tuple[int, ...], device, std: float = 1.0) -> torch.Tensor:
    a = std * math.sqrt(3.0)
    return (torch.rand(shape, device=device, dtype=torch.float32) * 2.0 - 1.0) * a

# ─────────────────────────────────────────────────────────────────────────────
# Data-corruption dispatch
# ─────────────────────────────────────────────────────────────────────────────

def sample_noise(noise_type: str,
                 shape: Tuple[int, ...],
                 lam_t: torch.Tensor | None = None,
                 M:     int = 80,
                 device = "cpu") -> torch.Tensor:
    """Used by RosenblattForward.corrupt for data corruption.  E[eps]=0,
    Var[eps]=1.  Only 'gaussian' and 'rosenblatt' are accepted — the proxy
    is reserved for gradient noise (see sample_grad_noise)."""
    if noise_type == "gaussian":
        return sample_gaussian(shape, device)
    if noise_type == "rosenblatt":
        return sample_rosenblatt(shape, lam_t, M, device)
    if noise_type == "rosenblatt_product":
        raise ValueError(
            "rosenblatt_product is a gradient-noise proxy, not a data "
            "corruption distribution. Use 'rosenblatt' (LP expansion) for "
            "data corruption.")
    raise ValueError(f"Unknown noise_type: {noise_type!r}.")


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-noise dispatch (Experiments pi, rho, tau)
# ─────────────────────────────────────────────────────────────────────────────

_WARNED_LEGACY_ROSENBLATT = False


def sample_grad_noise(shape: Tuple[int, ...], dist: str, std: float, device) -> torch.Tensor:    
    """Gradient noise sampler for optimiser experiments.
    `dist='rosenblatt'` is accepted as a legacy alias for
    `'rosenblatt_product'` (since the existing codebase and `experiment.md`
    use that name).  A single warning is emitted at first use.  New code
    should use `'rosenblatt_product'` explicitly so that papers do not
    conflate the proxy with the actual Rosenblatt distribution that drives
    data corruption."""
    global _WARNED_LEGACY_ROSENBLATT

    if dist == "none":
        return torch.zeros(shape, device=device)
    if dist == "gaussian":
        return sample_gaussian(shape, device) * std
    if dist == "laplace":
        return sample_laplace(shape, device, std)
    if dist == "uniform":
        return sample_uniform(shape, device, std)
    if dist == "rosenblatt_product":
        # return sample_rosenblatt_product(shape, device, std)
        return sample_rosenblatt_proxy(shape, device, std)
    if dist == "rosenblatt":
        if not _WARNED_LEGACY_ROSENBLATT:
            warnings.warn(
                "sample_grad_noise(dist='rosenblatt') resolves to the "
                "product-of-Gaussians proxy, NOT the true Rosenblatt "
                "distribution. Rename to 'rosenblatt_product' in new code.",
                stacklevel=2)
            _WARNED_LEGACY_ROSENBLATT = True
        return sample_rosenblatt(shape, device, std)
    raise ValueError(f"Unknown gradient noise dist: {dist!r}")