"""
RosenblattForward factory with cached empirical E[Sigma^2].

In the original codebase only the unified `train()` estimated E[Sigma^2]
from data.  Every other training entry point fell back to the analytic
hint attached to each sigma factory, which differs by 30-40 % from the
empirical value on FashionMNIST.  The result was that c_in normalisation
diverged across experiments and rendered cross-experiment cumulant
comparisons invalid.

This module makes empirical estimation the only blessed path.  Results
are cached by (sigma name, dataset name) so the dataloader cost is paid
once per (Sigma, dataset) pair.

The `RosenblattForward` class is not redefined here; the factory accepts
the existing class so the rest of the codebase keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from density_simulation import eigenvalues_LP
from rcd.data import _get_dataset, _NORM_TF
from rcd.diffusion.noise import sample_noise


# Cache survives the lifetime of the Python process.  Use clear_eg2_cache()
# in tests that need a fresh estimate per call.
_EG2_CACHE: dict[tuple[str, str], float] = {}


def build_eigenvalues(H: float, M: int, device: torch.device) -> torch.Tensor:
    """Unit-variance LP eigenvalues for Rosenblatt noise."""
    lam = eigenvalues_LP(a=1.0 - H, K=M)
    var = 2.0 * np.sum(lam ** 2)
    if var > 0:
        lam = lam / np.sqrt(var)
    return torch.tensor(lam, dtype=torch.float32, device=device)


def estimate_eg2(sigma_fn,
                 dataset_name: str,
                 get_dataset_fn: Callable,
                 transform,
                 n_samples:  int = 5000,
                 batch_size: int = 512) -> float:
    """Estimate E[ ||Sigma(x0)||_F^2 / numel(x0) ] over n_samples training
    examples.  Matches the original `_estimate_eg2` in
    Rosenblatt_cold_diffusion_unified.py."""
    ds     = get_dataset_fn(dataset_name, train=True, tf=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    total  = 0.0
    count  = 0
    for x0, labels in loader:
        if getattr(sigma_fn, "needs_label", False):
            S = sigma_fn(x0, labels)
        else:
            S = sigma_fn(x0)
        total += (S ** 2).mean().item() * x0.size(0)
        count += x0.size(0)
        if count >= n_samples:
            break
    return total / max(count, 1)


def get_eg2(sigma_fn,
            dataset_name: str,
            get_dataset_fn: Callable,
            transform,
            n_samples: int = 5000) -> float:
    """Cached version of estimate_eg2."""
    key = (getattr(sigma_fn, "__name__", "anon"), dataset_name)
    if key not in _EG2_CACHE:
        _EG2_CACHE[key] = estimate_eg2(
            sigma_fn, dataset_name, get_dataset_fn, transform,
            n_samples=n_samples)
    return _EG2_CACHE[key]


def clear_eg2_cache() -> None:
    _EG2_CACHE.clear()


def build_forward(forward_cls,
                  sigma_fn,
                  *,
                  noise_type:    str,
                  H:             float,
                  M_eig:         int,
                  sigma_max:     float,
                  device,
                  dataset_name:  str,
                  get_dataset_fn: Callable,
                  transform,
                  estimate: bool = True) -> Any:
    """Construct a RosenblattForward instance with empirical E[Sigma^2].

    `forward_cls` is the existing `RosenblattForward` class (passed in so
    this module stays decoupled from the rest of the codebase).

    Pass `estimate=False` ONLY when sigma_fn is sigma_additive (the analytic
    eg2 = 1 is exact in that case)."""
    fwd = forward_cls(sigma_fn,
                      noise_type=noise_type,
                      H=H, M_eig=M_eig,
                      sigma_max=sigma_max,
                      device=device)
    if estimate:
        eg2 = get_eg2(sigma_fn, dataset_name, get_dataset_fn, transform)
        fwd.set_eg2(eg2)
    return fwd


# x0 -> per-pixel scale (B,C,H,W)
SigmaFn = Callable[[torch.Tensor], torch.Tensor]
# Two calling conventions:
#   Standard:      fn(x0)        -> (B,C,H,W)   fn.needs_label = False  (default)
#   Class-aware:   fn(x0, y)     -> (B,C,H,W)   fn.needs_label = True
# All sigma factories set fn.needs_label appropriately.
# RosenblattForward.corrupt(x0, t, y=None) routes accordingly.


def sigma_additive() -> SigmaFn:
    """Sigma = I  (trivial baseline, included only for ablation if needed)."""
    def fn(x0: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x0)
    fn.__name__    = "additive"
    fn.label       = r"$\Sigma = I$"
    fn.eg2         = 1.0
    fn.needs_label = False
    return fn


def sigma_multiplicative() -> SigmaFn:
    """
    Sigma(x0) = diag(g(x0)),  g_i(x) = (1 + |x_i|) / 2.
    Motivated by the Doss–Sussmann 1D proof: density guaranteed by
    g >= 1/2 > 0 (uniform ellipticity automatically satisfied).
    E[g^2] = E[(1+|U|)^2/4] = 7/12 for U ~ Uniform[-1,1].
    """
    def fn(x0: torch.Tensor) -> torch.Tensor:
        return (1. + x0.abs()) / 2.
    fn.__name__    = "multiplicative"
    fn.label       = r"$\Sigma = \mathrm{diag}(g(\mathbf{x}_0))$"
    fn.eg2         = 7.0 / 12.0
    fn.needs_label = False
    return fn


def sigma_anisotropic(H_field: float = 0.7,
                      img_shape: tuple = (1, 28, 28),
                      mode: str = "h_emphasis") -> SigmaFn:
    """
    Sigma = A  fixed diagonal, precomputed per-pixel scale.
    Answer Q1: what happens qualitatively as noise becomes anisotropic?

    modes
    -----
    h_emphasis : scale varies 1→3 left→right (horizontal anisotropy)
    v_emphasis : scale varies 1→3 top→bottom (vertical anisotropy)
    """
    C, H_img, W = img_shape
    A = torch.ones(C, H_img, W)
    if mode == "h_emphasis":
        A[:, :, :] = torch.linspace(1., 3., W).view(1, 1, W)
    elif mode == "v_emphasis":
        A[:, :, :] = torch.linspace(1., 3., H_img).view(1, H_img, 1)
    else:
        raise ValueError(f"Unknown anisotropic mode: {mode!r}")
    A /= A.mean()
    _A = A.clone()

    def fn(x0: torch.Tensor) -> torch.Tensor:
        return _A.to(x0.device).expand_as(x0)
    fn.__name__    = f"anisotropic_{mode}"
    fn.label       = rf"$\Sigma = A_{{\rm {mode}}}$"
    fn.eg2         = float((_A ** 2).mean().item())
    fn.needs_label = False
    return fn

def sigma_pca_whitened_conditional(class_vars: dict[int, torch.Tensor]) -> SigmaFn:
    """
    Class-conditional PCA whitening:
        Sigma(x0, y) = C_y^{-1/2}  (diagonal approx from per-class pixel variance)
 
    This is more principled than global whitening because each class has a
    different spatial variance structure (trousers: vertical strips; boots:
    lower half; T-shirts: torso).  Global whitening averages these out and
    produces a less informative Sigma.
 
    During training: y is the true label, so Sigma is exact.
    During generation with CFG: y is the target class, so Sigma matches the
    class being generated.  This is feasible because CFG already conditions on y.
 
    class_vars: dict mapping class index (0..9) to per-pixel variance (C,H,W),
                computed by compute_pixel_variance().
    """
    # Pre-compute and cache per-class scale tensors
    _scales: dict[int, torch.Tensor] = {}
    for cls, var in class_vars.items():
        std   = var.clamp(min=1e-4).sqrt()   # (C,H,W)
        scale = 1. / std
        scale = scale / scale.mean()         # normalise so mean scale = 1
        _scales[cls] = scale
 
    # Global fallback (mean over classes) for null-label (CFG unconditional pass)
    _global = torch.stack(list(_scales.values())).mean(0)
    _global = _global / _global.mean()
 
    def fn(x0: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is None:
            # Global fallback — used when no label is available (e.g. recorrupt_stochastic)
            s = _global.to(x0.device)
            while s.dim() < x0.dim(): s = s.unsqueeze(0)
            return s.expand_as(x0)
        B = x0.size(0)
        out = torch.empty_like(x0)
        for b in range(B):
            cls = int(y[b].item())
            s   = _scales.get(cls, _global).to(x0.device)
            while s.dim() < x0.dim(): s = s.unsqueeze(0)
            out[b] = s.squeeze(0)
        return out
 
    # Estimate E[Sigma^2] as mean over classes
    eg2 = float(torch.stack([(s**2).mean() for s in _scales.values()]).mean())
 
    fn.__name__    = "pca_whitened_conditional"
    fn.label       = r"$\Sigma=\hat{C}_y^{-1/2}$ (class-cond.)"
    fn.eg2         = eg2
    fn.needs_label = True
    return fn

def sigma_pca_whitened_global(global_var: torch.Tensor) -> SigmaFn:
    """
    Global PCA whitening: Sigma = C_data^{-1/2} (diagonal, from full training set).
    needs_label = False — no CFG geometry mismatch, clean comparison.
    global_var: (C,H,W) from dataset-wide pixel variance.
    """
    std   = global_var.clamp(min=1e-4).sqrt()
    scale = 1. / std
    scale = scale / scale.mean()
    _s    = scale.clone()

    def fn(x0: torch.Tensor) -> torch.Tensor:
        s = _s.to(x0.device)
        while s.dim() < x0.dim(): s = s.unsqueeze(0)
        return s.expand_as(x0)

    fn.__name__    = "pca_whitened_global"
    fn.label       = r"$\Sigma = \hat{C}^{-1/2}$ (global)"
    fn.eg2         = float((scale**2).mean())
    fn.needs_label = False
    return fn


def sigma_pca_whitened(class_vars: dict[int, torch.Tensor]) -> SigmaFn:
    """Backward-compatible name for class-conditional PCA whitening."""
    return sigma_pca_whitened_conditional(class_vars)

def sigma_edge_aware(sobel_strength: float = 2.0) -> SigmaFn:
    """
    Sigma(x0) = diag(|Sobel(x0)| / mean + base),  base=0.5.
    Noise is proportional to local gradient magnitude: more noise at edges,
    less in flat regions.  Density guaranteed because base > 0.
    """
    _sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3)
    _sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3)
    _base = 0.5
    _scale = sobel_strength

    def fn(x0: torch.Tensor) -> torch.Tensor:
        dev = x0.device
        sx  = _sobel_x.to(dev)
        sy  = _sobel_y.to(dev)
        # per-channel gradient magnitude
        gx  = F.conv2d(x0, sx.expand(x0.shape[1], 1, 3, 3),
                       padding=1, groups=x0.shape[1])
        gy  = F.conv2d(x0, sy.expand(x0.shape[1], 1, 3, 3),
                       padding=1, groups=x0.shape[1])
        mag = (gx**2 + gy**2).sqrt()
        # normalise by mean magnitude per image, add base
        m   = mag.flatten(1).mean(
            1, keepdim=True).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-4)
        return _base + _scale * mag / m
    fn.__name__    = "edge_aware"
    fn.label       = r"$\Sigma = \mathrm{diag}(|\nabla \mathbf{x}_0|)$"
    fn.eg2         = None   # callers should use empirical estimate_eg2
    fn.needs_label = False
    return fn


def compute_condition_pixel_variance(dataset_name: str, n_per_class: int = 5000) -> dict[int, torch.Tensor]:
    """
    Per-class per-pixel variance.
    Returns dict[int -> Tensor(C,H,W)] for classes 0..9.
 
    Uses ALL available training samples for each class (up to n_per_class),
    so estimates are stable.  For FashionMNIST each class has ~6000 training
    images so n_per_class=500 is more than sufficient.
    """
    ds   = _get_dataset(dataset_name, train=True, tf=_NORM_TF)
    # Bucket images by class
    buckets: dict[int,list] = {}
    for x,y in DataLoader(ds,batch_size=512,shuffle=True,num_workers=2):
        for xi,yi in zip(x,y):
            c = int(yi.item())
            if c not in buckets: buckets[c] = []
            if len(buckets[c]) < n_per_class:
                buckets[c].append(xi)
        if all(len(v) >= n_per_class for v in buckets.values()) and len(buckets) == 10:
            break
    class_vars = {}
    for c, imgs in buckets.items():
        stack         = torch.stack(imgs)          # (N,C,H,W)
        class_vars[c] =stack.var(dim=0)  # (C,H,W)
    return class_vars


def compute_global_pixel_variance(dataset_name: str, n_samples: int = 50000) -> torch.Tensor:
    """Dataset-wide per-pixel variance, returned as (C,H,W)."""
    ds = _get_dataset(dataset_name, train=True, tf=_NORM_TF)
    xs: list[torch.Tensor] = []
    seen = 0
    for x, _ in DataLoader(ds, batch_size=512, shuffle=True, num_workers=2):
        xs.append(x)
        seen += x.size(0)
        if seen >= n_samples:
            break
    stack = torch.cat(xs, dim=0)[:n_samples]
    return stack.var(dim=0)


compute_pixel_variance = compute_condition_pixel_variance


def sigma_h_emphasis() -> SigmaFn:
    return sigma_anisotropic(mode="h_emphasis")


def sigma_v_emphasis() -> SigmaFn:
    return sigma_anisotropic(mode="v_emphasis")



# ─────────────────────────────────────────────────────────────────────────────
# 5. Forward process
# ─────────────────────────────────────────────────────────────────────────────

class RosenblattForward:
    """
    Unified forward process supporting additive and multiplicative modes.

    Unified forward process:
        X_t = x0 + sigma(t) * Sigma(x0) * eps

    sigma_fn: callable x0 -> per-pixel scale tensor, broadcastable to x0.shape.
    Use the factory functions above (sigma_multiplicative, sigma_anisotropic, etc.)
    to construct the right sigma_fn.

    Bridges
    -------
    stochastic    -- X_{k+1} = x0_hat + sigma(t_{k+1}) * Sigma(x0_hat) * eps_fresh
                     CORRECT for all noise types (exact marginal match)
    deterministic -- X_{k+1} = x0_hat + r * (X_k - x0_hat)
                     BROKEN for Rosenblatt (cumulant amplification as t->0)
    hybrid        -- interpolation via eta schedule (PART_VI Remark)
    """

    def __init__(
        self,
        sigma_fn:   SigmaFn,
        noise_type: str   = "rosenblatt",
        H:          float = 0.7,
        M_eig:      int   = 80,
        sigma_max:  float = 16.0,
        device:     torch.device | str = "cpu",
    ) -> None:
        self.sigma_fn   = sigma_fn
        self.noise_type = noise_type
        self.H          = float(H)
        self.M_eig      = M_eig
        self.sigma_max  = float(sigma_max)
        self.device     = device
        self.name       = getattr(sigma_fn, "__name__", "custom")
        self.label      = getattr(sigma_fn, "label",    self.name)
        eg2_hint        = getattr(sigma_fn, "eg2", 1.0)
        self._eg2       = 1.0 if eg2_hint is None else float(eg2_hint)
        self.lam_t      = (build_eigenvalues(H, M_eig, device)
                           if noise_type == "rosenblatt" else None)

    def set_eg2(self, eg2: float) -> None:
        """Override E[Sigma(x0)^2] estimated from training data."""
        self._eg2 = float(eg2)

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Scalar sigma(t) = sigma_max * t^H."""
        return self.sigma_max * t.clamp(min=1e-6).pow(self.H)

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        """Input normalisation: 1/sqrt(1 + sigma(t)^2 * E[Sigma^2])."""
        return (1. + self.sigma_t(t)**2 * self._eg2).pow(-0.5)

    def corrupt(self, x0: torch.Tensor,
                t: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_t, eps, sig_scalar) where x_t = x0 + sig * Sigma(x0) * eps.
        sig_scalar has shape (B, 1, 1, 1) for image inputs.
        """
        t   = t.to(x0.device, dtype=torch.float32)
        sig = self.sigma_t(t).view(-1, *([1]*(x0.dim()-1)))
        eps = sample_noise(self.noise_type, x0.shape, self.lam_t,
                           self.M_eig, x0.device)
        S   = self.sigma_fn(x0, y) if getattr(self.sigma_fn,"needs_label",False) else self.sigma_fn(x0)
        x_t = x0 + sig * S * eps
        return x_t, eps, sig    

    # ---- re-corruption (reverse step) ----------------------------------------

    def recorrupt_stochastic(self, x0_hat: torch.Tensor,
                              t_next: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        FIXED stochastic bridge re-corruption.

        Draws fresh eps ~ noise_type independently of everything else and
        applies the forward model at level t_next.  This guarantees that the
        distribution of X_{t_{k+1}} exactly matches the training distribution
        p_{t_{k+1}}, regardless of the estimation error in x0_hat.
        """
        x_next, _, _ = self.corrupt(x0_hat, t_next, y=y)
        return x_next

    def recorrupt_deterministic(self, x_cur: torch.Tensor, x0_hat: torch.Tensor,
                                t_cur: torch.Tensor,
                                t_next: torch.Tensor) -> torch.Tensor:
        """BROKEN for Rosenblatt (cumulant amplification). Kept for ablation."""
        sc = self.sigma_t(t_cur).view(-1, *([1]*(x_cur.dim()-1)))
        sn = self.sigma_t(t_next).view(-1, *([1]*(x_cur.dim()-1)))
        return x0_hat + (sn / sc.clamp(min=1e-5)) * (x_cur - x0_hat)

    def recorrupt_hybrid(self, x_cur: torch.Tensor, x0_hat: torch.Tensor,
                         t_cur: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Hybrid bridge (PART_VI Remark):
            eta = (s_{k+1}/s_k)^{H/(H+1)}
            X_{k+1} = x0_hat + s_{k+1} * Sigma(x0_hat) *
                      [sqrt(1-eta^2)*eps_hat + eta*eps_fresh]
        """
        s_cur   = self.sigma_t(t_cur).view(-1, *([1] * (x_cur.dim() - 1)))
        s_next  = self.sigma_t(t_next).view(-1, *([1] * (x_cur.dim() - 1)))
        r       = (s_next / s_cur.clamp(min=1e-5)).clamp(0., 1.)
        eta     = r.pow(self.H / (self.H + 1.))
        S_hat   = self.sigma_fn(x0_hat, y) if getattr(self.sigma_fn, "needs_label", False) else self.sigma_fn(x0_hat)
        eps_hat = (x_cur - x0_hat) / (s_cur * S_hat).clamp(min=1e-5)
        eps_new = sample_noise(self.noise_type, x0_hat.shape,
                               self.lam_t, self.M_eig, x0_hat.device)
        mix     = (1. - eta ** 2).clamp(min=0.).sqrt() * eps_hat + eta * eps_new
        return x0_hat + s_next * S_hat * mix
