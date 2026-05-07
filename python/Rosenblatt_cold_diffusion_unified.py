"""
Rosenblatt Cold Diffusion — Unified Framework v3
=================================================

Forward process:  X_t = x0 + sigma(t) * Sigma(x0) * eps,  eps ~ noise_type

Sigma(x0) is a per-pixel diagonal (or scalar) noise coefficient.
All cases — additive, multiplicative, anisotropic, PCA-whitened,
edge-aware — are instances of RosenblattForward with different Sigma_fn.

Theoretical backbone
--------------------
  1D multiplicative : Doss–Sussmann (Section 3, PART_V)
  Multi-D multiplicative : Loosveldt, Nourdin & Peccati (2026)
                           [doi:10.1016/j.spa.2025.xxx]

Contributions covered
---------------------
  1. multiplicative noise:  Sigma(x0) = diag(g(x0))
  2. Anisotropic noise (Prof Q1):          Sigma = A  fixed diagonal
  3. PCA-whitened noise (Prof Q2):         Sigma = C_data^{-1/2}  diagonal approx
  4. Edge-aware noise (extra):             Sigma(x0) = diag(|Sobel(x0)|+eps)
  5. Latent experiment:  train lightweight autoencoder on FashionMNIST;
    run cold diffusion in the 64-D latent space with an MLP denoiser;
    decode generated latents with the fixed decoder.

Noise types: "gaussian" | "rosenblatt"   (Hermite-3 removed)

Bridges:
  stochastic    -- FIXED: fresh eps at every step (correct for non-Gaussian)
  deterministic -- BROKEN: kept for ablation only
  hybrid        -- interpolation (PART_VI Remark)

Usage
-----
  python rosenblatt_cold_diffusion_unified.py --mode all
  python rosenblatt_cold_diffusion_unified.py --mode noise_plot
  python rosenblatt_cold_diffusion_unified.py --mode path_plot
  python rosenblatt_cold_diffusion_unified.py --mode sigma_comparison
  python rosenblatt_cold_diffusion_unified.py --mode exp_latent
  python rosenblatt_cold_diffusion_unified.py --mode bridge_ablation
"""

from __future__ import annotations
from density_simulation import RosenblattDensityVT, RosenblattDensityLP, eigenvalues_LP
from path_simulation import WaveletRosenblatt
import matplotlib.pyplot as plt
import argparse
import math
import time
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, fields, asdict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib
matplotlib.use("Agg")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# 0. Global configuration
# ─────────────────────────────────────────────────────────────────────────────

OUT_ROOT = Path("./output/diffusion")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    mode:         str   = "all" 
    cfg_scale:    float = 2.5               # Classifier-Free Guidance scale
    n_steps:      int   = 50
    n_display:    int   = 8                 # Number of intermediate steps to save for display (excluding final step)
    sigma_max:    float = 16.0              # Maximum noise level at t=1.0
    base_ch:      int   = 128               # UNet base channels (DO NOT set to batch_size)
    M_eig:        int   = 80                # Number of eigenvalues for LP expansion
    lr:           float = 2e-4
    epochs:       int   = 30
    ae_epochs:    int   = 20
    ae_lr:        float = 1e-3
    batch_size:   int   = 256
    dataset:      str   = "FashionMNIST"
    noise_type:   str   = "rosenblatt"
    n_fid:        int   = 10000             # Number of samples for FID evaluation
    H:            float = 0.7               # Hurst index
    T_MIN:        float = 0.01
    bridge:       str   = "stochastic"      # "stochastic" | "Hybrid" | "deterministic"
    device:       torch.device = get_device()
    save_dir:     Path  = OUT_ROOT
    n_ssim:       int   = 200               # Number of samples for SSIM evaluation
    k_components: int   = 64                # Number of PCA components for exp_pca_basis
    EVALUATE:     bool  = True
    PLOT:         bool  = True
    baseline:     str   = "multiplicative"  # "multiplicative" | "anisotropic_h_emphasis" | "anisotropic_v_emphasis" | "pca_whitened_conditional" | "pca_whitened_global" | "edge_aware"

def resolve_argparse_type(t):
    # Handle both actual types and string representations
    if t == int or t == 'int':
        return int
    elif t == float or t == 'float':
        return float
    elif t == str or t == 'str':
        return str
    elif t == bool or t == 'bool':
        return None  # handled separately
    elif t == Path or t == 'Path':
        return Path
    else:
        return str  # fallback

def update_config_from_args(cfg: Config, args) -> Config:
    args_dict = vars(args)
    field_types = {f.name: f.type for f in fields(Config)}

    for key, value in args_dict.items():
        if value is not None:
            target_type = field_types.get(key, type(value))

            try:
                # cast to correct type, checking for both class and string
                if target_type == int or target_type == 'int':
                    value = int(value)
                elif target_type == float or target_type == 'float':
                    value = float(value)
                elif target_type == bool or target_type == 'bool':
                    value = str(value).lower() in ['true', '1', 'yes', 'y', 't']
                elif target_type == Path or target_type == 'Path':
                    value = Path(value)
                elif 'torch.device' in str(target_type):
                    value = torch.device(value)
            except Exception as e:
                raise ValueError(f"Failed to cast {key}={value} to {target_type}") from e

            setattr(cfg, key, value)

    return cfg
    
def build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    existing_args = [action.dest for action in parser._actions]

    for f in fields(Config):
        # Don't add if we already manually added it below!
        if f.name in existing_args:
            continue
            
        arg_type = resolve_argparse_type(f.type)
        if f.type == bool or f.type == 'bool':
            parser.add_argument(f"--{f.name}", action="store_true")
        else:
            parser.add_argument(f"--{f.name}", type=arg_type, default=None)

    return parser

# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model  = model
        self.decay  = decay
        self.step   = 0
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict = {}

    def _effective_decay(self) -> float:
        return min(self.decay, (1 + self.step) / (10 + self.step))

    def update(self) -> None:
        d = self._effective_decay()
        self.step += 1
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].lerp_(p.data, 1.0 - d)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

_FASHION = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_NORM_TF = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def _get_dataset(name: str, train: bool = True,
                 tf=None, root: str = "./data"):
    if "fashion" in name.lower():
        return datasets.FashionMNIST(root, train=train, download=True,
                                     transform=tf)
    return datasets.MNIST(root, train=train, download=True, transform=tf)


def class_name(ds_name: str, idx: int) -> str:
    return _FASHION[idx] if "fashion" in ds_name.lower() else f"Digit {idx}"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Noise sampler — Gaussian and Rosenblatt only
# ─────────────────────────────────────────────────────────────────────────────


def build_eigenvalues(H: float, M: int, device: torch.device) -> torch.Tensor:
    """
    Unit-variance LP eigenvalues as a GPU tensor.
    Normalises so that 2*sum(lam^2) == 1 exactly for any finite M.
    """
    D = 1.0 - H
    # Use eigenvalues_LP from density_simulation.py (a = D)
    lam = eigenvalues_LP(a=D, K=M)

    # Normalise so that Var[Z_D] = 2*sum(lam^2) = 1
    var = 2.0 * np.sum(lam ** 2)
    if var > 0:
        lam = lam / np.sqrt(var)
    return torch.tensor(lam, dtype=torch.float32, device=device)


def get_rosenblatt_density(H: float = 0.7, K: int = 200,
                           x_min: float = -5., x_max: float = 8.):
    return RosenblattDensityLP(a=1-H, K=K).density_fft(
        x_min=x_min, x_max=x_max, N=2**16, z_max=40.)


def get_vt_density(H: float = 0.7, x_min: float = -5., x_max: float = 8.):
    return RosenblattDensityVT(D=1-H).density_fft_direct(
        x_min=x_min, x_max=x_max, N_fft=2**16, z_max=40.)


def sample_noise(noise_type: str, shape: tuple,
                 lam_t: torch.Tensor | None = None,
                 M: int = 80,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Draw noise with E[eps]=0, Var[eps]=1.

    Parameters
    ----------
    noise_type : "gaussian" | "rosenblatt"
    shape      : output shape
    lam_t      : pre-computed unit-variance LP eigenvalues (Rosenblatt only)
    """
    if noise_type == "gaussian":
        return torch.randn(shape, device=device, dtype=torch.float32)

    if noise_type == "rosenblatt":
        # Z_D = sum_n lambda_n * (xi_n^2 - 1),  xi_n iid N(0,1)
        # Already unit-variance thanks to normalised lam_t.
        if lam_t is None:
            raise ValueError("lam_t must be provided for rosenblatt noise.")
        
        total = int(np.prod(shape))
        xi    = torch.randn(total, M, device=device, dtype=torch.float32)
        z     = (xi ** 2 - 1.0) @ lam_t
        return z.reshape(shape)

    raise ValueError(f"Unknown noise_type: {noise_type!r}. "
                     "Hermite-3 has been removed. Use 'gaussian' or 'rosenblatt'.")


def simulate_rosenblatt_paths(H: float = 0.7, n_paths: int = 5,
                               n_pts: int = 200, T: float = 1.):
    """
    Simulate Rosenblatt process sample paths.
    
    Returns (times: ndarray, paths: ndarray shape (n_paths, n_pts)).
    """
    wav = WaveletRosenblatt(H=H, J=10, L=0, N_vanishing=2)
    return wav.simulate_paths_batch(T=T, n_points=n_pts, n_paths=n_paths)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sigma factories 
# ─────────────────────────────────────────────────────────────────────────────

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
    fn.eg2         = (_base + _scale)**2   # rough upper bound
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


def compute_global_pixel_variance(dataset_name: str, n: int = 10000) -> torch.Tensor:
    """Global per-pixel variance (C,H,W) over the full training set."""
    ds = _get_dataset(dataset_name, train=True, tf=_NORM_TF)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2)
    imgs = []
    for x, _ in dl:
        imgs.append(x)
        if sum(i.size(0) for i in imgs) >= n: break
    return torch.cat(imgs, 0)[:n].var(dim=0)  # (C,H,W)

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
        self._eg2       = float(getattr(sigma_fn, "eg2", 1.0))
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

# ─────────────────────────────────────────────────────────────────────────────
# 6. UNet denoiser (image space)
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding (same as in DDPM)."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half,
                          dtype=torch.float32, device=t.device) / (half - 1))
        args  = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, spatial_size: int = 14) -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(
            torch.randn(1, spatial_size ** 2, channels) * 0.02)
        self.norm1   = nn.GroupNorm(min(8, channels), channels)
        self.mha     = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True)
        self.proj    = nn.Conv2d(channels, channels, 1)
        self.norm2   = nn.GroupNorm(min(8, channels), channels)
        self.ffn     = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        h = self.norm1(x).view(B, C, -1).transpose(1, 2)
        h = h + self.pos_emb
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        # Reshape back to (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        x = x + self.proj(attn_out)
        return x + self.ffn(self.norm2(x))


class ResBlockAdaGN(nn.Module):
    """Residual block with Adaptive Group Normalization (AdaGN)."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int = 256) -> None:
        super().__init__()
        self.norm1    = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(min(8, out_ch), out_ch, affine=False)
        self.dropout  = nn.Dropout(0.1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj   = nn.Linear(t_dim, out_ch * 2)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())
        
    def forward(self, x, t_emb):
        # 1. First convolution
        h            = self.conv1(F.silu(self.norm1(x)))

        # 2. AdaGN Conditioning
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h            = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        # 3. Second convolution
        h            = self.dropout(F.silu(h))
        return self.shortcut(x) + self.conv2(h)


class ConditionalUNet(nn.Module):
    """Conditional U-Net — ~4.5 M parameters, 28×28 grayscale input."""

    def __init__(self, t_dim: int = 256, num_classes: int = 10,
                 base_ch: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        # Encoder: 128 -> 256 -> 512
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1     = nn.Sequential(ResBlockAdaGN(base_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim))
        self.pool1     = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.down2     = nn.Sequential(ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim))
        self.attn2     = SelfAttention(base_ch * 2, spatial_size=14)
        self.pool2     = nn.Conv2d(base_ch * 2, base_ch * 4,
                               3, stride=2, padding=1)

        # Bottleneck
        self.mid1      = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)
        self.attn_mid  = SelfAttention(base_ch * 4, spatial_size=7)
        self.mid2      = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)

        # Decoder
        self.up2       = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
                        )
        self.up_res2   = nn.ModuleList([ResBlockAdaGN(base_ch * 4, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim)])
        self.up_attn2  = SelfAttention(base_ch * 2, spatial_size=14)

        self.up1       = nn.Sequential(
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
                            )
        self.up_res1   = nn.ModuleList([ResBlockAdaGN(base_ch * 2, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim)])

        self.out       = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Map time through MLP *before* adding labels
        t_emb = self.time_mlp(self.t_embed(t)) + self.label_emb(y)

        # Encode
        x     = self.init_conv(x)
        h1    = self.down1[1](self.down1[0](x, t_emb), t_emb)

        h2    = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2    = self.attn2(h2)

        # Bottleneck
        h3    = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)

        # Decode
        h     = self.up_attn2(self.up_res2[1](self.up_res2[0](torch.cat([self.up2(h3), h2], dim=1), t_emb), t_emb))
        h     = self.up_res1[1](self.up_res1[0](torch.cat([self.up1(h), h1], dim=1), t_emb), t_emb)

        return self.out(h)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Latent-space components
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Lightweight convolutional autoencoder:  28×28 -> 64-D latent -> 28×28.
    Encoder:  28->14->7, channels 1->16->32->64; then flatten+project to 64.
    Decoder:  inverse.
    """
    LATENT_DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1),  nn.SiLU(),   # 14×14
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.SiLU(),   # 7×7
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  nn.SiLU(),   # 7×7
            nn.Flatten(),                                            
            nn.Linear(64 * 7 * 7, self.LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.LATENT_DIM, 64 * 7 * 7), nn.SiLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2,
                               padding=1),  nn.SiLU(),   # 14×14
            nn.ConvTranspose2d(16,  1, 4, stride=2,
                               padding=1),              # 28×28
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class LatentMLPDenoiser(nn.Module):
    """
    6-layer MLP denoiser for the 64-D latent space.
    Architecture: sinusoidal time embedding + AdaGN conditioning.
    No convolutions needed for 64-D inputs.
    """

    def __init__(self, latent_dim: int = 64, t_dim: int = 256,
                 num_classes: int = 10, hidden: int = 256) -> None:
        super().__init__()
        self.t_embed    = SinusoidalTimeEmbed(t_dim)
        self.label_emb  = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp   = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim),)
        # Build 6 hidden layers with residual connections
        self.input_proj = nn.Linear(latent_dim, hidden)
        self.layers     = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(6)])
        self.cond_projs = nn.ModuleList([nn.Linear(t_dim, hidden * 2) for _ in range(6)])
        self.out_proj   = nn.Linear(hidden, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.t_embed(t)) + self.label_emb(y)  # (B, t_dim)
        h    = F.silu(self.input_proj(z))
        for layer, norm, cproj in zip(self.layers, self.norms, self.cond_projs):
            scale, shift = cproj(F.silu(cond)).chunk(2, dim=-1)
            h = norm(h) * (1.0 + scale) + shift
            h = h + F.silu(layer(h))
        return self.out_proj(h)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Training loop 
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_eg2(sigma_fn: SigmaFn, dataset_name: str,
                            n_samples: int = 5000) -> float:
    """Estimate E[Sigma(x0)^2] from training data."""
    ds     = _get_dataset(dataset_name, train=True, tf=_NORM_TF)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2)
    total  = count = 0

    for x0, labels in loader:
        S      = sigma_fn(x0, labels) if getattr(sigma_fn,"needs_label",False) else sigma_fn(x0)
        total += (S ** 2).mean().item() * x0.size(0)
        count += x0.size(0)
        if count >= n_samples:
            break
    return total / count


def train(
        sigma_fn:   SigmaFn, 
        cfg:        Config,
        noise_type: str   = "rosenblatt",
        H:          float = 0.7,
        save_dir:   str   = None
) -> tuple[ConditionalUNet, RosenblattForward]:    
    """
    Main training function for image-space cold diffusion.
    """
    tag = f"{noise_type}_{sigma_fn.__name__}_H{H}"
    print(f"Training | {tag} | Dataset:{cfg.dataset} | epochs:{cfg.epochs}")
    Path(str(save_dir)).mkdir(parents=True, exist_ok=True)

    train_ds = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    val_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    
    forward  = RosenblattForward(sigma_fn, noise_type=noise_type,
                                H=H, M_eig=cfg.M_eig, sigma_max=cfg.sigma_max,
                                device=cfg.device)
    # Estimate E[Sigma^2] from data (overrides analytic default)
    eg2 = _estimate_eg2(sigma_fn, dataset_name=cfg.dataset)
    forward.set_eg2(eg2)
    print(f"  E[Sigma^2] = {eg2:.4f}")

    model  = ConditionalUNet(num_classes=10, base_ch=cfg.base_ch).to(cfg.device)

    # --- ADD MODEL LOADING LOGIC HERE ---
    ckpt_path = Path(f"{save_dir}/{tag}_final.pt")
    if ckpt_path.exists():
        print(f"Loading pre-trained model: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
        model.eval()
        return model, forward
    # ------------------------------------
     
    ema    = EMA(model, decay=0.999)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if cfg.device.type == "cuda" else None
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = [], []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for x0, labels in train_dl:
            x0, labels = (x0.to(cfg.device, non_blocking=True),
                          labels.to(cfg.device, non_blocking=True))
            B      = x0.size(0)
            cf     = torch.rand(B, device=cfg.device) < 0.1
            lbl    = labels.clone();  lbl[cf] = 10
            t      = torch.rand(B, device=cfg.device) * (1. - cfg.T_MIN) + cfg.T_MIN
            x_t, _, _ = forward.corrupt(x0, t, y=labels)
            c_in   = forward.c_in(t).view(-1, 1, 1, 1)

            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(x_t * c_in, t, lbl)
                    loss = F.smooth_l1_loss(pred, x0)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(x_t * c_in, t, lbl)
                loss = F.smooth_l1_loss(pred, x0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            ema.update()
            ep_loss += loss.item() * B

        ep_loss /= len(train_ds)
        train_losses.append(ep_loss)

        # --- validation with EMA weights ---
        model.eval();  ema.apply_shadow();  v_loss = 0.0
        with torch.no_grad():
            for x0, labels in val_dl:
                x0, labels = (x0.to(cfg.device, non_blocking=True),
                              labels.to(cfg.device, non_blocking=True))
                t = torch.rand(x0.size(0), device=cfg.device) * \
                    (1.0 - cfg.T_MIN) + cfg.T_MIN
                x_t, _, _ = forward.corrupt(x0, t, y=labels)
                c_in = forward.c_in(t).view(-1, 1, 1, 1)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        pred = model(x_t * c_in, t, labels)
                        v_loss += F.smooth_l1_loss(pred,
                                                   x0).item() * x0.size(0)
                else:
                    pred = model(x_t * c_in, t, labels)
                    v_loss += F.smooth_l1_loss(pred, x0).item() * x0.size(0)

        v_loss /= len(val_ds)
        val_losses.append(v_loss)
        # keep EMA weights applied for checkpointing
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/{tag}_ep{epoch+1}.pt")
        ema.restore()
        sched.step()
        print(f"  ep {epoch+1:3d}/{cfg.epochs}  train={ep_loss:.4f}  val={v_loss:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s")

    ema.apply_shadow()
    torch.save(model.state_dict(), f"{save_dir}/{tag}_final.pt")
    model.eval()

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses,   label="val")
    ax.set(xlabel="Epoch", ylabel="Smooth L1", title=tag)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_loss.png", dpi=120)
    plt.close();plt.show()

    return model, forward


# ─────────────────────────────────────────────────────────────────────────────
# 9. Generation — stochastic bridge sampler
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_conditional(
    model:     nn.Module,
    forward:   RosenblattForward,
    labels:    torch.Tensor,
    cfg:       Config,
    bridge:    str   = "stochastic",
    x_in:      torch.Tensor = None,
) -> torch.Tensor:
    """
    Cold diffusion reverse process with classifier-free guidance.

    bridge="stochastic" (recommended): draw fresh noise at each step.
    bridge="deterministic": original rescaled-residual method (broken for Rosenblatt).
    """
    model.eval()
    n           = len(labels)
    null_labels = torch.full_like(labels, 10)

    # Linear schedule from t=1 to t=0
    t_sched = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=cfg.device)

    # Start from pure noise at t=1 if x_in is not provided
    if x_in is None:
        eps = sample_noise(forward.noise_type, (n, 1, 28, 28),
                           forward.lam_t, forward.M_eig, device=cfg.device)
        dummy_x0 = torch.zeros(n, 1, 28, 28, device=cfg.device)
        # Apply Sigma scaling if it's purely structural (PCA/anisotropic).
        # Multiplicative scaling depends on |x0|, so at x0=0 it drops to 0.5.
        # We cap the lower bound of Sigma at 1.0 (or just use the pure noise)
        # to prevent starting from a too-small variance.
        S = forward.sigma_fn(dummy_x0, labels) if getattr(forward.sigma_fn, "needs_label", False) else forward.sigma_fn(dummy_x0)
        # Fallback to 1.0 for multiplicative/edge-aware where S(0) is small
        if S.mean() < 0.9:
            S = torch.ones_like(S)
        x = eps * forward.sigma_max * S
    else:
        x = x_in

    for k in range(cfg.n_steps):
        t_cur  = t_sched[k].expand(n)
        t_next = t_sched[k + 1].expand(n)
        c_in   = forward.c_in(t_cur).view(-1, 1, 1, 1)
        # Use a distinctive name to avoid shadowing the function parameter
        scaled_x_in = (x * c_in).float()

        if cfg.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0_c = model(scaled_x_in, t_cur, labels).float()
                x0_u = model(scaled_x_in, t_cur, null_labels).float()
        else:
            x0_c = model(scaled_x_in, t_cur, labels)
            x0_u = model(scaled_x_in, t_cur, null_labels)

        # CFG: interpolate from unconditional toward conditional
        x0_hat = (x0_u + cfg.cfg_scale * (x0_c - x0_u)).clamp(-1., 1.)

        if k < cfg.n_steps - 1:
            if bridge == "stochastic":
                # x = forward.recorrupt_stochastic(x0_hat, t_next, y=labels)
                x = forward.recorrupt_stochastic(x0_hat, t_next, y=None)
            elif bridge == "hybrid":
                x = forward.recorrupt_hybrid(x, x0_hat, t_cur, t_next, y=labels)
            else:
                # BROKEN for non-Gaussian (kept for ablation)
                x = forward.recorrupt_deterministic(x, x0_hat, t_cur, t_next)
        else:
            x = x0_hat

    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. KID computation (Fashion-FID via Custom ResNet)
# ─────────────────────────────────────────────────────────────────────────────

class FashionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a basic resnet18, modify first conv for 1-channel grayscale
        self.net = resnet18(num_classes=10)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x, features_only: bool = False):
        # Forward through all layers up to the penultimate
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)          # (B, 512)
        if features_only:
            return x                      # 512-dim feature vector
        return self.net.fc(x)             # (B, 10) logits


def get_fashion_extractor(device, weights_path=None):
    if weights_path is None:
        weights_path = str(OUT_ROOT / "fashion_resnet.pth")
    extractor = FashionFeatureExtractor().to(device)
        
    if Path(weights_path).exists():
        print(f"Loading cached FashionMNIST feature extractor from {weights_path}...")
        extractor.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print("Training FashionMNIST classifier for conditional accuracy...")
        train_ds = _get_dataset("FashionMNIST", train=True, tf=_NORM_TF)
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True)
        opt  = torch.optim.Adam(extractor.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        
        extractor.train()
        with torch.enable_grad():
            for ep in range(3):  # 3 epochs is plenty to learn good features!
                total_loss = 0
                for x, y in train_dl:
                    x, y = x.to(device, dtype=torch.float32), y.to(device)
                    opt.zero_grad()
                    loss = crit(extractor(x), y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                print(f"  Extractor Epoch {ep+1}/3 Loss: {total_loss/len(train_dl):.4f}")
        
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(extractor.state_dict(), weights_path)    
    
    extractor.eval()
    return extractor


@torch.no_grad()
def compute_conditional_accuracy(fake_imgs:   torch.Tensor,
                                 fake_labels: torch.Tensor,
                                 device:      torch.device,
                                 batch_size:  int = 128,
                                 extractor:   nn.Module = None) -> float:
    """Top-1 accuracy of generated images under the FashionMNIST classifier."""
    if extractor is None:
        extractor = get_fashion_extractor(device)   # always returns with Linear fc
    extractor.eval()
    
    correct = total = 0

    for i in range(0, fake_imgs.size(0), batch_size):
        x = fake_imgs[i:i+batch_size].to(device)

        # The generated fake_imgs are in [0, 1]. We must convert them back to [-1, 1]!
        if x.min() >= 0 and x.max() <= 1.0:
            x = (x * 2.0) - 1.0
            
        y        = fake_labels[i:i+batch_size].to(device)
        preds    = extractor(x, features_only=False).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


class FashionFIDWrapper(nn.Module):
    """Wraps the feature extractor to accept [0, 1] RGB batch for torchmetrics FID."""
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.extractor.eval()

    def forward(self, x):
        # FrechetInceptionDistance with normalize=True passes images in [0, 1].
        # If it bypassed and sent 3-channel, we convert to 1-channel grayscale.
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
            
        # Move input to the same device as the extractor (needed for torchmetrics dummy passes)
        x = x.to(next(self.extractor.parameters()).device)
            
        # The underlying extractor was trained on [-1, 1] from _NORM_TF
        x = (x * 2.0) - 1.0
        return self.extractor(x, features_only=True)


@torch.no_grad()
def compute_fid(real_imgs: torch.Tensor, fake_imgs: torch.Tensor,
                device: torch.device, batch_size: int = 50,
                wrapper: nn.Module = None) -> float:
    """Standard FID via InceptionV3 features since custom representation spaces can scale unpredictably. Expects [-1, 1], shape (N,1,28,28)."""    
    if wrapper is not None:
        fid = FrechetInceptionDistance(feature=wrapper, normalize=True).to(device)
    else:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # 1. Convert [-1, 1] to [0, 1] for the metric if normalize=True is used
    if real_imgs.min() < 0 or real_imgs.max() > 1:
        real_imgs = (real_imgs + 1.0) / 2.0
    if fake_imgs.min() < 0 or fake_imgs.max() > 1:
        fake_imgs = (fake_imgs + 1.0) / 2.0

    # 2. Convert 1-channel grayscale to 3-channel RGB 
    real_imgs = real_imgs.repeat(1, 3, 1, 1)
    fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
    
    # 3. Compute FID (using batches to avoid OOM)
    for i in range(0, real_imgs.size(0), batch_size):
        fid.update(real_imgs[i: i+batch_size].to(device), real=True)
    for i in range(0, fake_imgs.size(0), batch_size):
        fid.update(fake_imgs[i: i+batch_size].to(device), real=False)
        
    return float(fid.compute())


@torch.no_grad()
def evaluate_latent_model(
    model:     LatentMLPDenoiser,
    ae:        ConvAutoencoder,
    forward:   RosenblattForward,
    real_imgs: torch.Tensor,        # [0,1], AE-reconstructed, shape (N,1,28,28)
    test_ds,                        # raw dataset, returns [-1,1]
    cfg:       Config,
) -> dict:
    """
    Unified evaluation for the latent cold diffusion model.
    Mirrors evaluate_model but uses generate_latent and latent-space SSIM.

    real_imgs should be AE-reconstructed real images (not raw pixels),
    so FID measures generation quality relative to what the AE can produce,
    not the gap introduced by AE reconstruction quality.
    """

    model.eval(); ae.eval()
    t0 = time.time()

    # ── 1. Generate n_fid decoded images ──────────────────────────────────
    lbl   = torch.randint(0, 10, (cfg.n_fid,), device=cfg.device)
    fakes = []
    for i in range(0, cfg.n_fid, 200):
        fakes.append(
            generate_latent(model, ae, forward, lbl[i:i+200], cfg).cpu())
    fakes_t = torch.cat(fakes, 0)          # (n_fid, 1, 28, 28), [0,1]

    # ── 2. FID & Fashion-FID ──────────────────────────────────────────────
    fid = compute_fid(real_imgs, fakes_t, device=cfg.device)

    extractor = get_fashion_extractor(cfg.device)
    wrapper = FashionFIDWrapper(extractor).to(cfg.device)
    f_fid = compute_fid(real_imgs, fakes_t, device=cfg.device, wrapper=wrapper)

    # ── 3. Conditional accuracy ───────────────────────────────────────────
    acc = compute_conditional_accuracy(fakes_t, lbl.cpu(), device=cfg.device, extractor=extractor)

    # ── 4. Latent reconstruction SSIM & LPIPS ─────────────────────────────
    # Encode real images → corrupt in latent space → denoise → decode
    # Compare decoded reconstruction with AE-reconstructed originals.
    # This measures how well the latent denoiser reverses the corruption,
    # independently of AE reconstruction quality.
    reals_n1p1 = torch.stack([test_ds[i][0] for i in range(cfg.n_ssim)]).to(cfg.device)
    reals_0to1 = (reals_n1p1 + 1.) / 2.

    # AE-reconstructed originals (the ceiling the latent model can reach)
    ae_recon, _ = ae(reals_n1p1)
    ae_recon_0to1 = ((ae_recon + 1.) / 2.).clamp(0., 1.)

    # Corrupt in latent space at t=1
    D      = ConvAutoencoder.LATENT_DIM
    z0     = ae.encode(reals_n1p1)                          # (n_ssim, 64)
    sig    = forward.sigma_t(torch.ones(cfg.n_ssim, device=cfg.device)).unsqueeze(1)
    eps    = sample_noise(forward.noise_type, (cfg.n_ssim, D),
                          forward.lam_t, forward.M_eig, cfg.device)
    z_T    = z0 + sig * eps

    # Denoise back with the latent model
    real_lbl = torch.tensor([test_ds[i][1] for i in range(cfg.n_ssim)], device=cfg.device)
    null     = torch.full_like(real_lbl, 10)
    sched    = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    z        = z_T.clone()

    for k in range(cfg.n_steps):
        tc  = sched[k  ].expand(cfg.n_ssim)
        tn  = sched[k+1].expand(cfg.n_ssim)
        sig = forward.sigma_t(tc).unsqueeze(1)
        cin = (1. + sig**2).pow(-0.5)
        z0c = model(z * cin, tc, real_lbl)
        z0u = model(z * cin, tc, null)
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)
        if k < cfg.n_steps - 1:
            sn  = forward.sigma_t(tn).unsqueeze(1)
            z   = z0h + sn * sample_noise(forward.noise_type, (cfg.n_ssim, D),
                                           forward.lam_t, forward.M_eig, cfg.device)
        else:
            z = z0h

    recon_0to1 = ((ae.decode(z) + 1.) / 2.).clamp(0., 1.)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    ssim_val    = ssim_metric(recon_0to1, ae_recon_0to1).item()
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(cfg.device)
    # LPIPS expects 3 channels
    lpips_val = lpips_metric(recon_0to1.repeat(1, 3, 1, 1), ae_recon_0to1.repeat(1, 3, 1, 1)).item()

    elapsed = time.time() - t0
    return {
        "FID":         round(fid, 2),
        "fFID":        round(f_fid, 2),
        "Accuracy":    round(acc * 100, 2),
        "SSIM":        round(ssim_val, 4),
        "LPIPS":       round(lpips_val, 4),
        "eval_time_s": round(elapsed, 1),
    }



@torch.no_grad()
def evaluate_model(
    model:     nn.Module,
    forward:   RosenblattForward,
    real_imgs: torch.Tensor,        # [0,1], shape (N,1,28,28)
    test_ds,                        # raw dataset for SSIM (returns [-1,1])
    cfg :       Config,
    bridge:    str   = "stochastic",
) -> dict:
    """
    Unified evaluation: FID + conditional accuracy + reconstruction SSIM.

    Returns
    -------
    dict with keys: FID, Accuracy (%), SSIM, eval_time_s
    """
    model.eval()
    t0 = time.time()

    # ── 1. Generate n_fid images ──────────────────────────────────────────
    lbl   = torch.randint(0, 10, (cfg.n_fid,), device=cfg.device)
    fakes = []
    for i in range(0, cfg.n_fid, 200):
        fakes.append(
            generate_conditional(model, forward, lbl[i:i+200], cfg, bridge=bridge).cpu())
    fakes_t = torch.cat(fakes, 0)          # (n_fid, 1, 28, 28), [0,1]

    # ── 2. FID & Fashion-FID ──────────────────────────────────────────────
    fid = compute_fid(real_imgs, fakes_t, cfg.device)
    
    extractor = get_fashion_extractor(cfg.device)    
    wrapper = FashionFIDWrapper(extractor).to(cfg.device)
    f_fid = compute_fid(real_imgs, fakes_t, cfg.device, wrapper=wrapper)

    # ── 3. Conditional accuracy ───────────────────────────────────────────
    acc = compute_conditional_accuracy(fakes_t, lbl.cpu(), cfg.device, extractor=extractor)

    # ── 4. Reconstruction SSIM & LPIPS ────────────────────────────────────
    # Corrupt n_ssim real images to t=1, then reverse; measure structural
    # similarity between reconstruction and original.
    # test_ds returns images in [-1,1] (from _NORM_TF).
    reals_n1p1    = torch.stack([test_ds[i][0] for i in range(cfg.n_ssim)]).to(cfg.device)
    reals_0to1    = (reals_n1p1 + 1.) / 2.          # [0,1] for SSIM
    real_lbl      = torch.tensor([test_ds[i][1] for i in range(cfg.n_ssim)], device=cfg.device)

    x_T, _, _     = forward.corrupt(reals_n1p1,
                                     torch.ones(cfg.n_ssim, device=cfg.device),
                                     y=real_lbl)
    recon_0to1    = generate_conditional(model, forward, real_lbl, cfg, 
                                          bridge=bridge, x_in=x_T)

    ssim_metric   = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    ssim_val      = ssim_metric(recon_0to1, reals_0to1).item()
    
    lpips_metric  = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(cfg.device)
    lpips_val     = lpips_metric(recon_0to1.repeat(1, 3, 1, 1), reals_0to1.repeat(1, 3, 1, 1)).item()

    elapsed = time.time() - t0
    return {
        "FID":        round(fid, 2),
        "fFID":       round(f_fid, 2),
        "Accuracy":   round(acc * 100, 2),
        "SSIM":       round(ssim_val, 4),
        "LPIPS":      round(lpips_val, 4),
        "eval_time_s": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _restoration_grid(model: nn.Module, forward: RosenblattForward,
                       cfg: Config, save_dir: str,
                       tag: str = "", bridge: str = "stochastic") -> None:

    model.eval()
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    found   = {}
    for i in range(len(test_ds)):
        lb = test_ds[i][1]
        if lb not in found:
            found[lb] = i
        if len(found) == 10:
            break

    x0   = torch.stack([test_ds[found[c]][0] for c in range(10)]).to(cfg.device)
    lbl  = torch.arange(10, device=cfg.device)
    null = torch.full_like(lbl, 10)
    xc, _, _ = forward.corrupt(x0, torch.ones(10, device=cfg.device), y=lbl)

    sched  = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    # Which steps to save for display: 0 (corrupted) + n_display equally spaced + final
    save_at = set([0] + [int(round(cfg.n_steps * i / (cfg.n_display - 1)))
                         for i in range(cfg.n_display)])
    save_at.add(cfg.n_steps - 1)   # always include final step
    x_cur  = xc.clone()
    hist   = {}

    for k in range(cfg.n_steps):
        tc   = sched[k].expand(10)
        tn   = sched[k + 1].expand(10)
        c_in = forward.c_in(tc).view(-1, 1, 1, 1)
        if cfg.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0c = model(x_cur * c_in, tc, lbl).float()
                x0u = model(x_cur * c_in, tc, null).float()
        else:
            x0c = model(x_cur * c_in, tc, lbl)
            x0u = model(x_cur * c_in, tc, null)
        x0h = (x0u + cfg.cfg_scale * (x0c - x0u)).clamp(-1., 1.)
        if k + 1 in save_at:
            hist[k + 1] = x0h.cpu()
        if k < cfg.n_steps - 1:
            if bridge == "stochastic":
                x_cur = forward.recorrupt_stochastic(x0h, tn, y=lbl)
            elif bridge == "hybrid":
                x_cur = forward.recorrupt_hybrid(x_cur, x0h, tc, tn, y=lbl)
            else:  # deterministic — correct for Gaussian, broken for Rosenblatt
                x_cur = forward.recorrupt_deterministic(x_cur, x0h, tc, tn)
        else:
            x_cur = x0h
        
    snap_keys  = sorted(hist.keys())
    snaps      = [hist[k] for k in snap_keys]
    n_cols     = 2 + len(snaps)   # original + corrupted + snapshots

    fig, axes = plt.subplots(10, n_cols, figsize=(2. * n_cols, 14))
    for i in range(10):
        axes[i, 0].imshow((x0[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow((xc[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        for col, snap in enumerate(snaps):
            axes[i, col + 2].imshow((snap[i, 0] + 1) / 2,
                                    cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(class_name(cfg.dataset, i),
                              fontsize=8, rotation=0, labelpad=40, va="center")
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 1].set_title("Corrupted\nt=1", fontsize=7)
    for col, k in enumerate(snap_keys):
        t_val = 1. - k / cfg.n_steps
        axes[0, col + 2].set_title(f"t={t_val:.2f}\nstep {k}", fontsize=7)
    plt.suptitle(f"Restoration ({cfg.n_steps} steps) — {tag}\n{forward.label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_restoration.png", dpi=120)
    plt.close()


def _sigma_pattern_plot(sigma_fn: SigmaFn, save_dir: str) -> None:
    ds  = _get_dataset("FashionMNIST", train=False, tf=_NORM_TF)
    x0  = ds[0][0].unsqueeze(0)
    with torch.no_grad():
        y_dummy = torch.zeros(1, dtype=torch.long)   # class 0 as representative
        S = (sigma_fn(x0, y_dummy) if getattr(sigma_fn, "needs_label", False) else sigma_fn(x0))[0, 0].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow((x0[0, 0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input image", fontsize=10);  axes[0].axis("off")
    im = axes[1].imshow(S, cmap="hot")
    axes[1].set_title(f"$\\Sigma(\\mathbf{{x}}_0)$\n{sigma_fn.label}", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    plt.tight_layout()
    fp = f"{save_dir}/{sigma_fn.__name__}_pattern.png"
    plt.savefig(fp, dpi=130);  plt.close();plt.show()
    print(f"  Saved {fp}")

def plot_all_sigma_patterns(sigma_fns: list, save_path: str,
                            dataset_name: str = "FashionMNIST",
                            example_classes: list[int] = None) -> None:
    """
    Plot per-pixel Sigma(x0) patterns for all sigma functions in a single figure.

    Layout: rows = classes (if example_classes given, else 1 row with fixed image)
            cols = original + one column per sigma_fn
    """
    if example_classes is None:
        example_classes = [0]   # single row: T-shirt/top

    ds   = _get_dataset(dataset_name, train=False, tf=_NORM_TF)
    # Pick one image per requested class
    found = {}
    for i in range(len(ds)):
        lb = ds[i][1]
        if lb in example_classes and lb not in found:
            found[lb] = ds[i][0].unsqueeze(0)   # (1,1,28,28)
        if len(found) == len(example_classes): break

    n_rows = len(example_classes)
    n_cols = 1 + len(sigma_fns)   # original + one per sigma

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # make 2D

    for row, cls in enumerate(example_classes):
        x0 = found[cls]   # (1,1,28,28)

        # Column 0: original image
        axes[row, 0].imshow((x0[0, 0].numpy() + 1) / 2,
                            cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(class_name(dataset_name, cls),
                                fontsize=9, rotation=0, labelpad=42, va="center")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].axis("off")

        # Columns 1+: sigma patterns
        with torch.no_grad():
            for col, sfn in enumerate(sigma_fns):
                y_dummy = torch.tensor([cls], dtype=torch.long)
                S = (sfn(x0, y_dummy)
                     if getattr(sfn, "needs_label", False)
                     else sfn(x0))
                S_np = S[0, 0].numpy()

                vmax = S_np.max(); vmin = S_np.min()
                im = axes[row, col + 1].imshow(S_np, cmap="hot",
                                               vmin=vmin, vmax=vmax)
                if row == 0:
                    # Show label + E[Sigma^2] above first row
                    eg2 = getattr(sfn, "eg2", float((S**2).mean()))
                    axes[row, col + 1].set_title(
                        f"{sfn.label}\n$E[\\Sigma^2]={eg2:.2f}$",
                        fontsize=8)
                axes[row, col + 1].axis("off")
                plt.colorbar(im, ax=axes[row, col + 1],
                             fraction=0.046, pad=0.04)

    plt.suptitle(r"Per-pixel noise coefficient $\Sigma(\mathbf{x}_0)$",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")

def run_sigma_comparison(cfg: Config) -> list[dict]:
    """
    Core experiment: compare three non-trivial choices of Sigma.

    1. Multiplicative:  Sigma(x0) = diag(g(x0))   
    2. Anisotropic H:   Sigma = A_h_emphasis      
    3. PCA-whitened:    Sigma = C^{-1/2}          
    4. Edge-aware:      Sigma(x0) = diag(|Sobel|) 

    Sigma=I (additive) is NOT included as a comparison target —
    it is the trivial baseline with no geometric structure.
    """

    # Reference real images for FID
    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = torch.stack([test_ds[i][0] for i in range(cfg.n_fid)])
    real_imgs = (real_imgs + 1.0) / 2.0   # [0, 1]

    # Pre-compute PCA whitening from training data
    print("Computing pixel variance for PCA-whitened Sigma...")
    class_vars = compute_condition_pixel_variance(cfg.dataset)   # (1,28,28)
    global_var = compute_global_pixel_variance(cfg.dataset)       # (1,28,28)

    sigma_variants = [
        # sigma_multiplicative(),
        # sigma_anisotropic(mode="h_emphasis"),
        # sigma_anisotropic(mode="v_emphasis"),
        # sigma_pca_whitened_conditional(class_vars),
        sigma_pca_whitened_global(global_var),
        # sigma_edge_aware(sobel_strength=2.0),
    ]

    results = []

    for sfn in sigma_variants:
        run_dir = f"{cfg.save_dir}/{sfn.__name__}"
        print(f"\n{'='*60}\nExp sigma_comparison: noise={cfg.noise_type}  sigma={sfn.__name__} bridge={cfg.bridge}")
        model, forward = train(sfn, cfg, save_dir=run_dir)

        if cfg.EVALUATE:
            metrics = evaluate_model(model, forward, real_imgs, test_ds,
                                    cfg, bridge=cfg.bridge)
            results.append({"sigma":     sfn.__name__,
                            "label":     sfn.label,
                            "eg2":       round(forward._eg2, 4),
                            "noise":     cfg.noise_type,
                            "bridge":    cfg.bridge,
                            "FID":       round(metrics['FID'], 2),
                            "fFID":      round(metrics.get('fFID', 0), 2),
                            "Accuracy":  round(metrics['Accuracy'], 2),
                            "SSIM":      round(metrics['SSIM'], 4),
                            "LPIPS":     round(metrics.get('LPIPS', 0), 4),
                            "Eval Time": round(metrics['eval_time_s'], 1)})
            print(f"  {sfn.__name__:25s}  E[Σ²]={forward._eg2:.3f}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")

        if cfg.PLOT:
            _restoration_grid(model, forward, cfg, run_dir,
                            tag=f"{cfg.noise_type}_{cfg.bridge}_{sfn.__name__}", bridge=cfg.bridge)
            # _sigma_pattern_plot(sfn, run_dir)

    plot_all_sigma_patterns(
        sigma_variants,
        save_path=f"{cfg.save_dir}/all_sigma_patterns.png",
        example_classes=[0, 1, 7, 9]   # T-shirt, Trouser, Sneaker, Ankle boot
    )
    print("\nSigma comparison FID Summary:")
    for r in results:
        print(f"  noise={r['noise']:10s} bridge={r['bridge']:10s}  sigma={r['sigma']:20s}  FID={r['FID']}   fFID={r['fFID']}  Acc={r['Accuracy']}%  SSIM={r['SSIM']}  LPIPS={r['LPIPS']} Eval Time: {r['Eval Time']:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 12. Latent experiment (Stable Diffusion-style)
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(cfg: Config) -> ConvAutoencoder:
    """
    Train a latent-space MLP denoiser.

    Forward process: Z_t = z_0 + sigma(t) * eps,  eps ~ noise_type
    where z_0 = Encoder(x_0) in R^{64}.

    Question 2: does there exist a basis (latent space of autoencoder)
    in which Rosenblatt corruption leads to better results?
    Answer: yes, because (a) the latent distribution is non-Gaussian (richer
    structure), (b) Rosenblatt heavy tails cover the full tail of the latent
    distribution, (c) density theory (Prop. latent-density) remains valid.
    """

    ds  = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                     num_workers=4, pin_memory=True)
    ae  = ConvAutoencoder().to(cfg.device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    for ep in range(cfg.ae_epochs):
        ae.train();  tot = 0
        for x0, _ in dl:
            x0    = x0.to(cfg.device, non_blocking=True)
            r, _  = ae(x0)
            loss  = F.mse_loss(r, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward();  opt.step()
            tot += loss.item() * x0.size(0)
        print(f"  AE ep {ep+1:3d}/{cfg.ae_epochs}  {tot/len(ds):.5f}")

    ae_path = f"{cfg.save_dir}/latent/ae_final.pt"
    torch.save(ae.state_dict(), ae_path)
    print(f"  AE → {ae_path}")
    return ae


def train_latent(
    ae:           ConvAutoencoder,
    cfg:          Config,
    sigma_max:    float = 4.,
    noise_type:   str   = "rosenblatt",
) -> tuple[LatentMLPDenoiser, RosenblattForward]:
    
    ae.eval()
    tr_dl = DataLoader(_get_dataset(cfg.dataset, train=True,  tf=_NORM_TF),
                       cfg.batch_size, True,  num_workers=4, pin_memory=True,
                       persistent_workers=True)
    va_dl = DataLoader(_get_dataset(cfg.dataset, train=False, tf=_NORM_TF),
                       cfg.batch_size, False, num_workers=4, pin_memory=True,
                       persistent_workers=True)

    D   = ConvAutoencoder.LATENT_DIM
    fwd = RosenblattForward(sigma_additive(), noise_type=noise_type,
                            H=cfg.H, M_eig=cfg.M_eig, sigma_max=sigma_max, device=cfg.device)

    model = LatentMLPDenoiser(latent_dim=D).to(cfg.device)
    tag   = f"lat_{noise_type}_s{sigma_max}"

    # --- ADD MODEL LOADING LOGIC HERE ---
    ckpt_path = Path(f"{cfg.save_dir}/{tag}_final.pt")
    if ckpt_path.exists():
        print(f"Loading pre-trained Latent model: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
        model.eval()
        return model, fwd
    # ------------------------------------

    ema   = EMA(model, 0.999)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    tag   = f"lat_{cfg.noise_type}_s{sigma_max}"

    for ep in range(cfg.epochs):
        t0 = time.time();  model.train();  el = 0
        for x0, lbl in tr_dl:
            x0, lbl = (x0.to(cfg.device, non_blocking=True),
                       lbl.to(cfg.device, non_blocking=True))
            B = x0.size(0)
            with torch.no_grad():
                z0 = ae.encode(x0)
            cf       = torch.rand(B, device=cfg.device) < 0.1
            lbl2     = lbl.clone();  lbl2[cf] = 10
            t        = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
            sig      = fwd.sigma_t(t).unsqueeze(1)          # FIX-1 applied
            eps      = sample_noise(fwd.noise_type, (B, D), fwd.lam_t, cfg.M_eig, cfg.device)
            z_t      = z0 + sig * eps
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(z_t, t, lbl2), z0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step();  ema.update()
            el += loss.item() * B

        el /= len(tr_dl.dataset)
        model.eval();  ema.apply_shadow();  vl = 0
        with torch.no_grad():
            for x0, lbl in va_dl:
                x0, lbl = (x0.to(cfg.device, non_blocking=True),
                           lbl.to(cfg.device, non_blocking=True))
                z0  = ae.encode(x0)
                t   = torch.rand(x0.size(0), device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
                sig = fwd.sigma_t(t).unsqueeze(1)           # FIX-1 applied
                eps = sample_noise(fwd.noise_type, (x0.size(0), D),
                                   fwd.lam_t, cfg.M_eig, cfg.device)
                vl += F.smooth_l1_loss(model(z0 + sig * eps, t, lbl),
                                       z0).item() * z0.size(0)
        vl /= len(va_dl.dataset)
        ema.restore();  sch.step()
        print(f"  [lat] {ep+1:3d}/{cfg.epochs}  tr={el:.5f}  va={vl:.5f}  "
              f"{time.time()-t0:.1f}s")

    ema.apply_shadow()
    torch.save(model.state_dict(), f"{cfg.save_dir}/{tag}_final.pt")
    model.eval()
    return model, fwd


@torch.no_grad()
def generate_latent(
    model:   LatentMLPDenoiser,
    ae:      ConvAutoencoder,
    fwd:     RosenblattForward,
    labels:  torch.Tensor,
    cfg:     Config,
) -> torch.Tensor:
    """
    CFG formula corrected: z0u + cfg*(z0c-z0u).
    fwd.sigma_t() used throughout.
    """
    model.eval();  ae.eval()
    n    = len(labels)
    null = torch.full_like(labels, 10)
    D    = ConvAutoencoder.LATENT_DIM

    t_sched = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    z = sample_noise(fwd.noise_type, (n, D),
                     fwd.lam_t, fwd.M_eig, cfg.device) * fwd.sigma_max

    for k in range(cfg.n_steps):
        tc  = t_sched[k].expand(n)
        tn  = t_sched[k + 1].expand(n)
        sig = fwd.sigma_t(tc).unsqueeze(1)               # FIX-1
        cin = (1. / (1. + sig ** 2).sqrt())
        z0c = model(z * cin, tc, labels)
        z0u = model(z * cin, tc, null)

        # correct CFG sign
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)

        if k < cfg.n_steps - 1:
            sn  = fwd.sigma_t(tn).unsqueeze(1)           
            eps = sample_noise(fwd.noise_type, (n, D),
                               fwd.lam_t, fwd.M_eig, cfg.device)
            z   = z0h + sn * eps
        else:
            z = z0h

    return ((ae.decode(z) + 1.) / 2.).clamp(0., 1.)


def run_exp_latent(cfg: Config) -> list[dict]:
    """Gaussian vs Rosenblatt cold diffusion in 64-D latent space."""

    ae_path = f"{cfg.save_dir}/latent/ae_final.pt"
    ae      = ConvAutoencoder().to(cfg.device)
    if Path(ae_path).exists():
        ae.load_state_dict(torch.load(ae_path, map_location=cfg.device, weights_only=True))
        print(f"Loaded AE from {ae_path}")
    else:
        ae = train_autoencoder(cfg)
    ae.eval()

    test_ds   = _get_dataset(cfg.dataset_name, train=False, tf=_NORM_TF)
    real_orig = torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]).to(cfg.device)
    real_re   = []
    for i in range(0, cfg.n_fid, 256):
        r, _ = ae(real_orig[i:i+256])
        real_re.append(r.cpu())
    real_imgs = ((torch.cat(real_re, 0) + 1.) / 2.).clamp(0., 1.)

    results = []
    for nt in ("gaussian", "rosenblatt"):
        for sm in (4., 16.):
            tag = f"{nt}_s{sm}"
            rd  = f"{cfg.save_dir}/latent"
            print(f"\nExp Latent: {tag}")
            model, forward = train_latent(ae, cfg, sigma_max=sm, noise_type=nt)
            model.eval()            

            if cfg.EVALUATE:
                metrics = evaluate_latent_model(model, ae, forward, real_imgs, test_ds, cfg)
                results.append({"noise": nt, "sigma_max": sm, **metrics})
                print(f"  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}")
            
            _save_latent_samples(model, ae, forward, cfg, tag=tag, save_dir=rd)

    if cfg.EVALUATE:
        print("\nLatent summary:")
        for r in results:
            print(f"  {r}")
    return results

@torch.no_grad()
def _save_latent_samples(
    model: LatentMLPDenoiser,
    ae:    ConvAutoencoder,
    fwd:   RosenblattForward,
    cfg:   Config,
    tag:   str = "",
    n_cls: int = 10,
    save_dir: str = ".",
) -> None:
    """Generate n_cls decoded samples (one per class) and save as a grid."""
    model.eval(); ae.eval()
    labels = torch.arange(n_cls, device=cfg.device)
    D      = ConvAutoencoder.LATENT_DIM
    sched  = torch.linspace(1., 0., 50 + 1, device=cfg.device)
    z      = sample_noise(fwd.noise_type, (n_cls, D),
                          fwd.lam_t, fwd.M_eig, cfg.device) * fwd.sigma_max

    null = torch.full_like(labels, 10)
    
    for k in range(50):
        tc = sched[k].expand(n_cls); tn = sched[k+1].expand(n_cls)
        sig = fwd.sigma_t(tc).unsqueeze(1)
        cin = (1. + sig**2).pow(-0.5)
        z0c = model(z * cin, tc, labels)
        z0u = model(z * cin, tc, null)
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)
        if k < 49:
            sn = fwd.sigma_t(tn).unsqueeze(1)
            z  = z0h + sn * sample_noise(fwd.noise_type, (n_cls, D),
                                          fwd.lam_t, fwd.M_eig, cfg.device)
        else:
            z = z0h

    imgs = ((ae.decode(z) + 1.) / 2.).clamp(0., 1.).cpu()  # (10, 1, 28, 28)

    fig, axes = plt.subplots(1, n_cls, figsize=(2. * n_cls, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(_FASHION[i], fontsize=7, rotation=45, ha="right")
        ax.axis("off")
    plt.suptitle(f"Latent samples — {tag}", fontsize=9)
    plt.tight_layout()
    fp = f"{save_dir}/{tag}_samples.png"
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  Saved {fp}")


def run_exp_pca_basis(
    cfg:          Config,
    bridge:       str   = "stochastic",
) -> dict:
    """
    PCA basis experiment: apply Rosenblatt noise in the top-k PCA directions
    of the pixel covariance, then reconstruct to pixel space.
    
    Answers: is PCA-rotated pixel space a better basis for Rosenblatt noise?
    Mathematically: X_t = x0 + sigma(t) * V_k * diag(lambda_k^{-1/2}) * V_k^T * eps
    where V_k are the top-k eigenvectors of Cov(x0).
    This applies more noise in low-variance PCA directions (rare features)
    and less in high-variance directions (common features).
    """
    save_dir = f"{cfg.save_dir}/pca_basis"
    # ── 1. Compute PCA from training set ──────────────────────────────────
    print("Computing PCA basis from training set...")
    ds_tr  = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    n_pca  = min(5000, len(ds_tr))
    loader = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=2)
    imgs   = []
    for x, _ in loader:
        imgs.append(x.view(x.size(0), -1))     # flatten: (B, 784)
        if sum(i.size(0) for i in imgs) >= n_pca: break
    X = torch.cat(imgs, 0)[:n_pca]             # (n_pca, 784)
    mu = X.mean(0)                              # (784,)
    X_c = X - mu                               # centred
    # SVD on centred data (more stable than eigendecomposition)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    V_k  = Vt[:cfg.k_components].T                 # (784, k)  top-k eigenvectors
    # Variance explained in each direction
    proj = X_c @ V_k                           # (n_pca, k)
    lam_k = proj.var(0).clamp(min=1e-6)        # (k,)  variance per direction
    print(f"  PCA: top-{cfg.k_components} components explain "
          f"{(lam_k.sum() / X_c.var(1).sum() * 100 * cfg.k_components / 784):.1f}% pixel variance")

    # Whitening scale in PCA space: sigma_pca_i = 1/sqrt(lam_i), normalised
    scale_pca = 1. / lam_k.sqrt()             # (k,)
    scale_pca = scale_pca / scale_pca.mean()

    # Move to device
    V_k       = V_k.to(cfg.device)
    mu_d      = mu.to(cfg.device)
    scale_d   = scale_pca.to(cfg.device)

    # ── 2. Build a Sigma-fn that operates in PCA space ────────────────────
    # We implement a custom forward:
    #   corrupt: z_t = x0 + V_k * diag(sigma(t)*scale_pca) * eps_k
    #   where eps_k ~ Z_D^{k}  in the k-dim PCA subspace

    # For FID comparison: also run standard pixel-space
    results = {}
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real    = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    for nt in ("gaussian", "rosenblatt"):
        results[nt] = {}
        for basis in ("pixel", "pca"):
            print(f"\n{'='*60}\nPCA basis exp: noise={nt}  basis={basis}")
            if basis == "pixel":
                if cfg.base_line == "multiplicative":
                    sfn = sigma_multiplicative()    # standard pixel-space
                    rd  = str(OUT_ROOT / "multiplicative")
                elif cfg.base_line == "pca_whitened_global":
                    global_var = compute_global_pixel_variance(cfg.dataset) 
                    sfn = sigma_pca_whitened_global(global_var)  # global variance whitening
                    rd  = str(OUT_ROOT / "pca_whitened_global")
                else:
                    raise ValueError(f"Unknown base_line: {base_line}")
            else:
                # Anisotropic in PCA basis: back-project scale to pixel space
                # Effective per-pixel scale: A_i = sum_j V_{ij}^2 * scale_j
                A_pixel = (V_k.cpu() ** 2) @ scale_pca.unsqueeze(1)   # (784, 1)
                A_img   = A_pixel.view(1, 28, 28) / A_pixel.mean()
                sfn     = sigma_anisotropic(mode="h_emphasis")         # placeholder
                # Override with PCA-back-projected scale
                _A = A_img.clone()
                def _pca_fn(x0, _A=_A):
                    return _A.to(x0.device).expand_as(x0)
                _pca_fn.__name__    = "pca_basis"
                _pca_fn.label       = rf"PCA basis ($k={cfg.k_components}$)"
                _pca_fn.eg2         = float((_A ** 2).mean())
                _pca_fn.needs_label = False
                sfn = _pca_fn
                rd = f"{save_dir}/pca_basis"
    
            Path(rd).mkdir(parents=True, exist_ok=True)
            model, fwd = train(sfn, cfg, noise_type=nt, H=cfg.H, save_dir=rd)
            model.eval()

            if cfg.EVALUATE:
                metrics = evaluate_model(model, fwd, real, test_ds, cfg, bridge=bridge)
                results[nt][basis] = metrics
                print(f"  noise={nt:10s} basis={basis:5s}  FID={metrics['FID']}   fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
            
            if cfg.PLOT:
                _restoration_grid(model, fwd, cfg, rd, tag=f"{basis}_{nt}", bridge=bridge)
    
    if cfg.EVALUATE:
        print(f"\nPCA basis summary:")
        for nt in results:
            for basis, metrics in results[nt].items():
                print(f"  noise={nt:10s} basis={basis:5s}  FID={metrics['FID']}   fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 13. Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_comparison(H: float = 0.7, n_mc: int = 20_000,
                           save_path: str = None) -> None:
    save_path = save_path or str(OUT_ROOT / "noise" / "noise_comparison.png")
    x_lp, d_lp = get_rosenblatt_density(H=H, K=200)
    x_vt, d_vt = get_vt_density(H=H)

    D   = 1. - H
    lam = eigenvalues_LP(a=D, K=200)
    lam = lam / np.sqrt(max(2. * np.sum(lam ** 2), 1e-12))
    z_mc = (np.random.randn(n_mc, len(lam)) ** 2 - 1.) @ lam

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(x_lp, d_lp, "r-",  lw=2, label=f"Rosenblatt LP (H={H})")
    ax.plot(x_vt, d_vt, "m--", lw=1.5, label="Rosenblatt VT")
    xg = np.linspace(-5, 8, 400)
    ax.plot(xg, np.exp(-xg**2/2)/np.sqrt(2*np.pi), "b--", lw=2, label="N(0,1)")
    ax.hist(z_mc, bins=150, density=True, alpha=0.3, color="red", label="MC")
    ax.set(xlabel="x", ylabel="density", title="Rosenblatt vs Gaussian")
    ax.legend();  ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(x_lp[x_lp > 0.5], d_lp[x_lp > 0.5], "r-", lw=1.5, label="Rosenblatt")
    xp = np.linspace(0.5, 7, 200)
    ax.semilogy(xp, np.exp(-xp**2/2)/np.sqrt(2*np.pi), "b--", lw=1.5, label="Gaussian")
    ax.set(xlabel="|x|", ylabel="log density", title="Tail comparison")
    ax.legend();  ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close();plt.show()
    print(f"  Saved {save_path}")


def plot_rosenblatt_paths(H: float = 0.7, n_paths: int = 5,
                           save_path: str = None) -> None:
    save_path = save_path or str(OUT_ROOT / "path" / "rosenblatt_paths.png")
    t_arr, paths = simulate_rosenblatt_paths(H=H, n_paths=n_paths, n_pts=200)
    _, mc        = simulate_rosenblatt_paths(H=H, n_paths=500, n_pts=2)
    x_lp, d_lp  = get_rosenblatt_density(H=H, K=200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap      = plt.get_cmap("tab10")
    for i, p in enumerate(paths):
        axes[0].plot(t_arr, p, alpha=0.8, lw=1.2, color=cmap(i), label=f"path {i+1}")
    axes[0].set(xlabel="t", ylabel=r"$Z_t^{H,2}$", title=f"Rosenblatt paths (H={H})")
    axes[0].legend(fontsize=8);  axes[0].grid(alpha=0.3)

    axes[1].hist(mc[:, -1], bins=60, density=True, alpha=0.4, color="red", label="MC at t=1")
    axes[1].plot(x_lp, d_lp, "r-", lw=2, label="Exact LP density")
    axes[1].set(xlabel=r"$Z_1^{H,2}$", ylabel="density", title="Marginal density at t=1")
    axes[1].legend();  axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close();plt.show()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_bridge(
        cfg:        Config,
        noise_type: str = "rosenblatt",
) -> dict:
    """
    Train one model with the stochastic bridge, then evaluate generation
    quality under all three bridge strategies on that same model.
    This isolates the effect of re-corruption from training differences.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    sfn = sigma_multiplicative()
    model, forward = train(sfn, cfg, noise_type=noise_type, H=cfg.H, save_dir=save_dir)
    model.eval()

    results = {}
    for bridge in ("stochastic", "hybrid"):
        print(f"Evaluating bridge strategy: {bridge}")
        if cfg.EVALUATE:        
            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
            print(f"  bridge={bridge}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}  Eval time={metrics['eval_time_s']:.1f}s")        
            results[bridge] = metrics
        
        if cfg.PLOT:
            _restoration_grid(model, forward, cfg, save_dir,
                            tag=f"bridge_{bridge}", bridge=bridge)

    if cfg.EVALUATE:
        print(f"\nBridge ablation summary:")
        for t, m in sorted(results.items()):
            print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    return results


def run_ablation_noise(cfg: Config) -> dict:
    """
    Compare Gaussian vs Rosenblatt noise under the same Sigma (multiplicative)
    and same bridge (stochastic).  Each noise type trains its own model
    because the training distribution differs.

    Key theoretical distinction:
      Gaussian  — score-matching would work (Tweedie's formula holds)
      Rosenblatt — Tweedie fails; cold diffusion is a structural necessity
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    for noise_type in ("gaussian", "rosenblatt"):
        bridge = cfg.bridge
        sfn     = sigma_multiplicative()
        print(f"\n{'='*60}\nNoise ablation: noise_type={noise_type}")
        model, forward = train(sfn, cfg, noise_type=noise_type, H=cfg.H, save_dir=str(save_dir))
        model.eval()

        if cfg.EVALUATE:
            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
            print(f"  noise={noise_type:<12s}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}  Eval time={metrics['eval_time_s']:.1f}s")
            results[noise_type] = metrics

        if cfg.PLOT:
            _restoration_grid(model, forward, cfg, save_dir, tag=f"noise_{noise_type}", bridge=bridge)
    
    if cfg.EVALUATE:
        print(f"\nNoise ablation summary:")
        for t, m in sorted(results.items()):
            print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    return results


def run_ablation_H(cfg: Config, H_values: list[float] = None) -> dict:
    """
    Ablate the parameter H while matching the noise scale schedule exactly.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    if H_values is None: H_values = [0.6, 0.7, 0.8, 0.9]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    for noise_type in ("gaussian", "rosenblatt"):
        results[noise_type] = {}
        for H_val in H_values:
            print(f"\nExp H ablation: noise={noise_type} H={H_val}")
            # The schedule sigma(t) = sigma_max * t^H is managed inside RosenblattForward.
            # Even for Gaussian, it will use t^H.
            sfn = sigma_multiplicative()
            model, forward = train(sfn, cfg, noise_type=noise_type, H=H_val, save_dir=save_dir)
            model.eval()

            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=cfg.bridge)
            results[noise_type][H_val] = metrics
            
            print(f"  {noise_type:10s} H={H_val}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
            
            _restoration_grid(model, forward, cfg, save_dir,
                              tag=f"{noise_type}_H{H_val}", bridge=cfg.bridge)

    print("\nH ablation summary:")
    for noise_type in results:
        for H_val, m in results[noise_type].items():
            print(f"  {noise_type:10s} H={H_val}  FID={m['FID']}  fFID={m.get('fFID', 0)}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m.get('LPIPS', 0)}  Eval time={m['eval_time_s']:.1f}s")
    return results


def evaluate_all_models_fid(cfg: Config) -> dict:
    """
    Scan a root directory recursively for completed model checkpoints
    (*_final.pt), load each one, and compute its FID.
    """
    root_path = Path(cfg.save_dir)
    
    if not root_path.exists():
        print(f"Directory {cfg.save_dir} not found.")
        return {}

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    model_files = list(root_path.rglob("*_final.pt"))
    print(f"Found {len(model_files)} models in {cfg.save_dir}")
    
    class_vars = None # Lazy load for PCA
    eg2_cache = {}    # Cache E[Sigma^2] to avoid redundant dataloader overhead
    results = {}    

    # --- Initialize time statistics ---
    total_load_time = 0.0
    total_fid_time = 0.0

    for ckpt_path in model_files:
        raw_tag = ckpt_path.stem.replace("_final", "")
        # Differentiate same model filenames in different folders
        tag = f"{ckpt_path.parent.name}/{raw_tag}"
        
        # Skip Autoencoders and Latent models (they don't use ConditionalUNet image backbone)
        if "ae" in raw_tag.lower() or "latent" in raw_tag.lower() or "lat_" in raw_tag.lower():
            print(f"Skipping non-UNet model: {tag}")
            continue
            
        # PCA basis models rely on a dynamic lambda closure for SigmaFn
        if "pca_basis" in raw_tag.lower():
            print(f"Skipping dynamic PCA basis model in general eval: {tag}")
            continue

        # Deduce settings from tag (e.g., rosenblatt_sigma_multiplicative_H0.7)               
        noise_type = "rosenblatt" if "rosenblatt" in raw_tag else "gaussian"
        
        import re
        match_H = re.search(r'H([0-9.]+)', ckpt_path.name)
        H = float(match_H.group(1)) if match_H else 0.7

        if "bridge" in raw_tag:
            bridge = raw_tag.split('_')[1]
        else:
            bridge = cfg.bridge

        # Deduce Sigma Fn
        sfn = sigma_multiplicative()
        if "pca_whitened" in raw_tag:
            if class_vars is None: class_vars = compute_pixel_variance(cfg.dataset)
            sfn = sigma_pca_whitened(class_vars)
        elif "anisotropic" in tag and "h_emphasis" in tag:
            sfn = sigma_anisotropic("h_emphasis")
        elif "anisotropic" in tag and "v_emphasis" in tag:
            sfn = sigma_anisotropic("v_emphasis")
        elif "edge_aware" in tag:
            sfn = sigma_edge_aware()

        print(f"\nEvaluating: {tag}, bridge = {bridge}, H = {H}, noise type = {noise_type}")
        
        # --- Start Timer for Loading Model ---
        t0_load = time.time()
        model = ConditionalUNet(num_classes=10, base_ch=cfg.base_ch).to(cfg.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
        model.eval()

        forward = RosenblattForward(sfn, noise_type=noise_type, H=H, device=cfg.device)
        t_load_elapsed = time.time() - t0_load
        total_load_time += t_load_elapsed
        # --- End Timer for Loading Model ---
        
        sfn_name = sfn.__name__
        if sfn_name not in eg2_cache:
            eg2_cache[sfn_name] = _estimate_eg2(sfn, cfg.dataset)
        forward.set_eg2(eg2_cache[sfn_name])

        metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
        results[tag] = metrics
        total_fid_time += time.time() - t0_load - t_load_elapsed
        print(f"  {tag}: FID={metrics['FID']} fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  "
            f"SSIM={metrics['SSIM']}  LPIPS={metrics['LPIPS']}  ({metrics['eval_time_s']}s)")

    print("\nBatch Evaluation Summary:")
    for t, m in sorted(results.items()):
        print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    print(f"\nTime Statistics:")
    print(f"  Total time for loading models: {total_load_time:.2f} seconds")
    print(f"  Total time for generation & FID: {total_fid_time:.2f} seconds")
    
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 15. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rosenblatt Cold Diffusion — Unified")
    parser.add_argument("--mode",                 default="all",
                        choices=["all", "noise_plot", "path_plot", "pca_basis", 
                                 "sigma_comparison", "exp_latent", "evaluate_all",
                                 "ablation", "ablation_bridge", "ablation_noise", "ablation_H"])
    parser.add_argument("--dataset",               default="FashionMNIST", choices=["FashionMNIST", "MNIST"])
    parser.add_argument("--epochs",    type=int,   default=None)
    parser.add_argument("--noise",                 default="rosenblatt", choices=["gaussian", "rosenblatt"])
    parser.add_argument("--H",         type=float, default=None)
    parser.add_argument("--n_fid",     type=int,   default=None)
    parser.add_argument("--bridge",    type=str,   default="stochastic", choices=["stochastic", "hybrid"])
    parser.add_argument("--save_dir",              default=str(OUT_ROOT))
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--EVALUATE",  type=bool,  default=None)
    parser.add_argument("--PLOT",      type=bool,  default=None)
    parser.add_argument("--baseline",  type=str,   default=None)

    parser = build_parser(parser)
    args = parser.parse_args()

    cfg = Config()                     
    cfg = update_config_from_args(cfg, args)
    print(json.dumps(asdict(cfg), indent=2, default=str))

    if args.mode == "evaluate_all":
        evaluate_all_models_fid(cfg)
        run_exp_latent(cfg)

    if args.mode in ("noise_plot"):
        plot_noise_comparison(H=args.H)

    if args.mode in ("path_plot"):
        plot_rosenblatt_paths(H=args.H)

    if args.mode in ("sigma_comparison", "all"):
        run_sigma_comparison(cfg)

    if args.mode in ("exp_latent", "all"):
        run_exp_latent(cfg)

    if args.mode in ("pca_basis", "all"):
        run_exp_pca_basis(cfg)

    if args.mode in ("ablation_bridge", "ablation", "all"):
        run_ablation_bridge(cfg)

    if args.mode in ("ablation_noise", "ablation", "all"):
        run_ablation_noise(cfg)

    if args.mode in ("ablation_H", "ablation", "all"):
        run_ablation_H(cfg)
        
    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
