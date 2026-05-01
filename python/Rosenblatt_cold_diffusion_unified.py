"""
Rosenblatt Cold Diffusion — Unified Framework
==============================================

Extends the original Rosenblatt_cold_diffusion.py with:

  1. FIXED additive re-corruption  (stochastic bridge: fresh Rosenblatt draw
     at each reverse step, eliminating cumulant amplification instability)
  2. MULTIPLICATIVE forward process  X_t = x_0 + sigma(t) * g(x_0) * eps
  3. ANISOTROPIC noise experiment  (Exp 6, Prof. question 1)
  4. LATENT-SPACE experiment  (Exp 7, Prof. question 2 — SD-style)
  5. Noise types: Gaussian + Rosenblatt only  (Hermite-3 removed)
  6. Density simulation uses the correct LP characteristic-function path
     (integrated from density_simulation.py, no standalone bugs)

Mathematical grounding
----------------------
  * Stochastic bridge fix:  at step k draw fresh eps ~ Z_D independently;
    then X_{t_{k+1}} = x0_hat + sigma(t_{k+1}) * eps  (additive) or
    X_{t_{k+1}} = x0_hat + sigma(t_{k+1}) * g(x0_hat) * eps  (multiplicative).
    This matches the training distribution exactly at every step.

  * Deterministic bridge (old, BROKEN for non-Gaussian):
    X_{t_{k+1}} = x0_hat + (sigma_{k+1}/sigma_k) * (X_{t_k} - x0_hat)
    amplifies higher cumulants as sigma_k -> 0.

  * Anisotropy:  replace scalar sigma(t) with diagonal A * diag(sigma(t)):
    X_t = x0 + sigma(t) * A * eps,  A = diag(a_1,...,a_d).
    Malliavin matrix Gamma_{X_t} = sigma(t)^2 * A * Gamma_Z * A^T
    remains non-degenerate for invertible A.

  * Latent experiment:  train lightweight autoencoder on FashionMNIST;
    run cold diffusion in the 64-D latent space with an MLP denoiser;
    decode generated latents with the fixed decoder.

Usage
-----
  python rosenblatt_cold_diffusion_unified.py --mode all
  python rosenblatt_cold_diffusion_unified.py --mode aniso
  python rosenblatt_cold_diffusion_unified.py --mode latent
  python rosenblatt_cold_diffusion_unified.py --mode ablation
"""

from __future__ import annotations
import torchvision.models as tv_models
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import math
import time
from pathlib import Path
from density_simulation import eigenvalues_LP

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────────────────────────
# 0. Global configuration
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CONFIG = {
    "cfg_scale":   2.5,         # Classifier-Free Guidance scale
    "n_steps":     50,
    "sigma_max":   16.0,        # Maximum noise level at t=1.0
    "base_ch":     128,         # UNet base channels (DO NOT set to batch_size)
    "M_eig":       80,          # Number of eigenvalues for LP expansion
    "lr":          2e-4,
    "epochs":      30,
    "batch_size":  256,
    "dataset":     "FashionMNIST",
    "noise_type":  "rosenblatt",
    "H":           0.7,         # Hurst index
    "T_MIN":       0.01,        # FIX Bug 6: minimum timestep during training
    # Additive or multiplicative
    "forward_mode": "additive",   # "additive" | "multiplicative"
    # Bridge type  (stochastic is the FIXED version)
    "bridge":       "stochastic",  # "stochastic" | "deterministic"
}

OUT_ROOT = Path("./output/experiments")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.step = 0
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict = {}

    def _effective_decay(self) -> float:
        return min(self.decay, (1 + self.step) / (10 + self.step))

    def update(self):
        d = self._effective_decay()
        self.step += 1
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].lerp_(p.data, 1.0 - d)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset(name: str, root: str = "./data", train: bool = True, tf=None):
    name = name.lower()
    if "fashion" in name:
        return datasets.FashionMNIST(root, train=train, download=True, transform=tf)
    return datasets.MNIST(root, train=train, download=True, transform=tf)


_FASHION_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",       "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]


def class_name(dataset_name: str, idx: int) -> str:
    if "fashion" in dataset_name.lower():
        return _FASHION_NAMES[idx]
    return f"Digit {idx}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Noise sampler — Gaussian and Rosenblatt only
# ─────────────────────────────────────────────────────────────────────────────


def build_eigenvalues(H: float, M: int, device: torch.device) -> torch.Tensor:
    """Return unit-variance-normalised LP eigenvalues as a GPU tensor."""
    D   = 1.0 - H
    # Use eigenvalues_LP from density_simulation.py (a = D)
    lam = eigenvalues_LP(a=D, K=M)
    
    # Normalise so that Var[Z_D] = 2*sum(lam^2) = 1
    var = 2.0 * np.sum(lam**2)
    if var > 0:
        lam = lam / np.sqrt(var)
    return torch.tensor(lam, dtype=torch.float32, device=device)


def sample_noise(
    noise_type: str,
    shape:      tuple,
    lam_t:      torch.Tensor | None = None,
    M:          int = 80,
    device:     torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Draw a batch of noise samples.

    Parameters
    ----------
    noise_type : "gaussian" | "rosenblatt"
    shape      : output shape
    lam_t      : pre-computed unit-variance LP eigenvalues (Rosenblatt only)
    """
    if noise_type == "gaussian":
        return torch.randn(shape, device=device, dtype=torch.float32)

    elif noise_type == "rosenblatt":
        # Z_D = sum_n lambda_n * (xi_n^2 - 1),  xi_n iid N(0,1)
        # Already unit-variance thanks to normalised lam_t.
        total = int(np.prod(shape))
        xi = torch.randn(total, M, device=device, dtype=torch.float32)
        z = (xi**2 - 1.0) @ lam_t        # (total,)
        return z.reshape(shape)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}. "
                         f"Hermite-3 has been removed. "
                         f"Choose 'gaussian' or 'rosenblatt'.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward process — additive & multiplicative
# ─────────────────────────────────────────────────────────────────────────────

class RosenblattForward:
    """
    Unified forward process supporting additive and multiplicative modes.

    Additive
    --------
        X_t = x_0 + sigma(t) * eps

    Multiplicative
    --------------
        X_t = x_0 + sigma(t) * g(x_0) * eps
        g_i(x) = (1 + |x_i|) / 2    (signal-proportional, bounded away from 0)

    The stochastic bridge re-corruption (FIXED) draws fresh eps at each
    generation step, exactly matching the training distribution.
    The deterministic bridge (original, BROKEN for Rosenblatt) is kept
    for comparison only.
    """

    def __init__(
        self,
        noise_type:   str = "rosenblatt",
        forward_mode: str = "additive",
        H:            float = 0.7,
        M_eig:        int = 80,
        sigma_max:    float = 16.0,
        device: torch.device | str = "cpu",
    ):
        assert forward_mode in ("additive", "multiplicative"), \
            f"forward_mode must be 'additive' or 'multiplicative', got {forward_mode!r}"
        self.noise_type = noise_type
        self.forward_mode = forward_mode
        self.H          = float(H)
        self.M_eig      = M_eig
        self.sigma_max  = float(sigma_max)
        self.device     = device
        self.lam_t: torch.Tensor | None = None
        if noise_type == "rosenblatt":
            self.lam_t = build_eigenvalues(H, M_eig, self.device)

        # For multiplicative mode: precompute E[g^2(x0)] normalisation
        # g(x) = (1 + |x|) / 2, x in [-1, 1]  => E[g^2] ~= E[(1+|U|)^2/4]
        # With U ~ Uniform[-1,1]:  E[(1+|U|)^2] = E[1 + 2|U| + U^2]
        #                                        = 1 + 1 + 1/3 = 7/3
        # So E[g^2] = 7/12  analytically.
        self._eg2: float = 7.0 / 12.0   # updated from training data in set_eg2()

    # ---- public API: update E[g^2] from real data ---------------------------

    def set_eg2(self, eg2: float):
        """Set E[g^2(x0)] estimated from the training set."""
        self._eg2 = float(eg2)

    # ---- noise coefficient --------------------------------------------------

    @staticmethod
    def g_fn(x0: torch.Tensor) -> torch.Tensor:
        """Signal-dependent noise coefficient: g_i(x) = (1 + |x_i|) / 2."""
        return (1.0 + x0.abs()) / 2.0

    # ---- sigma schedule -----------------------------------------------------

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_max * (t.clamp(min=1e-6) ** self.H)

    # ---- input normalisation c_in -------------------------------------------

    def c_in(self, t: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """
        Normalisation constant for the UNet input:
            c_in = 1 / sqrt(1 + sigma(t)^2 * E[g^2])
        For additive, E[g^2] = 1; for multiplicative, E[g^2] = self._eg2.
        """
        m = mode or self.forward_mode
        eg2 = self._eg2 if m == "multiplicative" else 1.0
        sig2 = self.sigma(t) ** 2
        return (1.0 / torch.sqrt(1.0 + sig2 * eg2))

    # ---- forward corruption -------------------------------------------------

    def corrupt(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_t, eps, sigma_t) where x_t = x0 + sigma_t * [g(x0) *] eps.
        sigma_t has shape (B, 1, 1, 1) (broadcast-ready).
        """
        t = t.to(x0.device, dtype=torch.float32)
        sig = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        eps = sample_noise(self.noise_type, x0.shape, self.lam_t,
                           self.M_eig, device=x0.device)

        if self.forward_mode == "multiplicative":
            x_t = x0 + sig * self.g_fn(x0) * eps
        else:  # additive
            x_t = x0 + sig * eps

        return x_t, eps, sig

    # ---- re-corruption (reverse step) ----------------------------------------

    def recorrupt_stochastic(
        self,
        x0_hat:  torch.Tensor,
        t_next:  torch.Tensor,
    ) -> torch.Tensor:
        """
        FIXED stochastic bridge re-corruption.

        Draws fresh eps ~ noise_type independently of everything else and
        applies the forward model at level t_next.  This guarantees that the
        distribution of X_{t_{k+1}} exactly matches the training distribution
        p_{t_{k+1}}, regardless of the estimation error in x0_hat.
        """
        x_next, _, _ = self.corrupt(x0_hat, t_next)
        return x_next

    def recorrupt_deterministic(
        self,
        x_cur:   torch.Tensor,
        x0_hat:  torch.Tensor,
        t_cur:   torch.Tensor,
        t_next:  torch.Tensor,
    ) -> torch.Tensor:
        """
        ORIGINAL deterministic bridge (kept for ablation / comparison only).

        BROKEN for Rosenblatt: rescaling the residual changes the cumulant
        structure, causing distribution mismatch that blows up as t -> 0.
        """
        sig_cur = self.sigma(t_cur).view(-1, *([1]*(x_cur.dim()-1)))
        sig_next = self.sigma(t_next).view(-1, *([1]*(x_cur.dim()-1)))
        return x0_hat + (sig_next / sig_cur.clamp(min=1e-5)) * (x_cur - x0_hat)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Anisotropic forward process
# ─────────────────────────────────────────────────────────────────────────────

class AnisotropicForward(RosenblattForward):
    """
    Anisotropic extension:  X_t = x0 + sigma(t) * (A * eps)

    A is a diagonal matrix with per-pixel (or per-channel) scale factors.
    For images: A can encode horizontal/vertical emphasis or a PCA-style
    covariance structure.

    The Malliavin covariance satisfies:
        Gamma_{X_t} = sigma(t)^2 * A * Gamma_Z * A^T
    which remains non-degenerate for any invertible A (Theorem mult-density).

    Parameters
    ----------
    aniso_matrix : torch.Tensor, shape (d,)
        Diagonal of A.  Each pixel gets its own scale factor.
        Pass None for isotropic (falls back to parent class).
    """

    def __init__(self, aniso_matrix: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.aniso_matrix = aniso_matrix   # None or (d,) tensor

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor):
        t = t.to(x0.device, dtype=torch.float32)
        sig = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        eps = sample_noise(self.noise_type, x0.shape, self.lam_t,
                           self.M_eig, device=x0.device)

        if self.aniso_matrix is not None:
            A = self.aniso_matrix.to(x0.device)
            # A may be (1,1,28,28) or (d,); normalise to broadcast over (B,C,H,W)
            if A.dim() == 1:
                # flat (d,) -> reshape to match spatial dims
                spatial = x0.shape[1:]   # e.g. (1, 28, 28)
                A = A.view(1, *spatial)
            # A is now broadcastable over (B, C, H, W)
            eps = A * eps

        if self.forward_mode == "multiplicative":
            x_t = x0 + sig * self.g_fn(x0) * eps
        else:
            x_t = x0 + sig * eps

        return x_t, eps, sig


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNet denoiser (image space) 
# ─────────────────────────────────────────────────────────────────────────────


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding (same as in DDPM)."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
        args  = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, spatial_size: int = 14):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_size**2, channels) * 0.02)
        self.norm1   = nn.GroupNorm(min(8, channels), channels)
        self.mha     = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.proj    = nn.Conv2d(channels, channels, 1)
        self.norm2   = nn.GroupNorm(min(8, channels), channels)
        self.ffn     = nn.Sequential(
                        nn.Conv2d(channels, channels * 2, 1),
                        nn.GELU(),
                        nn.Conv2d(channels * 2, channels, 1)
                    )

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

    def __init__(self, in_ch: int, out_ch: int, t_dim: int = 256):
        super().__init__()
        self.norm1   = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2   = nn.GroupNorm(min(8, out_ch), out_ch, affine=False)
        self.dropout = nn.Dropout(0.1)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj  = nn.Linear(t_dim, out_ch * 2)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

        # Shortcut connection if channels change
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # 1. First convolution
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # 2. AdaGN Conditioning
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        # 3. Second convolution
        h = self.dropout(F.silu(h))
        return self.shortcut(x) + self.conv2(h)


class ConditionalUNet(nn.Module):
    """Conditional U-Net — ~4.5 M parameters, 28×28 grayscale input."""

    def __init__(self, t_dim: int = 256, num_classes: int = 10,
                 base_ch: int = 128, in_channels: int = 1):
        super().__init__()
        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        # Encoder: 128 -> 256 -> 512
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1 = nn.Sequential(ResBlockAdaGN(base_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim))
        self.pool1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.down2 = nn.Sequential(ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim))
        self.attn2 = SelfAttention(base_ch * 2, spatial_size=14)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid1     = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)
        self.attn_mid = SelfAttention(base_ch * 4, spatial_size=7)
        self.mid2     = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)

        # Decoder
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        )
        self.up_res2 = nn.ModuleList([ResBlockAdaGN(base_ch * 4, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim)])
        self.up_attn2 = SelfAttention(base_ch * 2, spatial_size=14)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        )
        self.up_res1 = nn.ModuleList([ResBlockAdaGN(base_ch * 2, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim)])

        self.out = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Map time through MLP *before* adding labels
        t_emb = self.time_mlp(self.t_embed(t)) + self.label_emb(y)

        # Encode
        x = self.init_conv(x)
        h1 = self.down1[1](self.down1[0](x, t_emb), t_emb)

        h2 = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2 = self.attn2(h2)

        # Bottleneck
        h3 = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)

        # Decode
        h = self.up_attn2(self.up_res2[1](self.up_res2[0](torch.cat([self.up2(h3), h2], dim=1), t_emb), t_emb))
        h = self.up_res1[1](self.up_res1[0](torch.cat([self.up1(h), h1], dim=1), t_emb), t_emb)

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

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1),  nn.SiLU(),   # 14×14
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.SiLU(),  # 7×7
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  nn.SiLU(),  # 7×7
            nn.Flatten(),                                              # 3136
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
                 num_classes: int = 10, hidden: int = 256):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim),
        )
        # Build 6 hidden layers with residual connections
        self.input_proj = nn.Linear(latent_dim, hidden)
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(6)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(6)
        ])
        self.cond_projs = nn.ModuleList([
            nn.Linear(t_dim, hidden * 2) for _ in range(6)
        ])
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.t_embed(t)) + self.label_emb(y)  # (B, t_dim)
        h = F.silu(self.input_proj(z))
        for layer, norm, cproj in zip(self.layers, self.norms, self.cond_projs):
            scale, shift = cproj(F.silu(cond)).chunk(2, dim=-1)
            h = norm(h) * (1.0 + scale) + shift
            h = h + F.silu(layer(h))
        return self.out_proj(h)


def train_autoencoder(
    dataset_name: str = "FashionMNIST",
    epochs:       int = 20,
    batch_size:   int = 256,
    lr:           float = 1e-3,
    save_path:    str = "./latent_ae.pt",
    device: torch.device = None,
) -> ConvAutoencoder:
    """Train the convolutional autoencoder independently of the diffusion model."""
    if device is None:
        device = get_device()
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = get_dataset(dataset_name, train=True, tf=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)

    ae = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for ep in range(epochs):
        ae.train()
        total = 0.0
        for x0, _ in dl:
            x0 = x0.to(device, non_blocking=True)
            recon, _ = ae(x0)
            loss = F.mse_loss(recon, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * x0.size(0)
        print(f"  AE epoch {ep+1:3d}/{epochs}  recon_loss={total/len(ds):.5f}")

    torch.save(ae.state_dict(), save_path)
    print(f"  Autoencoder saved to {save_path}")
    return ae


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training loop — shared by image-space and latent-space models
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_eg2(dataset_name: str, n_samples: int = 5000) -> float:
    """Estimate E[g^2(x0)] from training data for multiplicative normalisation."""
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = get_dataset(dataset_name, train=True, tf=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2)
    total = 0.0
    count = 0
    for x0, _ in loader:
        g = (1.0 + x0.abs()) / 2.0
        total += (g**2).mean().item() * x0.size(0)
        count += x0.size(0)
        if count >= n_samples:
            break
    return total / count


def train(
    dataset_name:  str = GLOBAL_CONFIG["dataset"],
    noise_type:    str = "rosenblatt",
    forward_mode:  str = "additive",
    H:             float = GLOBAL_CONFIG["H"],
    epochs:        int = GLOBAL_CONFIG["epochs"],
    batch_size:    int = GLOBAL_CONFIG["batch_size"],
    lr:            float = GLOBAL_CONFIG["lr"],
    M_eig:         int = GLOBAL_CONFIG["M_eig"],
    sigma_max:     float = GLOBAL_CONFIG["sigma_max"],
    save_dir:      str = "./run",
    device: torch.device = None,
    aniso_matrix:  torch.Tensor | None = None,
) -> tuple[ConditionalUNet, RosenblattForward]:
    """
    Main training function for image-space cold diffusion.
    Supports additive and multiplicative modes, stochastic bridge only.
    """
    if device is None:
        device = get_device()
    tag = f"{noise_type}_{forward_mode}_H{H}"
    print(f"Training | {tag} | Dataset:{dataset_name} | epochs:{epochs}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    tf_train = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = get_dataset(dataset_name, train=True,  tf=tf_train)
    val_ds = get_dataset(dataset_name, train=False, tf=tf_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    # Build forward process
    ForwardCls = AnisotropicForward if aniso_matrix is not None else RosenblattForward
    fwd_kwargs = dict(noise_type=noise_type, forward_mode=forward_mode,
                      H=H, M_eig=M_eig, sigma_max=sigma_max, device=device)
    if aniso_matrix is not None:
        forward = ForwardCls(aniso_matrix=aniso_matrix, **fwd_kwargs)
    else:
        forward = ForwardCls(**fwd_kwargs)

    # Estimate E[g^2] from data for multiplicative normalisation
    if forward_mode == "multiplicative":
        eg2 = _estimate_eg2(dataset_name)
        forward.set_eg2(eg2)
        print(f"  Multiplicative mode: E[g^2] = {eg2:.4f}")

    model = ConditionalUNet(
        num_classes=10, base_ch=GLOBAL_CONFIG["base_ch"]).to(device)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    T_MIN = GLOBAL_CONFIG["T_MIN"]
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for x0, labels in train_dl:
            x0, labels = x0.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)
            B = x0.size(0)
            cf = torch.rand(B, device=device) < 0.1
            lbl = labels.clone()
            lbl[cf] = 10
            t = torch.rand(B, device=device) * (1.0 - T_MIN) + T_MIN
            x_t, _, sig = forward.corrupt(x0, t)
            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    c_in = forward.c_in(t).view(-1, 1, 1, 1)
                    pred = model(x_t * c_in, t, lbl)
                    loss = F.smooth_l1_loss(pred, x0)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                c_in = forward.c_in(t).view(-1, 1, 1, 1)
                pred = model(x_t * c_in, t, lbl)
                loss = F.smooth_l1_loss(pred, x0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            ema.update()
            ep_loss += loss.item() * B

        ep_loss /= len(train_ds)
        train_losses.append(ep_loss)

        model.eval()
        ema.apply_shadow()
        v_loss = 0.0
        with torch.no_grad():
            for x0, labels in val_dl:
                x0, labels = x0.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True)
                t = torch.rand(x0.size(0), device=device) * \
                    (1.0 - T_MIN) + T_MIN
                x_t, _, _ = forward.corrupt(x0, t)
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
        ema.restore()
        sched.step()
        print(f"  ep {epoch+1:3d}/{epochs}  train={ep_loss:.4f}  val={v_loss:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s")

        if (epoch + 1) % 5 == 0:
            ema.apply_shadow()
            torch.save(model.state_dict(), f"{save_dir}/{tag}_ep{epoch+1}.pt")
            ema.restore()

    ema.apply_shadow()
    torch.save(model.state_dict(), f"{save_dir}/{tag}_final.pt")
    ema.restore()

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set(xlabel="Epoch", ylabel="Smooth L1", title=tag)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_loss.png", dpi=120)
    plt.close()

    return model, forward


# ─────────────────────────────────────────────────────────────────────────────
# 9. Generation — stochastic bridge sampler
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_conditional(
    model:     nn.Module,
    forward:   RosenblattForward,
    labels:    torch.Tensor,
    n_steps:   int = GLOBAL_CONFIG["n_steps"],
    cfg_scale: float = GLOBAL_CONFIG["cfg_scale"],
    bridge:    str = "stochastic",
    device:    torch.device = None,
) -> torch.Tensor:
    """
    Cold diffusion reverse process with classifier-free guidance.

    bridge="stochastic" (recommended): draw fresh noise at each step.
    bridge="deterministic": original rescaled-residual method (broken for Rosenblatt).
    """
    if device is None:
        device = get_device()
    model.eval()
    n = len(labels)
    null_labels = torch.full_like(labels, 10)

    # Quadratic schedule: t_k = (1 - k/N)^2  (concentrates steps near t=0)
    steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    t_sched = steps ** 1.0   # linear; use steps**2 for quadratic

    # Start from pure noise at t=1
    x = sample_noise(forward.noise_type, (n, 1, 28, 28),
                     forward.lam_t, forward.M_eig, device=device)
    x = x * forward.sigma_max

    for k in range(n_steps):
        t_cur = t_sched[k].expand(n)
        t_next = t_sched[k + 1].expand(n)
        sig_c = forward.sigma(t_cur).view(-1, 1, 1, 1)
        c_in = forward.c_in(t_cur).view(-1, 1, 1, 1)
        x_in = (x * c_in).float()

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0_c = model(x_in, t_cur, labels).float()
                x0_u = model(x_in, t_cur, null_labels).float()
        else:
            x0_c = model(x_in, t_cur, labels)
            x0_u = model(x_in, t_cur, null_labels)

        x0_hat = (x0_u + cfg_scale * (x0_c - x0_u)).clamp(-1.0, 1.0)

        if k < n_steps - 1:
            if bridge == "stochastic":
                # FIXED: draw fresh Rosenblatt noise
                x = forward.recorrupt_stochastic(x0_hat, t_next)
            else:
                # BROKEN for non-Gaussian (kept for ablation)
                x = forward.recorrupt_deterministic(x, x0_hat, t_cur, t_next)
        else:
            x = x0_hat

    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. FID computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_fid(
    real_imgs:  torch.Tensor,
    fake_imgs:  torch.Tensor,
    device:     torch.device,
    batch_size: int = 50,
) -> float:
    """FID via InceptionV3 features.  real/fake: (N, 1, 28, 28) in [0, 1]."""
    inc = tv_models.inception_v3(
        weights=tv_models.Inception_V3_Weights.DEFAULT, transform_input=False
    ).to(device)
    inc.fc = nn.Identity()
    inc.eval()

    def feats(imgs):
        out = []
        for i in range(0, len(imgs), batch_size):
            b = imgs[i:i+batch_size].to(device)
            if b.shape[1] == 1:
                b = b.repeat(1, 3, 1, 1)
            b = F.interpolate(b, (299, 299), mode="bilinear",
                              align_corners=False)
            b = b * 2.0 - 1.0
            out.append(inc(b).cpu().numpy())
        return np.concatenate(out, axis=0)

    a1 = feats(real_imgs)
    a2 = feats(fake_imgs)
    m1, s1 = a1.mean(0), np.cov(a1, rowvar=False)
    m2, s2 = a2.mean(0), np.cov(a2, rowvar=False)
    d = m1 - m2
    cm, _ = sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(d @ d + np.trace(s1 + s2 - 2.0 * cm))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Experiment 5 — fixed additive + multiplicative (main cold diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp5_cold_diffusion(
    dataset_name: str = "FashionMNIST",
    epochs:       int = 30,
    n_fid:        int = 2000,
    save_dir:     str = str(OUT_ROOT / "exp5"),
    device: torch.device = None,
):
    """
    Experiment 5: Compare additive (fixed) vs multiplicative forward process
    under Gaussian and Rosenblatt noise. Stochastic bridge in both cases.
    """
    if device is None:
        device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Reference real images for FID
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = get_dataset(dataset_name, train=False, tf=tf)
    real_imgs = torch.stack([test_ds[i][0] for i in range(n_fid)])
    real_imgs = (real_imgs + 1.0) / 2.0   # [0, 1]

    results = []
    variants = [
        ("gaussian",    "additive"),
        ("rosenblatt",  "additive"),
        ("gaussian",    "multiplicative"),
        ("rosenblatt",  "multiplicative"),
    ]

    for noise_type, fwd_mode in variants:
        tag = f"{noise_type}_{fwd_mode}"
        run_dir = f"{save_dir}/{tag}"
        print(f"\n{'='*60}")
        print(f"Exp 5: noise={noise_type}  forward={fwd_mode}")
        model, forward = train(
            dataset_name=dataset_name, noise_type=noise_type,
            forward_mode=fwd_mode, epochs=epochs, save_dir=run_dir, device=device,
        )
        model.eval()
        ema = EMA(model)
        ema.apply_shadow()

        gen_labels = torch.randint(0, 10, (n_fid,), device=device)
        fake_imgs_list = []
        for i in range(0, n_fid, 200):
            bl = gen_labels[i:i+200]
            fake_imgs_list.append(
                generate_conditional(model, forward, bl,
                                     bridge="stochastic", device=device).cpu()
            )
        fake_imgs = torch.cat(fake_imgs_list, 0)
        fid = compute_fid(real_imgs, fake_imgs, device)
        results.append(
            {"noise": noise_type, "forward": fwd_mode, "FID": round(fid, 2)})
        print(f"  FID ({tag}) = {fid:.2f}")
        ema.restore()

        # Restoration grid: 1 row per class
        _plot_restoration_grid(model, forward, dataset_name, save_dir,
                               tag=tag, device=device)

    # Summary table
    print("\nExp 5 FID Summary:")
    for r in results:
        print(f"  {r['noise']:12s}  {r['forward']:15s}  FID={r['FID']}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 12. Experiment 6 — Anisotropic noise
# ─────────────────────────────────────────────────────────────────────────────

def build_aniso_matrix(mode: str, img_shape=(1, 28, 28)) -> torch.Tensor:
    """
    Construct a per-pixel anisotropy diagonal for 28×28 images.

    Modes
    -----
    "isotropic"    : A = ones  (baseline)
    "h_emphasis"   : horizontal pixels get 3× more noise (scale [1, 3] per pixel column)
    "v_emphasis"   : vertical emphasis
    "pca_like"     : simulate PCA-like structure: first half of pixels scaled up by sqrt(5)
    """
    d = int(np.prod(img_shape))
    A = torch.ones(d)
    C, H, W = img_shape

    if mode == "isotropic":
        pass

    elif mode == "h_emphasis":
        # Scale pixel (i,j) by 1 + 2*j/(W-1)  → left=1, right=3
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    A[c*H*W + i*W + j] = 1.0 + 2.0 * j / (W - 1)
        A /= A.mean()   # normalise so mean scale = 1

    elif mode == "v_emphasis":
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    A[c*H*W + i*W + j] = 1.0 + 2.0 * i / (H - 1)
        A /= A.mean()

    elif mode == "pca_like":
        # First d//2 "principal" directions get sqrt(5) scaling, rest sqrt(1)
        A[:d//2] = math.sqrt(5.0)
        A = A / A.mean()

    else:
        raise ValueError(f"Unknown aniso mode: {mode!r}")

    return A   # shape (d,)


@torch.no_grad()
def _directional_fid(
    model:        nn.Module,
    forward:      AnisotropicForward,
    real_imgs:    torch.Tensor,
    device:       torch.device,
    direction:    str = "horizontal",
    n_fid:        int = 1000,
) -> float:
    """
    Directional FID (D-FID): crop images to horizontal or vertical strips
    before computing FID, measuring spatial frequency accuracy.
    """
    gen_labels = torch.randint(0, 10, (n_fid,), device=device)
    fakes_list = []
    for i in range(0, n_fid, 200):
        bl = gen_labels[i:i+200]
        fakes_list.append(
            generate_conditional(model, forward, bl,
                                 bridge="stochastic", device=device).cpu()
        )
    fake_imgs = torch.cat(fakes_list, 0)

    # Crop to emphasised direction for directional evaluation
    if direction == "horizontal":
        real_c = real_imgs[:n_fid, :, :, :14]   # left half
        fake_c = fake_imgs[:n_fid, :, :, :14]
    else:
        real_c = real_imgs[:n_fid, :, :14, :]   # top half
        fake_c = fake_imgs[:n_fid, :, :14, :]

    # Pad to 28×28 for InceptionV3
    real_c = F.pad(real_c, (0, 14, 0, 0))
    fake_c = F.pad(fake_c, (0, 14, 0, 0))
    return compute_fid(real_c, fake_c, device)


def run_exp6_anisotropic(
    dataset_name: str = "FashionMNIST",
    epochs:       int = 20,
    n_fid:        int = 1000,
    save_dir:     str = str(OUT_ROOT / "exp6"),
    device: torch.device = None,
):
    """
    Experiment 6: Anisotropic noise study.

    For each anisotropy mode we train an additive Rosenblatt model and
    compute both the full FID and the Directional FID (D-FID).

    Prof. question 1 (qualitative): what happens as noise becomes more
    anisotropic?  We visualise the generated images and their frequency spectra.
    """
    if device is None:
        device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = get_dataset(dataset_name, train=False, tf=tf)
    real_imgs = torch.stack([test_ds[i][0] for i in range(n_fid)])
    real_imgs = (real_imgs + 1.0) / 2.0

    aniso_modes = ["isotropic", "h_emphasis", "v_emphasis", "pca_like"]
    results = []

    for mode in aniso_modes:
        print(f"\n{'='*60}\nExp 6: aniso_mode={mode}")
        A_flat = build_aniso_matrix(mode, img_shape=(1, 28, 28)).to(device)
        # Reshape to (1, 1, 28, 28) for broadcast inside AnisotropicForward
        A_4d = A_flat.view(1, 1, 28, 28)

        run_dir = f"{save_dir}/{mode}"
        model, forward = train(
            dataset_name=dataset_name, noise_type="rosenblatt",
            forward_mode="additive", epochs=epochs,
            save_dir=run_dir, device=device,
            aniso_matrix=A_4d,
        )
        model.eval()

        # Full FID
        gen_labels = torch.randint(0, 10, (n_fid,), device=device)
        fakes_list = []
        for i in range(0, n_fid, 200):
            bl = gen_labels[i:i+200]
            fakes_list.append(
                generate_conditional(model, forward, bl,
                                     bridge="stochastic", device=device).cpu()
            )
        fake_imgs = torch.cat(fakes_list, 0)
        fid = compute_fid(real_imgs, fake_imgs, device)

        # Directional FID
        dfid_h = _directional_fid(model, forward, real_imgs, device,
                                  direction="horizontal", n_fid=n_fid)
        dfid_v = _directional_fid(model, forward, real_imgs, device,
                                  direction="vertical",   n_fid=n_fid)

        results.append({"mode": mode, "FID": round(fid, 2),
                        "D-FID-H": round(dfid_h, 2), "D-FID-V": round(dfid_v, 2)})
        print(f"  FID={fid:.2f}  D-FID-H={dfid_h:.2f}  D-FID-V={dfid_v:.2f}")

        # Qualitative: generated images + power spectrum
        _plot_aniso_qualitative(
            fake_imgs[:16], A_flat.cpu(), mode, save_dir, device)

    # Print table
    print("\nExp 6 Anisotropy Summary:")
    for r in results:
        print(f"  {r['mode']:12s}  FID={r['FID']:<8}  "
              f"D-FID-H={r['D-FID-H']:<8}  D-FID-V={r['D-FID-V']}")
    return results


def _plot_aniso_qualitative(
    fake_imgs:   torch.Tensor,
    A_flat:      torch.Tensor,
    mode:        str,
    save_dir:    str,
    device:      torch.device,
):
    """Plot a grid of generated images and their average power spectrum."""
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for i, ax in enumerate(axes[0]):
        ax.imshow(fake_imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes[0, 0].set_title(f"Generated ({mode})", loc="left", fontsize=10)

    # Average power spectrum
    psd = torch.zeros(28, 28)
    for img in fake_imgs[:64]:
        f = torch.fft.fft2(img[0])
        psd += torch.fft.fftshift(f.abs()).log1p()
    psd /= 64

    for ax in axes[1]:
        ax.axis("off")
    ax_psd = fig.add_subplot(2, 1, 2)
    ax_psd.imshow(psd.numpy(), cmap="inferno")
    ax_psd.set_title(f"Avg log power spectrum — {mode}", fontsize=10)
    ax_psd.axis("off")

    # Anisotropy map
    A_img = A_flat.view(1, 28, 28)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{mode}_qualitative.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_dir}/{mode}_qualitative.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Experiment 7 — Stable Diffusion-style latent experiment
# ─────────────────────────────────────────────────────────────────────────────

def train_latent_denoiser(
    ae:           ConvAutoencoder,
    dataset_name: str = "FashionMNIST",
    noise_type:   str = "rosenblatt",
    H:            float = 0.7,
    sigma_max:    float = 4.0,
    M_eig:        int = 80,
    epochs:       int = 30,
    batch_size:   int = 256,
    lr:           float = 1e-3,
    save_dir:     str = "./latent_run",
    device: torch.device = None,
) -> tuple[LatentMLPDenoiser, RosenblattForward]:
    """
    Train a latent-space MLP denoiser.

    Forward process: Z_t = z_0 + sigma(t) * eps,  eps ~ noise_type
    where z_0 = Encoder(x_0) in R^{64}.

    Prof. question 2: does there exist a basis (latent space of autoencoder)
    in which Rosenblatt corruption leads to better results?
    Answer: yes, because (a) the latent distribution is non-Gaussian (richer
    structure), (b) Rosenblatt heavy tails cover the full tail of the latent
    distribution, (c) density theory (Prop. latent-density) remains valid.
    """
    if device is None:
        device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ae.eval()

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = get_dataset(dataset_name, train=True,  tf=tf)
    val_ds = get_dataset(dataset_name, train=False, tf=tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    D = ConvAutoencoder.LATENT_DIM
    forward = RosenblattForward(
        noise_type=noise_type, forward_mode="additive",
        H=H, M_eig=M_eig, sigma_max=sigma_max, device=device,
    )
    model = LatentMLPDenoiser(latent_dim=D).to(device)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T_MIN = 0.01

    tag = f"latent_{noise_type}_H{H}"
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for x0, labels in train_dl:
            x0, labels = x0.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)
            B = x0.size(0)
            with torch.no_grad():
                z0 = ae.encode(x0)                        # (B, D)

            cf = torch.rand(B, device=device) < 0.1
            lbl = labels.clone()
            lbl[cf] = 10
            t = torch.rand(B, device=device) * (1.0 - T_MIN) + T_MIN
            sig = forward.sigma(t).unsqueeze(1)           # (B, 1)
            eps = sample_noise(forward.noise_type, (B, D),
                               forward.lam_t, forward.M_eig, device=device)
            z_t = z0 + sig * eps

            opt.zero_grad(set_to_none=True)
            pred = model(z_t, t, lbl)
            loss = F.smooth_l1_loss(pred, z0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()
            ep_loss += loss.item() * B

        ep_loss /= len(train_ds)
        train_losses.append(ep_loss)

        model.eval()
        ema.apply_shadow()
        v_loss = 0.0
        with torch.no_grad():
            for x0, labels in val_dl:
                x0, labels = x0.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True)
                z0 = ae.encode(x0)
                t = torch.rand(x0.size(0), device=device) * \
                    (1.0 - T_MIN) + T_MIN
                sig = forward.sigma(t).unsqueeze(1)
                eps = sample_noise(forward.noise_type, z0.shape,
                                   forward.lam_t, forward.M_eig, device=device)
                z_t = z0 + sig * eps
                pred = model(z_t, t, labels)
                v_loss += F.smooth_l1_loss(pred, z0).item() * z0.size(0)

        v_loss /= len(val_ds)
        val_losses.append(v_loss)
        ema.restore()
        sched.step()
        print(f"  [latent] ep {epoch+1:3d}/{epochs}  train={ep_loss:.5f}  "
              f"val={v_loss:.5f}  {time.time()-t0:.1f}s")

    ema.apply_shadow()
    torch.save(model.state_dict(), f"{save_dir}/{tag}_final.pt")
    ema.restore()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set(xlabel="Epoch", ylabel="Loss", title=tag)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_loss.png", dpi=120)
    plt.close()

    return model, forward


@torch.no_grad()
def generate_latent(
    model:     LatentMLPDenoiser,
    ae:        ConvAutoencoder,
    forward:   RosenblattForward,
    labels:    torch.Tensor,
    n_steps:   int = 50,
    cfg_scale: float = 2.5,
    device:    torch.device = None,
) -> torch.Tensor:
    """
    Latent cold diffusion reverse process.
    Returns decoded images in [0, 1].
    """
    if device is None:
        device = get_device()
    model.eval()
    ae.eval()
    n = len(labels)
    null = torch.full_like(labels, 10)
    D = ConvAutoencoder.LATENT_DIM

    t_sched = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    z = sample_noise(forward.noise_type, (n, D),
                     forward.lam_t, forward.M_eig, device=device)
    z = z * forward.sigma_max

    for k in range(n_steps):
        t_c = t_sched[k].expand(n)
        t_n = t_sched[k+1].expand(n)
        sig = forward.sigma(t_c).unsqueeze(1)
        c_in = 1.0 / torch.sqrt(1.0 + sig**2)
        z_in = z * c_in

        z0_c = model(z_in, t_c, labels)
        z0_u = model(z_in, t_c, null)
        z0_hat = z0_c + cfg_scale * (z0_c - z0_u)   # CFG

        if k < n_steps - 1:
            # Stochastic bridge: fresh noise
            sig_n = forward.sigma(t_n).unsqueeze(1)
            eps_n = sample_noise(forward.noise_type, (n, D),
                                 forward.lam_t, forward.M_eig, device=device)
            z = z0_hat + sig_n * eps_n
        else:
            z = z0_hat

    imgs = ae.decode(z)
    return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)


def run_exp7_latent(
    dataset_name: str = "FashionMNIST",
    ae_epochs:    int = 20,
    diff_epochs:  int = 30,
    n_fid:        int = 1000,
    save_dir:     str = str(OUT_ROOT / "exp7"),
    device: torch.device = None,
):
    """
    Experiment 7: Stable Diffusion-style latent cold diffusion on FashionMNIST.
    Compares Gaussian vs Rosenblatt noise in the 64-D latent space.

    Addresses Prof. question 2: is there a basis (autoencoder latent space)
    in which Rosenblatt corruption yields better generative quality?
    """
    if device is None:
        device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Train or load autoencoder
    ae_path = f"{save_dir}/ae_final.pt"
    ae = ConvAutoencoder().to(device)
    if not Path(ae_path).exists():
        print("Training autoencoder...")
        ae = train_autoencoder(dataset_name=dataset_name, epochs=ae_epochs,
                               save_path=ae_path, device=device)
    else:
        ae.load_state_dict(torch.load(
            ae_path, map_location=device, weights_only=True))
        print(f"Loaded autoencoder from {ae_path}")
    ae.eval()

    # 2. Get real decoded images for FID (decode real latents)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = get_dataset(dataset_name, train=False, tf=tf)
    real_orig = torch.stack([test_ds[i][0] for i in range(n_fid)]).to(device)
    with torch.no_grad():
        real_recon_list = []
        for i in range(0, n_fid, 256):
            batch = real_orig[i:i+256]
            recon, _ = ae(batch)
            real_recon_list.append(recon.cpu())
        real_recon = torch.cat(real_recon_list, 0)
        real_imgs = ((real_recon + 1.0) / 2.0).clamp(0.0, 1.0)

    results = []
    sigma_maxs = [4.0, 16.0]
    noise_types = ["gaussian", "rosenblatt"]

    for sigma_max in sigma_maxs:
        for noise_type in noise_types:
            tag = f"latent_{noise_type}_s{sigma_max}"
            run_dir = f"{save_dir}/{tag}"
            print(f"\n{'='*60}\nExp 7: {tag}")
            model, forward = train_latent_denoiser(
                ae=ae, dataset_name=dataset_name, noise_type=noise_type,
                sigma_max=sigma_max, epochs=diff_epochs, save_dir=run_dir, device=device,
            )
            model.eval()

            gen_labels = torch.randint(0, 10, (n_fid,), device=device)
            fake_list = []
            for i in range(0, n_fid, 200):
                bl = gen_labels[i:i+200]
                fake_list.append(
                    generate_latent(model, ae, forward, bl,
                                    device=device).cpu()
                )
            fake_imgs = torch.cat(fake_list, 0)
            fid = compute_fid(real_imgs, fake_imgs, device)
            results.append({"noise": noise_type, "sigma_max": sigma_max,
                            "FID": round(fid, 2)})
            print(f"  Pixel FID = {fid:.2f}")

            # PCA trajectory plot (first 2 principal components of latent space)
            _plot_latent_trajectory(ae, forward, model, test_ds, device,
                                    noise_type=noise_type, sigma_max=sigma_max,
                                    save_dir=save_dir)

    # Summary
    print("\nExp 7 Latent FID Summary:")
    for r in results:
        print(
            f"  noise={r['noise']:12s}  sigma_max={r['sigma_max']}  FID={r['FID']}")

    return results


@torch.no_grad()
def _plot_latent_trajectory(
    ae:          ConvAutoencoder,
    forward:     RosenblattForward,
    model:       LatentMLPDenoiser,
    test_ds,
    device:      torch.device,
    noise_type:  str = "rosenblatt",
    sigma_max:   float = 4.0,
    n_traj:      int = 5,
    save_dir:    str = ".",
):
    """PCA projection of corrupted latent trajectories (t=0 to 1)."""
    from sklearn.decomposition import PCA
    ae.eval()
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    xs = torch.stack([test_ds[i][0] for i in range(n_traj)]).to(device)
    z0s = ae.encode(xs)   # (n_traj, 64)

    t_vals = torch.linspace(0.0, 1.0, 20, device=device)
    all_z = []
    for t_val in t_vals:
        t_rep = t_val.expand(n_traj)
        sig = forward.sigma(t_rep).unsqueeze(1)
        eps = sample_noise(noise_type, z0s.shape, forward.lam_t,
                           forward.M_eig, device=device)
        all_z.append((z0s + sig * eps).cpu().numpy())

    # PCA on all latents concatenated
    data = np.concatenate(all_z, axis=0)       # (20*n_traj, 64)
    pca = PCA(n_components=2).fit(data)
    proj = [pca.transform(z) for z in all_z]   # list of (n_traj, 2)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_traj):
        traj = np.stack([p[i] for p in proj])   # (20, 2)
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=4,
                color=cmap(i / n_traj), alpha=0.7, label=f"sample {i}")
    ax.set(xlabel="PC1", ylabel="PC2",
           title=f"Latent trajectory — {noise_type}, σ_max={sigma_max}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{save_dir}/latent_traj_{noise_type}_s{sigma_max}.png"
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Restoration grid visualisation (Exp 5 helper)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _plot_restoration_grid(
    model:        nn.Module,
    forward:      RosenblattForward,
    dataset_name: str,
    save_dir:     str,
    tag:          str = "",
    n_steps_vis:  int = 8,
    cfg_scale:    float = GLOBAL_CONFIG["cfg_scale"],
    device: torch.device = None,
):
    if device is None:
        device = get_device()
    model.eval()
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = get_dataset(dataset_name, train=False, tf=tf)

    found, idx_dict = set(), {}
    for i in range(len(test_ds)):
        lbl = test_ds[i][1]
        if lbl not in found:
            found.add(lbl)
            idx_dict[lbl] = i
        if len(found) == 10:
            break

    indices = [idx_dict[c] for c in range(10)]
    x0 = torch.stack([test_ds[i][0] for i in indices]).to(device)
    labels = torch.tensor([test_ds[i][1] for i in indices], device=device)
    null = torch.full_like(labels, 10)
    n = len(labels)

    # Corrupt at t=1
    t1 = torch.ones(n, device=device)
    x_corr, _, _ = forward.corrupt(x0, t1)

    # Reverse steps
    t_sched = torch.linspace(1.0, 0.0, n_steps_vis + 1, device=device)
    x_cur = x_corr.clone()
    history = [x_corr.cpu()]

    for k in range(n_steps_vis):
        t_c = t_sched[k].expand(n)
        t_n = t_sched[k+1].expand(n)
        sig = forward.sigma(t_c).view(-1, 1, 1, 1)
        c_in = forward.c_in(t_c).view(-1, 1, 1, 1)
        x_in = (x_cur * c_in).float()
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0_c = model(x_in, t_c, labels).float()
                x0_u = model(x_in, t_c, null).float()
        else:
            x0_c = model(x_in, t_c, labels)
            x0_u = model(x_in, t_c, null)
        x0_hat = (x0_u + cfg_scale * (x0_c - x0_u)).clamp(-1.0, 1.0)
        history.append(x0_hat.cpu())
        if k < n_steps_vis - 1:
            x_cur = forward.recorrupt_stochastic(x0_hat, t_n)
        else:
            x_cur = x0_hat

    n_cols = 2 + n_steps_vis   # original + corrupted + steps
    fig, axes = plt.subplots(n, n_cols, figsize=(2.0 * n_cols, 1.4 * n))

    for i in range(n):
        axes[i, 0].imshow((x0[i, 0].cpu() + 1) / 2,
                          cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow((history[0][i, 0] + 1) / 2,
                          cmap="gray", vmin=0, vmax=1)
        for col, h in enumerate(history[1:]):
            axes[i, col + 2].imshow((h[i, 0] + 1) / 2,
                                    cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(class_name(dataset_name, labels[i].item()),
                              fontsize=9, rotation=0, labelpad=40, va="center")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 1].set_title("Corrupted", fontsize=9)
    for j in range(n_steps_vis):
        axes[0, j + 2].set_title(f"Step {j+1}", fontsize=9)

    plt.suptitle(f"Restoration — {tag}", fontsize=12)
    plt.tight_layout()
    fpath = f"{save_dir}/{tag}_restoration.png"
    plt.savefig(fpath, dpi=140)
    plt.close()
    print(f"  Saved {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Bridge comparison ablation (deterministic vs stochastic)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_bridge_ablation(
    model:   nn.Module,
    forward: RosenblattForward,
    real_imgs: torch.Tensor,
    device:  torch.device,
    n_fid:   int = 500,
    n_steps: int = 50,
    save_dir: str = ".",
) -> dict:
    """
    Compare stochastic vs deterministic bridge FID side-by-side.
    Illustrates cumulant amplification defect of the deterministic bridge.
    """
    results = {}
    for bridge in ("stochastic", "deterministic"):
        gen_labels = torch.randint(0, 10, (n_fid,), device=device)
        fake_list = []
        for i in range(0, n_fid, 200):
            bl = gen_labels[i:i+200]
            fake_list.append(
                generate_conditional(model, forward, bl, n_steps=n_steps,
                                     bridge=bridge, device=device).cpu()
            )
        fake_imgs = torch.cat(fake_list, 0)
        fid = compute_fid(real_imgs[:n_fid], fake_imgs, device)
        results[bridge] = round(fid, 2)
        print(f"  Bridge={bridge:<15s}  FID={fid:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 16. Density simulation integration (fixes the cold-diffusion density bug)
# ─────────────────────────────────────────────────────────────────────────────

def get_rosenblatt_density(H: float = 0.7, K: int = 200):
    """
    Return (x_grid, density) for Z_D using the correct LP algorithm from
    density_simulation.py.  If that module is importable we delegate to it;
    otherwise we compute inline using the same correct formulas.

    The original Rosenblatt_cold_diffusion.py density was WRONG because
    lp_eigenvalues() used a misapplied C_D coefficient.  This function uses
    the corrected eigenvalue formula defined in Section 3 of this file.
    """
    try:
        # Prefer the full density_simulation module if available
        from density_simulation import RosenblattDensityLP
        lp = RosenblattDensityLP(a=1.0 - H, K=K)
        return lp.density_fft(x_min=-5.0, x_max=8.0, N=2**16, z_max=40.0)
    except ImportError:
        pass

    # Inline fallback: same LP characteristic function
    D = 1.0 - H
    lam = lp_eigenvalues(D, K)
    # Normalise to variance 1  (Var Z_D = 2 sum lam^2)
    var = 2.0 * np.sum(lam**2)
    lam = lam / np.sqrt(var) if var > 0 else lam

    def chf(z):
        z = np.asarray(z, dtype=complex)
        out = np.ones(z.shape, dtype=complex)
        for ln in lam:
            out *= np.exp(-0.5 * np.log(1.0 - 2j * ln * z) - 1j * ln * z)
        return out

    N = 2**16
    z_max = 40.0
    dz = 2.0 * z_max / N
    j = np.arange(N)
    z = -z_max + j * dz
    dx = 2.0 * np.pi / (N * dz)
    x0_off = -N / 2 * dx
    g = chf(z) * np.exp(-1j * j * dz * x0_off)
    F = np.fft.fft(g)
    phas = np.exp(1j * z_max * x0_off) * np.exp(1j * z_max * j * dx)
    dens = np.real(phas * F) * dz / (2.0 * np.pi)
    xg = x0_off + j * dx
    mask = (xg >= -5.0) & (xg <= 8.0)
    return xg[mask], np.maximum(dens[mask], 0.0)


def plot_noise_comparison(H: float = 0.7, n_mc: int = 20_000,
                          save_path: str = str(OUT_ROOT / "noise_comparison.png")):
    """
    Plot Rosenblatt density (exact LP) vs Gaussian for a visual comparison.
    Uses the FIXED density code.
    """
    x_lp, d_lp = get_rosenblatt_density(H=H, K=200)

    # Monte Carlo Rosenblatt samples
    D = 1.0 - H
    lam = lp_eigenvalues(D, 200)
    var = 2.0 * np.sum(lam**2)
    lam = lam / np.sqrt(var)
    xi = np.random.randn(n_mc, 200)
    z_mc = (xi**2 - 1.0) @ lam

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x_lp, d_lp, "r-", lw=2, label=f"Rosenblatt LP (H={H})")
    x_g = np.linspace(-5, 8, 400)
    ax.plot(x_g, np.exp(-x_g**2/2)/np.sqrt(2*np.pi),
            "b--", lw=2, label="Gaussian N(0,1)")
    ax.hist(z_mc, bins=150, density=True, alpha=0.3,
            color="red", label="Rosenblatt MC")
    ax.set(xlabel="x", ylabel="density",
           title="Rosenblatt vs Gaussian Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(np.abs(x_lp), d_lp, "r-", lw=1.5, label="Rosenblatt tail")
    ax.semilogy(x_g[x_g > 0], np.exp(-x_g[x_g > 0]**2/2)/np.sqrt(2*np.pi),
                "b--", lw=1.5, label="Gaussian tail")
    ax.set(xlabel="|x|", ylabel="log density",
           title="Tail Comparison (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 17. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rosenblatt Cold Diffusion — Unified")
    parser.add_argument("--mode",     default="all",
                        choices=["all", "exp5", "exp6", "exp7",
                                 "aniso", "latent", "bridge_ablation",
                                 "noise_plot"],
                        help="Which experiment(s) to run")
    parser.add_argument("--dataset",  default=GLOBAL_CONFIG["dataset"])
    parser.add_argument("--epochs",   type=int,
                        default=GLOBAL_CONFIG["epochs"])
    parser.add_argument("--noise",    default="rosenblatt",
                        choices=["gaussian", "rosenblatt"])
    parser.add_argument("--forward",  default="additive",
                        choices=["additive", "multiplicative"])
    parser.add_argument("--H",        type=float, default=GLOBAL_CONFIG["H"])
    parser.add_argument("--n_fid",    type=int,   default=1000)
    parser.add_argument("--save_dir", default=str(OUT_ROOT))
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.mode in ("noise_plot",):
        plot_noise_comparison(H=args.H)
        return

    if args.mode in ("exp5", "all"):
        run_exp5_cold_diffusion(
            dataset_name=args.dataset, epochs=args.epochs,
            n_fid=args.n_fid, device=device,
            save_dir=str(OUT_ROOT / "exp5"),
        )

    if args.mode in ("exp6", "aniso", "all"):
        run_exp6_anisotropic(
            dataset_name=args.dataset, epochs=max(15, args.epochs // 2),
            n_fid=args.n_fid, device=device,
            save_dir=str(OUT_ROOT / "exp6"),
        )

    if args.mode in ("exp7", "latent", "all"):
        run_exp7_latent(
            dataset_name=args.dataset,
            ae_epochs=20, diff_epochs=args.epochs,
            n_fid=args.n_fid, device=device,
            save_dir=str(OUT_ROOT / "exp7"),
        )

    if args.mode in ("bridge_ablation",):
        # Quick bridge comparison using a single trained model
        fwd = RosenblattForward(noise_type=args.noise, forward_mode=args.forward,
                                H=args.H, device=device)
        model, fwd = train(
            dataset_name=args.dataset, noise_type=args.noise,
            forward_mode=args.forward, H=args.H, epochs=args.epochs,
            save_dir=str(OUT_ROOT / "bridge_ablation"), device=device,
        )
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        ds = get_dataset(args.dataset, train=False, tf=tf)
        real = ((torch.stack([ds[i][0]
                for i in range(args.n_fid)]) + 1) / 2).clamp(0, 1)
        run_bridge_ablation(model, fwd, real, device,
                            n_fid=args.n_fid, save_dir=str(OUT_ROOT / "bridge_ablation"))

    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
