from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model  = model
        self.decay  = decay
        self.step   = 0
        self.shadow: dict[str, torch.Tensor] = {
            n: p.data.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def to(self, device: torch.device | str) -> "EMA":
        with torch.no_grad():
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
            self.backup = {k: v.to(device) for k, v in self.backup.items()}
        return self

    def _effective_decay(self) -> float:
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    def update(self) -> None:
        d = self._effective_decay()
        self.step += 1
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    if self.shadow[n].device != p.device:
                        self.shadow[n] = self.shadow[n].to(p.device)
                    self.shadow[n].lerp_(p.data, 1.0 - d)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
            "step":   int(self.step),
            "decay":  float(self.decay),
        }

    def load_state_dict(
        self,
        state: dict,
        device: torch.device | str | None = None,
        strict: bool = True,
    ) -> None:
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = "cpu"
        self.step  = int(state.get("step", 0))
        self.decay = float(state.get("decay", self.decay))
        self.backup = {}
        loaded_shadow = state["shadow"]
        if strict:
            self.shadow = {k: v.to(device) for k, v in loaded_shadow.items()}
        else:
            self.shadow = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if n in loaded_shadow:
                        self.shadow[n] = loaded_shadow[n].to(device)
                    else:
                        print(f"  [EMA] Initializing missing shadow parameter: {n}")
                        self.shadow[n] = p.data.detach().clone().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_norm(norm_type: str, channels: int, affine: bool = True) -> nn.Module:
    if norm_type == "group8":
        return nn.GroupNorm(min(8, channels), channels, affine=affine)
    elif norm_type == "group4":
        return nn.GroupNorm(min(4, channels), channels, affine=affine)
    elif norm_type == "group1":
        return nn.GroupNorm(1, channels, affine=affine)
    elif norm_type == "batch":
        return nn.BatchNorm2d(channels, affine=affine)
    elif norm_type == "none":
        return nn.Identity()
    else:
        return nn.GroupNorm(min(8, channels), channels, affine=affine)


def _make_act(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn   == "silu": return nn.SiLU()
    elif act_fn == "relu": return nn.ReLU()
    elif act_fn == "gelu": return nn.GELU()
    elif act_fn == "tanh": return nn.Tanh()
    elif act_fn == "mish": return nn.Mish()
    else:                  return nn.SiLU()


# ─────────────────────────────────────────────────────────────────────────────
# Base Blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4,
                 spatial_size: int = 14, norm_type: str = "group8") -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_size ** 2, channels) * 0.02)
        self.norm1   = _make_norm(norm_type, channels, affine=True)
        self.mha     = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj    = nn.Conv2d(channels, channels, 1)
        self.norm2   = _make_norm(norm_type, channels, affine=True)
        self.ffn     = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm1(x).view(B, C, -1).transpose(1, 2) + self.pos_emb
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        x = x + self.proj(attn_out)
        return x + self.ffn(self.norm2(x))


class ResBlockAdaGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int = 256,
                 norm_type: str = "group8", act_fn: str = "silu") -> None:
        super().__init__()
        self._act    = _make_act(act_fn)
        self.norm1   = _make_norm(norm_type, in_ch, affine=True)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2   = _make_norm(norm_type, out_ch, affine=False)
        self.dropout = nn.Dropout(0.1)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj  = nn.Linear(t_dim, out_ch * 2)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self._act(self.norm1(x)))
        scale, shift = self.t_proj(self._act(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.dropout(self._act(h))
        return self.shortcut(x) + self.conv2(h)


# ─────────────────────────────────────────────────────────────────────────────
# ConditionalUNet
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net with configurable bottleneck width, skip connections,
    normalization type, and activation function.

    Bug fixed: self.bneck_ch is now set as a public attribute so experiment
    runners (e.g. run_gaussianity) can read the actual bottleneck channel count.
    """

    def __init__(self, t_dim: int = 256, num_classes: int = 10,
                 base_ch: int = 128, in_channels: int = 1,
                 bottleneck_factor: float = 1.0,
                 use_skip_h1: bool = True, use_skip_h2: bool = True,
                 norm_type: str = "group8", act_fn: str = "silu") -> None:
        super().__init__()
        self.use_skip_h1 = use_skip_h1
        self.use_skip_h2 = use_skip_h2

        # FIX: expose bneck_ch as a public attribute for downstream analysis
        self.bneck_ch = max(base_ch, int(round(4 * base_ch * bottleneck_factor)))
        enc2_ch       = 2 * base_ch

        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            _make_act(act_fn),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Encoder
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1     = nn.Sequential(
            ResBlockAdaGN(base_ch,  base_ch,  t_dim, norm_type, act_fn),
            ResBlockAdaGN(base_ch,  base_ch,  t_dim, norm_type, act_fn),
        )
        self.pool1     = nn.Conv2d(base_ch, enc2_ch, 3, stride=2, padding=1)
        self.down2     = nn.Sequential(
            ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim, norm_type, act_fn),
            ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim, norm_type, act_fn),
        )
        self.attn2     = SelfAttention(enc2_ch, spatial_size=14, norm_type=norm_type)
        self.pool2     = nn.Conv2d(enc2_ch, self.bneck_ch, 3, stride=2, padding=1)

        # Bottleneck
        self.mid1     = ResBlockAdaGN(self.bneck_ch, self.bneck_ch, t_dim, norm_type, act_fn)
        self.attn_mid = SelfAttention(self.bneck_ch, spatial_size=7, norm_type=norm_type)
        self.mid2     = ResBlockAdaGN(self.bneck_ch, self.bneck_ch, t_dim, norm_type, act_fn)

        # Decoder
        self.up2    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.bneck_ch, enc2_ch, 3, padding=1),
        )
        cat2_ch     = (2 * enc2_ch) if self.use_skip_h2 else enc2_ch
        self.up_res2  = nn.ModuleList([
            ResBlockAdaGN(cat2_ch, enc2_ch, t_dim, norm_type, act_fn),
            ResBlockAdaGN(enc2_ch, enc2_ch, t_dim, norm_type, act_fn),
        ])
        self.up_attn2 = SelfAttention(enc2_ch, spatial_size=14, norm_type=norm_type)

        self.up1    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc2_ch, base_ch, 3, padding=1),
        )
        cat1_ch     = (2 * base_ch) if self.use_skip_h1 else base_ch
        self.up_res1  = nn.ModuleList([
            ResBlockAdaGN(cat1_ch, base_ch, t_dim, norm_type, act_fn),
            ResBlockAdaGN(base_ch, base_ch, t_dim, norm_type, act_fn),
        ])

        self.out = nn.Sequential(
            _make_norm(norm_type, base_ch, affine=True),
            _make_act(act_fn),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor, t_emb: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.init_conv(x)
        h1 = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2 = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2 = self.attn2(h2)
        h3 = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)
        return h3, h2, h1

    def decode(self, h3: torch.Tensor, h2: torch.Tensor, h1: torch.Tensor,
               t_emb: torch.Tensor) -> torch.Tensor:
        up2_x = self.up2(h3)
        h     = torch.cat([up2_x, h2], dim=1) if self.use_skip_h2 else up2_x
        h     = self.up_attn2(self.up_res2[1](self.up_res2[0](h, t_emb), t_emb))
        up1_x = self.up1(h)
        h     = torch.cat([up1_x, h1], dim=1) if self.use_skip_h1 else up1_x
        h     = self.up_res1[1](self.up_res1[0](h, t_emb), t_emb)
        return self.out(h)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb      = self.time_mlp(self.t_embed(t)) + self.label_emb(y)
        h3, h2, h1 = self.encode(x, t_emb)
        return self.decode(h3, h2, h1, t_emb)


# ─────────────────────────────────────────────────────────────────────────────
# ConvAutoencoder
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """28×28 → 64-D latent → 28×28 convolutional autoencoder."""
    LATENT_DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.LATENT_DIM, 64 * 7 * 7), nn.SiLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(16,  1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# ─────────────────────────────────────────────────────────────────────────────
# LatentMLPDenoiser
# ─────────────────────────────────────────────────────────────────────────────

class LatentMLPDenoiser(nn.Module):
    """6-layer MLP denoiser for the 64-D latent space."""

    def __init__(self, latent_dim: int = 64, t_dim: int = 256,
                 num_classes: int = 10, hidden: int = 256) -> None:
        super().__init__()
        self.t_embed    = SinusoidalTimeEmbed(t_dim)
        self.label_emb  = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim),
        )
        self.input_proj = nn.Linear(latent_dim, hidden)
        self.layers     = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(6)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(6)])
        self.cond_projs = nn.ModuleList([nn.Linear(t_dim, hidden * 2) for _ in range(6)])
        self.out_proj   = nn.Linear(hidden, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.t_embed(t)) + self.label_emb(y)
        h    = F.silu(self.input_proj(z))
        for layer, norm, cproj in zip(self.layers, self.norms, self.cond_projs):
            scale, shift = cproj(F.silu(cond)).chunk(2, dim=-1)
            h = norm(h) * (1.0 + scale) + shift
            h = h + F.silu(layer(h))
        return self.out_proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# SkipZeroWrapper
# ─────────────────────────────────────────────────────────────────────────────

class SkipZeroWrapper(nn.Module):
    """
    Wraps a trained ConditionalUNet and zeros specified skip connections
    at inference time WITHOUT retraining.
    """

    def __init__(self, model: nn.Module,
                 zero_h1: bool = False, zero_h2: bool = False) -> None:
        super().__init__()
        self.model   = model
        self.zero_h1 = zero_h1
        self.zero_h2 = zero_h2

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb      = self.model.time_mlp(self.model.t_embed(t)) + self.model.label_emb(y)
        h3, h2, h1 = self.model.encode(x, t_emb)
        if self.zero_h2: h2 = torch.zeros_like(h2)
        if self.zero_h1: h1 = torch.zeros_like(h1)
        return self.model.decode(h3, h2, h1, t_emb)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)