import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self.t_embed   = SinusoidalTimeEmbed(t_dim) # 1 * 256
        self.label_emb = nn.Embedding(num_classes + 1, t_dim) # 11 * 256
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

class ConditionalUNetFlexible(nn.Module):
    """
    Identical to ConditionalUNet but with a variable bottleneck channel count.
    
    bottleneck_factor : float
        Scales the bottleneck channels relative to the standard 4 × base_ch.
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
        self.time_mlp  = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1     = nn.Sequential(ResBlockAdaGN(base_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim))
        self.pool1     = nn.Conv2d(base_ch, enc2_ch, 3, stride=2, padding=1)

        self.down2     = nn.Sequential(ResBlockAdaGN(enc2_ch,  enc2_ch,  t_dim), ResBlockAdaGN(enc2_ch, enc2_ch, t_dim))
        self.attn2     = SelfAttention(enc2_ch, spatial_size=14)
        self.pool2     = nn.Conv2d(enc2_ch, bneck_ch, 3, stride=2, padding=1)

        self.mid1      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)
        self.attn_mid  = SelfAttention(bneck_ch, spatial_size=7)
        self.mid2      = ResBlockAdaGN(bneck_ch, bneck_ch, t_dim)

        self.up2       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(bneck_ch, enc2_ch, 3, padding=1)
        )
        self.up_res2   = nn.ModuleList([ResBlockAdaGN(bneck_ch + enc2_ch, enc2_ch, t_dim), ResBlockAdaGN(enc2_ch, enc2_ch, t_dim)])
        self.up_attn2  = SelfAttention(enc2_ch, spatial_size=14)

        self.up1       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(enc2_ch, base_ch, 3, padding=1)
        )
        self.up_res1   = nn.ModuleList([ResBlockAdaGN(enc2_ch + base_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim)])

        self.out       = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.t_embed(t)) + self.label_emb(y)

        x     = self.init_conv(x)
        h1    = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2    = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2    = self.attn2(h2)
        h3    = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)
        
        h     = self.up_attn2(self.up_res2[1](self.up_res2[0](torch.cat([self.up2(h3), h2], dim=1), t_emb), t_emb))
        h     = self.up_res1[1](self.up_res1[0](torch.cat([self.up1(h), h1], dim=1), t_emb), t_emb)
        return self.out(h)

class ConditionalUNetAblation(nn.Module):
    """
    Like ConditionalUNet but allows disabling skip connections.
    """
    def __init__(self, t_dim: int = 256, num_classes: int = 10, base_ch: int = 128, in_channels: int = 1,
                 use_skip_h1: bool = True, use_skip_h2: bool = True) -> None:
        super().__init__()
        self.use_skip_h1 = use_skip_h1
        self.use_skip_h2 = use_skip_h2
        
        self.t_embed   = SinusoidalTimeEmbed(t_dim)
        self.label_emb = nn.Embedding(num_classes + 1, t_dim)
        self.time_mlp  = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1     = nn.Sequential(ResBlockAdaGN(base_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim))
        self.pool1     = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.down2     = nn.Sequential(ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim))
        self.attn2     = SelfAttention(base_ch * 2, spatial_size=14)
        self.pool2     = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        self.mid1      = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)
        self.attn_mid  = SelfAttention(base_ch * 4, spatial_size=7)
        self.mid2      = ResBlockAdaGN(base_ch * 4, base_ch * 4, t_dim)

        self.up2       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        )
        
        cat2_ch = (base_ch * 4) if self.use_skip_h2 else (base_ch * 2)
        self.up_res2   = nn.ModuleList([ResBlockAdaGN(cat2_ch, base_ch * 2, t_dim), ResBlockAdaGN(base_ch * 2, base_ch * 2, t_dim)])
        self.up_attn2  = SelfAttention(base_ch * 2, spatial_size=14)

        self.up1       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        )
        cat1_ch = (base_ch * 2) if self.use_skip_h1 else base_ch
        self.up_res1   = nn.ModuleList([ResBlockAdaGN(cat1_ch, base_ch, t_dim), ResBlockAdaGN(base_ch, base_ch, t_dim)])

        self.out       = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.t_embed(t)) + self.label_emb(y)

        x     = self.init_conv(x)
        h1    = self.down1[1](self.down1[0](x, t_emb), t_emb)
        h2    = self.down2[1](self.down2[0](self.pool1(h1), t_emb), t_emb)
        h2    = self.attn2(h2)
        h3    = self.mid2(self.attn_mid(self.mid1(self.pool2(h2), t_emb)), t_emb)

        up2_x = self.up2(h3)
        h     = torch.cat([up2_x, h2], dim=1) if self.use_skip_h2 else up2_x
        h     = self.up_attn2(self.up_res2[1](self.up_res2[0](h, t_emb), t_emb))

        up1_x = self.up1(h)
        h     = torch.cat([up1_x, h1], dim=1) if self.use_skip_h1 else up1_x
        h     = self.up_res1[1](self.up_res1[0](h, t_emb), t_emb)

        return self.out(h)

