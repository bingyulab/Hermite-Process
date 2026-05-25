import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import SinusoidalTimeEmbed


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
