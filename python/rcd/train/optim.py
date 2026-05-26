
import torch
import torch.nn as nn
from torch.optim import Optimizer

from rcd.data.config import Config
from rcd.train.noise import sample_grad_noise

# ─────────────────────────────────────────────────────────────────────────────
# 1. Optimizers
# ─────────────────────────────────────────────────────────────────────────────

class LionOptimizer(Optimizer):
    """Lion: Evolved Sign Momentum.  κ₄(updates) ≈ -2 (maximally anti-Gaussian)."""

    def __init__(self, params, lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            β1, β2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                c = β1 * m + (1.0 - β1) * g
                p.mul_(1.0 - lr * wd).add_(torch.sign(c), alpha=-lr)
                m.mul_(β2).add_(g, alpha=1.0 - β2)
        return loss


class CustomNoiseAdamW(torch.optim.AdamW):
    """AdamW with configurable additive gradient noise.
    FIX: delegates to sample_grad_noise() instead of a private duplicate."""

    def __init__(self, params, lr: float = 1e-3,
                 noise_std: float = 1e-4, noise_dist: str = "none", **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.noise_std  = noise_std
        self.noise_dist = noise_dist

    @torch.no_grad()
    def step(self, closure=None):
        if self.noise_std > 0 and self.noise_dist != "none":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        # FIX: use the canonical sample_grad_noise from noise.py
                        p.grad.add_(
                            sample_grad_noise(p.grad.shape, self.noise_dist,
                                              self.noise_std, p.grad.device)
                        )
        return super().step(closure)


def _make_optimizer(opt_name: str, model: nn.Module, cfg: Config,
                    noise_std: float = 0.0, noise_dist: str = "none") -> Optimizer:
    """Factory for all tested optimizers."""
    lr, wd = cfg.lr, 1e-4
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "lion":
        return LionOptimizer(model.parameters(), lr=lr / 5.0,
                             betas=(0.9, 0.99), weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr * 10,
                                momentum=0.9, weight_decay=wd, nesterov=True)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr,
                                    alpha=0.99, weight_decay=wd)
    elif opt_name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "noise_adamw":
        return CustomNoiseAdamW(model.parameters(), lr=lr,
                                noise_std=noise_std, noise_dist=noise_dist,
                                weight_decay=wd)
    raise ValueError(f"Unknown opt_name: {opt_name!r}")
