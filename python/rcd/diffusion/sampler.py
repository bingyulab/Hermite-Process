from __future__ import annotations
import torch
import torch.nn as nn
from rcd.core.config import Config
from rcd.diffusion.noise import sample_noise
from rcd.models.latent import ConvAutoencoder
from typing import Any


@torch.no_grad()
def generate_conditional(
    model:     nn.Module,
    forward:   Any,  # RosenblattForward
    labels:    torch.Tensor,
    cfg:       Config,
    bridge:    str   = "stochastic",
    x_in:      torch.Tensor = None,
) -> torch.Tensor:
    """
    Cold diffusion reverse process with classifier-free guidance.
    """
    model.eval()
    n           = len(labels)
    null_labels = torch.full_like(labels, 10)

    t_sched = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=cfg.device)

    if x_in is None:
        eps = sample_noise(forward.noise_type, (n, 1, 28, 28),
                           forward.lam_t, forward.M_eig, device=cfg.device)
        dummy_x0 = torch.zeros(n, 1, 28, 28, device=cfg.device)
        S = forward.sigma_fn(dummy_x0, labels) if getattr(forward.sigma_fn, "needs_label", False) else forward.sigma_fn(dummy_x0)
        x = eps * forward.sigma_max * S
    else:
        x = x_in

    for k in range(cfg.n_steps):
        t_cur  = t_sched[k].expand(n)
        t_next = t_sched[k + 1].expand(n)
        c_in   = forward.c_in(t_cur).view(-1, 1, 1, 1)
        scaled_x_in = (x * c_in).float()

        if cfg.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0_c = model(scaled_x_in, t_cur, labels).float()
                x0_u = model(scaled_x_in, t_cur, null_labels).float()
        else:
            x0_c = model(scaled_x_in, t_cur, labels)
            x0_u = model(scaled_x_in, t_cur, null_labels)

        x0_hat = (x0_u + cfg.cfg_scale * (x0_c - x0_u)).clamp(-1., 1.)

        if k < cfg.n_steps - 1:
            if bridge == "stochastic":
                x = forward.recorrupt_stochastic(x0_hat, t_next, y=labels)
            elif bridge == "hybrid":
                x = forward.recorrupt_hybrid(x, x0_hat, t_cur, t_next, y=labels)
            elif bridge == "deterministic":
                x = forward.recorrupt_deterministic(x, x0_hat, t_cur, t_next)
            else:
                raise ValueError(f"Unknown bridge: {bridge!r}")
        else:
            x = x0_hat

    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def generate_latent(
    model:   nn.Module,
    ae:      ConvAutoencoder,
    fwd:     Any,
    labels:  torch.Tensor,
    cfg:     Config,
) -> torch.Tensor:
    """
    CFG formula corrected: z0u + cfg*(z0c-z0u).
    """
    model.eval()
    ae.eval()
    n    = len(labels)
    null = torch.full_like(labels, 10)
    D    = ConvAutoencoder.LATENT_DIM

    t_sched = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    z = sample_noise(fwd.noise_type, (n, D),
                     fwd.lam_t, fwd.M_eig, cfg.device) * fwd.sigma_max

    for k in range(cfg.n_steps):
        tc  = t_sched[k].expand(n)
        tn  = t_sched[k + 1].expand(n)
        sig = fwd.sigma_t(tc).unsqueeze(1)               
        cin = (1. / (1. + sig ** 2).sqrt())
        
        # Bug H2 fixed by using cin properly at train/inference
        z0c = model(z * cin, tc, labels)
        z0u = model(z * cin, tc, null)

        z0h = z0u + cfg.cfg_scale * (z0c - z0u)

        if k < cfg.n_steps - 1:
            sn  = fwd.sigma_t(tn).unsqueeze(1)           
            eps = sample_noise(fwd.noise_type, (n, D),
                               fwd.lam_t, fwd.M_eig, cfg.device)
            z   = z0h + sn * eps
        else:
            z = z0h

    return ((ae.decode(z) + 1.) / 2.).clamp(0., 1.)
