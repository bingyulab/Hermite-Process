#!/usr/bin/env python3
"""
experiment_optimizer.py
=======================
Investigates the relationship between optimisation algorithm choice
and the Gaussianization of internal representations.

Theoretical backbone
────────────────────
Adam maintains v_t ≈ E[g²] and updates θ ← θ - α m̂/(√v̂ + ε).
The effective per-parameter step size is m̂_k/√v̂_k ≈ g_k/σ_{g,k}, which
is a standardised (whitened) version of the gradient.  Whitened gradients
are approximately standard-normal by construction.  This imposes spherical
Gaussian geometry on the parameter update trajectory, independent of the
loss function.

Lion uses sign(β₁m + (1-β₁)g), which is Bernoulli(½) for symmetric
gradient distributions.  Bernoulli(½) has κ₄ = -2 — the minimum possible
kurtosis, maximally anti-Gaussian.  If the optimiser's update distribution
propagates into the representation geometry, Lion should produce sub-Gaussian
(κ₄ < 0) or at least less-Gaussian bottleneck representations than Adam.

SGD uses the raw gradient without whitening; its representation geometry
should be closer to the gradient's natural distribution (approximately
Gaussian for L₂, non-Gaussian for L₁ or heavy-tailed data).

Rosenblatt-SGLD injects Rosenblatt-distributed noise into the gradient.
Rosenblatt noise has κ₄ > 0 (heavy-tailed).  This test asks:
  (a) Does heavy-tailed gradient noise prevent Gaussianization?
  (b) Does it find flatter/better minima (heavy-tail SGD benefits)?
Both (a) and (b) would be novel results for the thesis.

Five experiments
────────────────
  ο  Optimiser comparison
       AdamW, Lion, SGD+momentum, RMSprop, NAdam
       Measure: bottleneck κ₄, layer trace, val loss, flatness

  π  Gradient noise distribution  (CustomNoiseAdamW)
       Gaussian / Laplace / Rosenblatt / Uniform noise at varying σ
       Measure: bottleneck κ₄ as a function of noise type and σ
       Key: does gradient noise distribution propagate to representation?

  ρ  Rosenblatt-SGLD landscape analysis
       Fine-tune a pre-trained AdamW model with Rosenblatt gradient noise
       Measure: test loss trajectory, sharpness before/after, κ₄ before/after
       Key: does Rosenblatt noise help escape sharp minima AND change geometry?

  σ  Adam whitening effect visualisation  (analytical + empirical)
       Track per-parameter update std before/after Adam normalisation
       Measure: Wasserstein distance of effective updates from N(0,1)
       Confirms the theoretical mechanism for Adam Gaussianization

  τ  Gradient κ₄ evolution during training
       Log gradient κ₄ at the bottleneck every N steps for each optimiser
       Reveals whether gradient non-Gaussianity and representation
       non-Gaussianity are correlated across training

Usage:
    python Experiment_Optimizer.py --save_dir output/diffusion/multiplicative --mode all
    python Experiment_Optimizer.py --save_dir output/diffusion/multiplicative --mode omicron
    python Experiment_Optimizer.py --save_dir output/diffusion/multiplicative --mode rho --noise_type rosenblatt
    python Experiment_Optimizer.py --save_dir output/diffusion/multiplicative --mode sigma
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# ── local imports ──────────────────────────────────────────────────────────────
from Rosenblatt_cold_diffusion_unified import (
    Config,
    EMA,
    RosenblattForward,
    _NORM_TF,
    _get_dataset,
    sigma_multiplicative,
)
from Experiment_Gaussianity import (
    compute_marginal_cumulants,
    set_global_seed,
    load_or_train_variant,
)
from Experiment_Ablation import (
    ConditionalUNetAblation,
    compute_loss,
    measure_bottleneck,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"gaussian": "#3A7EBF", "rosenblatt": "#E07B39"}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Optimizer catalogue
# ─────────────────────────────────────────────────────────────────────────────

OPT_LABELS: dict[str, str] = {
    "adamw":      "AdamW (baseline)",
    "lion":       "Lion (sign-based, anti-Gaussian)",
    "sgd":        "SGD + momentum (no whitening)",
    # "rmsprop":    "RMSprop (whitening, no momentum)",
    # "nadam":      "NAdam (Nesterov + Adam)",
}

NOISE_LABELS: dict[str, str] = {
    "none":        "No noise (baseline AdamW)",
    "gaussian":    "Gaussian gradient noise",
    "laplace":     "Laplace gradient noise",
    "rosenblatt":  "Rosenblatt gradient noise",
    # "uniform":     "Uniform gradient noise",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Custom optimizers
# ─────────────────────────────────────────────────────────────────────────────

class LionOptimizer(Optimizer):
    """
    Lion: Evolved Sign Momentum.
    Chen et al. (2023), arXiv:2302.06675.

    Update rule (weight decay folded in):
        c_t    = β₁ · m_{t-1} + (1 - β₁) · g_t
        θ_t    = θ_{t-1} · (1 - η·λ) - η · sign(c_t)
        m_t    = β₂ · m_{t-1} + (1 - β₂) · g_t

    Key property: sign(c_t) ∈ {-1, +1} (Bernoulli with κ₄ = -2).
    Update magnitude is constant: always η, independent of |g|.
    This is the theoretical opposite of Adam's gradient whitening:
    instead of normalising to unit variance (Gaussian), Lion normalises
    to unit L∞ norm (maximally non-Gaussian).
    """

    def __init__(self, params, lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            β1, β2  = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Interpolated direction for this step's update
                c = β1 * m + (1.0 - β1) * g

                # Weight decay + sign update
                p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(c), alpha=-lr)

                # Maintain momentum (note: different β₂ from update)
                m.mul_(β2).add_(g, alpha=1.0 - β2)

        return loss


class CustomNoiseAdamW(torch.optim.AdamW):
    """
    AdamW with configurable additive gradient noise.

    Supported noise distributions:
        gaussian   : ε ~ N(0, σ²)
        laplace    : ε ~ Laplace(0, σ/√2)   (matched variance)
        rosenblatt : ε ≈ product-of-Gaussians (κ₄ ≈ 6, unit variance × σ)
        uniform    : ε ~ Uniform(-σ√3, σ√3) (matched variance)
        none       : no noise (standard AdamW)

    The noise is added to p.grad BEFORE the Adam update, so Adam's second-
    moment normalisation operates on the noisy gradient.  This tests whether
    gradient noise distribution (not just gradient signal) propagates into
    the representation geometry.
    """

    def __init__(self, params, lr: float = 1e-3,
                 noise_std: float = 1e-4,
                 noise_dist: str = "none",
                 **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.noise_std  = noise_std
        self.noise_dist = noise_dist
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        if self.noise_std > 0 and self.noise_dist != "none":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad.add_(_sample_grad_noise(
                        p.grad.shape, self.noise_dist,
                        self.noise_std, p.grad.device))
        return super().step(closure)


def _sample_grad_noise(shape: tuple, dist: str,
                       std: float, device: torch.device) -> torch.Tensor:
    """
    Sample gradient noise of given shape, distribution, and std.

    All distributions are normalised to have variance std².

    rosenblatt approximation: product of two independent N(0,1) variables.
        E[g₁g₂] = 0, Var[g₁g₂] = 1, κ₄ = 6.
        This faithfully reproduces the key qualitative property of Rosenblatt
        marginals (zero mean, unit variance, κ₄ > 0 / heavy tails).
    """
    n = math.prod(shape)
    if dist == "gaussian":
        return torch.randn(shape, device=device) * std

    elif dist == "laplace":
        # Laplace(0, b) with b = std/√2 → Var = 2b² = std²
        b = std / math.sqrt(2.0)
        u = torch.rand(shape, device=device) - 0.5
        return -b * u.sign() * u.abs().log()    # inverse CDF transform

    elif dist == "rosenblatt":
        # Product of two independent N(0,1): κ₄ = 6, unit variance
        g1 = torch.randn(shape, device=device)
        g2 = torch.randn(shape, device=device)
        noise = g1 * g2                          # Var = 1
        return noise * std                       # Var = std²

    elif dist == "uniform":
        # Uniform(-a, a) with a = std·√3 → Var = std²
        a = std * math.sqrt(3.0)
        return (torch.rand(shape, device=device) * 2.0 - 1.0) * a

    else:
        raise ValueError(f"Unknown noise dist: {dist!r}")


def _make_optimizer(opt_name: str, model: nn.Module,
                    cfg: Config,
                    noise_std:  float = 0.0,
                    noise_dist: str   = "none") -> Optimizer:
    """
    Factory for all tested optimisers.
    All use the same effective learning rate and weight decay as the
    baseline AdamW to ensure fair comparison.
    """
    lr = cfg.lr
    wd = 1e-4

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    elif opt_name == "lion":
        # Lion requires larger lr (typical ratio: lr_lion ≈ lr_adam / 3–10)
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

    else:
        raise ValueError(f"Unknown opt_name: {opt_name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient statistics tracker
# ─────────────────────────────────────────────────────────────────────────────

class GradientTracker:
    """
    Captures gradient distribution at the bottleneck (mid2) during training
    via a backward hook.

    Usage:
        tracker = GradientTracker(model, log_every=50)
        ... training loop ...
        tracker.remove()
        df = tracker.get_log()   # list of dicts with step, kappa3, kappa4, norm
    """

    def __init__(self, model: nn.Module, log_every: int = 50):
        self._log_every = log_every
        self._step      = 0
        self._log:  list[dict] = []
        # Hook on mid2's output gradient (∂L/∂h3)
        self._handle = model.mid2.register_full_backward_hook(self._hook)

    def _hook(self, module, grad_input, grad_output):
        self._step += 1
        if self._step % self._log_every != 0:
            return
        g = grad_output[0].detach().float()   # (B, C, H, W)
        # Spatial average → (B, C); treat as N=B samples in D=C dimensions
        g_flat = g.mean(dim=(-2, -1)).cpu()
        if g_flat.numel() < 4:
            return
        cum = compute_marginal_cumulants(g_flat)
        self._log.append({
            "step":     self._step,
            "kappa4":   cum["mean_kappa4"],
            "kappa3":   cum["mean_abs_kappa3"],
            "grad_norm":g.norm().item(),
        })

    def remove(self):
        self._handle.remove()

    def get_log(self) -> list[dict]:
        return list(self._log)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shared training loop (optimizer-aware)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_optimizer(
        model:            nn.Module,
        fwd:              RosenblattForward,
        cfg:              Config,
        ckpt_path:        Path,
        opt_name:         str   = "adamw",
    loss_type:        str   = "huber",
        noise_type:       str   = "rosenblatt",
        noise_std:        float = 0.0,
        noise_dist:       str   = "none",
        log_grads:        bool  = False,
        log_every:        int   = 100,
        tag:              str   = "opt",
) -> tuple[nn.Module, list[dict], list[dict]]:
    """
    Train `model` with the specified optimiser, saving to `ckpt_path`.

    Returns (ema_model, train_history, grad_log).
    train_history: list of {ep, tr_loss, va_loss, va_l1, va_l2}
    grad_log:      list of {step, kappa4, kappa3, grad_norm}  (if log_grads)
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed(getattr(cfg, "seed", 42))

    # Resume if possible
    start_ep = 0
    for ep_i in range(cfg.epochs - 1, 0, -1):
        resume = ckpt_path.parent / f"{ckpt_path.stem}_ep{ep_i}.pt"
        if resume.exists():
            print(f"  [{tag}] Resuming from ep {ep_i}")
            model.load_state_dict(torch.load(resume, map_location=cfg.device,
                                             weights_only=True))
            start_ep = ep_i
            break

    tr_dl = DataLoader(_get_dataset(cfg.dataset, train=True,  tf=_NORM_TF),
                       cfg.batch_size, True,  num_workers=4, pin_memory=True,
                       persistent_workers=True, drop_last=True)
    va_dl = DataLoader(_get_dataset(cfg.dataset, train=False, tf=_NORM_TF),
                       cfg.batch_size, False, num_workers=2, pin_memory=True,
                       persistent_workers=True)

    opt = _make_optimizer(opt_name, model, cfg, noise_std, noise_dist)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs)
    for _ in range(start_ep):
        sch.step()
    ema = EMA(model, 0.999)

    tracker:  GradientTracker | None = None
    grad_log: list[dict]             = []
    if log_grads:
        tracker = GradientTracker(model, log_every=log_every)

    history:  list[dict] = []
    use_amp   = cfg.device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    for ep in range(start_ep, cfg.epochs):
        t0 = time.time();  model.train();  el = 0.0
        for x0, lbl in tr_dl:
            x0, lbl = (x0.to(cfg.device, non_blocking=True),
                       lbl.to(cfg.device, non_blocking=True))
            B    = x0.size(0)
            cf   = torch.rand(B, device=cfg.device) < 0.1
            lbl2 = lbl.clone();  lbl2[cf] = 10
            t    = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
            x_t, _, _ = fwd.corrupt(x0, t, y=lbl2)
            c_in = fwd.c_in(t).view(-1, 1, 1, 1)
            opt.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = compute_loss(model(x_t * c_in, t, lbl2), x0, loss_type)
                if not torch.isfinite(loss):
                    print(f"  [{tag}] Warning: non-finite training loss; skipping batch")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = compute_loss(model(x_t * c_in, t, lbl2), x0, loss_type)
                if not torch.isfinite(loss):
                    print(f"  [{tag}] Warning: non-finite training loss; skipping batch")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ema.update()
            el += loss.item() * B

        el /= len(tr_dl.dataset)
        model.eval();  ema.apply_shadow()

        vl = vl_l1 = vl_l2 = 0.0
        with torch.no_grad():
            for x0, lbl in va_dl:
                x0, lbl = (x0.to(cfg.device, non_blocking=True),
                           lbl.to(cfg.device, non_blocking=True))
                B = x0.size(0)
                t = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
                x_t, _, _ = fwd.corrupt(x0, t, y=lbl)
                c_in = fwd.c_in(t).view(-1, 1, 1, 1)
                pred = model(x_t * c_in, t, lbl).float()
                vl    += compute_loss(pred, x0.float(), loss_type).item() * B
                vl_l1 += F.l1_loss(pred, x0.float()).item() * B
                vl_l2 += F.mse_loss(pred, x0.float()).item() * B

        N = len(va_dl.dataset)
        vl /= N;  vl_l1 /= N;  vl_l2 /= N
        ema.restore();  sch.step()

        history.append({"ep": ep+1, "tr_loss": el,
                         "va_loss": vl, "va_l1": vl_l1, "va_l2": vl_l2})
        print(f"  [{tag}] ep {ep+1:2d}/{cfg.epochs}  "
              f"tr={el:.5f}  va_l1={vl_l1:.5f}  {time.time()-t0:.1f}s")

        if (ep + 1) % 5 == 0 and (ep + 1) < cfg.epochs:
            ema.apply_shadow()
            torch.save(model.state_dict(),
                       ckpt_path.parent / f"{ckpt_path.stem}_ep{ep+1}.pt")
            ema.restore()

    if tracker is not None:
        grad_log = tracker.get_log()
        tracker.remove()

    ema.apply_shadow()
    torch.save(model.state_dict(), ckpt_path)
    model.eval()
    return model, history, grad_log


def load_or_train_opt(
        variant_tag:   str,
        cfg:           Config,
        save_dir:      Path,
        opt_name:      str   = "adamw",
        noise_type:    str   = "rosenblatt",
        noise_std:     float = 0.0,
        noise_dist:    str   = "none",
        log_grads:     bool  = False,
        use_pretrained_baseline: bool = False,
) -> tuple[nn.Module, RosenblattForward, list[dict], list[dict]]:
    model, fwd, extras = load_or_train_variant(
        variant_tag,
        lambda: ConditionalUNetAblation(num_classes=10, base_ch=cfg.base_ch),
        cfg,
        save_dir,
        ckpt_subdir="optimizer_ablation",
        train_fn=train_with_optimizer,
        train_kwargs={
            "opt_name": opt_name,
            "loss_type": "huber",
            "noise_type": noise_type,
            "noise_std": noise_std,
            "noise_dist": noise_dist,
            "log_grads": log_grads,
            "tag": variant_tag,
        },
        noise_type=noise_type,
        use_pretrained_baseline=use_pretrained_baseline,
    )
    history, grad_log = (list(extras[0]), list(extras[1])) if len(extras) >= 2 else ([], [])
    return model, fwd, history, grad_log


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss landscape / sharpness analysis
# ─────────────────────────────────────────────────────────────────────────────

def measure_sharpness(
        model:           nn.Module,
        fwd:             RosenblattForward,
        test_ds,
        cfg:             Config,
        perturb_sigma:   float = 0.01,
        n_perturbations: int   = 30,
    loss_type:       str   = "huber",
) -> dict[str, float]:
    """
    Gaussian perturbation sharpness measure.

    For each of n_perturbations random unit-sphere perturbations scaled by
    perturb_sigma, compute the change in validation loss.

    sharpness = E[|ΔL|] / perturb_sigma

    Interpretation: higher sharpness → sharper minimum → worse generalisation
    (Keskar et al. 2017; Dziugaite & Roy 2017).

    Also computes the full distribution of ΔL to check for asymmetry
    (negative ΔL means the perturbation accidentally found a better point,
    i.e., the minimum is in a wide flat valley).
    """
    set_global_seed(getattr(cfg, "seed", 42))
    model.eval()
    loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=2)

    # Baseline loss
    def _eval_loss():
        total = 0.0; n = 0
        with torch.no_grad():
            for x0, lbl in loader:
                x0, lbl = x0.to(cfg.device), lbl.to(cfg.device)
                B  = x0.size(0)
                t  = torch.full((B,), 0.5, device=cfg.device)
                xt, _, _ = fwd.corrupt(x0, t, y=lbl)
                cin = fwd.c_in(t).view(-1, 1, 1, 1)
                pred = model(xt * cin, t, lbl).float()
                total += compute_loss(pred, x0.float(), loss_type).item() * B
                n     += B
        return total / n

    baseline = _eval_loss()

    # Save original state
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    delta_losses: list[float] = []
    for _ in range(n_perturbations):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * perturb_sigma)
        perturbed = _eval_loss()
        delta_losses.append(perturbed - baseline)
        model.load_state_dict(orig_state)

    dl = np.array(delta_losses)
    return {
        "baseline_loss":   baseline,
        "sharpness":       float(np.mean(np.abs(dl)) / perturb_sigma),
        "mean_delta":      float(dl.mean()),
        "std_delta":       float(dl.std()),
        "frac_negative":   float((dl < 0).mean()),   # fraction in better basin
        "perturb_sigma":   perturb_sigma,
    }


def measure_update_whiteness(
        model:      nn.Module,
        fwd:        RosenblattForward,
        train_ds,
        cfg:        Config,
        opt_name:   str   = "adamw",
        n_batches:  int   = 50,
) -> dict[str, Any]:
    """
    Experiment σ core measurement: tracks the distribution of effective
    parameter updates at the bottleneck layer for one pass of n_batches.

    For Adam:  effective_update_k = m̂_k / (√v̂_k + ε)
               → should be ≈ N(0,1) per-coordinate
    For Lion:  effective_update_k = sign(c_k) ∈ {-1, +1}
               → Bernoulli(½), κ₄ = -2
    For SGD:   effective_update_k = g_k
               → same distribution as the gradient

    Returns per-coordinate statistics of the effective update over n_batches.
    """
    set_global_seed(getattr(cfg, "seed", 42))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                        shuffle=True, num_workers=2)

    # Collect effective updates by hooking the parameter tensors' deltas
    # We'll track mid2.conv2.weight as representative of the bottleneck
    target_param = model.mid2.conv2.weight  # (C_out, C_in, k, k)

    updates_collected: list[torch.Tensor] = []
    prev_param = target_param.detach().clone()

    opt = _make_optimizer(opt_name, model, cfg)
    model.train()
    sfn = sigma_multiplicative()

    for i, (x0, lbl) in enumerate(loader):
        if i >= n_batches:
            break
        x0, lbl = x0.to(cfg.device), lbl.to(cfg.device)
        B = x0.size(0)
        t = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
        xt, _, _ = fwd.corrupt(x0, t, y=lbl)
        cin = fwd.c_in(t).view(-1, 1, 1, 1)
        opt.zero_grad(set_to_none=True)
        loss = F.huber_loss(model(xt * cin, t, lbl), x0)
        loss.backward()

        # Save gradient before update
        grad = target_param.grad.detach().flatten().cpu()

        opt.step()
        curr_param = target_param.detach().clone()

        # Effective update = parameter change (captures the optimizer's output)
        effective = (curr_param - prev_param).flatten().cpu()
        updates_collected.append(effective)
        prev_param = curr_param

    model.eval()
    updates = torch.stack(updates_collected)   # (n_batches, D)
    cum = compute_marginal_cumulants(updates)

    # Per-coordinate std of effective updates (should be ≈ 1 for Adam whitening)
    per_coord_std  = updates.std(dim=0).numpy()

    return {
        "kappa4_updates":     cum["mean_kappa4"],
        "kappa3_updates":     cum["mean_abs_kappa3"],
        "std_updates_mean":   float(per_coord_std.mean()),
        "update_std_cv":      float(per_coord_std.std() / (per_coord_std.mean() + 1e-8)),
        "frac_nong_updates":  cum["frac_non_gauss"],
        # Wasserstein-1 distance of per-batch update distribution from N(0,1)
        # approximated by comparing quantiles
        "w1_from_normal":     _w1_from_normal(updates.flatten().numpy()),
    }


def _w1_from_normal(arr: np.ndarray, n_quantiles: int = 100) -> float:
    """
    Approximate W₁(empirical, N(0,1)) via quantile comparison.
    Lower = closer to standard normal.
    """
    # Standardise first
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    # Empirical quantiles
    q_emp = np.percentile(arr, np.linspace(1, 99, n_quantiles))
    # Theoretical N(0,1) quantiles
    from scipy.stats import norm
    q_nor = norm.ppf(np.linspace(0.01, 0.99, n_quantiles))
    return float(np.mean(np.abs(q_emp - q_nor)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Experiment ο — Optimiser comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OmicronResult:
    opt_name:    str
    noise_type:  str
    label:       str
    # Bottleneck geometry
    bn_kappa4:   float
    bn_kappa3:   float
    bn_pr:       float
    bn_mardia_z: float
    # Reconstruction quality
    val_l1:      float
    val_l2:      float
    # Landscape
    sharpness:   float
    frac_neg:    float   # fraction of perturbations that find better loss
    # Gradient update statistics (Experiment σ)
    update_k4:   float
    update_w1:   float
    update_std_cv: float


def run_experiment_omicron(
        cfg:        Config,
        save_dir:   Path,
        opt_names:  list[str] | None = None,
        noise_types:list[str] | None = None,
) -> list[OmicronResult]:
    """
    Train identical models with different optimisers, measure representation
    geometry and loss landscape sharpness.

    Theoretical predictions:
        AdamW:   bn_kappa4 ≈ 0  (gradient whitening → Gaussian)
        Lion:    bn_kappa4 < 0? (sign updates → sub-Gaussian)
        SGD:     bn_kappa4 ≠ 0? (no whitening → retains gradient distribution)
        RMSprop: bn_kappa4 ≈ 0  (shares Adam's whitening)
        NAdam:   bn_kappa4 ≈ 0  (Adam variant)
    """
    if opt_names   is None: opt_names   = list(OPT_LABELS)
    if noise_types is None: noise_types = ["gaussian", "rosenblatt"]

    print("\n" + "═" * 72)
    print("Experiment ο — Optimiser Comparison")
    print("═" * 72)
    print("Theory: Adam whitens gradients → Gaussian geometry.")
    print("        Lion uses sign(·)      → Bernoulli (anti-Gaussian)?")
    print("        SGD: no whitening      → preserves gradient distribution?")

    set_global_seed(getattr(cfg, "seed", 42))

    test_ds  = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    train_ds = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    rows: list[OmicronResult] = []

    for noise_type in noise_types:
        fwd = _make_fwd(noise_type, cfg)
        for opt_name in opt_names:
            tag = f"omicron_{noise_type}_{opt_name}"
            print(f"\n── noise={noise_type}  opt={opt_name} ────────────────────")

            model, fwd2, _, _ = load_or_train_opt(
                tag, cfg, save_dir,
                opt_name=opt_name, noise_type=noise_type,
                log_grads=False,
                use_pretrained_baseline=(opt_name == "adamw"))

            # Bottleneck cumulants
            m = measure_bottleneck(model, fwd2, test_ds, cfg)

            # Landscape sharpness
            sharp = measure_sharpness(model, fwd2, test_ds, cfg)

            # Effective update distribution (σ experiment)
            upd = measure_update_whiteness(model, fwd2, train_ds, cfg,
                                           opt_name=opt_name, n_batches=30)

            row = OmicronResult(
                opt_name    = opt_name,
                noise_type  = noise_type,
                label       = OPT_LABELS[opt_name],
                bn_kappa4   = m["kappa4"],
                bn_kappa3   = m["kappa3"],
                bn_pr       = m["pr"],
                bn_mardia_z = m["mardia_z"],
                val_l1      = m["val_l1"],
                val_l2      = m["val_l2"],
                sharpness   = sharp["sharpness"],
                frac_neg    = sharp["frac_negative"],
                update_k4   = upd["kappa4_updates"],
                update_w1   = upd["w1_from_normal"],
                update_std_cv = upd["update_std_cv"],
            )
            rows.append(row)
            print(f"  bn_κ4={m['kappa4']:+.3f}  PR={m['pr']:.1f}  "
                  f"Z={m['mardia_z']:+.2f}  sharp={sharp['sharpness']:.4f}"
                  f"  upd_κ4={upd['kappa4_updates']:+.3f}")

            _save_omicron_csv(rows, save_dir / "omicron_optimizer.csv")

    _save_omicron_csv(rows,  save_dir / "omicron_optimizer.csv")
    _save_omicron_latex(rows, save_dir / "omicron_optimizer.tex")
    _plot_omicron(rows,       save_dir)
    return rows


def _save_omicron_csv(rows: list[OmicronResult], path: Path) -> None:
    fields = list(OmicronResult.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v
                        for k, v in asdict(r).items()})


def _save_omicron_latex(rows: list[OmicronResult], path: Path) -> None:
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Experiment~$\omicron$: optimiser comparison.",
        r"Theory predicts Adam/RMSprop whitening $\Rightarrow$ Gaussian",
        r"($\bar\kappa_4\approx 0$); Lion sign updates $\Rightarrow$",
        r"sub-Gaussian ($\bar\kappa_4 < 0$)?}",
        r"\label{tab:omicron}",
        r"\begin{tabular}{ll rr rr rr}",
        r"\toprule",
        r"Noise & Optimiser & $\bar\kappa_4^{\rm bn}$ & PR & Mardia-$Z$"
        r" & $L_1$ & Sharp & $\kappa_4^{\rm upd}$ \\",
        r"\midrule",
    ]
    prev_nt = None
    for r in rows:
        if prev_nt is not None and r.noise_type != prev_nt:
            lines.append(r"\midrule")
        prev_nt = r.noise_type
        lines.append(
            f"{r.noise_type} & {r.label.split('(')[0].strip()} & "
            f"{r.bn_kappa4:+.3f} & {r.bn_pr:.1f} & {r.bn_mardia_z:+.2f} & "
            f"{r.val_l1:.4f} & {r.sharpness:.4f} & {r.update_k4:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Experiment π — Gradient noise distribution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PiResult:
    noise_dist:  str
    noise_std:   float
    noise_type:  str
    label:       str
    bn_kappa4:   float
    bn_kappa3:   float
    bn_pr:       float
    bn_mardia_z: float
    val_l1:      float


def run_experiment_pi(
        cfg:         Config,
        save_dir:    Path,
        noise_dists: list[str]   | None = None,
        noise_stds:  list[float] | None = None,
) -> list[PiResult]:
    """
    Gradient noise distribution ablation.

    Key question: does the DISTRIBUTION of injected gradient noise propagate
    into the representation geometry?

    If yes:
        gaussian gradient noise  → more Gaussian bottleneck
        rosenblatt gradient noise → more non-Gaussian (κ₄ > 0) bottleneck
        lion-like (Bernoulli) noise → sub-Gaussian (κ₄ < 0) bottleneck?

    If no (κ₄ ≈ 0 regardless): Adam's second-moment normalisation washes out
    the gradient noise distribution before it can affect representations.
    This would be the stronger result: Adam imposes Gaussian geometry even
    when the gradient noise is Rosenblatt.
    """
    if noise_dists is None: noise_dists = list(NOISE_LABELS)
    if noise_stds  is None: noise_stds  = [0.0, 1e-4, 1e-3, 1e-2]

    print("\n" + "═" * 72)
    print("Experiment π — Gradient Noise Distribution")
    print("═" * 72)
    print("Does Rosenblatt gradient noise propagate to representation geometry?")
    print("Or does Adam's second-moment normalisation absorb it?")

    set_global_seed(getattr(cfg, "seed", 42))

    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    rows: list[PiResult] = []

    for noise_type in ("gaussian", "rosenblatt"):
        fwd = _make_fwd(noise_type, cfg)
        for dist in noise_dists:
            for std in (noise_stds if dist != "none" else [0.0]):
                tag = f"pi_{noise_type}_{dist}_std{str(std).replace('.','p')}"
                print(f"\n── noise={noise_type}  grad_dist={dist}  σ={std}")

                model, fwd2, _, _ = load_or_train_opt(
                    tag, cfg, save_dir,
                    opt_name="noise_adamw",
                    noise_type=noise_type,
                    noise_std=std,
                    noise_dist=dist,
                    use_pretrained_baseline=(dist == "none"))

                m = measure_bottleneck(model, fwd2, test_ds, cfg)
                rows.append(PiResult(
                    noise_dist  = dist,
                    noise_std   = std,
                    noise_type  = noise_type,
                    label       = NOISE_LABELS.get(dist, dist),
                    bn_kappa4   = m["kappa4"],
                    bn_kappa3   = m["kappa3"],
                    bn_pr       = m["pr"],
                    bn_mardia_z = m["mardia_z"],
                    val_l1      = m["val_l1"],
                ))
                print(f"  κ₄={m['kappa4']:+.3f}  PR={m['pr']:.1f}  "
                      f"Z={m['mardia_z']:+.2f}  L1={m['val_l1']:.4f}")

                _save_pi_csv(rows, save_dir / "pi_grad_noise.csv")

    _save_pi_csv(rows, save_dir / "pi_grad_noise.csv")
    _plot_pi(rows, save_dir)
    return rows


def _save_pi_csv(rows: list[PiResult], path: Path) -> None:
    fields = list(PiResult.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v
                        for k, v in asdict(r).items()})


# ─────────────────────────────────────────────────────────────────────────────
# 7. Experiment ρ — Rosenblatt-SGLD landscape analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RhoResult:
    phase:          str   # "before" / "after"
    noise_type:     str
    grad_noise:     str
    # Bottleneck geometry
    bn_kappa4:      float
    bn_kappa3:      float
    bn_pr:          float
    bn_mardia_z:    float
    # Landscape
    val_l1:         float
    sharpness:      float
    frac_neg:       float


def run_experiment_rho(
        cfg:         Config,
        save_dir:    Path,
        fine_tune_epochs: int   = 10,
        noise_stds:  list[float]| None = None,
) -> list[RhoResult]:
    """
    Experiment ρ: Rosenblatt-SGLD landscape analysis.

    Protocol:
    1. Load a pre-trained AdamW model (from omicron or from beta experiment).
    2. Measure BEFORE: bottleneck κ₄ and landscape sharpness.
    3. Fine-tune for fine_tune_epochs with CustomNoiseAdamW injecting
       Rosenblatt gradient noise at σ=noise_std.
    4. Measure AFTER: bottleneck κ₄ and landscape sharpness.

    Key comparisons:
        Gaussian gradient noise  (control)
        Rosenblatt gradient noise (treatment)
        No noise                 (stability baseline)

    Two-level outcome:
        (a) Sharpness: does Rosenblatt noise help escape sharp minima?
            (heavy-tail SGD literature predicts: yes)
        (b) Geometry:  does Rosenblatt noise change representation κ₄?
            (gradient-distribution → representation-distribution: maybe yes)

    If (a) yes and (b) no: Rosenblatt noise helps optimisation WITHOUT
    changing the Gaussian character of representations.  Suggests the two
    phenomena (optimisation landscape + representation geometry) are
    causally decoupled.

    If (a) yes and (b) yes: Rosenblatt noise both helps optimisation AND
    makes representations less Gaussian.  Strong mechanistic claim.
    """
    if noise_stds is None: noise_stds = [1e-3]

    print("\n" + "═" * 72)
    print("Experiment ρ — Rosenblatt-SGLD Landscape Analysis")
    print("  Does Rosenblatt gradient noise help escape sharp minima (heavy-tail SGD)?")
    print("  Does it change the bottleneck geometry?")
    print("═" * 72)

    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    rows: list[RhoResult] = []

    for noise_type in ("gaussian", "rosenblatt"):
        fwd = _make_fwd(noise_type, cfg)

        # Load or train the baseline AdamW model
        base_tag  = f"omicron_{noise_type}_adamw"
        base_ckpt = save_dir / "optimizer_ablation" / f"{base_tag}_final.pt"
        base_model = ConditionalUNetAblation(num_classes=10,
                                              base_ch=cfg.base_ch).to(cfg.device)
        if base_ckpt.exists():
            print(f"\n  Loading baseline AdamW: {base_ckpt}")
            base_model.load_state_dict(
                torch.load(base_ckpt, map_location=cfg.device, weights_only=True))
        else:
            print(f"\n  Baseline not found; training AdamW first …")
            base_model, fwd, _, _ = load_or_train_opt(
                base_tag, cfg, save_dir, opt_name="adamw",
                noise_type=noise_type,
                use_pretrained_baseline=True)
        base_model.eval()

        for grad_noise in ("none", "gaussian", "rosenblatt"):
            for std in (noise_stds if grad_noise != "none" else [0.0]):
                print(f"\n── data_noise={noise_type}  grad_noise={grad_noise}"
                      f"  σ={std} ────────────────")

                # Measure BEFORE fine-tuning
                m_before = measure_bottleneck(base_model, fwd, test_ds, cfg)
                s_before = measure_sharpness(base_model, fwd, test_ds, cfg)

                rows.append(RhoResult(
                    phase       = "before",
                    noise_type  = noise_type,
                    grad_noise  = f"{grad_noise}(σ={std})",
                    bn_kappa4   = m_before["kappa4"],
                    bn_kappa3   = m_before["kappa3"],
                    bn_pr       = m_before["pr"],
                    bn_mardia_z = m_before["mardia_z"],
                    val_l1      = m_before["val_l1"],
                    sharpness   = s_before["sharpness"],
                    frac_neg    = s_before["frac_negative"],
                ))
                print(f"  BEFORE: κ4={m_before['kappa4']:+.3f}  "
                      f"sharp={s_before['sharpness']:.4f}")

                # Fine-tune with Rosenblatt/Gaussian/no gradient noise
                ft_model = ConditionalUNetAblation(
                    num_classes=10, base_ch=cfg.base_ch).to(cfg.device)
                ft_model.load_state_dict(base_model.state_dict())

                ft_cfg       = Config()
                ft_cfg.__dict__.update(cfg.__dict__)
                ft_cfg.epochs = fine_tune_epochs
                ft_cfg.lr     = cfg.lr / 5.0   # lower lr for fine-tuning

                ft_tag   = (f"rho_{noise_type}_{grad_noise}_"
                            f"std{str(std).replace('.','p')}_ft")
                ft_ckpt  = save_dir / "optimizer_ablation" / f"{ft_tag}_final.pt"

                if ft_ckpt.exists():
                    ft_model.load_state_dict(
                        torch.load(ft_ckpt, map_location=cfg.device,
                                   weights_only=True))
                else:
                    ft_model, _, _ = train_with_optimizer(
                        ft_model, fwd, ft_cfg, ft_ckpt,
                        opt_name="noise_adamw",
                                loss_type="huber",
                        noise_type=noise_type,
                        noise_std=std,
                        noise_dist=grad_noise,
                        tag=ft_tag)

                ft_model.eval()

                # Measure AFTER fine-tuning
                m_after = measure_bottleneck(ft_model, fwd, test_ds, cfg)
                s_after = measure_sharpness(ft_model, fwd, test_ds, cfg)

                rows.append(RhoResult(
                    phase       = "after",
                    noise_type  = noise_type,
                    grad_noise  = f"{grad_noise}(σ={std})",
                    bn_kappa4   = m_after["kappa4"],
                    bn_kappa3   = m_after["kappa3"],
                    bn_pr       = m_after["pr"],
                    bn_mardia_z = m_after["mardia_z"],
                    val_l1      = m_after["val_l1"],
                    sharpness   = s_after["sharpness"],
                    frac_neg    = s_after["frac_negative"],
                ))
                print(f"  AFTER:  κ4={m_after['kappa4']:+.3f}  "
                      f"sharp={s_after['sharpness']:.4f}  "
                      f"Δsharp={s_after['sharpness']-s_before['sharpness']:+.4f}  "
                      f"Δκ4={m_after['kappa4']-m_before['kappa4']:+.4f}")

                _save_rho_csv(rows, save_dir / "rho_rosenblatt_sgld.csv")

    _save_rho_csv(rows, save_dir / "rho_rosenblatt_sgld.csv")
    _save_rho_latex(rows, save_dir / "rho_rosenblatt_sgld.tex")
    _plot_rho(rows, save_dir)
    return rows


def _save_rho_csv(rows: list[RhoResult], path: Path) -> None:
    fields = list(RhoResult.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v
                        for k, v in asdict(r).items()})


def _save_rho_latex(rows: list[RhoResult], path: Path) -> None:
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Experiment~$\rho$: Rosenblatt-SGLD fine-tuning.",
        r"Before/after comparison of bottleneck $\bar\kappa_4$ and",
        r"landscape sharpness.  A negative $\Delta$sharpness indicates",
        r"the optimiser found a flatter (better-generalising) minimum.}",
        r"\label{tab:rho}",
        r"\begin{tabular}{lll rr rr rr}",
        r"\toprule",
        r"Data noise & Grad noise & Phase & $\bar\kappa_4$ & $\Delta\kappa_4$"
        r" & Sharpness & $\Delta$Sharp & $L_1$ & Frac$(-) \\",
        r"\midrule",
    ]
    # Group by (noise_type, grad_noise) pairs
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r.noise_type, r.grad_noise)
        if key not in seen:
            seen[key] = {"before": None, "after": None}
        seen[key][r.phase] = r

    for key, phases in seen.items():
        bf, af = phases.get("before"), phases.get("after")
        if bf is None or af is None:
            continue
        dk4   = af.bn_kappa4  - bf.bn_kappa4
        dsharp= af.sharpness  - bf.sharpness
        lines.append(
            f"{bf.noise_type} & {bf.grad_noise} & before & "
            f"{bf.bn_kappa4:+.3f} & — & {bf.sharpness:.4f} & — & "
            f"{bf.val_l1:.4f} & {bf.frac_neg:.2f} \\\\")
        lines.append(
            f" & & after & "
            f"{af.bn_kappa4:+.3f} & {dk4:+.3f} & {af.sharpness:.4f} & "
            f"{dsharp:+.4f} & {af.val_l1:.4f} & {af.frac_neg:.2f} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Experiment τ — Gradient κ₄ evolution during training
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_tau_grad_evolution(
        cfg:         Config,
        save_dir:    Path,
        opt_names:   list[str] | None = None,
        log_every:   int = 50,
) -> dict[str, list[dict]]:
    """
    Track gradient κ₄ at the bottleneck (mid2) during training for each
    optimiser.  Reveals whether gradient non-Gaussianity and representation
    non-Gaussianity are correlated.

    Expected:
        AdamW:  gradient κ₄ → 0 quickly (whitening normalises heavy tails)
        Lion:   gradient κ₄ → -2 quickly (sign always gives Bernoulli)
        SGD:    gradient κ₄ tracks loss curvature (non-trivially non-Gaussian)
    """
    if opt_names is None: opt_names = ["adamw", "lion", "sgd"]

    print("\n" + "═" * 72)
    print("Experiment τ — Gradient κ₄ Evolution During Training")
    print("═" * 72)

    tau_dir = save_dir / "tau_grad_evolution"
    tau_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}

    for opt_name in opt_names:
        tag = f"tau_rosenblatt_{opt_name}"
        print(f"\n── opt={opt_name} ────────────────────────────────────────────")

        fwd = _make_fwd("rosenblatt", cfg)
        ckpt = tau_dir / f"{tag}_final.pt"
        model = ConditionalUNetAblation(num_classes=10,
                                         base_ch=cfg.base_ch).to(cfg.device)

        if ckpt.exists():
            model.load_state_dict(
                torch.load(ckpt, map_location=cfg.device, weights_only=True))
            model.eval()
            # Load gradient log if exists
            grad_csv = tau_dir / f"{tag}_gradlog.csv"
            if grad_csv.exists():
                with open(grad_csv) as f:
                    results[opt_name] = list(csv.DictReader(f))
                    for r in results[opt_name]:
                        r["step"]   = int(r["step"])
                        r["kappa4"] = float(r["kappa4"])
                print(f"  Loaded gradient log: {len(results[opt_name])} entries")
                continue

        _, history, grad_log = train_with_optimizer(
            model, fwd, cfg, ckpt,
            opt_name=opt_name, loss_type="huber",
            noise_type="rosenblatt",
            log_grads=True, log_every=log_every, tag=tag)

        results[opt_name] = grad_log

        # Save gradient log
        grad_csv = tau_dir / f"{tag}_gradlog.csv"
        with open(grad_csv, "w", newline="") as f:
            if grad_log:
                w = csv.DictWriter(f, fieldnames=grad_log[0].keys())
                w.writeheader();  w.writerows(grad_log)
        print(f"  Gradient log: {len(grad_log)} entries")

    _plot_tau(results, save_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_omicron(rows: list[OmicronResult], save_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if r.noise_type == noise_type]
        if not sub: continue
        col = COLORS[noise_type]
        x   = np.arange(len(sub))
        off = 0.2 if noise_type == "rosenblatt" else -0.2
        lbl = noise_type.capitalize()

        axes[0,0].bar(x+off, [r.bn_kappa4   for r in sub], 0.35, color=col, alpha=0.8, label=lbl)
        axes[0,1].bar(x+off, [r.sharpness   for r in sub], 0.35, color=col, alpha=0.8, label=lbl)
        axes[0,2].bar(x+off, [r.update_k4   for r in sub], 0.35, color=col, alpha=0.8, label=lbl)
        axes[1,0].bar(x+off, [r.update_w1   for r in sub], 0.35, color=col, alpha=0.8, label=lbl)
        axes[1,1].bar(x+off, [r.bn_pr       for r in sub], 0.35, color=col, alpha=0.8, label=lbl)
        axes[1,2].bar(x+off, [r.val_l1      for r in sub], 0.35, color=col, alpha=0.8, label=lbl)

    xlabels = [r.label.split("(")[0].strip() for r in
               [r for r in rows if r.noise_type == "rosenblatt"]]

    configs = [
        (axes[0,0], "$\\bar\\kappa_4$ at bottleneck",
         "Theory: Adam→0, Lion→<0, SGD→depends on gradient"),
        (axes[0,1], "Sharpness $\\mathbb{E}[|\\Delta L|]/\\sigma$",
         "Lower = flatter minimum"),
        (axes[0,2], "$\\kappa_4$ of effective parameter updates",
         "Theory: Adam→0 (whitening), Lion→-2 (Bernoulli)"),
        (axes[1,0], "$W_1$ (updates, $\\mathcal{N}(0,1)$)",
         "Lower = more Gaussian effective updates"),
        (axes[1,1], "Bottleneck PR",
         "Effective dimensionality"),
        (axes[1,2], "$L_1$ val loss",
         "Reconstruction quality"),
    ]
    for ax, ylabel, title in configs:
        ax.axhline(0, color="red", lw=1.0, ls="--", alpha=0.6)
        if xlabels:
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)

    # Annotate Adam whitening
    axes[0,2].text(0.5, 0.95,
                   "Adam: W₁ ≈ 0 (whitened → Gaussian)\n"
                   "Lion: κ₄ ≈ -2 (Bernoulli, anti-Gaussian)\n"
                   "SGD:  κ₄ depends on loss",
                   transform=axes[0,2].transAxes, fontsize=7, va="top",
                   bbox=dict(boxstyle="round", fc="lightyellow", ec="gold", alpha=0.9))

    fig.suptitle("Experiment ο: Optimiser comparison — "
                 "does Adam whitening impose Gaussian geometry?", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(save_dir / f"omicron_optimizer.{ext}",
                    bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  → Saved omicron_optimizer.pdf/png")


def _plot_pi(rows: list[PiResult], save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if r.noise_type == noise_type]
        if not sub: continue
        col = COLORS[noise_type]

        # Group by noise_dist, average over std
        dists = sorted({r.noise_dist for r in sub})
        for d in dists:
            dsub = sorted([r for r in sub if r.noise_dist == d],
                          key=lambda r: r.noise_std)
            if not dsub: continue
            stds  = [r.noise_std  for r in dsub]
            k4s   = [r.bn_kappa4  for r in dsub]
            l1s   = [r.val_l1     for r in dsub]
            axes[0].plot(stds, k4s, marker="o", lw=1.8, color=col,
                         label=f"{noise_type}/{d}")
            axes[1].plot(stds, l1s, marker="o", lw=1.8, color=col,
                         label=f"{noise_type}/{d}")

    for ax, ylabel, title in [
        (axes[0], "$\\bar\\kappa_4$ at bottleneck",
         "Does gradient noise dist propagate to representation?\n"
         "(Rosenblatt noise → higher κ₄? Or Adam absorbs it?)"),
        (axes[1], "$L_1$ val loss",
         "Reconstruction quality vs gradient noise σ"),
    ]:
        ax.axhline(0, color="red", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("Gradient noise σ")
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Experiment π: Gradient noise distribution", fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(save_dir / f"pi_grad_noise.{ext}",
                    bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  → Saved pi_grad_noise.pdf/png")


def _plot_rho(rows: list[RhoResult], save_dir: Path) -> None:
    """
    Before/after arrows for sharpness and κ₄.
    Each condition gets a separate arrow from (before) to (after).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = {
        "none(σ=0)":       ("gray",     "None (baseline)"),
        "gaussian(σ=0.001)":("#3A7EBF", "Gaussian noise"),
        "rosenblatt(σ=0.001)":("#E07B39","Rosenblatt noise"),
    }

    for noise_type in ("gaussian", "rosenblatt"):
        sub = [r for r in rows if r.noise_type == noise_type]
        if not sub: continue
        marker = "o" if noise_type == "gaussian" else "s"
        y_offset = 0.0 if noise_type == "gaussian" else 0.02

        for grad_noise, (col, lbl) in configs.items():
            bf = next((r for r in sub if r.phase == "before" and
                       r.grad_noise == grad_noise), None)
            af = next((r for r in sub if r.phase == "after" and
                       r.grad_noise == grad_noise), None)
            if bf is None or af is None:
                continue
            for ax_idx, (bval, aval, ax) in enumerate([
                (bf.sharpness,  af.sharpness,  axes[0]),
                (bf.bn_kappa4,  af.bn_kappa4,  axes[1]),
            ]):
                ydelta = (1 if ax_idx == 0 else 0) * y_offset
                ax.annotate("", xy=(aval, ax_idx + ydelta),
                            xytext=(bval, ax_idx + ydelta),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
                ax.plot([bval], [ax_idx + ydelta], "o", color=col, ms=6)
                ax.plot([aval], [ax_idx + ydelta], "s", color=col, ms=6)

    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Gaussian data", "Rosenblatt data"])
    axes[0].set_xlabel("Sharpness (lower = flatter minimum)")
    axes[0].set_title("Experiment ρ: Sharpness before→after Rosenblatt-SGLD\n"
                      "Left dot = before, right dot = after", fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Gaussian data", "Rosenblatt data"])
    axes[1].set_xlabel("Bottleneck $\\bar\\kappa_4$")
    axes[1].set_title("κ₄ before→after Rosenblatt-SGLD\n"
                      "Does Rosenblatt noise un-Gaussianize representations?", fontsize=9)
    axes[1].axvline(0, color="red", lw=1.0, ls="--", alpha=0.6)
    axes[1].grid(axis="x", alpha=0.3)

    # Legend
    handles = [plt.Line2D([0],[0], color=col, lw=2, label=lbl)
               for _, (col, lbl) in configs.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Experiment ρ: Rosenblatt-SGLD — "
                 "escape sharp minima AND change geometry?", fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(save_dir / f"rho_rosenblatt_sgld.{ext}",
                    bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  → Saved rho_rosenblatt_sgld.pdf/png")


def _plot_tau(results: dict[str, list[dict]], save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    opt_colors = {
        "adamw":  "#3A7EBF",
        "lion":   "#E07B39",
        "sgd":    "#55A868",
        "rmsprop":"#C44E52",
    }

    for opt_name, log in results.items():
        if not log: continue
        col  = opt_colors.get(opt_name, "#888888")
        steps = [d["step"]    for d in log]
        k4s   = [d["kappa4"]  for d in log]
        norms = [d["grad_norm"]for d in log]

        axes[0].plot(steps, k4s,   color=col, lw=1.5,
                     label=OPT_LABELS.get(opt_name, opt_name).split("(")[0].strip())
        axes[1].plot(steps, norms, color=col, lw=1.5,
                     label=OPT_LABELS.get(opt_name, opt_name).split("(")[0].strip())

    axes[0].axhline(0,  color="red",  lw=1.0, ls="--", alpha=0.6,
                    label="$\\kappa_4=0$")
    axes[0].axhline(-2, color="gray", lw=0.8, ls=":", alpha=0.5,
                    label="$\\kappa_4=-2$ (Bernoulli)")

    for ax, ylabel, title in [
        (axes[0], "Gradient $\\kappa_4$ at bottleneck",
         "Gradient κ₄ evolution — does Adam normalise toward Gaussian?\n"
         "Lion should converge to κ₄ ≈ -2"),
        (axes[1], "Gradient L2 norm at bottleneck",
         "Gradient norm evolution"),
    ]:
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Experiment τ: Gradient κ₄ evolution during training\n"
                 "(Rosenblatt model, Smooth L1 loss)", fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(save_dir / f"tau_grad_evolution.{ext}",
                    bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  → Saved tau_grad_evolution.pdf/png")


def plot_optimizer_summary(save_dir: Path) -> None:
    """Four-panel summary combining ο, π, ρ key results."""
    fig = plt.figure(figsize=(16, 9))
    gs  = mgs.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])   # ο: optimizer comparison
    ax2 = fig.add_subplot(gs[0, 1])   # σ: update whiteness
    ax3 = fig.add_subplot(gs[1, 0])   # ρ: sharpness delta
    ax4 = fig.add_subplot(gs[1, 1])   # π: gradient noise effect

    def _try_load(path: Path, cls):
        if not path.exists(): return []
        with open(path) as f:
            rows = []
            for row in csv.DictReader(f):
                try:
                    kwargs = {}
                    for k, v in row.items():
                        try: kwargs[k] = float(v)
                        except: kwargs[k] = v
                    rows.append(cls(**kwargs))
                except Exception: pass
        return rows

    omicron_rows = _try_load(save_dir/"omicron_optimizer.csv", OmicronResult)
    pi_rows      = _try_load(save_dir/"pi_grad_noise.csv",    PiResult)
    rho_rows     = _try_load(save_dir/"rho_rosenblatt_sgld.csv", RhoResult)

    # Panel 1: Optimizer bottleneck κ4
    sub = [r for r in omicron_rows if r.noise_type == "rosenblatt"]
    if sub:
        x = np.arange(len(sub))
        ax1.bar(x, [r.bn_kappa4 for r in sub],
                color=[COLORS["rosenblatt"]] * len(sub), alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([r.label.split("(")[0].strip() for r in sub],
                            rotation=35, ha="right", fontsize=7.5)
        ax1.axhline(0, color="red", lw=1, ls="--", alpha=0.6)
        ax1.set_ylabel("$\\bar\\kappa_4$ at bottleneck")
        ax1.set_title("(A)  Optimiser comparison (ο)\n"
                      "Does Adam whitening → Gaussian?", fontweight="bold", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Update whiteness (W1 from N(0,1))
    sub = [r for r in omicron_rows if r.noise_type == "rosenblatt"]
    if sub:
        x = np.arange(len(sub))
        ax2.bar(x, [r.update_w1 for r in sub],
                color=COLORS["gaussian"], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r.label.split("(")[0].strip() for r in sub],
                            rotation=35, ha="right", fontsize=7.5)
        ax2.set_ylabel("$W_1$(updates, $\\mathcal{N}(0,1)$)")
        ax2.set_title("(B)  Update distribution whiteness (σ)\n"
                      "Lower = more Gaussian effective updates",
                      fontweight="bold", fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Rosenblatt-SGLD sharpness change
    rho_pairs: dict[str, dict] = {}
    for r in rho_rows:
        if r.noise_type != "rosenblatt": continue
        k = r.grad_noise
        if k not in rho_pairs: rho_pairs[k] = {}
        rho_pairs[k][r.phase] = r
    if rho_pairs:
        labels = list(rho_pairs.keys())
        delta_sharp = []
        delta_k4    = []
        for lbl in labels:
            bf = rho_pairs[lbl].get("before")
            af = rho_pairs[lbl].get("after")
            if bf and af:
                delta_sharp.append(af.sharpness - bf.sharpness)
                delta_k4.append(af.bn_kappa4 - bf.bn_kappa4)
            else:
                delta_sharp.append(0); delta_k4.append(0)
        x = np.arange(len(labels))
        ax3.bar(x - 0.2, delta_sharp, 0.38, color=COLORS["rosenblatt"],
                alpha=0.8, label="ΔSharpness")
        ax3_r = ax3.twinx()
        ax3_r.bar(x + 0.2, delta_k4, 0.38, color=COLORS["gaussian"],
                  alpha=0.6, label="Δκ₄")
        ax3.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax3.set_ylabel("ΔSharpness (neg = flatter)", fontsize=8.5)
        ax3_r.set_ylabel("Δκ₄ (neg = more Gaussian)", fontsize=8.5, color="gray")
        ax3.set_title("(C)  Rosenblatt-SGLD before/after (ρ)\n"
                      "Rosenblatt data noise, varying gradient noise",
                      fontweight="bold", fontsize=9)
        ax3.grid(alpha=0.3)

    # Panel 4: π gradient noise
    sub = [r for r in pi_rows if r.noise_type == "rosenblatt"
           and r.noise_dist in ("none","gaussian","rosenblatt")]
    if sub:
        for dist in ("none", "gaussian", "rosenblatt"):
            dsub = sorted([r for r in sub if r.noise_dist == dist],
                          key=lambda r: r.noise_std)
            if dsub:
                col = {"none":"gray","gaussian":COLORS["gaussian"],
                       "rosenblatt":COLORS["rosenblatt"]}[dist]
                ax4.plot([r.noise_std for r in dsub],
                         [r.bn_kappa4 for r in dsub],
                         color=col, marker="o", lw=1.8, label=dist)
        ax4.axhline(0, color="red", lw=1.0, ls="--", alpha=0.6)
        ax4.set_xlabel("Gradient noise σ")
        ax4.set_ylabel("$\\bar\\kappa_4$ at bottleneck")
        ax4.set_title("(D)  Gradient noise type (π)\n"
                      "Does noise dist propagate to representations?",
                      fontweight="bold", fontsize=9)
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

    fig.suptitle("Optimizer geometry summary: "
                 "Adam whitening → Gaussian representations; "
                 "Lion → anti-Gaussian; Rosenblatt-SGLD → landscape benefits?",
                 fontsize=10)
    for ext in ("pdf", "png"):
        plt.savefig(save_dir / f"optimizer_summary.{ext}",
                    bbox_inches="tight", dpi=160)
    plt.close()
    print(f"  → Saved optimizer_summary.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_fwd(noise_type: str, cfg: Config) -> RosenblattForward:
    sfn = sigma_multiplicative()
    fwd = RosenblattForward(sfn, noise_type=noise_type,
                            H=cfg.H, device=cfg.device,
                            sigma_max=cfg.sigma_max)
    fwd.set_eg2(float(getattr(sfn, "eg2", 1.0)))
    return fwd


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(args: argparse.Namespace) -> tuple[Config, Path]:
    cfg = Config()
    cfg.dataset     = args.dataset
    cfg.epochs      = args.epochs
    cfg.batch_size  = 128
    cfg.no_evaluate = True
    cfg.no_plot     = True
    cfg.n_steps     = 50
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return cfg, save_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optimizer geometry experiments for Rosenblatt Cold Diffusion")
    p.add_argument("--save_dir",  default="output/diffusion/multiplicative")
    p.add_argument("--mode",      default="all",
                   choices=["all", "omicron", "pi", "rho", "sigma",
                             "tau", "summary"],
                   help="Experiment to run:\n"
                        "  omicron : optimiser comparison (Adam/Lion/SGD/…)\n"
                        "  pi      : gradient noise distribution\n"
                        "  rho     : Rosenblatt-SGLD landscape analysis\n"
                        "  sigma   : Adam whitening visualisation\n"
                        "  tau     : gradient κ₄ evolution during training\n"
                        "  summary : regenerate summary plot from CSVs")
    p.add_argument("--dataset",   default="FashionMNIST",
                   choices=["FashionMNIST", "MNIST"])
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--noise_type",default="both",
                   choices=["gaussian", "rosenblatt", "both"])
    p.add_argument("--opt_names", nargs="+", default=None)
    p.add_argument("--quick",     action="store_true",
                   help="Faster run: fewer epochs, fewer variants")
    args = p.parse_args()

    if args.quick:
        args.epochs = 8
        if args.opt_names is None:
            args.opt_names = ["adamw", "lion", "sgd"]

    noise_types = (["gaussian", "rosenblatt"] if args.noise_type == "both"
                   else [args.noise_type])
    cfg, save_dir = _make_cfg(args)

    print(f"Save dir  : {save_dir}")
    print(f"Mode      : {args.mode}")
    print(f"Dataset   : {cfg.dataset}")
    print(f"Epochs    : {cfg.epochs}  Device: {cfg.device}")
    print(f"Optimisers: {args.opt_names or list(OPT_LABELS)}")

    t0 = time.time()

    if args.mode in ("omicron", "all"):
        run_experiment_omicron(cfg, save_dir,
                               opt_names=args.opt_names,
                               noise_types=noise_types)

    if args.mode in ("pi", "all"):
        run_experiment_pi(cfg, save_dir)

    if args.mode in ("rho", "all"):
        run_experiment_rho(cfg, save_dir)

    if args.mode in ("tau", "all"):
        run_experiment_tau_grad_evolution(cfg, save_dir,
                                          opt_names=args.opt_names or
                                          ["adamw", "lion", "sgd"])

    if args.mode in ("summary", "all"):
        plot_optimizer_summary(save_dir)

    print(f"\nAll optimizer experiments complete in {time.time()-t0:.1f}s")
    print(f"Results → {save_dir}/optimizer_ablation/")


if __name__ == "__main__":
    main()