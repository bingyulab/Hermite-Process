"""
Unified diffusion training loop.

Replaces the five near-identical training routines in the original codebase:

    train                  (image space, unified.py)
    train_latent           (latent space, unified.py)
    train_flexible_unet    (variable bottleneck width, Gaussianity.py)
    train_ablation_model   (norm/act/skip variants, Ablation.py)
    train_with_optimizer   (optimiser comparison, Optimizer.py)

Invariants enforced here:

  1. Full-state checkpoints (rcd.checkpoints) — model + EMA shadow +
     optimiser + scheduler + AMP scaler.  Resume from a tag_epN.pt is exact.

  2. EMA shadow is checkpointed independently of the model.  On resume the
     shadow is restored to the value it had at save time, NOT inferred from
     EMA-applied weights (which was the silent bug in every original loop).

  3. The `*_final.pt` written at the end stores EMA-APPLIED weights as
     `model`, so legacy callers that load only `state_dict["model"]` still
     get the smoothed model.  The raw weights and EMA shadow are recoverable
     from `state_dict["ema"]` for any caller that wants exact resume.

  4. `apply_c_in` is a mandatory argument.  Train-time and inference-time
     input normalisation must match.  Passing `apply_c_in=False` is a
     deliberate research choice, not a default.

  5. AMP non-finite-loss skip path does NOT advance the scaler (advancing
     without a corresponding backward causes scale drift).

  6. `grad_hook(step, model, opt)` is invoked AFTER ema.update so gradient
     trackers see the same step count as the optimiser.

  7. Custom `optimizer_factory(model, cfg) -> Optimizer` permits SGD / Lion /
     RMSprop / custom variants without further refactoring of this loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .ema         import EMA
from ..tracker.checkpoints import save_full, load_full, find_latest_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Loss dispatch (unified across the four legacy variants)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(pred: torch.Tensor,
                 target: torch.Tensor,
                 loss_type: str) -> torch.Tensor:
    """Single source of truth for the loss functions used across all scripts.

    Accepted names
    --------------
    'huber'   | 'smooth_l1'   — F.smooth_l1_loss
    'l1'                       — F.l1_loss
    'l2'      | 'mse'          — F.mse_loss
    'elastic'                  — 0.5 L1 + 0.5 L2
    'quantile'                 — pinball loss with tau=0.5  (== MAE)
    'qNN'    (NN in 1..99)     — pinball loss with tau=NN/100
    """
    if loss_type in ("huber", "smooth_l1"):
        return F.smooth_l1_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type in ("l2", "mse"):
        return F.mse_loss(pred, target)
    if loss_type == "elastic":
        return 0.5 * F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
    if loss_type == "quantile":
        err = target - pred
        return torch.max(0.5 * err, -0.5 * err).mean()
    if loss_type.startswith("q") and loss_type[1:].isdigit():
        tau = int(loss_type[1:]) / 100.0
        err = target - pred
        return torch.max(tau * err, (tau - 1.0) * err).mean()
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Default optimiser
# ─────────────────────────────────────────────────────────────────────────────

OptimizerFactory = Callable[[nn.Module, "Config"], Optimizer]


def default_adamw_factory(model: nn.Module, cfg) -> Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_diffusion(
    model:             nn.Module,
    fwd,                                                    # RosenblattForward
    train_ds,
    val_ds,
    cfg,
    ckpt_path:         str | Path,
    *,
    tag:               str   = "model",
    loss_type:         str   = "huber",
    optimizer_factory: Optional[OptimizerFactory] = None,
    cfg_dropout:       float = 0.1,
    grad_clip:         float = 1.0,
    ckpt_every:        int   = 5,
    encode_x0:         Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    apply_c_in:        bool  = True,
    grad_hook:         Optional[Callable[[int, nn.Module, Optimizer], None]] = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """
    Parameters
    ----------
    model         denoiser network
    fwd           RosenblattForward (already configured with empirical E[Sigma^2])
    train_ds      training dataset
    val_ds        validation dataset
    cfg           Config-like object with attributes:
                   batch_size, device, epochs, lr, T_MIN
    ckpt_path     final checkpoint path (also defines tag_epN.pt search root)
    tag           printed log prefix
    loss_type     see compute_loss
    optimizer_factory
                  callable(model, cfg) -> Optimizer; default AdamW(cfg.lr, wd=1e-4)
    cfg_dropout   classifier-free guidance label-dropout probability
    grad_clip     L2 clip threshold; 0 disables clipping
    ckpt_every    save *_epN.pt every this many epochs
    encode_x0     if provided, applied to each x0 batch to obtain the
                  denoising target; used for latent-space training
    apply_c_in    if True, divide model input by fwd.c_in(t) before passing
                  to the network.  Must match the inference convention.
    grad_hook     called as grad_hook(step, model, opt) after each
                  optimiser step; used by Experiment tau

    Returns
    -------
    model      trained network, eval mode, EMA weights applied
    history    {"train": [...], "val": [...], "epochs": [...]}
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True)

    # ── optimiser / scheduler / EMA / AMP ─────────────────────────────────
    opt_factory = optimizer_factory or default_adamw_factory
    opt    = opt_factory(model, cfg)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ema    = EMA(model, decay=0.999)
    use_amp = (cfg.device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # ── resume from highest tag_epN.pt under ckpt_path.parent ─────────────
    start_epoch  = 0
    final_stem   = ckpt_path.stem.replace("_final", "")
    latest       = find_latest_epoch(ckpt_path.parent, final_stem)
    inference_only = False

    if latest is not None:
        latest_path, ep_n = latest
        print(f"  [{tag}] Resuming from {latest_path.name} (epoch {ep_n})")
        start_epoch, _ = load_full(latest_path, model,
                                    ema=ema, opt=opt, sch=sch, scaler=scaler,
                                    device=cfg.device)
    elif ckpt_path.exists():
        # Final exists but no intermediate -> already finished; load for inference.
        print(f"  [{tag}] Loading completed checkpoint: {ckpt_path}")
        load_full(ckpt_path, model, ema=ema, device=cfg.device)
        inference_only = True

    if inference_only or start_epoch >= cfg.epochs:
        model.eval()
        return model, {"train": [], "val": [], "epochs": []}

    history: dict[str, list[float]] = {"train": [], "val": [], "epochs": []}
    global_step = 0

    # ── main training loop ────────────────────────────────────────────────
    for ep in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()
        ep_loss   = 0.0
        n_batches = 0
        n_skipped = 0

        for x0_raw, labels in train_dl:
            x0_raw = x0_raw.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            B = x0_raw.size(0)

            if encode_x0 is not None:
                with torch.no_grad():
                    target = encode_x0(x0_raw)
            else:
                target = x0_raw

            # Classifier-free guidance: with probability cfg_dropout, replace label with "no label" token (10).
            cf  = torch.rand(B, device=cfg.device) < cfg_dropout
            lbl = labels.clone(); lbl[cf] = cfg.num_classes
            t   = torch.rand(B, device=cfg.device) * (1.0 - cfg.T_MIN) + cfg.T_MIN

            x_t, _, _ = fwd.corrupt(target, t, y=lbl)
            if apply_c_in:
                view_shape = (-1,) + (1,) * (x_t.dim() - 1)
                c_in = fwd.c_in(t).view(*view_shape)
                model_in = x_t * c_in
            else:
                model_in = x_t

            opt.zero_grad(set_to_none=True)
            step_executed = True

            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = model(model_in, t, lbl)
                    loss = compute_loss(pred, target, loss_type)
                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue                                   # do NOT scaler.update()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Dynamic check for AMP step execution
                initial_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                # If the scale factor drops, it implies inf/nan gradients caused a skipped step
                if scaler.get_scale() < initial_scale:
                    step_executed = False
                    n_skipped += 1
            else:
                pred = model(model_in, t, lbl)
                loss = compute_loss(pred, target, loss_type)
                if not torch.isfinite(loss):
                    n_skipped += 1
                    continue
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            if step_executed:
                ema.update()
                global_step += 1
                if grad_hook is not None:
                    grad_hook(global_step, model, opt)

                ep_loss   += loss.item() * B
                n_batches += B

        train_loss = ep_loss / max(n_batches, 1)
        history["train"].append(train_loss)

        # ── validation under EMA weights ──────────────────────────────────
        model.eval()
        ema.apply_shadow()
        v_loss = 0.0
        v_n    = 0
        # Create a deterministic generator for validation time steps
        val_gen = torch.Generator(device=cfg.device).manual_seed(42)

        with torch.no_grad():
            for x0_raw, labels in val_dl:
                x0_raw = x0_raw.to(cfg.device, non_blocking=True)
                labels = labels.to(cfg.device, non_blocking=True)
                B = x0_raw.size(0)
                target = (encode_x0(x0_raw) if encode_x0 is not None else x0_raw)
                # Deterministic sampling over validation epochs
                t = torch.rand(B, device=cfg.device, generator=val_gen) * (1.0 - cfg.T_MIN) + cfg.T_MIN

                x_t, _, _ = fwd.corrupt(target, t, y=labels)
                if apply_c_in:
                    view_shape = (-1,) + (1,) * (x_t.dim() - 1)
                    c_in = fwd.c_in(t).view(*view_shape)
                    model_in = x_t * c_in
                else:
                    model_in = x_t
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(model_in, t, labels)
                        v_loss += compute_loss(pred, target, loss_type).item() * B
                else:
                    pred = model(model_in, t, labels)
                    v_loss += compute_loss(pred, target, loss_type).item() * B
                v_n += B
        v_loss /= max(v_n, 1)
        history["val"].append(v_loss)
        history["epochs"].append(ep + 1)

        # ── intermediate checkpoint: RAW weights + full optimiser state ──
        # restore raw weights so the saved model represents the actual
        # training trajectory; resume reconstructs the EMA shadow from
        # the checkpoint, not from EMA-applied weights.
        ema.restore()
        if ((ep + 1) % ckpt_every == 0) and ((ep + 1) < cfg.epochs):
            save_full(ckpt_path.parent / f"{final_stem}_ep{ep+1}.pt",
                      model, ema=ema, opt=opt, sch=sch, scaler=scaler,
                      epoch=ep + 1,
                      extras={"tag": tag, "loss_type": loss_type})

        sch.step()
        skip_note = f"  (skipped {n_skipped} bad/AMP steps)" if n_skipped else ""
        print(f"  [{tag}] ep {ep+1:3d}/{cfg.epochs}  "
              f"tr={train_loss:.5f}  va={v_loss:.5f}  "
              f"lr={sch.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.1f}s{skip_note}")

    # ── final: save EMA-APPLIED weights so legacy loaders work ────────────
    ema.apply_shadow()
    save_full(ckpt_path, model, ema=ema, opt=opt, sch=sch, scaler=scaler,
              epoch=cfg.epochs,
              extras={"tag": tag, "loss_type": loss_type})
    model.eval()
    return model, history
