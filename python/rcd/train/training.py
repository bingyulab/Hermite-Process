"""
Unified Diffusion Training Pipeline.

Bug fixes applied vs uploaded version
──────────────────────────────────────
1. All imports use the correct intra-package paths:
     rcd.train.forward   (not rcd.diffusion)
     rcd.train.noise     (not rcd.diffusion)
     rcd.train.models    (not rcd.models.latent / .ema)
     rcd.train.checkpoints (not ..tracker.checkpoints)
2. GradientTracker.step() was called in the training loop but the method
   does not exist — the hook increments self._step automatically.
   The erroneous `if tracker: tracker.step()` call is removed.
3. Private `_sample_grad_noise` duplication removed; CustomNoiseAdamW now
   delegates to `sample_grad_noise` from rcd.train.noise.
4. The validation autocast context now consistently uses
   torch.amp.autocast / contextlib.nullcontext (no bare torch.autocast).
5. train_latent_model accepts an optional `model` argument so checkpoints
   can supply a pre-instantiated LatentMLPDenoiser; falls back to
   creating one if None is passed.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from rcd.data.config import Config
from rcd.data.datasets import _NORM_TF, _get_dataset
from rcd.train.forward import RosenblattForward
from rcd.train.noise import sample_noise
from rcd.train.models import ConvAutoencoder, LatentMLPDenoiser, EMA
from rcd.train.checkpoints import save_full, load_full, find_latest_epoch
from rcd.train.optim import _make_optimizer
from rcd.evaluation.measurement import GradientTracker

# ─────────────────────────────────────────────────────────────────────────────
# 3. Generation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    fwd: Any,
    labels: torch.Tensor,
    cfg: Config,
    bridge: str = "stochastic",
    x_in: Optional[torch.Tensor] = None,
    t_start: float = 1.0,
    ae: Optional[nn.Module] = None,
    apply_c_in: bool = True,
) -> torch.Tensor:
    """Unified cold diffusion reverse process with classifier-free guidance."""
    model.eval()
    if ae is not None:
        ae.eval()

    n           = len(labels)
    null_labels = torch.full_like(labels, getattr(cfg, "num_classes", 10))
    steps   = max(1, int(cfg.n_steps * t_start))
    t_sched = torch.linspace(t_start, 0.0, steps + 1, device=cfg.device)

    if x_in is not None:
        if ae is not None and x_in.ndim == 4:
            x = ae.encode(x_in)
        else:
            x = x_in.clone() 
    elif ae is not None:
        D   = ae.LATENT_DIM
        x   = sample_noise(fwd.noise_type, (n, D), fwd.lam_t, fwd.M_eig,
                           cfg.device) * fwd.sigma_max * (t_start ** fwd.H)
    else:
        eps     = sample_noise(fwd.noise_type, (n, 1, 28, 28),
                               fwd.lam_t, fwd.M_eig, device=cfg.device)
        dummy   = torch.zeros(n, 1, 28, 28, device=cfg.device)
        S       = (fwd.sigma_fn(dummy, labels)
                   if getattr(fwd.sigma_fn, "needs_label", False)
                   else fwd.sigma_fn(dummy))
        # Fallback to 1.0 for multiplicative/edge-aware where S(0) is small
        if S.mean() < 0.9:
            S = torch.ones_like(S)
        
        x = eps * fwd.sigma_max * S * (t_start ** fwd.H)

    use_amp = cfg.device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    for k in range(steps):
        t_cur  = t_sched[k].expand(n)
        t_next = t_sched[k + 1].expand(n)

        if ae is not None and bridge == "stochastic":
            sig   = fwd.sigma_t(t_cur).unsqueeze(1)
            c_in  = (1.0 + sig ** 2).pow(-0.5)
        else:
            c_in  = fwd.c_in(t_cur).view(-1, 1, 1, 1)

        if not apply_c_in:
            c_in = torch.ones_like(c_in)
            
        with amp_ctx:
            x0_c = model(x * c_in, t_cur, labels).float()
            x0_u = model(x * c_in, t_cur, null_labels).float()

        x0_hat = x0_u + cfg.cfg_scale * (x0_c - x0_u)
        if ae is None:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        if k < steps - 1:
            if ae is not None:
                sn  = fwd.sigma_t(t_next).unsqueeze(1)
                eps = sample_noise(fwd.noise_type, (n, ae.LATENT_DIM),
                                   fwd.lam_t, fwd.M_eig, cfg.device)
                x = x0_hat + sn * eps
            elif bridge == "stochastic":
                x = fwd.recorrupt_stochastic(x0_hat, t_next, y=labels)
            elif bridge == "hybrid":
                x = fwd.recorrupt_hybrid(x, x0_hat, t_cur, t_next, y=labels)
            elif bridge == "deterministic":
                x = fwd.recorrupt_deterministic(x, x0_hat, t_cur, t_next)
            else:
                raise ValueError(f"Unknown bridge: {bridge!r}")
        else:
            x = x0_hat

    if ae is not None:
        x = ae.decode(x)

    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trainer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedDiffusionTrainer:
    """Training engine with AMP, EMA, checkpointing, and optional gradient tracking."""

    def __init__(self, cfg: Config, model: nn.Module,
                 fwd: Optional[RosenblattForward] = None,
                 optimizer_factory: Optional[Callable] = None):
        self.cfg      = cfg
        self.model    = model.to(cfg.device)
        self.fwd      = fwd
        self.device   = cfg.device
        self.opt_factory = optimizer_factory or self._default_adamw_factory

    @staticmethod
    def _default_adamw_factory(model: nn.Module, cfg: Config) -> Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    @staticmethod
    def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                     loss_type: str) -> torch.Tensor:
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
        if loss_type.startswith("quantile_"):
            tau = float(loss_type.split("_", 1)[1])
            err = target - pred
            return torch.max(tau * err, (tau - 1.0) * err).mean()
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def fit(
        self,
        train_ds: Dataset,
        val_ds:   Dataset,
        ckpt_path: str | Path,
        *,
        tag:           str            = "model",
        loss_type:     str            = "huber",
        corrupt_fn:    Optional[Callable] = None,
        encode_x0:     Optional[Callable] = None,
        apply_c_in:    bool           = True,
        log_grads:     bool           = False,
        log_every:     int            = 100,
        extra_val_metrics: bool       = False,
        grad_clip:     float          = 1.0,
        ckpt_every:    int            = 5,
        cfg_dropout:   float          = 0.1,
    ):
        if not self.fwd and not corrupt_fn:
            raise ValueError(
                "Must provide either an initialized RosenblattForward or corrupt_fn."
            )
        print(f"[train] Starting training with tag={tag}, loss={loss_type}, corrupt_fn={corrupt_fn is not None}, encode_x0={encode_x0 is not None}")
        print(f"  Dataset: {self.cfg.dataset}, Epochs: {self.cfg.epochs}, Batch Size: {self.cfg.batch_size}, LR: {self.cfg.lr}, Device: {self.device}")
        print(f"  Corruption: {corrupt_fn.__name__ if corrupt_fn else 'RosenblattForward'}, Apply c_in: {apply_c_in}, Grad Tracking: {log_grads}, Extra Val Metrics: {extra_val_metrics}")

        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_fn = corrupt_fn or self.fwd.corrupt

        train_dl = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_dl = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        opt    = self.opt_factory(self.model, self.cfg)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        ema    = EMA(self.model, decay=0.999)
        use_amp = self.device.type == "cuda"
        scaler  = torch.amp.GradScaler("cuda") if use_amp else None

        # FIX: GradientTracker only — do NOT call tracker.step() in the loop
        tracker = GradientTracker(self.model, log_every=log_every) if log_grads else None

        start_epoch, inference_only = 0, False
        final_stem = ckpt_path.stem.replace("_final", "")
        latest = find_latest_epoch(ckpt_path.parent, final_stem)

        if latest is not None:
            print(f"  [{tag}] Resuming from {latest[0].name} (epoch {latest[1]})")
            start_epoch, _ = load_full(
                latest[0], self.model, ema=ema, opt=opt, sch=sch,
                scaler=scaler, device=self.device,
            )
        elif ckpt_path.exists():
            print(f"  [{tag}] Loading completed checkpoint: {ckpt_path}")
            load_full(ckpt_path, self.model, ema=ema, device=self.device)
            inference_only = True

        history: dict[str, list] = {
            "train": [], "val": [], "val_l1": [], "val_l2": [], "epochs": [],
        }

        if inference_only or start_epoch >= self.cfg.epochs:
            self.model.eval()
            return self.model, history, (tracker.get_log() if tracker else [])

        amp_train = torch.amp.autocast("cuda") if use_amp else nullcontext()
        # FIX: use nullcontext for CPU validation (not bare torch.autocast)
        amp_val   = torch.amp.autocast("cuda") if use_amp else nullcontext()

        for ep in range(start_epoch, self.cfg.epochs):
            t0 = time.time()
            self.model.train()
            ep_loss, n_batches, n_skipped = 0.0, 0, 0

            for x0_raw, labels in train_dl:
                x0_raw = x0_raw.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                B      = x0_raw.size(0)

                target = encode_x0(x0_raw) if encode_x0 else x0_raw
                cf     = torch.rand(B, device=self.device) < cfg_dropout
                lbl    = labels.clone()
                lbl[cf] = getattr(self.cfg, "num_classes", 10)
                t = (torch.rand(B, device=self.device)
                     * (1.0 - self.cfg.T_MIN) + self.cfg.T_MIN)

                corrupt_out = corrupt_fn(target, t, y=lbl)
                x_t = corrupt_out[0] if isinstance(corrupt_out, tuple) else corrupt_out

                c_view = (-1,) + (1,) * (x_t.dim() - 1)
                model_in = (x_t * self.fwd.c_in(t).view(*c_view)
                            if apply_c_in and self.fwd else x_t)

                opt.zero_grad(set_to_none=True)
                step_executed = True

                if use_amp:
                    with amp_train:
                        pred = self.model(model_in, t, lbl)
                        loss = self.compute_loss(pred, target, loss_type)
                    if not torch.isfinite(loss):
                        n_skipped += 1
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    initial_scale = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() < initial_scale:
                        step_executed = False
                        n_skipped += 1
                else:
                    pred = self.model(model_in, t, lbl)
                    loss = self.compute_loss(pred, target, loss_type)
                    if not torch.isfinite(loss):
                        n_skipped += 1
                        continue
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    opt.step()

                if step_executed:
                    ema.update()
                    # FIX: removed tracker.step() — the hook fires automatically
                    ep_loss  += loss.item() * B
                    n_batches += B

            train_loss = ep_loss / max(n_batches, 1)
            history["train"].append(train_loss)

            # Validation
            self.model.eval()
            ema.apply_shadow()
            v_loss, v_l1, v_l2, v_n = 0.0, 0.0, 0.0, 0
            val_gen = torch.Generator(device="cpu").manual_seed(42)

            with torch.no_grad():
                for x0_raw, labels in val_dl:
                    x0_raw = x0_raw.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    B      = x0_raw.size(0)
                    target = encode_x0(x0_raw) if encode_x0 else x0_raw
                    t = (torch.rand(B, generator=val_gen)
                         .to(self.device) * (1.0 - self.cfg.T_MIN) + self.cfg.T_MIN)

                    corrupt_out = corrupt_fn(target, t, y=labels)
                    x_t = corrupt_out[0] if isinstance(corrupt_out, tuple) else corrupt_out
                    c_view = (-1,) + (1,) * (x_t.dim() - 1)
                    model_in = (x_t * self.fwd.c_in(t).view(*c_view)
                                if apply_c_in and self.fwd else x_t)

                    with amp_val:
                        pred = self.model(model_in, t, labels)
                        v_loss += self.compute_loss(pred, target, loss_type).item() * B
                        if extra_val_metrics:
                            v_l1 += F.l1_loss(pred, target).item() * B
                            v_l2 += F.mse_loss(pred, target).item() * B
                    v_n += B

            history["val"].append(v_loss / max(v_n, 1))
            if extra_val_metrics:
                history["val_l1"].append(v_l1 / max(v_n, 1))
                history["val_l2"].append(v_l2 / max(v_n, 1))
            history["epochs"].append(ep + 1)

            ema.restore()

            if (ep + 1) % ckpt_every == 0 and (ep + 1) < self.cfg.epochs:
                save_full(
                    ckpt_path.parent / f"{final_stem}_ep{ep+1}.pt",
                    self.model, ema=ema, opt=opt, sch=sch, scaler=scaler,
                    epoch=ep + 1, extras={"tag": tag, "loss_type": loss_type},
                )

            sch.step()
            skip_note = f"  (skipped {n_skipped} bad steps)" if n_skipped else ""
            print(
                f"  [{tag}] ep {ep+1:3d}/{self.cfg.epochs}"
                f"  tr={train_loss:.5f}"
                f"  va={history['val'][-1]:.5f}"
                f"  lr={sch.get_last_lr()[0]:.2e}"
                f"  {time.time()-t0:.1f}s{skip_note}"
            )

        ema.apply_shadow()
        save_full(
            ckpt_path, self.model, ema=ema, opt=opt, sch=sch, scaler=scaler,
            epoch=self.cfg.epochs, extras={"tag": tag, "loss_type": loss_type},
        )
        print(f"  [{tag}] Saved → {ckpt_path}")
        self.model.eval()

        grad_log = tracker.get_log() if tracker else []
        if tracker:
            tracker.remove()

        if extra_val_metrics:
            formatted = [
                {"ep": ep, "tr_loss": tr, "va_loss": va, "va_l1": l1, "va_l2": l2}
                for ep, tr, va, l1, l2 in zip(
                    history["epochs"], history["train"], history["val"],
                    history["val_l1"], history["val_l2"],
                )
            ]
            return self.model, formatted, grad_log

        return self.model, history, grad_log


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def train_standard(cfg: Config, model: nn.Module, fwd: RosenblattForward,
                   ckpt_path: Path, **kwargs):
    """Train a ConditionalUNet with default AdamW."""
    trainer   = UnifiedDiffusionTrainer(cfg, model, fwd)
    train_ds  = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    val_ds    = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    return trainer.fit(train_ds, val_ds, ckpt_path, **kwargs)


def train_with_optimizer(
    cfg: Config,
    model: nn.Module,
    fwd: RosenblattForward,
    ckpt_path: Path,
    *,
    opt_name:   str   = "adamw",
    noise_std:  float = 0.0,
    noise_dist: str   = "none",
    **kwargs,
):
    """Train a model with a specified optimizer (for experiment ο/π/ρ)."""
    def _opt_factory(m: nn.Module, c: Config) -> Optimizer:
        return _make_optimizer(opt_name, m, c,
                               noise_std=noise_std, noise_dist=noise_dist)

    trainer  = UnifiedDiffusionTrainer(cfg, model, fwd,
                                        optimizer_factory=_opt_factory)
    train_ds = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    val_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    return trainer.fit(train_ds, val_ds, ckpt_path, **kwargs)


def train_latent_model(
    ae: nn.Module,
    cfg: Config,
    sigma_max: float  = 4.0,
    noise_type: str   = "rosenblatt",
    model: Optional[nn.Module] = None,   # FIX: accept pre-built model
) -> tuple[nn.Module, RosenblattForward]:
    """
    Train (or fine-tune) a LatentMLPDenoiser.

    FIX: accepts an optional `model` argument so the checkpoint layer can
    supply a pre-instantiated denoiser.  If None, creates one here.
    """
    from rcd.train.forward import sigma_additive

    ae.eval()
    fwd = RosenblattForward(
        sigma_additive(), noise_type=noise_type,
        H=cfg.H, M_eig=cfg.M_eig, sigma_max=sigma_max, device=cfg.device,
    )
    if model is None:
        model = LatentMLPDenoiser(latent_dim=ae.LATENT_DIM)

    tag      = f"lat_{noise_type}_s{sigma_max}"
    ckpt_dir = Path(getattr(cfg, "ckpt_dir", cfg.save_dir)) / "latent"
    ckpt_path = ckpt_dir / f"{tag}_final.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _latent_corrupt(z0, t, y=None):
        sig = fwd.sigma_t(t).unsqueeze(1)
        eps = sample_noise(fwd.noise_type, z0.shape,
                           fwd.lam_t, cfg.M_eig, cfg.device)
        return z0 + sig * eps, eps, sig

    trainer  = UnifiedDiffusionTrainer(cfg, model, fwd)
    train_ds = _get_dataset(cfg.dataset, train=True,  tf=_NORM_TF)
    val_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)

    trainer.fit(
        train_ds, val_ds, ckpt_path, tag=tag, loss_type="smooth_l1",
        encode_x0=lambda x: ae.encode(x).detach(),
        corrupt_fn=_latent_corrupt, 
    )
    return trainer.model, fwd


def train_autoencoder(cfg: Config) -> nn.Module:
    """Train a ConvAutoencoder and save to checkpoints/latent/ae_final.pt."""
    ds  = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                     num_workers=4, pin_memory=True)

    ae  = ConvAutoencoder().to(cfg.device)
    opt = torch.optim.Adam(ae.parameters(),
                            lr=getattr(cfg, "ae_lr", 1e-3))

    for ep in range(getattr(cfg, "ae_epochs", 10)):
        ae.train()
        tot = 0
        for x0, _ in dl:
            x0        = x0.to(cfg.device, non_blocking=True)
            recon, _  = ae(x0)
            loss      = F.mse_loss(recon, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x0.size(0)
        print(f"  AE ep {ep+1:3d}/{cfg.ae_epochs}  {tot/len(ds):.5f}")

    ae_path = (Path(getattr(cfg, "ckpt_dir", cfg.save_dir))
               / "latent" / "ae_final.pt")
    ae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), ae_path)
    print(f"  AE → {ae_path}")
    return ae