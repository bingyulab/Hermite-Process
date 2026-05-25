"""
Parametric activation measurement.

Replaces four near-duplicate routines from the original codebase:

    extract_all_stages           (Experiment_Gaussianity.run_experiment_alpha)
    extract_full_layer_trace     (Experiment_Gaussianity.run_experiment_gamma)
    measure_bottleneck           (Experiment_Ablation.measure_bottleneck)
    _extract_bottleneck_at_t     (Experiment_Ablation.run_experiment_theta)

These four were ALMOST identical but differed in the choices of `t_corrupt`,
`t_eval`, and whether activations from the conditional / unconditional
forward passes were mixed in the hook store (audit item H6).  The cumulants
reported in alpha and gamma therefore measured different things despite
identical naming.

`measure_layers()` makes those choices explicit arguments.  Existing
behaviour is recoverable as follows:

    extract_all_stages           condition='both',  t_corrupt=1, t_eval=T_MIN
                                  (legacy; mixes conditional+unconditional)
    extract_full_layer_trace     condition='null',  t_corrupt=1, t_eval=T_MIN
    measure_bottleneck           condition='null',  t_corrupt=1, t_eval=T_MIN
    _extract_bottleneck_at_t     condition='null',  t_corrupt=t,  t_eval=t

Use `condition='null'` by default.  `condition='both'` is provided for
backwards compatibility with the original alpha tables only.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Activation store
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Forward-hook accumulator.  Concatenates per-batch tensors and
    optionally spatial-averages 4-D tensors to (B, C)."""

    def __init__(self, spatial_pool: bool = True) -> None:
        self._parts: list[torch.Tensor] = []
        self._spatial_pool = spatial_pool

    def hook_fn(self, module, _input, output) -> None:
        x = output.detach().float().cpu()
        if self._spatial_pool and x.dim() == 4:
            x = x.mean(dim=(-2, -1))
        elif x.dim() > 2:
            x = x.flatten(1)
        self._parts.append(x)

    def get(self) -> torch.Tensor:
        if not self._parts:
            return torch.empty(0)
        return torch.cat(self._parts, dim=0)

    def clear(self) -> None:
        self._parts.clear()


@contextmanager
def capture_layer(module: nn.Module, spatial_pool: bool = True):
    store  = ActivationStore(spatial_pool=spatial_pool)
    handle = module.register_forward_hook(store.hook_fn)
    try:
        yield store
    finally:
        handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Layer-key catalogue (mirrors UNET_LAYER_KEYS in Experiment_Gaussianity)
# ─────────────────────────────────────────────────────────────────────────────

UNET_LAYER_KEYS: list[str] = [
    "init_conv",
    "down1_0",  "down1_1",
    "pool1",
    "down2_0",  "down2_1",   "attn2",
    "pool2",
    "mid1",     "attn_mid",  "mid2",
    "up_res2_0","up_res2_1", "up_attn2",
    "up_res1_0","up_res1_1",
    "out",
]


def get_unet_modules(model: nn.Module,
                     keys: Optional[list[str]] = None) -> dict[str, nn.Module]:
    """Map layer keys to the corresponding modules of a ConditionalUNet
    (or ConditionalUNetFlexible / ConditionalUNetAblation, which expose the
    same attribute names)."""
    if keys is None:
        keys = UNET_LAYER_KEYS
    accessor = {
        "init_conv":  lambda m: m.init_conv,
        "down1_0":    lambda m: m.down1[0],
        "down1_1":    lambda m: m.down1[1],
        "pool1":      lambda m: m.pool1,
        "down2_0":    lambda m: m.down2[0],
        "down2_1":    lambda m: m.down2[1],
        "attn2":      lambda m: m.attn2,
        "pool2":      lambda m: m.pool2,
        "mid1":       lambda m: m.mid1,
        "attn_mid":   lambda m: m.attn_mid,
        "mid2":       lambda m: m.mid2,
        "up_res2_0":  lambda m: m.up_res2[0],
        "up_res2_1":  lambda m: m.up_res2[1],
        "up_attn2":   lambda m: m.up_attn2,
        "up_res1_0":  lambda m: m.up_res1[0],
        "up_res1_1":  lambda m: m.up_res1[1],
        "out":        lambda m: m.out,
    }
    out: dict[str, nn.Module] = {}
    for k in keys:
        if k not in accessor:
            continue
        try:
            out[k] = accessor[k](model)
        except (AttributeError, IndexError):
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Parametric measurement
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_layers(
    model:        nn.Module,
    layer_modules:dict[str, nn.Module],
    fwd,
    test_ds,
    cfg,
    *,
    t_corrupt:    float = 1.0,
    t_eval:       float = 0.01,
    condition:    str   = "null",          # 'label' | 'null' | 'both'
    apply_c_in:   bool  = True,
    n_samples:    int   = 2000,
    spatial_pool: bool  = True,
    cfg_scale:    Optional[float] = None,  # if not None, also evaluate CFG mix
) -> dict[str, torch.Tensor]:
    """Capture per-layer activations under a controlled measurement regime.

    Parameters
    ----------
    t_corrupt   corruption level applied to x0 to produce x_T
    t_eval      time argument given to the model
                (t_corrupt == t_eval ==> self-consistent measurement;
                 t_corrupt = 1, t_eval = T_MIN reproduces gamma's probe;
                 t_corrupt = t_eval = t reproduces theta's probe.)
    condition   'label' uses the true class label
                'null'  uses the unconditional (class 10) label
                'both'  runs BOTH and lets the hook store accumulate the
                        union of activations (matches the legacy behaviour
                        of extract_all_stages; produces a 50/50 mixture of
                        conditional and null activations -- generally NOT
                        a meaningful distribution; provided for backwards
                        compatibility only.)
    apply_c_in  divide model input by fwd.c_in(t_eval) before forwarding

    Returns
    -------
    dict {key: (n_samples * num_passes, D) tensor}, where num_passes is 1
    for 'label' or 'null' and 2 for 'both'.
    """
    if condition not in ("label", "null", "both"):
        raise ValueError(f"condition must be 'label'|'null'|'both', got {condition!r}")

    model.eval()
    device = cfg.device
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)

    stores  = {k: ActivationStore(spatial_pool=spatial_pool) for k in layer_modules}
    handles = [m.register_forward_hook(stores[k].hook_fn)
               for k, m in layer_modules.items()]

    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if device.type == "cuda" else nullcontext

    try:
        n_done = 0
        for x0, y in loader:
            if n_done >= n_samples:
                break
            B  = x0.size(0)
            x0 = x0.to(device); y = y.to(device)

            t_corr = torch.full((B,), float(t_corrupt), device=device)
            x_T, _, _ = fwd.corrupt(x0, t_corr, y=y)

            t_in = torch.full((B,), float(t_eval), device=device)
            if apply_c_in:
                view_shape = (-1,) + (1,) * (x_T.dim() - 1)
                c_in     = fwd.c_in(t_in).view(*view_shape)
                model_in = x_T * c_in
            else:
                model_in = x_T

            null = torch.full_like(y, 10)
            ctx  = autocast_ctx() if device.type == "cuda" else nullcontext()
            with ctx:
                if condition in ("label", "both"):
                    _ = model(model_in, t_in, y)
                if condition in ("null", "both"):
                    _ = model(model_in, t_in, null)

            n_done += B
    finally:
        for h in handles:
            h.remove()

    out = {}
    for k, store in stores.items():
        a = store.get()
        # Per-pass cap.  With condition='both' the store has 2*N entries;
        # we keep 2*n_samples so each pass contributes n_samples.
        cap = n_samples * (2 if condition == "both" else 1)
        out[k] = a[:cap]
    return out


@torch.no_grad()
def measure_layer(model, layer_module, fwd, test_ds, cfg, **kwargs) -> torch.Tensor:
    """Single-layer convenience wrapper."""
    out = measure_layers(model, {"_": layer_module}, fwd, test_ds, cfg, **kwargs)
    return out["_"]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction helper for the legacy 'val_l1' metric
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_loss_ood(
    model: nn.Module,
    fwd,
    test_ds,
    cfg,
    *,
    t_corrupt: float = 1.0,
    t_eval:    float = 0.01,
    n_samples: int   = 2000,
    apply_c_in: bool = True,
) -> dict[str, float]:
    """Reconstruction loss between unconditional model output and the raw x0,
    given heavy corruption at t_corrupt and the model conditioned on t_eval.

    This is what the legacy `measure_bottleneck` returned under the name
    `val_l1` / `val_l2`.  It is NOT the training-time validation loss
    (which uses random t and matched corruption); callers that want the
    true validation loss should use `train_diffusion`'s history instead.
    """
    model.eval()
    device = cfg.device
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)

    raw_l: list[torch.Tensor] = []
    pred_l: list[torch.Tensor] = []
    n_done = 0

    autocast_ctx = (lambda: torch.amp.autocast("cuda")) if device.type == "cuda" else nullcontext

    for x0, y in loader:
        if n_done >= n_samples:
            break
        B  = x0.size(0)
        x0 = x0.to(device); y = y.to(device)
        raw_l.append(x0.view(B, -1).cpu())

        t_c = torch.full((B,), float(t_corrupt), device=device)
        x_T, _, _ = fwd.corrupt(x0, t_c, y=y)

        t_e = torch.full((B,), float(t_eval), device=device)
        if apply_c_in:
            view_shape = (-1,) + (1,) * (x_T.dim() - 1)
            c_in     = fwd.c_in(t_e).view(*view_shape)
            model_in = x_T * c_in
        else:
            model_in = x_T

        null = torch.full_like(y, 10)
        ctx  = autocast_ctx() if device.type == "cuda" else nullcontext()
        with ctx:
            x0h = model(model_in, t_e, null).float()
        pred_l.append(x0h.view(B, -1).cpu())
        n_done += B

    raw  = torch.cat(raw_l,  0)[:n_samples]
    pred = torch.cat(pred_l, 0)[:n_samples]

    return {
        "l1":    torch.nn.functional.l1_loss(pred, raw).item(),
        "l2":    torch.nn.functional.mse_loss(pred, raw).item(),
        "huber": torch.nn.functional.smooth_l1_loss(pred, raw).item(),
    }