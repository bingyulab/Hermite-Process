"""
Parametric activation measurement.

Patches in this version
───────────────────────
1. t-mismatch fix. measure_bottleneck / measure_decoder_kappa4 / _bottleneck_acts_at_t
   now corrupt AND condition the model at the same t (t=1). The previous code
   corrupted at t=1 but conditioned at t=T_MIN with c_in(T_MIN), which is
   off-distribution relative to both training (matched t) and generation
   (t=1 at the first step). extract_pipeline_stages now sets t_eval=1.0.
2. Spatial-pooling artifact fix. ActivationStore gains an explicit `reduce`
   mode {"mean","center","flatten","none"}. measure_bottleneck captures the
   raw (N,C,H,W) tensor and reports kappa4 under THREE reductions:
       mean   — spatial average over (H,W)  [CLT-suppressed; old number]
       center — single central spatial cell [no averaging]
       unit   — every (channel,location) marginal [no averaging]
   κ4≈0 on the spatial mean does NOT prove marginal Gaussianity; the unit and
   center views are the decisive test. For 2D activations the three coincide.
3. measure_update_whiteness W1 computed in float64 with clamped erfinv (the
   float32 version returned inf because 2q-1 rounded to ±1).
4. extract_full_layer_trace gains a `spatial_pool` argument so gamma can be
   rerun per-unit (spatial_pool=False) without editing call sites.

These three high-level helpers (measure_bottleneck, measure_sharpness,
measure_update_whiteness) were imported by the optimizer/ablation runners but
never defined in the original package.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from rcd.evaluation.gaussianity import (
    compute_marginal_cumulants, compute_spectrum_stats, mardia_statistics,
    covariance_whiteness, js_divergence_from_gaussian,
)
from rcd.experiments.registry import (
    ExperimentRecord, GaussianityStats, UNET_LAYER_KEYS, DECODER_KEYS,
)


class GradientTracker:
    """Captures gradient distribution at mid2 during training via a backward hook.

    The hook increments self._step internally. The training loop must NOT call
    tracker.step() — that method does not exist.

    NOTE: gradients are spatially averaged here (g.mean over H,W). This is the
    same reduction applied to activations in the "mean" view, so the
    gradient-vs-activation kurtosis CONTRAST is fair: both are pooled identically.
    """

    def __init__(self, model: nn.Module, log_every: int = 50):
        self._log_every = log_every
        self._step = 0
        self._log: list[dict] = []
        self._handle = model.mid2.register_full_backward_hook(self._hook)

    def _hook(self, module, grad_input, grad_output):
        self._step += 1
        if self._step % self._log_every != 0:
            return
        g      = grad_output[0].detach().float()
        g_flat = g.mean(dim=(-2, -1)).cpu()
        if g_flat.numel() < 4:
            return
        cum = compute_marginal_cumulants(g_flat.unsqueeze(0) if g_flat.dim() == 1
                                         else g_flat)
        self._log.append({
            "step":      self._step,
            "kappa4":    cum["mean_kappa4"],
            "kappa3":    cum["mean_abs_kappa3"],
            "grad_norm": g.norm().item(),
        })

    def remove(self):
        self._handle.remove()

    def get_log(self) -> list[dict]:
        return list(self._log)

# ─────────────────────────────────────────────────────────────────────────────
# Hook infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Captures forward activations under a chosen spatial reduction.

    reduce:
        "mean"    — spatial average over (H,W) → (B, C)   [CLT-suppressed]
        "center"  — central spatial cell       → (B, C)   [no averaging]
        "flatten" — every (channel,location)   → (B, C*H*W) [per-unit marginal]
        "none"    — keep the raw (B, C, H, W) tensor (caller reshapes)
    `spatial_pool` is kept for backward compatibility: True→"mean", False→"flatten".
    """

    def __init__(self, spatial_pool: bool = True, reduce: Optional[str] = None) -> None:
        if reduce is None:
            reduce = "mean" if spatial_pool else "flatten"
        if reduce not in ("mean", "center", "flatten", "none"):
            raise ValueError(f"reduce must be mean|center|flatten|none, got {reduce!r}")
        self._reduce = reduce
        self._parts: List[torch.Tensor] = []

    def hook_fn(self, module, input, output) -> None:
        x = output.detach().float().cpu()
        if x.dim() == 4:
            if self._reduce == "mean":
                x = x.mean(dim=(-2, -1))                 # (B, C)
            elif self._reduce == "center":
                H, W = x.shape[-2], x.shape[-1]
                x = x[:, :, H // 2, W // 2]              # (B, C)
            elif self._reduce == "flatten":
                x = x.flatten(1)                         # (B, C*H*W)
            # "none": keep (B, C, H, W)
        elif x.dim() > 2 and self._reduce != "none":
            x = x.flatten(1)
        self._parts.append(x)

    def get(self) -> torch.Tensor:
        return torch.cat(self._parts, dim=0) if self._parts else torch.empty(0)

    def clear(self) -> None:
        self._parts.clear()


@contextmanager
def capture_activations(modules: Dict[str, nn.Module],
                        spatial_pool: bool = True,
                        reduce: Optional[str] = None):
    stores  = {k: ActivationStore(spatial_pool=spatial_pool, reduce=reduce)
               for k in modules}
    handles = [m.register_forward_hook(stores[k].hook_fn)
               for k, m in modules.items() if m is not None]
    try:
        yield stores
    finally:
        for h in handles:
            h.remove()


def get_unet_modules(model: nn.Module,
                     keys: Optional[List[str]] = None) -> Dict[str, nn.Module]:
    if keys is None:
        keys = UNET_LAYER_KEYS
    accessor = {
        "init_conv":  lambda m: getattr(m, "init_conv", None),
        "down1_0":    lambda m: m.down1[0] if hasattr(m, "down1") and len(m.down1) > 0 else None,
        "down1_1":    lambda m: m.down1[1] if hasattr(m, "down1") and len(m.down1) > 1 else None,
        "pool1":      lambda m: getattr(m, "pool1", None),
        "down2_0":    lambda m: m.down2[0] if hasattr(m, "down2") and len(m.down2) > 0 else None,
        "down2_1":    lambda m: m.down2[1] if hasattr(m, "down2") and len(m.down2) > 1 else None,
        "attn2":      lambda m: getattr(m, "attn2", None),
        "pool2":      lambda m: getattr(m, "pool2", None),
        "mid1":       lambda m: getattr(m, "mid1", None),
        "attn_mid":   lambda m: getattr(m, "attn_mid", None),
        "mid2":       lambda m: getattr(m, "mid2", None),
        "up_res2_0":  lambda m: m.up_res2[0] if hasattr(m, "up_res2") and len(m.up_res2) > 0 else None,
        "up_res2_1":  lambda m: m.up_res2[1] if hasattr(m, "up_res2") and len(m.up_res2) > 1 else None,
        "up_attn2":   lambda m: getattr(m, "up_attn2", None),
        "up_res1_0":  lambda m: m.up_res1[0] if hasattr(m, "up_res1") and len(m.up_res1) > 0 else None,
        "up_res1_1":  lambda m: m.up_res1[1] if hasattr(m, "up_res1") and len(m.up_res1) > 1 else None,
        "out":        lambda m: getattr(m, "out", None),
    }
    return {k: mod for k in keys
            if k in accessor and isinstance(mod := accessor[k](model), nn.Module)}


# ─────────────────────────────────────────────────────────────────────────────
# Three-view kurtosis decomposition (anti-CLT-artifact)
# ─────────────────────────────────────────────────────────────────────────────

def _summ(X: torch.Tensor) -> Dict[str, float]:
    """Cumulant + spectrum + Mardia summary for one (N, D) matrix."""
    if X.numel() == 0 or X.dim() != 2 or X.size(0) < 4:
        nan = float("nan")
        return {"kappa4": nan, "kappa3": nan, "std_k4": nan,
                "frac_nong": nan, "pr": nan, "mardia_z": nan}
    cum  = compute_marginal_cumulants(X)
    spec = compute_spectrum_stats(X)
    mard = mardia_statistics(X, use_pca=True)
    return {
        "kappa4":    cum["mean_kappa4"],
        "kappa3":    cum["mean_abs_kappa3"],
        "std_k4":    cum["std_kappa4"],
        "frac_nong": cum["frac_non_gauss"],
        "pr":        spec["pr"],
        "mardia_z":  mard["b2p_z"],
    }


def kappa4_three_views(acts: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Given activations (N, C, H, W) raw, or (N, D) already reduced, return
    the cumulant summary under three reductions:
        mean   — spatial average (CLT-suppressed)
        center — central spatial cell (no averaging)
        unit   — every (channel,location) marginal (no averaging)
    For 2D input all three are identical (no spatial axis to reduce)."""
    if acts.numel() == 0:
        empty = _summ(torch.empty(0))
        return {"mean": dict(empty), "center": dict(empty), "unit": dict(empty)}

    if acts.dim() == 4:
        N, C, H, W = acts.shape
        views = {
            "mean":   acts.mean(dim=(-2, -1)),
            "center": acts[:, :, H // 2, W // 2],
            "unit":   acts.reshape(N, C * H * W),
        }
    else:
        v = acts if acts.dim() == 2 else acts.reshape(acts.size(0), -1)
        views = {"mean": v, "center": v, "unit": v}

    return {name: _summ(X) for name, X in views.items()}


def _flat_views(metrics_prefix: str, views: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten the three-view dict into CSV-ready keys, e.g.
    kappa4_mean / kappa4_center / kappa4_unit, mardia_z_unit, pr_unit, ..."""
    out: Dict[str, float] = {}
    for view in ("mean", "center", "unit"):
        for stat in ("kappa4", "kappa3", "pr", "mardia_z", "frac_nong"):
            out[f"{stat}_{view}"] = views[view][stat]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# High-level measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_bottleneck(
    model:  nn.Module,
    fwd:    Any,
    test_ds: Any,
    cfg:    Any,
) -> Dict[str, float]:
    """
    Capture mid2 (bottleneck) activations and compute cumulants.

    The canonical keys (kappa3/kappa4/std_k4/frac_nong/pr/mardia_z) report the
    SPATIAL-MEAN view, preserving backward compatibility with existing records.
    The decisive per-unit and center views are added as kappa4_unit /
    kappa4_center / mardia_z_unit / pr_unit / frac_nong_unit (and the full
    *_mean/*_center/*_unit set).

    FIX: corruption AND conditioning both at t=1 (matched), with c_in(t=1).
    """
    model.eval()
    device   = cfg.device
    loader   = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                          shuffle=False, num_workers=2)
    amp_ctx  = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    bn_mod   = getattr(model, "mid2", None)
    if bn_mod is None:
        raise AttributeError("model has no .mid2 attribute — not a ConditionalUNet?")

    raw_list, pred_list = [], []
    n_done = 0

    # reduce="none" → keep the raw (B,C,H,W) tensor so we can build all 3 views.
    with capture_activations({"mid2": bn_mod}, reduce="none") as stores:
        for x0, y in loader:
            if n_done >= cfg.n_samples:
                break
            B  = x0.size(0)
            x0, y = x0.to(device), y.to(device)
            t_one = torch.ones(B, device=device)                       # FIX: matched t
            x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
            c_in = fwd.c_in(t_one).view(-1, 1, 1, 1)                    # FIX: c_in at same t
            with amp_ctx:
                pred = model(x_T * c_in, t_one, y).float()             # FIX: condition at same t
            raw_list.append(x0.view(B, -1).cpu())
            pred_list.append(pred.view(B, -1).cpu())
            n_done += B

    acts = stores["mid2"].get()[:cfg.n_samples]      # (N,C,H,W) or (N,C)
    raw  = torch.cat(raw_list,  0)[:cfg.n_samples]
    pred = torch.cat(pred_list, 0)[:cfg.n_samples]

    views = kappa4_three_views(acts)
    mean_v = views["mean"]

    out: Dict[str, float] = {
        # canonical keys = MEAN view (unchanged schema; clearly the pooled number)
        "kappa3":    mean_v["kappa3"],
        "kappa4":    mean_v["kappa4"],
        "std_k4":    mean_v["std_k4"],
        "frac_nong": mean_v["frac_nong"],
        "pr":        mean_v["pr"],
        "mardia_z":  mean_v["mardia_z"],
        "val_l1":    F.l1_loss(pred, raw).item(),
        "val_l2":    F.mse_loss(pred, raw).item(),
        "val_huber": F.smooth_l1_loss(pred, raw).item(),
    }
    out.update(_flat_views("", views))               # adds *_mean / *_center / *_unit
    return out


def measure_sharpness(
    model:           nn.Module,
    fwd:             "RosenblattForward",
    test_ds,
    cfg:             "Config",
    perturb_sigma:   float = 0.01,
    n_perturbations: int   = 30,
    max_samples:     int   = 2000,
) -> dict[str, float]:
    """
    Gaussian-perturbation sharpness. Caches a data subset on device and uses
    in-place weight restore. Independent of the optimizer's learning rate
    (probes the final weights), but note final weights still depend on lr.
    """
    model.eval()
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    cached_inputs, cached_x0, cached_lbl, cached_t = [], [], [], []
    samples_collected = 0
    with torch.no_grad():
        for x0, lbl in loader:
            if samples_collected >= max_samples:
                break
            B = x0.size(0)
            x0, lbl = x0.to(cfg.device), lbl.to(cfg.device)
            t = torch.full((B,), 0.5, device=cfg.device)
            xt, _, _ = fwd.corrupt(x0, t, y=lbl)
            cin = fwd.c_in(t).view(-1, 1, 1, 1)
            x_in = (xt * cin).float()
            cached_inputs.append(x_in)
            cached_x0.append(x0.float())
            cached_lbl.append(lbl)
            cached_t.append(t)
            samples_collected += B

    def _eval_loss_cached():
        total, n = 0.0, 0
        is_cuda = cfg.device.type == "cuda"
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=is_cuda):
                for x_in, x0_f, lbl, t in zip(cached_inputs, cached_x0, cached_lbl, cached_t):
                    B = x_in.size(0)
                    pred = model(x_in, t, lbl).float()
                    total += F.smooth_l1_loss(pred, x0_f, reduction="mean").item() * B
                    n += B
        return total / n

    baseline = _eval_loss_cached()
    orig_params = [p.clone() for p in model.parameters()]

    delta_losses: list[float] = []
    for _ in range(n_perturbations):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * perturb_sigma)
        perturbed = _eval_loss_cached()
        delta_losses.append(perturbed - baseline)
        with torch.no_grad():
            for p, orig_p in zip(model.parameters(), orig_params):
                p.copy_(orig_p)

    dl = np.array(delta_losses)
    return {
        "baseline_loss":   baseline,
        "sharpness":       float(np.mean(np.abs(dl)) / perturb_sigma),
        "mean_delta":      float(dl.mean()),
        "std_delta":       float(dl.std()),
        "frac_negative":   float((dl < 0).mean()),
        "perturb_sigma":   perturb_sigma,
    }


def measure_update_whiteness(
    model:    nn.Module,
    fwd:      Any,
    train_ds: Any,
    cfg:      Any,
    opt_name: str = "adamw",
    n_batches: int = 30,
) -> Dict[str, float]:
    """
    Run n_batches of the chosen optimizer on a copy of the model and collect
    per-parameter update vectors. Returns kappa4_updates, w1_from_normal,
    update_std_cv.
    """
    from rcd.train.optim import _make_optimizer
    import copy

    m_copy = copy.deepcopy(model).to(cfg.device)
    m_copy.train()
    opt    = _make_optimizer(opt_name, m_copy, cfg)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=2)

    params_before = {n: p.data.clone() for n, p in m_copy.named_parameters()
                     if p.requires_grad}
    device = cfg.device
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    for i, (x0, y) in enumerate(loader):
        if i >= n_batches:
            break
        x0, y = x0.to(device), y.to(device)
        B = x0.size(0)
        t = torch.rand(B, device=device) * (1.0 - cfg.T_MIN) + cfg.T_MIN
        x_t, _, _ = fwd.corrupt(x0, t, y=y)
        c_in = fwd.c_in(t).view(-1, 1, 1, 1)
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            pred = m_copy(x_t * c_in, t, y)
            loss = F.smooth_l1_loss(pred, x0)
        loss.backward()
        opt.step()

    updates = []
    for n, p in m_copy.named_parameters():
        if p.requires_grad and n in params_before:
            delta = (p.data - params_before[n]).flatten().cpu()
            updates.append(delta)

    if not updates:
        nan = float("nan")
        return {"kappa4_updates": nan, "w1_from_normal": nan, "update_std_cv": nan}

    all_updates = torch.cat(updates)
    std = all_updates.std().item()
    if std < 1e-12:
        return {"kappa4_updates": float("nan"),
                "w1_from_normal": float("nan"),
                "update_std_cv":  float("nan")}

    z   = (all_updates - all_updates.mean()) / std
    k4  = float(((z ** 4).mean() - 3.0).item())

    # W1 vs N(0,1) via sorted quantile matching — FIX: float64 + clamped erfinv.
    n    = len(z)
    z_s  = z.sort().values.double()
    q    = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=torch.float64)
    arg  = (2.0 * q - 1.0).clamp(-1.0 + 1e-12, 1.0 - 1e-12)
    gauss_q = torch.erfinv(arg) * (2.0 ** 0.5)
    w1   = float((z_s - gauss_q).abs().mean().item())

    stds = torch.tensor([u.std().item() for u in updates])
    cv   = float(stds.std().item() / (stds.mean().item() + 1e-12))

    return {"kappa4_updates": k4, "w1_from_normal": w1, "update_std_cv": cv}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrappers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_layers(
    model: nn.Module, layer_modules: Dict[str, nn.Module], fwd: Any,
    test_ds: Any, cfg: Any, *, t_corrupt: float = 1.0,
    t_eval: Optional[float] = None, condition: str = "null",
    apply_c_in: bool = True, n_samples: int = 2000,
    spatial_pool: bool = True,
) -> Dict[str, torch.Tensor]:
    return extract_representation_diagnostics(
        model, fwd, test_ds, cfg, n_samples=n_samples,
        layer_modules=layer_modules, spatial_pool=spatial_pool,
        t_corrupt=t_corrupt, t_eval=t_eval, condition=condition,
        apply_c_in=apply_c_in, capture_reconstructions=False,
    )["activations"]


@torch.no_grad()
def extract_full_layer_trace(
    model: nn.Module, forward: Any, test_ds: Any, cfg: Any,
    n_samples: int = 2000, t_eval: Optional[float] = None,
    spatial_pool: bool = True,
) -> Dict[str, torch.Tensor]:
    """Layer-by-layer activation capture. Set spatial_pool=False to get the
    per-unit (un-pooled) marginals required for a CLT-artifact-free κ4 trace.
    FIX: default t_eval is now 1.0 (matched to t_corrupt=1.0)."""
    return measure_layers(
        model=model, layer_modules=get_unet_modules(model), fwd=forward,
        test_ds=test_ds, cfg=cfg, t_corrupt=1.0,
        t_eval=1.0 if t_eval is None else t_eval,
        condition="null", n_samples=n_samples, spatial_pool=spatial_pool,
    )


@torch.no_grad()
def extract_representation_diagnostics(
    model: nn.Module, fwd: Any, test_ds: Any, cfg: Any,
    *, n_samples: int = 2000, model_type: str = "pixel",
    ae: Optional[nn.Module] = None,
    layer_keys: Optional[List[str]] = None,
    layer_modules: Optional[Dict[str, nn.Module]] = None,
    spatial_pool: bool = True, t_corrupt: float = 1.0,
    t_eval: Optional[float] = None, condition: str = "null",
    apply_c_in: bool = True, capture_reconstructions: bool = False,
    run_generation_trajectory: bool = False,
    sample_noise_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    if condition not in ("label", "null", "both"):
        raise ValueError(f"condition must be 'label'|'null'|'both', got {condition!r}")
    if model_type == "latent" and ae is None:
        raise ValueError("ae required for latent mode")

    model.eval()
    if ae is not None:
        ae.eval()

    device  = cfg.device
    # FIX: matched-t default. If t_eval is None, condition at t_corrupt, not T_MIN.
    t_val   = float(t_corrupt) if t_eval is None else float(t_eval)
    loader  = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                         shuffle=False, num_workers=2)
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    if layer_modules is None:
        if layer_keys is not None:
            layer_modules = get_unet_modules(model, keys=layer_keys)
        elif model_type == "latent":
            hook_layer = (ae.layers[3] if hasattr(ae, "layers")
                          else (model.layers[3] if hasattr(model, "layers") else None))
            layer_modules = {"mlp_mid": hook_layer} if hook_layer else {}
        else:
            layer_modules = get_unet_modules(model)

    raw_list, struct_list, corrupt_list, recon_list = [], [], [], []
    n_done = 0

    with capture_activations(layer_modules, spatial_pool=spatial_pool) as stores:
        for x0, y in loader:
            if n_done >= n_samples:
                break
            B = x0.size(0)
            x0, y = x0.to(device), y.to(device)
            null_lbl = torch.full_like(y, 10)
            raw_list.append(x0.view(B, -1).cpu())

            t_c = torch.full((B,), float(t_corrupt), device=device)
            if model_type == "latent":
                z0 = ae.encode(x0)
                struct_list.append(z0.cpu())
                sig = fwd.sigma_t(t_c).unsqueeze(1)
                if sample_noise_fn:
                    from rcd.train.noise import sample_noise
                    eps = sample_noise_fn(fwd.noise_type, (B, ae.LATENT_DIM),
                                          fwd.lam_t, fwd.M_eig, device)
                else:
                    from rcd.train.noise import sample_noise
                    eps = sample_noise(fwd.noise_type, (B, ae.LATENT_DIM),
                                       fwd.lam_t, fwd.M_eig, device)
                v_corrupt = z0 + sig * eps
            else:
                struct_list.append(x0.view(B, -1).cpu())
                v_corrupt, _, _ = fwd.corrupt(x0, t_c, y=y)
            corrupt_list.append(v_corrupt.view(B, -1).cpu())

            t_in = torch.full((B,), float(t_val), device=device)
            if apply_c_in:
                if model_type == "latent":
                    c = (1.0 + fwd.sigma_t(t_in).unsqueeze(1) ** 2).pow(-0.5)
                else:
                    c = fwd.c_in(t_in).view((-1,) + (1,) * (v_corrupt.dim() - 1))
                model_in = v_corrupt * c
            else:
                model_in = v_corrupt

            with amp_ctx:
                out_c = model(model_in, t_in, y).float()   if condition in ("label", "both") else None
                out_u = model(model_in, t_in, null_lbl).float() if condition in ("null", "both")  else None

            if capture_reconstructions:
                pred = out_u if out_u is not None else out_c
                if condition == "both" and getattr(cfg, "cfg_scale", 1.0) != 1.0:
                    pred = out_u + cfg.cfg_scale * (out_c - out_u)
                if model_type == "pixel":
                    pred = pred.clamp(-1.0, 1.0)
                recon_list.append(pred.view(B, -1).cpu())
            n_done += B

    results = {
        "activations":      {k: s.get()[:n_samples] for k, s in stores.items()},
        "raw_images":       torch.cat(raw_list,    0)[:n_samples],
        "inputs":           torch.cat(struct_list,  0)[:n_samples],
        "corrupted_inputs": torch.cat(corrupt_list, 0)[:n_samples],
    }
    if capture_reconstructions and recon_list:
        results["reconstructions"] = torch.cat(recon_list, 0)[:n_samples]

    if run_generation_trajectory and model_type == "pixel":
        import math
        from rcd.train.noise import sample_noise
        _noise_fn = sample_noise_fn if sample_noise_fn else sample_noise

        mid_t05_list = []
        labels_gen = torch.arange(10, device=device).repeat(
            math.ceil(n_samples / 10)
        )[:n_samples]

        n_steps_half = cfg.n_steps // 2
        t_full  = torch.linspace(1.0, 0.0, cfg.n_steps + 1, device=device)
        t_half  = t_full[:n_steps_half + 1]

        chunk = min(256, n_samples)
        collected = 0
        while collected < n_samples:
            n_now = min(chunk, n_samples - collected)
            lbl   = labels_gen[collected: collected + n_now]
            null  = torch.full_like(lbl, 10)

            eps = _noise_fn(fwd.noise_type, (n_now, 1, 28, 28),
                            fwd.lam_t, fwd.M_eig, device)
            x   = eps * fwd.sigma_max

            for k in range(len(t_half) - 1):
                tc   = t_half[k].expand(n_now)
                tn   = t_half[k + 1].expand(n_now)
                c_in = fwd.c_in(tc).view(-1, 1, 1, 1)
                with amp_ctx:
                    xc = model(x * c_in, tc, lbl).float()
                    xu = model(x * c_in, tc, null).float()
                x0h = (xu + cfg.cfg_scale * (xc - xu)).clamp(-1., 1.)
                x = fwd.recorrupt_stochastic(x0h, tn, y=lbl)

            mid_t05_list.append(x.view(n_now, -1).cpu())
            collected += n_now

        results["mid_t05"] = torch.cat(mid_t05_list, 0)[:n_samples]
    return results


@torch.no_grad()
def extract_pipeline_stages(
    model: torch.nn.Module,
    forward: Any,
    test_ds: Any,
    cfg: Any,
    n_samples: int = 2000,
    mode: str = "image",
    ae: torch.nn.Module | None = None,
    sample_noise_fn: Any | None = None,
    spatial_pool: bool = True,
) -> dict[str, torch.Tensor | None]:
    """FIX: t_eval=1.0 (matched to t_corrupt). Pass spatial_pool=False to obtain
    per-unit (un-pooled) bottleneck activations for the CLT-artifact-free test."""
    m_type = "pixel" if mode == "image" else "latent"
    data   = extract_representation_diagnostics(
        model, forward, test_ds, cfg, n_samples=n_samples,
        model_type=m_type, ae=ae, t_corrupt=1.0,
        t_eval=1.0,
        condition="null",
        capture_reconstructions=True,
        run_generation_trajectory=(mode == "image"),
        sample_noise_fn=sample_noise_fn,
        spatial_pool=spatial_pool,
    )
    if mode == "image":
        return {
            "input":      data.get("inputs"),
            "corrupted":  data.get("corrupted_inputs"),
            "bottleneck": data.get("activations", {}).get("mid2"),
            "mid_t05":    data.get("mid_t05"),
            "x0hat":      data.get("reconstructions"),
        }
    else:
        return {
            "image_input":  data.get("raw_images"),
            "latent_z0":    data.get("inputs"),
            "latent_corr":  data.get("corrupted_inputs"),
            "mlp_mid":      data.get("activations", {}).get("mlp_mid"),
            "latent_x0hat": data.get("reconstructions"),
        }


def _analyse_stage(
    acts:       torch.Tensor,
    model_name: str,
    stage_key:  str,
    label:      str,
) -> tuple[object, np.ndarray]:
    """Compute cumulants + Mardia for one stage; return (record, k4_array)."""
    if acts.numel() == 0:
        return (ExperimentRecord(
            experiment_type="alpha", noise_type=model_name.lower(),
            model_name=model_name, label=label,
        ), np.array([]))

    cum  = compute_marginal_cumulants(acts)
    spec = compute_spectrum_stats(acts)
    wh   = covariance_whiteness(acts)
    mard = mardia_statistics(acts, use_pca=True)
    js   = js_divergence_from_gaussian(acts)

    record = ExperimentRecord(
        experiment_type="alpha",
        noise_type=model_name.lower(),
        model_name=model_name,
        label=label,
        config={"stage": stage_key},
    )
    record.dist = GaussianityStats(
        k3=cum["mean_abs_kappa3"], k4=cum["mean_kappa4"],
        std_k4=cum["std_kappa4"], max_k4=cum["max_kappa4"],
        frac_nong=cum["frac_non_gauss"],
        pr=spec["pr"], effective_rank=spec["effective_rank"],
        whiteness=wh, mardia_z=mard["b2p_z"],
        mardia_b2p=mard["b2p"], mardia_b2p_exp=mard["b2p_exp"],
        js_gauss=js,
    )
    record.N    = cum["N"]
    record.D    = cum["D"]
    record.stage = stage_key
    record.mean_k4 = cum["mean_kappa4"]
    record.mardia_b2p_z = mard["b2p_z"]
    record.mardia_b2p   = mard["b2p"]
    record.mardia_b2p_exp = mard["b2p_exp"]

    return record, np.array(cum["kappa4"])


@torch.no_grad()
def _bottleneck_acts_at_t(
    model: torch.nn.Module, fwd, test_ds, cfg,
    t_corrupt: float = 1.0,
    reduce: str = "none",
) -> torch.Tensor:
    """Collect mid2 activations with corruption AND conditioning at t_corrupt
    (matched). Default reduce='none' returns the raw (N,C,H,W) tensor so the
    caller can build the three views."""
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)
    bn_mod = getattr(model, "mid2", None)
    n_done = 0
    with capture_activations({"mid2": bn_mod}, reduce=reduce) as stores:
        for x0, y in loader:
            if n_done >= cfg.n_samples:
                break
            x0, y = x0.to(cfg.device), y.to(cfg.device)
            B = x0.size(0)
            t_c = torch.full((B,), t_corrupt, device=cfg.device)
            x_T, _, _ = fwd.corrupt(x0, t_c, y=y)
            c_in = fwd.c_in(t_c).view(-1, 1, 1, 1)
            model(x_T * c_in, t_c, y)
            n_done += B
    return stores["mid2"].get()[:cfg.n_samples]


@torch.no_grad()
def measure_decoder_kappa4(model, fwd, test_ds, cfg, n_samples: int | None = None) -> dict[str, float]:
    """Decoder-layer spatial-mean κ4. FIX: corrupt and condition both at t=1."""
    n_samples = n_samples or cfg.n_samples
    dec_mods = {k: v for k, v in get_unet_modules(model).items() if k in DECODER_KEYS}
    with capture_activations(dec_mods) as stores:
        loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128), shuffle=False, num_workers=2)
        n_done = 0
        for x0, y in loader:
            if n_done >= n_samples:
                break
            x0, y = x0.to(cfg.device), y.to(cfg.device)
            B = x0.size(0)
            t_one = torch.ones(B, device=cfg.device)                   # FIX: matched t
            x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
            c_in = fwd.c_in(t_one).view(-1, 1, 1, 1)
            model(x_T * c_in, t_one, y)
            n_done += B
    return {
        k: float(compute_marginal_cumulants(stores[k].get()[:n_samples])["mean_kappa4"])
        for k in DECODER_KEYS if stores[k].get().numel() > 0
    }