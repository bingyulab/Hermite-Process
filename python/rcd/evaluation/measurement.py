"""
Parametric activation measurement.

Added vs uploaded version
──────────────────────────
measure_bottleneck()      — bottleneck κ4/κ3/PR/Mardia-Z + val losses
measure_sharpness()       — loss-landscape sharpness via random perturbations
measure_update_whiteness() — per-update W1-from-normal + kurtosis

These three were imported in run_optimizer.py and run_ablation.py but were
never defined anywhere in the package.
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

    FIX: The hook increments self._step internally.  The training loop must
    NOT call tracker.step() — that method does not exist and was erroneously
    present in the uploaded version.
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
# Hook infrastructure  (unchanged from uploaded version)
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    def __init__(self, spatial_pool: bool = True) -> None:
        self._parts: List[torch.Tensor] = []
        self._spatial_pool = spatial_pool

    def hook_fn(self, module, input, output) -> None:
        x = output.detach().float().cpu()
        if self._spatial_pool and x.dim() == 4:
            x = x.mean(dim=(-2, -1))
        elif x.dim() > 2:
            x = x.flatten(1)
        self._parts.append(x)

    def get(self) -> torch.Tensor:
        return torch.cat(self._parts, dim=0) if self._parts else torch.empty(0)

    def clear(self) -> None:
        self._parts.clear()


@contextmanager
def capture_activations(modules: Dict[str, nn.Module], spatial_pool: bool = True):
    stores  = {k: ActivationStore(spatial_pool=spatial_pool) for k in modules}
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
# New: high-level measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_bottleneck(
    model:  nn.Module,
    fwd:    Any,
    test_ds: Any,
    cfg:    Any,
    n_samples: int = 2000,
) -> Dict[str, float]:
    """
    Capture mid2 (bottleneck) activations and compute:
        kappa3, kappa4, std_k4, frac_nong, pr, mardia_z, val_l1, val_l2, val_huber

    Returns a flat dict that matches the keys expected by ExperimentRecord.
    """
    model.eval()
    device   = cfg.device
    loader   = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                          shuffle=False, num_workers=2)
    amp_ctx  = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    bn_mod   = getattr(model, "mid2", None)
    if bn_mod is None:
        raise AttributeError("model has no .mid2 attribute — not a ConditionalUNet?")

    acts_list, raw_list, pred_list = [], [], []
    n_done = 0

    with capture_activations({"mid2": bn_mod}) as stores:
        for x0, y in loader:
            if n_done >= n_samples:
                break
            B  = x0.size(0)
            x0, y = x0.to(device), y.to(device)
            t_min = torch.full((B,), cfg.T_MIN, device=device)
            t_one = torch.ones(B, device=device)
            x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
            c_in = fwd.c_in(t_min).view(-1, 1, 1, 1)
            with amp_ctx:
                pred = model(x_T * c_in, t_min, y).float()
            raw_list.append(x0.view(B, -1).cpu())
            pred_list.append(pred.view(B, -1).cpu())
            n_done += B

    acts = stores["mid2"].get()[:n_samples]
    raw  = torch.cat(raw_list,  0)[:n_samples]
    pred = torch.cat(pred_list, 0)[:n_samples]

    cum  = compute_marginal_cumulants(acts)
    spec = compute_spectrum_stats(acts)
    mard = mardia_statistics(acts, use_pca=True)

    return {
        "kappa3":   cum["mean_abs_kappa3"],
        "kappa4":   cum["mean_kappa4"],
        "std_k4":   cum["std_kappa4"],
        "frac_nong": cum["frac_non_gauss"],
        "pr":       spec["pr"],
        "mardia_z": mard["b2p_z"],
        "val_l1":   F.l1_loss(pred, raw).item(),
        "val_l2":   F.mse_loss(pred, raw).item(),
        "val_huber": F.smooth_l1_loss(pred, raw).item(),
    }


@torch.no_grad()
def measure_sharpness(
    model:   nn.Module,
    fwd:     Any,
    test_ds: Any,
    cfg:     Any,
    n_samples: int   = 512,
    n_dirs:    int   = 20,
    epsilon:   float = 1e-3,
) -> Dict[str, float]:
    """
    Estimate loss-landscape sharpness via random Gaussian directional
    perturbations of the model parameters.

    sharpness   = mean loss increase over n_dirs random unit-ball directions
    frac_negative = fraction of directions where loss decreased (flat/concave)
    """
    model.eval()
    device = cfg.device
    loader = DataLoader(test_ds, batch_size=min(n_samples, cfg.batch_size),
                        shuffle=True, num_workers=2)
    x0, y = next(iter(loader))
    x0, y = x0.to(device), y.to(device)
    B = x0.size(0)

    t_one = torch.ones(B, device=device)
    t_min = torch.full((B,), cfg.T_MIN, device=device)
    x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
    c_in = fwd.c_in(t_min).view(-1, 1, 1, 1)

    def _loss():
        with torch.no_grad():
            pred = model(x_T * c_in, t_min, y)
        return F.smooth_l1_loss(pred, x0).item()

    base_loss = _loss()
    params    = [p for p in model.parameters() if p.requires_grad]
    deltas: list[float] = []

    for _ in range(n_dirs):
        # Save & perturb
        saved = [p.data.clone() for p in params]
        norm  = 0.0
        noises = [torch.randn_like(p) for p in params]
        norm   = sum((n ** 2).sum().item() for n in noises) ** 0.5 + 1e-12
        for p, n in zip(params, noises):
            p.data.add_(n * (epsilon / norm))

        perturbed = _loss()
        deltas.append(perturbed - base_loss)

        # Restore
        for p, s in zip(params, saved):
            p.data.copy_(s)

    return {
        "sharpness":    float(sum(max(d, 0) for d in deltas) / n_dirs),
        "frac_negative": float(sum(d < 0 for d in deltas) / n_dirs),
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
    the per-parameter update vectors.  Returns:
        kappa4_updates   — excess kurtosis of the stacked updates
        w1_from_normal   — Wasserstein-1 distance of updates from N(0,1)
        update_std_cv    — coefficient of variation of per-param update stds
    """
    from rcd.train.training import _make_optimizer

    # Work on a throw-away copy
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

    # Collect updates
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

    # W1 approximation via sorted quantile matching
    n   = len(z)
    z_s = z.sort().values
    q   = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    # inverse normal CDF approximation
    gauss_q = torch.erfinv(2.0 * q - 1.0) * (2.0 ** 0.5)
    w1  = float((z_s - gauss_q).abs().mean().item())

    # CV of per-parameter-tensor update stds
    stds = torch.tensor([u.std().item() for u in updates])
    cv   = float(stds.std().item() / (stds.mean().item() + 1e-12))

    return {"kappa4_updates": k4, "w1_from_normal": w1, "update_std_cv": cv}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrappers  (unchanged from uploaded version)
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
) -> Dict[str, torch.Tensor]:
    return measure_layers(
        model=model, layer_modules=get_unet_modules(model), fwd=forward,
        test_ds=test_ds, cfg=cfg, t_corrupt=1.0, t_eval=t_eval,
        condition="null", n_samples=n_samples, spatial_pool=True,
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
    t_val   = getattr(cfg, "T_MIN", 0.001) if t_eval is None else t_eval
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

    return results


@torch.no_grad()
def extract_pipeline_stages(
    model: nn.Module, forward: Any, test_ds: Any, cfg: Any,
    n_samples: int = 2000, mode: str = "image",
    ae: Optional[nn.Module] = None,
    sample_noise_fn: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    m_type = "pixel" if mode == "image" else "latent"
    data   = extract_representation_diagnostics(
        model, forward, test_ds, cfg, n_samples=n_samples,
        model_type=m_type, ae=ae, t_corrupt=1.0,
        t_eval=getattr(cfg, "T_MIN", 0.001),
        condition="both" if mode == "image" else "null",
        capture_reconstructions=True,
        sample_noise_fn=sample_noise_fn,
    )
    if mode == "image":
        return {
            "input":      data["inputs"],
            "corrupted":  data["corrupted_inputs"],
            "bottleneck": data["activations"].get("mid2", torch.empty(0)),
            "x0hat":      data["reconstructions"],
        }
    else:
        return {
            "image_input":  data["raw_images"],
            "latent_z0":    data["inputs"],
            "latent_corr":  data["corrupted_inputs"],
            "mlp_mid":      data["activations"].get("mlp_mid", torch.empty(0)),
            "latent_x0hat": data.get("reconstructions", torch.empty(0)),
        }


def _analyse_stage(
    acts:       torch.Tensor,
    model_name: str,
    stage_key:  str,
    label:      str,
) -> tuple[object, np.ndarray]:
    """Compute cumulants + Mardia for one stage; return (record, k4_array)."""    
    if acts.numel() == 0:
        empty = GaussianityStats()
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
    # attach convenience attributes for print_cumulant_table
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
    t_corrupt: float = 1.0, n_samples: int = N_MEAS,
) -> torch.Tensor:
    """Collect mid2 activations with corruption at a specific t."""
    loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                        shuffle=False, num_workers=2)
    bn_mod = getattr(model, "mid2", None)
    n_done = 0
    with capture_activations({"mid2": bn_mod}) as stores:
        for x0, y in loader:
            if n_done >= n_samples:
                break
            x0, y = x0.to(cfg.device), y.to(cfg.device)
            B = x0.size(0)
            t_c = torch.full((B,), t_corrupt, device=cfg.device)
            x_T, _, _ = fwd.corrupt(x0, t_c, y=y)
            c_in = fwd.c_in(t_c).view(-1, 1, 1, 1)
            model(x_T * c_in, t_c, y)
            n_done += B
    return stores["mid2"].get()[:n_samples]

@torch.no_grad()
def measure_decoder_kappa4(model, fwd, test_ds, cfg, n_samples: int | None = None) -> dict[str, float]:
    n_samples = n_samples or cfg.n_meas
    dec_mods = {k: v for k, v in get_unet_modules(model).items() if k in DECODER_KEYS}
    acts: dict[str, torch.Tensor] = {}
    with capture_activations(dec_mods) as stores:
        loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128), shuffle=False, num_workers=2)
        n_done = 0
        for x0, y in loader:
            if n_done >= n_samples: break
            x0, y = x0.to(cfg.device), y.to(cfg.device)
            B = x0.size(0)
            t_one = torch.ones(B, device=cfg.device)
            t_min = torch.full((B,), cfg.T_MIN, device=cfg.device)
            x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
            c_in = fwd.c_in(t_min).view(-1, 1, 1, 1)
            model(x_T * c_in, t_min, y)
            n_done += B
    return {
        k: float(compute_marginal_cumulants(stores[k].get()[:n_samples])["mean_kappa4"])
        for k in DECODER_KEYS if stores[k].get().numel() > 0
    }