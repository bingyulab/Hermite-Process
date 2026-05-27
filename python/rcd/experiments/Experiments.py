"""
All experiment functions in one module.

Sections:
  1. Shared loaders and forward-builder defaults
  2. Generic parameter-sweep executor (`run_sweep`)
  3. Ablation experiments     : epsilon, zeta, kappa, mu, theta
  4. Optimizer experiments    : omicron, pi, rho, tau
  5. Gaussianity experiments  : alpha, beta, gamma, delta
  6. Cold ablation experiments: sigma_comparison, pca_basis, cold_latent,
                                cold_loss, cold_bridge, cold_h_sweep,
                                n_steps, cfg_scale

Conventions:
  * Every experiment function has signature `(cfg, ctx, runner) -> list`.
  * Sweep-shaped experiments are 15-25 lines: a `grid` declaration plus
    callbacks (model_factory, measure_fn, record_fn).
  * Probe-shaped experiments use explicit loops because their inner shape
    differs from "load -> measure -> record".
  * All measurement budgets come from `cfg.n_samples` and `cfg.n_samples`.
    No module-level constants are introduced.
  * Forward processes are built only through `_mult_fwd` or `build_forward_process`
    — no per-experiment forward construction logic.
  * Checkpoints flow only through `load_or_train(LoadRequest(...))`.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rcd.data.config import Config, override
from rcd.train.checkpoints import LoadRequest, load_full, load_or_train
from rcd.train.forward import (
    RosenblattForward, build_forward_process, compute_pixel_variance,
    sigma_additive, sigma_anisotropic, sigma_edge_aware,
    sigma_multiplicative, sigma_pca_whitened,
)
from rcd.train.models import (
    ConditionalUNet, ConvAutoencoder, LatentMLPDenoiser, SkipZeroWrapper,
)
from rcd.train.training import (
    train_autoencoder, train_latent_model, train_standard, train_with_optimizer,
)
from rcd.train.save import save_csv
from rcd.evaluation.measurement import (
    _analyse_stage, _bottleneck_acts_at_t, ActivationStore,
    extract_full_layer_trace, extract_pipeline_stages,
    measure_bottleneck, measure_decoder_kappa4,
    measure_sharpness, measure_update_whiteness,
)
from rcd.evaluation.gaussianity import (
    compute_marginal_cumulants, compute_spectrum_stats,
    covariance_whiteness, js_divergence_from_gaussian, mardia_statistics,
)
from rcd.evaluation.metrics import rigidity_test
from rcd.experiments.registry import (
    ACT_VARIANTS, BetaResult, ExperimentRecord, GaussianityStats,
    LandscapeStats, LAYER_LABELS, LayerStats, LOSS_VARIANTS, LossStats,
    NORM_VARIANTS, OPT_LABELS, SKIP_VARIANTS,
    STAGE_LABELS_LATENT, STAGE_LABELS_UNET, UNET_LAYER_KEYS,
)
from rcd.train.plotting import (
    plot_all_sigma_patterns, plot_kappa4_violins, plot_layer_profiles,
    plot_rigidity, plot_restoration_grid
)


# =============================================================================
# 1. Shared loaders and forward-builder defaults
# =============================================================================

def _mult_fwd(params: dict, cfg: Config) -> RosenblattForward:
    """Default forward process: multiplicative Σ + LP Rosenblatt."""
    return build_forward_process(
        sigma_multiplicative(), cfg,
        noise_type=params["noise_type"], H=cfg.H,
    )


def _baseline_ckpt(ctx, noise_type: str, cfg: Config) -> Path:
    """Canonical multiplicative baseline path used for checkpoint inheritance."""
    return Path(ctx.ckpt_dir)  / "cold_ablation" / \
           f"{noise_type}_multiplicative_H{cfg.H}_final.pt"


def _load_unet_baseline(cfg: Config, ctx, noise_type: str
                         ) -> tuple[nn.Module, RosenblattForward]:
    """Load or train the canonical multiplicative-Σ ConditionalUNet baseline."""
    tag = f"{noise_type}_multiplicative_H{cfg.H}"
    req = LoadRequest(
        tag=tag, cfg=cfg, save_dir=Path(ctx.ckpt_dir) , subdir="cold_ablation",
        model_factory=lambda: ConditionalUNet(num_classes=10, base_ch=cfg.base_ch),
        train_fn=lambda m, f, c, ck, t=tag: train_standard(
            c, m, f, ck, tag=t, loss_type="huber",
        ),
        fwd_builder=lambda c, nt=noise_type: build_forward_process(
            sigma_multiplicative(), c, noise_type=nt, H=c.H,
        ),
    )
    model, fwd, _ = load_or_train(req)
    return model, fwd


def _load_latent_pipeline(cfg: Config, ctx, noise_type: str
                            ) -> tuple[nn.Module, nn.Module, RosenblattForward]:
    """Load or train (autoencoder + latent MLP) for a given noise type."""
    ae = ConvAutoencoder().to(cfg.device)
    ae_path = Path(ctx.ckpt_dir) / "latent" / "ae_final.pt"
    if ae_path.exists():
        load_full(ae_path, ae, device=cfg.device, strict=False)
    else:
        ae = train_autoencoder(cfg)
        ae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ae.state_dict(), ae_path)
    ae.eval()

    fwd_lat = build_forward_process(
        sigma_additive(), cfg, noise_type=noise_type, H=cfg.H, estimate_eg2=False,
    )
    fwd_lat.set_eg2(1.0)

    tag = f"lat_{noise_type}_s{cfg.sigma_max}"
    req = LoadRequest(
        tag=tag, cfg=cfg, save_dir=ctx.run_dir, subdir="latent",
        fwd=fwd_lat,
        model_factory=lambda d=ae.LATENT_DIM: LatentMLPDenoiser(latent_dim=d),
        train_fn=lambda m, f, c, ck, ae=ae, nt=noise_type: train_latent_model(
            ae, c, sigma_max=c.sigma_max, noise_type=nt, model=m,
        ),
    )
    mlp, fwd_lat, _ = load_or_train(req)
    return ae, mlp, fwd_lat


# =============================================================================
# 2. Generic parameter-sweep executor
# =============================================================================

def run_sweep(
    cfg: Config, ctx, runner,
    *,
    name:             str,
    subdir:           str,
    grid:             list[dict],
    model_factory:    Callable[[dict, Config], nn.Module],
    measure_fn:       Callable[[nn.Module, Any, dict, Config, Any], dict],
    record_fn:        Callable[[dict, dict], Any],
    fwd_builder:      Callable[[dict, Config], Any] = _mult_fwd,
    train_fn:         Callable = train_standard,
    train_kwargs_fn:  Callable[[dict], dict] = lambda p: {"loss_type": "huber"},
    baseline_path_fn: Optional[Callable[[dict], Optional[Path]]] = None,
) -> list:
    """
    Iterate over `grid`, performing load_or_train -> measure -> record at
    each point. Streams the CSV after every iteration so partial runs are
    safe to resume.
    """
    rows: list = []
    csv_path = ctx.get_path("metric", f"{name}.csv")
    for params in grid:
        tag = f"{name}_{params['_id']}"
        baseline = baseline_path_fn(params) if baseline_path_fn else None
        train_kwargs = {"tag": tag, **train_kwargs_fn(params)}

        req = LoadRequest(
            tag=tag, cfg=cfg, save_dir=ctx.ckpt_dir, subdir=subdir,
            model_factory=_bind_factory(model_factory, params, cfg),
            train_fn=_bind_train(train_fn, train_kwargs),
            fwd_builder=_bind_fwd(fwd_builder, params),
            baseline_path=baseline,
        )
        
        model, fwd, _ = load_or_train(req)
        metrics = measure_fn(model, fwd, params, cfg, runner)
        rows.append(record_fn(params, metrics))
        ctx.logger.info(f"  [{name}] {params['label']:36s}  {_summary(metrics)}")
        _stream_csv(rows, csv_path)
        plot_restoration_grid(model, fwd, cfg, ctx.get_path("plot", f"{name}_{params['_id']}_restoration.png"))
    return rows


# Closure binders — these avoid the well-known "lambda-in-for-loop" capture bug.

def _bind_factory(factory, params, cfg):
    def _f():
        return factory(params, cfg)
    return _f


def _bind_train(train_fn, kwargs):
    def _f(model, fwd, cfg, ckpt_path):
        return train_fn(cfg, model, fwd, ckpt_path, **kwargs)
    return _f


def _bind_fwd(builder, params):
    def _f(cfg):
        return builder(params, cfg)
    return _f


def _stream_csv(rows: list, path: Path) -> None:
    flat = [
        r.flatten() if hasattr(r, "flatten")
        else (asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r))
        for r in rows
    ]
    save_csv(flat, path, silent=True)


def _summary(m: dict) -> str:
    parts: list[str] = []
    for key, fmt in (("kappa4",   "{:+.3f}"), ("kappa3",   "{:.3f}"),
                     ("pr",       "{:.1f}"),  ("mardia_z", "{:+.2f}"),
                     ("val_l1",   "{:.4f}"),  ("sharpness","{:.4f}"),
                     ("FID",      "{:.2f}"),  ("fFID",     "{:.2f}"),
                     ("Accuracy", "{:.2f}"),  ("SSIM",     "{:.4f}"),  
                     ("LPIPS",    "{:.4f}"),  ("eg2",      "{:.4f}"),
                     ("frac_nong","{:+.3f}"), ("frac_negative", "{:.3f}"),
                     ("effective_rank", "{:.4f}"), 
                     ):
        if key in m and isinstance(m[key], (int, float)):
            parts.append(f"{key}=" + fmt.format(m[key]))
    return "  ".join(parts)


# =============================================================================
# =============================================================================
# 3. Record-builder helpers — collapse repeated dataclass spelling
# =============================================================================

def _dist(m: dict) -> GaussianityStats:
    """Build a GaussianityStats from a measurement dict; absent keys → NaN."""
    g = m.get
    return GaussianityStats(
        k3            = g("kappa3",         float("nan")),
        k4            = g("kappa4",         float("nan")),
        std_k4        = g("std_k4",         float("nan")),
        max_k4        = g("max_k4",         float("nan")),
        frac_nong     = g("frac_nong",      float("nan")),
        pr            = g("pr",             float("nan")),
        mardia_z      = g("mardia_z",       float("nan")),
        mardia_b2p    = g("mardia_b2p",     float("nan")),
        mardia_b2p_exp= g("mardia_b2p_exp", float("nan")),
        effective_rank= g("effective_rank", float("nan")),
        whiteness     = g("whiteness",      float("nan")),
        js_gauss      = g("js_gauss",       float("nan")),
    )


def _loss_stats(m: dict) -> LossStats:
    """Build a LossStats from a measurement dict; absent keys → NaN."""
    g = m.get
    return LossStats(
        l1       = g("val_l1",    float("nan")),
        l2       = g("val_l2",    float("nan")),
        huber    = g("val_huber", float("nan")),
        mse      = g("mse",       float("nan")),
        mae      = g("mae",       float("nan")),
        quantile = g("quantile",  float("nan")),
    )


def _landscape(m: dict) -> LandscapeStats:
    """Build a LandscapeStats from a measurement dict; absent keys → NaN."""
    g = m.get
    return LandscapeStats(
        sharpness    = g("sharpness",        float("nan")),
        frac_neg     = g("frac_negative",    float("nan")),
        update_k4    = g("kappa4_updates",   float("nan")),
        update_w1    = g("w1_from_normal",   float("nan")),
        update_std_cv= g("update_std_cv",    float("nan")),
    )


# =============================================================================
# 4. Generic inference-time sweep + FID measurement
# =============================================================================

def _measure_fid(model, fwd, params, cfg, runner) -> dict:
    """Image-level FID/fFID/Accuracy/SSIM/LPIPS. Requires runner.evaluator."""
    bridge = params.get("bridge", cfg.bridge)
    return runner.evaluator.evaluate(
        model, fwd, runner.real_imgs, runner.test_ds, cfg, bridge=bridge,
    )


def _inference_time_sweep(
    cfg: Config, ctx, runner,
    *,
    name:         str,
    attr:         str,
    values:       Iterable,
    fixed_noise:  Optional[str] = None,
    label_fmt:    Callable[[str, Any], str] = lambda nt, v: f"{nt}/{v}",
) -> list[ExperimentRecord]:
    """
    Sweep an inference-only `cfg.<attr>` over `values`. The baseline UNet is
    loaded once per noise_type and the evaluator is re-run for each value.
    No training occurs inside this helper. Drives cold_bridge, n_steps,
    and cfg_scale.
    """
    rows: list[ExperimentRecord] = []
    csv_path = ctx.get_path("metric", f"{name}.csv")
    nts = (fixed_noise,) if fixed_noise else cfg.noise_types

    for noise_type in nts:
        model, fwd = _load_unet_baseline(cfg, ctx, noise_type)
        for v in values:
            with override(cfg, **{attr: v}):
                bridge = v if attr == "bridge" else cfg.bridge
                metrics = runner.evaluator.evaluate(
                    model, fwd, runner.real_imgs, runner.test_ds, cfg, bridge=bridge,
                )
            rows.append(ExperimentRecord(
                experiment_type=name, noise_type=noise_type,
                label=label_fmt(noise_type, v),
                config={attr: v}, extras=metrics,
            ))
            ctx.logger.info(
                f"  [{name}] {noise_type} {attr}={v}: FID={metrics['FID']:.2f} fFiD={metrics.get('fFID', float('nan')):.2f} Acc={metrics.get('Accuracy', float('nan')):.2f} SSIM={metrics.get('SSIM', float('nan')):.4f} LPIPS={metrics.get('LPIPS', float('nan')):.4f}"
            )
            _stream_csv(rows, csv_path)
    return rows


# =============================================================================
# 5. Ablation experiments
# =============================================================================

def run_experiment_epsilon(cfg, ctx, runner):
    """ε — Loss-function ablation. Sweep noise_type × loss_type."""
    grid = [
        {"_id": f"{nt}_loss_{lt}", "label": f"{nt}/{LOSS_VARIANTS[lt]}",
         "noise_type": nt, "loss_type": lt}
        for nt in cfg.noise_types for lt in cfg.loss_types
    ]
    return run_sweep(
        cfg, ctx, runner, name="epsilon", subdir="ablation", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        measure_fn=lambda m, f, p, c, r: measure_bottleneck(
            m, f, r.test_ds, c,
        ),
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="epsilon", noise_type=p["noise_type"],
            label=LOSS_VARIANTS[p["loss_type"]], config={"loss_type": p["loss_type"]},
            dist=_dist(m), loss=_loss_stats(m),
        ),
        train_kwargs_fn=lambda p: {"loss_type": p["loss_type"]},
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["loss_type"] == "huber" else None
        ),
    )


def run_experiment_zeta(cfg, ctx, runner):
    """ζ — Normalization ablation. Sweep noise_type × norm_type."""
    grid = [
        {"_id": f"{nt}_norm_{nrm}", "label": f"{nt}/{NORM_VARIANTS[nrm]}",
         "noise_type": nt, "norm_type": nrm}
        for nt in cfg.noise_types for nrm in cfg.norm_types
    ]
    return run_sweep(
        cfg, ctx, runner, name="zeta", subdir="ablation", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(
            num_classes=10, base_ch=c.base_ch, norm_type=p["norm_type"],
        ),
        measure_fn=lambda m, f, p, c, r: measure_bottleneck(
            m, f, r.test_ds, c,
        ),
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="zeta", noise_type=p["noise_type"],
            label=NORM_VARIANTS[p["norm_type"]], config={"norm_type": p["norm_type"]},
            dist=_dist(m), loss=_loss_stats(m),
        ),
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["norm_type"] == "group8" else None
        ),
    )


def run_experiment_kappa(cfg, ctx, runner):
    """κ — Activation-function ablation. Sweep noise_type × act_fn."""
    grid = [
        {"_id": f"{nt}_act_{af}", "label": f"{nt}/{ACT_VARIANTS[af]}",
         "noise_type": nt, "act_fn": af}
        for nt in cfg.noise_types for af in cfg.act_fns
    ]
    return run_sweep(
        cfg, ctx, runner, name="kappa", subdir="ablation", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(
            num_classes=10, base_ch=c.base_ch, act_fn=p["act_fn"],
        ),
        measure_fn=lambda m, f, p, c, r: measure_bottleneck(
            m, f, r.test_ds, c,
        ),
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="kappa", noise_type=p["noise_type"],
            label=ACT_VARIANTS[p["act_fn"]], config={"act_fn": p["act_fn"]},
            dist=_dist(m), loss=_loss_stats(m),
        ),
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["act_fn"] == "silu" else None
        ),
    )


def run_experiment_mu(cfg, ctx, runner):
    """
    μ — Skip-connection ablation. Probe-shaped: 4/5 variants share one baseline
    checkpoint via SkipZeroWrapper; only 'retrained_no_skip' has its own ckpt.
    """
    rows: list[ExperimentRecord] = []
    csv_path = ctx.get_path("metric", "mu.csv")

    for noise_type in cfg.noise_types:
        full_model, fwd = _load_unet_baseline(cfg, ctx, noise_type)

        retrain_tag = f"mu_{noise_type}_no_skip_retrained"
        retrain_req = LoadRequest(
            tag=retrain_tag, cfg=cfg, save_dir=ctx.run_dir, subdir="mu",
            model_factory=lambda: ConditionalUNet(
                num_classes=10, base_ch=cfg.base_ch,
                use_skip_h1=False, use_skip_h2=False,
            ),
            train_fn=lambda m, f, c, ck, t=retrain_tag: train_standard(
                c, m, f, ck, tag=t, loss_type="huber",
            ),
            fwd_builder=lambda c, nt=noise_type: build_forward_process(
                sigma_multiplicative(), c, noise_type=nt, H=c.H,
            ),
        )
        retrain_model, _, _ = load_or_train(retrain_req)

        for variant in SKIP_VARIANTS:
            if variant == "retrained_no_skip":
                eval_model = retrain_model
            elif variant == "full":
                eval_model = full_model
            else:
                eval_model = SkipZeroWrapper(
                    full_model,
                    zero_h1=variant in ("no_h1", "no_skip"),
                    zero_h2=variant in ("no_h2", "no_skip"),
                )

            m    = measure_bottleneck(eval_model, fwd, runner.test_ds, cfg)
            prof = measure_decoder_kappa4(eval_model, fwd, runner.test_ds, cfg, n_samples=cfg.n_samples)
            rows.append(ExperimentRecord(
                experiment_type="mu", noise_type=noise_type,
                label=SKIP_VARIANTS[variant], config={"variant": variant},
                dist=_dist(m), loss=_loss_stats(m),
                extras={f"dec_{k}_k4": v for k, v in prof.items()},
            ))
            ctx.logger.info(
                f"  [mu] {noise_type}/{variant:22s}  "
                f"κ4={m['kappa4']:+.3f}  L1={m['val_l1']:.4f}"
            )
            _stream_csv(rows, csv_path)
    return rows


def run_experiment_theta(cfg, ctx, runner):
    """θ — Time-conditional κ4. Probe shape: one baseline per noise_type, vary t."""
    rows: list[ExperimentRecord] = []
    csv_path = ctx.get_path("metric", "theta.csv")

    for noise_type in cfg.noise_types:
        model, fwd = _load_unet_baseline(cfg, ctx, noise_type)
        for t_val in cfg.t_values:
            acts = _bottleneck_acts_at_t(
                model, fwd, runner.test_ds, cfg,
                t_corrupt=t_val
            )
            cum = compute_marginal_cumulants(acts)
            mard = mardia_statistics(acts, use_pca=True)
            # Renamed for _dist consumption; compute_marginal_cumulants uses
            # `mean_kappa4` / `mean_abs_kappa3` rather than measure_bottleneck's
            # `kappa4` / `kappa3`.
            metrics = {
                "kappa3":   cum["mean_abs_kappa3"],
                "kappa4":   cum["mean_kappa4"],
                "mardia_z": mard["b2p_z"],
            }
            rows.append(ExperimentRecord(
                experiment_type="theta", noise_type=noise_type,
                label=f"t={t_val:.2f}", config={"t_value": t_val},
                dist=_dist(metrics),
            ))
            ctx.logger.info(
                f"  [theta] {noise_type}  t={t_val:.2f}  "
                f"κ4={cum['mean_kappa4']:+.3f}  Z={mard['b2p_z']:+.2f}"
            )
            _stream_csv(rows, csv_path)
    return rows


# =============================================================================
# 6. Optimizer experiments
# =============================================================================

def run_experiment_omicron(cfg, ctx, runner):
    """ο — Optimiser comparison. Sweep noise_type × opt_name, measure landscape."""
    opt_names = cfg.opt_names or list(OPT_LABELS.keys())[:4]
    grid = [
        {"_id": f"{nt}_{opt}", "label": f"{nt}/{OPT_LABELS.get(opt, opt)}",
         "noise_type": nt, "opt_name": opt}
        for nt in cfg.noise_types for opt in opt_names
    ]

    def measure(m, f, p, c, r):
        out = measure_bottleneck(m, f, r.test_ds, c)
        out.update(measure_sharpness(m, f, r.test_ds, c))
        out.update(measure_update_whiteness(
            m, f, r.train_ds, c, opt_name=p["opt_name"], n_batches=30,
        ))
        return out

    return run_sweep(
        cfg, ctx, runner, name="omicron", subdir="optimizer", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        measure_fn=measure,
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="omicron", noise_type=p["noise_type"],
            label=OPT_LABELS.get(p["opt_name"], p["opt_name"]),
            config={"opt_name": p["opt_name"]},
            dist=_dist(m), loss=_loss_stats(m), optim=_landscape(m),
        ),
        train_fn=train_with_optimizer,
        train_kwargs_fn=lambda p: {"opt_name": p["opt_name"], "loss_type": "huber"},
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["opt_name"] == "adamw" else None
        ),
    )


def run_experiment_pi(cfg, ctx, runner):
    """π — Gradient-noise distribution. Sweep noise_type × dist × std."""
    noise_dists = ("none", "gaussian", "rosenblatt_product")
    noise_stds  = (1e-4, 1e-3, 1e-2)
    grid = [
        {"_id": f"{nt}_{d}_std{str(s).replace('.', 'p')}",
         "label": f"{nt}/{d}/σ={s}",
         "noise_type": nt, "noise_dist": d, "noise_std": s}
        for nt in cfg.noise_types for d in noise_dists
        for s in (noise_stds if d != "none" else (0.0,))
    ]
    return run_sweep(
        cfg, ctx, runner, name="pi", subdir="optimizer", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        measure_fn=lambda m, f, p, c, r: measure_bottleneck(
            m, f, r.test_ds, c,
        ),
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="pi", noise_type=p["noise_type"],
            label=f"{p['noise_dist']}(σ={p['noise_std']})",
            config={"noise_dist": p["noise_dist"], "noise_std": p["noise_std"]},
            dist=_dist(m), loss=_loss_stats(m),
        ),
        train_fn=train_with_optimizer,
        train_kwargs_fn=lambda p: {
            "opt_name": "noise_adamw",
            "noise_std": p["noise_std"], "noise_dist": p["noise_dist"],
            "loss_type": "huber",
        },
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg)
            if (p["noise_dist"] == "none" and p["noise_std"] == 0.0) else None
        ),
    )


def run_experiment_rho(cfg, ctx, runner, fine_tune_epochs: int = 10):
    """
    ρ — Rosenblatt-SGLD: before/after fine-tune sharpness comparison.
    Probe shape: load baseline, measure BEFORE, fine-tune with grad-noise,
    measure AFTER.
    """
    rows: list[ExperimentRecord] = []
    csv_path = ctx.get_path("metric", "rho.csv")
    grad_noises = ("none", "gaussian", "rosenblatt_product")
    noise_stds  = (1e-3,)

    for noise_type in cfg.noise_types:
        base_model, fwd = _load_unet_baseline(cfg, ctx, noise_type)

        for grad_noise in grad_noises:
            for std in (noise_stds if grad_noise != "none" else (0.0,)):
                m_b = measure_bottleneck(base_model, fwd, runner.test_ds, cfg)
                s_b = measure_sharpness(base_model, fwd, runner.test_ds, cfg)
                rows.append(_rho_record(noise_type, "before", grad_noise, std, m_b, s_b))

                ft_model = copy.deepcopy(base_model).to(cfg.device)
                ft_tag   = f"rho_{noise_type}_{grad_noise}_std{str(std).replace('.','p')}_ft"
                ft_ckpt  = Path(ctx.ckpt_dir) / "rho" / f"{ft_tag}_final.pt"
                ft_ckpt.parent.mkdir(parents=True, exist_ok=True)

                with override(cfg, epochs=fine_tune_epochs, lr=cfg.lr / 5.0):
                    if not ft_ckpt.exists():
                        train_with_optimizer(
                            cfg, ft_model, fwd, ft_ckpt,
                            opt_name="noise_adamw",
                            noise_std=std, noise_dist=grad_noise,
                            tag=ft_tag, loss_type="huber",
                        )
                    else:
                        load_full(ft_ckpt, ft_model, device=cfg.device)
                ft_model.eval()

                m_a = measure_bottleneck(ft_model, fwd, runner.test_ds, cfg)
                s_a = measure_sharpness(ft_model, fwd, runner.test_ds, cfg)
                rows.append(_rho_record(noise_type, "after", grad_noise, std, m_a, s_a))

                ctx.logger.info(
                    f"  [rho] {noise_type} {grad_noise}(σ={std}): "
                    f"Δκ4={m_a['kappa4']-m_b['kappa4']:+.3f}  "
                    f"Δsharp={s_a['sharpness']-s_b['sharpness']:+.4f}"
                )
                _stream_csv(rows, csv_path)
    return rows


def _rho_record(noise_type, phase, grad_noise, std, m, s) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_type="rho", noise_type=noise_type,
        label=f"{grad_noise}(σ={std})/{phase}",
        config={"phase": phase, "grad_noise": grad_noise, "noise_std": std},
        dist=_dist(m), loss=_loss_stats(m), optim=_landscape(s),
    )


def run_experiment_tau(cfg, ctx, runner, log_every: int = 50):
    """
    τ — Gradient-κ4 trajectory during training. Probe shape: training itself
    is the measurement, so we use load_or_train's extras tuple.
    """
    opt_names = cfg.opt_names or ["adamw", "lion", "sgd"]
    rows: list[dict] = []
    csv_path = ctx.get_path("metric", "tau.csv")

    for opt_name in opt_names:
        tag = f"tau_rosenblatt_{opt_name}"
        train_kwargs = {
            "tag": tag, "opt_name": opt_name, "loss_type": "huber",
            "log_grads": True, "log_every": log_every,
        }
        req = LoadRequest(
            tag=tag, cfg=cfg, save_dir=ctx.run_dir, subdir="tau",
            model_factory=lambda: ConditionalUNet(num_classes=10, base_ch=cfg.base_ch),
            train_fn=_bind_train(train_with_optimizer, train_kwargs),
            fwd_builder=lambda c: build_forward_process(
                sigma_multiplicative(), c, noise_type="rosenblatt", H=c.H,
            ),
            # baseline_path REMOVED to ensure all optimizers start from scratch
        )
        _, _, extras = load_or_train(req)
        grad_log = extras[1] if len(extras) >= 2 else []
        ctx.logger.info(f"  [tau] {opt_name}: {len(grad_log)} grad-log entries")
        for entry in grad_log:
            rows.append({"opt_name": opt_name, **entry})
        _stream_csv(rows, csv_path)
    return rows


# =============================================================================
# 7. Gaussianity experiments
# =============================================================================

def run_experiment_alpha(cfg, ctx, runner):
    """α — Cumulant probe across UNet pixel stages and latent MLP stages."""
    rows: list = []
    all_kappa4: dict[str, dict[str, Any]] = {}
    csv_path = ctx.get_path("metric", "alpha.csv")

    for noise_type in cfg.noise_types:
        mname = noise_type.capitalize()
        all_kappa4[mname] = {}

        model, fwd = _load_unet_baseline(cfg, ctx, noise_type)
        unet_acts = extract_pipeline_stages(
            model, fwd, runner.test_ds, cfg,
            n_samples=cfg.n_samples, mode="image",
        )
        for stage_key, label in STAGE_LABELS_UNET.items():
            acts = unet_acts.get(stage_key, torch.empty(0))
            row, k4 = _analyse_stage(acts, mname, stage_key, label)
            rows.append(row)
            all_kappa4[mname][label] = k4
            ctx.logger.info(
                f"  [alpha] {mname} {label:22s}  "
                f"κ4={getattr(row, 'mean_k4', float('nan')):+.3f}"
            )

        ae, mlp, fwd_lat = _load_latent_pipeline(cfg, ctx, noise_type)
        lat_acts = extract_pipeline_stages(
            mlp, fwd_lat, runner.test_ds, cfg,
            n_samples=cfg.n_samples, mode="latent", ae=ae,
        )
        for stage_key, label in STAGE_LABELS_LATENT.items():
            acts = lat_acts.get(stage_key, torch.empty(0))
            row, k4 = _analyse_stage(acts, mname, stage_key, label)
            rows.append(row)
            all_kappa4[mname][label] = k4
            ctx.logger.info(
                f"  [alpha] {mname} {label:22s}  "
                f"κ4={getattr(row, 'mean_k4', float('nan')):+.3f}"
            )
        _stream_csv(rows, csv_path)

    plot_kappa4_violins(all_kappa4, ctx.get_path("plot", "alpha_kappa4_violins.png"))
    return rows


def run_experiment_beta(cfg, ctx, runner):
    """β — Bottleneck width vs Gaussianization. Sweep noise_type × bf."""
    grid = [
        {"_id": f"{nt}_bf{bf}", "label": f"{nt}/bf={bf}",
         "noise_type": nt, "bottleneck_factor": float(bf)}
        for nt in cfg.noise_types for bf in cfg.bf_list
    ]

    def measure(model, fwd, p, cfg, runner):
        m = _measure_beta_full(model, fwd, runner.test_ds, cfg)
        m["_rig"] = rigidity_test(
            model, fwd, runner.test_ds, cfg, sigma_levels=[0.3, 0.5, 1.0],
        )
        m["_bneck_ch"] = int(model.bneck_ch)
        return m

    def record(p, m):
        rig = m["_rig"]
        sk  = 0.5
        return BetaResult(
            noise_type=p["noise_type"],
            bottleneck_factor=p["bottleneck_factor"],
            bneck_ch=m["_bneck_ch"],
            mean_k4_input=m["k4_input"],
            mean_k4_bneck=m["k4_bneck"],
            std_k4_bneck=m["std_k4_bneck"],
            max_k4_bneck=m["max_k4_bneck"],
            frac_nong_bneck=m["frac_nong_bneck"],
            mean_k4_x0hat=m["k4_x0hat"],
            mardia_b2p_z=m["mardia_z_bneck"],
            mardia_b2p_z_x0hat=m["mardia_z_x0hat"],
            offline_loss_mse=m["offline_mse"],
            offline_loss_mae=m["offline_mae"],
            offline_loss_huber=m["offline_huber"],
            pr_bneck=m["pr_bneck"],
            effective_rank_bneck=m["effective_rank_bneck"],
            whiteness_bneck=m["whiteness_bneck"],
            js_gauss_bneck=m["js_gauss_bneck"],
            perturb_gauss_huber     =rig["gaussian"].get(sk,   float("nan")),
            perturb_laplace_huber   =rig["laplace"].get(sk,    float("nan")),
            perturb_rosenblatt_huber=rig["rosenblatt"].get(sk, float("nan")),
            perturb_t3_huber        =rig["student_t3"].get(sk, float("nan")),
        )

    return run_sweep(
        cfg, ctx, runner, name="beta", subdir="beta", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(
            num_classes=10, base_ch=c.base_ch,
            bottleneck_factor=p["bottleneck_factor"],
        ),
        measure_fn=measure, record_fn=record,
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg)
            if p["bottleneck_factor"] == 1.0 else None
        ),
    )


@torch.no_grad()
def _measure_beta_full(model, fwd, test_ds, cfg) -> dict:
    """β-specific extended measurement (input, bottleneck, x0hat statistics)."""
    device = cfg.device
    n = cfg.n_samples

    bn_store = ActivationStore(spatial_pool=True)
    handle = model.mid2.register_forward_hook(bn_store.hook_fn)

    raw_chunks, x0h_chunks = [], []
    n_done = 0
    loader = DataLoader(
        test_ds, batch_size=min(cfg.batch_size, 128),
        shuffle=False, num_workers=2,
    )
    for x0, y in loader:
        if n_done >= n:
            break
        x0, y = x0.to(device), y.to(device)
        B = x0.size(0)
        raw_chunks.append(x0.view(B, -1).cpu())
        t_one = torch.ones(B, device=device)
        x_T, _, _ = fwd.corrupt(x0, t_one, y=y)
        t_min = torch.full((B,), cfg.T_MIN, device=device)
        null = torch.full_like(y, 10)
        c_in = fwd.c_in(t_min).view(-1, 1, 1, 1)
        # By applying CFG, here do a double forward pass (x0c and x0u) inside a loop 
        x0_out = model(x_T * c_in, t_min, y)
        x0h = x0_out.clamp(-1.0, 1.0)
        x0h_chunks.append(x0h.view(B, -1).cpu())
        n_done += B
    handle.remove()

    bneck = bn_store.get()[:n]
    raw   = torch.cat(raw_chunks, 0)[:n]
    x0hat = torch.cat(x0h_chunks, 0)[:n]

    cum_bn  = compute_marginal_cumulants(bneck)
    cum_raw = compute_marginal_cumulants(raw)
    cum_x0h = compute_marginal_cumulants(x0hat)
    spec_bn = compute_spectrum_stats(bneck)
    mard_bn = mardia_statistics(bneck, use_pca=True)
    mard_x0 = mardia_statistics(x0hat, use_pca=True)

    return {
        "k4_input":             cum_raw["mean_kappa4"],
        "k4_bneck":             cum_bn["mean_kappa4"],
        "std_k4_bneck":         cum_bn["std_kappa4"],
        "max_k4_bneck":         cum_bn["max_kappa4"],
        "frac_nong_bneck":      cum_bn["frac_non_gauss"],
        "k4_x0hat":             cum_x0h["mean_kappa4"],
        "pr_bneck":             spec_bn["pr"],
        "effective_rank_bneck": spec_bn["effective_rank"],
        "mardia_z_bneck":       mard_bn["b2p_z"],
        "mardia_z_x0hat":       mard_x0["b2p_z"],
        "whiteness_bneck":      covariance_whiteness(bneck),
        "js_gauss_bneck":       js_divergence_from_gaussian(bneck),
        "offline_mse":          F.mse_loss(x0hat, raw).item(),
        "offline_mae":          F.l1_loss(x0hat, raw).item(),
        "offline_huber":        F.smooth_l1_loss(x0hat, raw).item(),
    }


def run_experiment_gamma(cfg, ctx, runner):
    """γ — Full layer-by-layer kurtosis trace."""
    rows: list[LayerStats] = []
    csv_path = ctx.get_path("metric", "gamma.csv")

    for noise_type in cfg.noise_types:
        mname = noise_type.capitalize()
        model, fwd = _load_unet_baseline(cfg, ctx, noise_type)
        trace = extract_full_layer_trace(
            model, fwd, runner.test_ds, cfg, n_samples=cfg.n_samples,
        )
        for depth, key in enumerate(UNET_LAYER_KEYS):
            acts = trace.get(key, torch.empty(0))
            if acts.numel() == 0:
                continue
            cum  = compute_marginal_cumulants(acts)
            spec = compute_spectrum_stats(acts)
            mard = mardia_statistics(acts, use_pca=True)
            rows.append(LayerStats(
                model=mname, noise_type=noise_type,
                layer_key=key, layer_label=LAYER_LABELS.get(key, key),
                depth_index=depth,
                mean_k4=cum["mean_kappa4"], std_k4=cum["std_kappa4"],
                frac_nong=cum["frac_non_gauss"],
                pr=spec["pr"], effective_rank=spec["effective_rank"],
                whiteness=covariance_whiteness(acts),
                mardia_b2p_z=mard["b2p_z"],
            ))
        ctx.logger.info(f"  [gamma] {mname}: {sum(1 for r in rows if r.model == mname)} layers")
        _stream_csv(rows, csv_path)

    plot_layer_profiles(rows, ctx.plot_dir)
    return rows


def run_experiment_delta(cfg, ctx, runner):
    """δ — Rigidity test on the bf=1.0 baseline at a fine σ grid."""
    
    rows: list[dict] = []
    csv_path = ctx.get_path("metric", "delta.csv")

    for noise_type in cfg.noise_types:
        mname = noise_type.capitalize()
        model, fwd = _load_unet_baseline(cfg, ctx, noise_type)
        rig = rigidity_test(model, fwd, runner.test_ds, cfg, sigma_levels=cfg.sigma_grid)

        plot_rigidity(
            rig, ctx.get_path("plot", f"delta_rigidity_{noise_type}.png"),
            title=f"Rigidity — {mname} model (bf=1.0)",
        )

        for sigma in cfg.sigma_grid:
            for kind in cfg.noise_kinds:
                rows.append({
                    "noise_type":   noise_type,
                    "perturbation": kind,
                    "sigma":        sigma,
                    "huber_loss":   rig[kind].get(sigma, float("nan")),
                })
        ctx.logger.info(f"  [delta] {mname}: {len(cfg.sigma_grid)} σ-points written")
        _stream_csv(rows, csv_path)
    return rows


# =============================================================================
# 8. Cold ablation experiments
# =============================================================================


def run_experiment_sigma_comparison(cfg, ctx, runner):
    """Cold — Sigma-factory comparison via image-level FID."""
    class_vars = compute_pixel_variance(cfg.dataset, conditional=True)
    global_var = compute_pixel_variance(cfg.dataset, conditional=False)
    sigmas = [
        ("multiplicative",  sigma_multiplicative()),
        ("anisotropic_h",   sigma_anisotropic(mode="h_emphasis")),
        ("anisotropic_v",   sigma_anisotropic(mode="v_emphasis")),
        ("pca_global",      sigma_pca_whitened(global_var)),
        ("pca_conditional", sigma_pca_whitened(class_vars)),
        ("edge_aware",      sigma_edge_aware()),
    ]
    grid = [
        {
            "_id": f"{cfg.noise_type}_{name}_H{cfg.H}", 
            "label": name, 
            "noise_type": cfg.noise_type, 
            "sigma": sigma
        }
        for name, sigma in sigmas
    ]
    rows = run_sweep(
        cfg, ctx, runner, name="sigma_comparison", subdir="sigma", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        fwd_builder=lambda p, c: build_forward_process(
            p["sigma"], c, noise_type=p["noise_type"], H=c.H,
        ),
        measure_fn=_measure_fid,
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="sigma_comparison", noise_type=p["noise_type"],
            label=p["label"], config={"sigma": p["label"]}, extras=m, 
        ),
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg)
            if p["label"] == "multiplicative" else None 
        ),
    )
    plot_all_sigma_patterns(
        [s for _, s in sigmas],
        ctx.get_path("sample", "all_sigma_patterns.png"),
        dataset_name=cfg.dataset,
    )
    return rows


def run_experiment_pca_basis(cfg, ctx, runner):
    """
    Cold (2c) — PCA vs Pixel basis. Narrower than sigma_comparison: only
    pixel-multiplicative against PCA-whitened Σ in global and class-conditional
    forms.
    """
    class_vars = compute_pixel_variance(cfg.dataset, conditional=True)
    global_var = compute_pixel_variance(cfg.dataset, conditional=False)
    bases = [
        ("pixel_multiplicative", sigma_multiplicative()),
        ("pca_global",           sigma_pca_whitened(global_var)),
        ("pca_class_cond",       sigma_pca_whitened(class_vars)),
    ]
    grid = [
        {"_id": f"{nt}_{name}", "label": f"{nt}/{name}",
         "noise_type": nt, "basis": name, "sigma": sigma}
        for nt in cfg.noise_types for name, sigma in bases
    ]
    return run_sweep(
        cfg, ctx, runner, name="pca_basis", subdir="pca_basis", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        fwd_builder=lambda p, c: build_forward_process(
            p["sigma"], c, noise_type=p["noise_type"], H=c.H,
        ),
        measure_fn=_measure_fid,
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="pca_basis", noise_type=p["noise_type"],
            label=p["label"], config={"basis": p["basis"]}, extras=m,
        ),
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg)
            if p["basis"] == "pixel_multiplicative" else None
        ),
    )


def run_experiment_cold_latent(cfg, ctx, runner):
    """Cold — Latent-space FID across noise_type × sigma_max."""
    rows: list[ExperimentRecord] = []
    csv_path = ctx.get_path("metric", "cold_latent.csv")

    for noise_type in cfg.noise_types:
        for sigma_max in cfg.sigma_maxs:
            with override(cfg, sigma_max=sigma_max):
                ae, mlp, fwd = _load_latent_pipeline(cfg, ctx, noise_type)
                metrics = runner.evaluator.evaluate(
                    mlp, fwd, runner.real_imgs, runner.test_ds, cfg,
                    bridge=cfg.bridge,
                )
            rows.append(ExperimentRecord(
                experiment_type="cold_latent", noise_type=noise_type,
                label=f"{noise_type}/σ_max={sigma_max}",
                config={"sigma_max": sigma_max}, extras=metrics,
            ))
            ctx.logger.info(
                f"  [cold_latent] {noise_type}/σ_max={sigma_max}: "
                f"FID={metrics['FID']:.2f} fFID={metrics.get('fFID', float('nan')):.2f} Acc={metrics.get('Accuracy', float('nan')):.1f}%  SSIM={metrics.get('SSIM', float('nan')):.4f}  LPIPS={metrics.get('LPIPS', float('nan')):.4f}"
            )
            _stream_csv(rows, csv_path)
    return rows


def run_experiment_cold_loss(cfg, ctx, runner):
    """
    Cold (2f) — Generation-loss ablation at the image level. Like ε but with
    FID/Accuracy/SSIM/LPIPS instead of bottleneck cumulants.
    """
    grid = [
        {"_id": f"{nt}_loss_{lt}", "label": f"{nt}/{LOSS_VARIANTS[lt]}",
         "noise_type": nt, "loss_type": lt}
        for nt in cfg.noise_types for lt in cfg.loss_types
    ]
    return run_sweep(
        cfg, ctx, runner, name="cold_loss", subdir="cold_loss", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        measure_fn=_measure_fid,
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="cold_loss", noise_type=p["noise_type"],
            label=p["label"], config={"loss_type": p["loss_type"]}, extras=m,
        ),
        train_kwargs_fn=lambda p: {"loss_type": p["loss_type"]},
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["loss_type"] == "huber" else None
        ),
    )


def run_experiment_cold_bridge(cfg, ctx, runner):
    """Cold — Bridge-strategy comparison on the fixed rosenblatt baseline."""
    return _inference_time_sweep(
        cfg, ctx, runner,
        name="cold_bridge", attr="bridge",
        values=("stochastic", "hybrid", "deterministic"),
        fixed_noise="rosenblatt",
        label_fmt=lambda nt, v: f"{nt}/{v}",
    )


def run_experiment_cold_h_sweep(cfg, ctx, runner):
    """Cold — Hurst exponent sweep with image-level FID."""
    grid = [
        {"_id": f"{nt}_H{H}", "label": f"{nt}/H={H}",
         "noise_type": nt, "H": float(H)}
        for nt in cfg.noise_types for H in cfg.h_values
    ]
    return run_sweep(
        cfg, ctx, runner, name="cold_h_sweep", subdir="cold_h", grid=grid,
        model_factory=lambda p, c: ConditionalUNet(num_classes=10, base_ch=c.base_ch),
        fwd_builder=lambda p, c: build_forward_process(
            sigma_multiplicative(), c, noise_type=p["noise_type"], H=p["H"],
        ),
        measure_fn=_measure_fid,
        record_fn=lambda p, m: ExperimentRecord(
            experiment_type="cold_h_sweep", noise_type=p["noise_type"],
            label=p["label"], config={"H": p["H"]}, extras=m,
        ),
        baseline_path_fn=lambda p: (
            _baseline_ckpt(ctx, p["noise_type"], cfg) if p["H"] == cfg.H else None
        ),
    )

def run_experiment_n_steps(cfg, ctx, runner):
    """Cold (2g) — Sampling-step ablation. Inference-only sweep over cfg.n_steps."""
    return _inference_time_sweep(
        cfg, ctx, runner,
        name="n_steps", attr="n_steps",
        values=cfg.n_steps_grid,
        label_fmt=lambda nt, v: f"{nt}/steps={v}",
    )


def run_experiment_cfg_scale(cfg, ctx, runner):
    """Cold (2h) — CFG-scale ablation. Inference-only sweep over cfg.cfg_scale."""
    return _inference_time_sweep(
        cfg, ctx, runner,
        name="cfg_scale", attr="cfg_scale",
        values=cfg.cfg_scale_grid,
        label_fmt=lambda nt, v: f"{nt}/w={v}",
    )