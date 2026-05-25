"""
rcd — Rosenblatt Cold Diffusion shared primitives.

Centralises the parts of the existing codebase that were duplicated across
Rosenblatt_cold_diffusion_unified.py, Experiment_Gaussianity.py,
Experiment_Ablation.py, and Experiment_Optimizer.py with subtle divergences.

Modules
-------
ema           EMA with state_dict / load_state_dict.
checkpoints   Full-state save/load (model + EMA + opt + sch + scaler + epoch).
noise         Noise samplers with explicit naming (true Rosenblatt vs
              product-of-Gaussians proxy).
forward       RosenblattForward factory with cached empirical E[Sigma^2].
overrides     Context-managed cfg overrides; no in-place leaks.
training      Unified train_diffusion() used by image and latent training.
measurement   Parametric layer / bottleneck activation measurement.
"""

from .diffusion import (
    EMA,
    RosenblattForward,
    build_forward,
    get_eg2,
    estimate_eg2,
    clear_eg2_cache,
    sample_noise,
    sample_gaussian,
    sample_rosenblatt,
    sample_rosenblatt_proxy,
    sample_grad_noise,
    sigma_additive,
    sigma_multiplicative,
    sigma_pca_whitened,
    sigma_pca_whitened_conditional,
    sigma_pca_whitened_global,
    sigma_anisotropic,
    sigma_edge_aware,
    sigma_h_emphasis,
    sigma_v_emphasis,
    compute_condition_pixel_variance,
    compute_global_pixel_variance,
    compute_pixel_variance,
    train_diffusion,
    compute_loss,
)
from .tracker.checkpoints import save_full, load_full, find_latest_epoch
from .core.overrides import override
from .evaluation.measurement import (
    ActivationStore,
    capture_layer,
    measure_layers,
    measure_layer,
    get_unet_modules,
    UNET_LAYER_KEYS,
)

__all__ = [
    "EMA",
    "save_full", "load_full", "find_latest_epoch",
    "sample_noise", "sample_gaussian", "sample_rosenblatt",
    "sample_rosenblatt_proxy", "sample_grad_noise",
    "RosenblattForward", "sigma_additive", "sigma_multiplicative",
    "sigma_pca_whitened", "sigma_pca_whitened_conditional",
    "sigma_pca_whitened_global", "sigma_anisotropic", "sigma_edge_aware",
    "sigma_h_emphasis", "sigma_v_emphasis",
    "compute_condition_pixel_variance", "compute_global_pixel_variance",
    "compute_pixel_variance",
    "build_forward", "get_eg2", "estimate_eg2", "clear_eg2_cache",
    "override",
    "train_diffusion", "compute_loss",
    "ActivationStore", "capture_layer", "measure_layers", "measure_layer",
    "get_unet_modules", "UNET_LAYER_KEYS",
]
