"""
Single source of truth for shared constants, label maps, and result dataclasses.

Nothing in `rcd/data/` or `rcd/train/` may redefine the symbols exported here.
Plotting, logging, measurement, and run scripts must import from this module.

Result dataclasses live here because they are produced by experiments and
consumed by save/plot/measure — placing them in datasets.py created a
circular dependency between data definitions and experiment record types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Literal, Tuple


# =============================================================================
# 1. Result dataclasses
# =============================================================================

@dataclass
class GaussianityStats:
    k3:             float = math.nan
    k4:             float = math.nan
    std_k4:         float = math.nan
    max_k4:         float = math.nan
    frac_nong:      float = math.nan
    pr:             float = math.nan
    mardia_z:       float = math.nan
    mardia_b2p:     float = math.nan
    mardia_b2p_exp: float = math.nan
    effective_rank: float = math.nan
    whiteness:      float = math.nan
    js_gauss:       float = math.nan


@dataclass
class LossStats:
    l1:       float = math.nan
    l2:       float = math.nan
    huber:    float = math.nan
    mse:      float = math.nan
    mae:      float = math.nan
    quantile: float = math.nan


@dataclass
class LandscapeStats:
    sharpness:     float = math.nan
    frac_neg:      float = math.nan
    update_k4:     float = math.nan
    update_w1:     float = math.nan
    update_std_cv: float = math.nan


@dataclass
class LayerStats:
    model:          str
    noise_type:     str
    layer_key:      str
    layer_label:    str
    depth_index:    int
    mean_k4:        float = math.nan
    mean_k4_unit:   float = math.nan
    std_k4:         float = math.nan
    frac_nong:      float = math.nan
    pr:             float = math.nan
    effective_rank: float = math.nan
    whiteness:      float = math.nan
    mardia_b2p_z:   float = math.nan


@dataclass
class BetaResult:
    noise_type:               str
    bottleneck_factor:        float
    bneck_ch:                 int
    mean_k4_input:            float = math.nan
    mean_k4_bneck:            float = math.nan
    mean_k4_bneck_center:     float = math.nan
    mean_k4_bneck_unit:       float = math.nan
    mardia_b2p_z_bneck_unit:  float = math.nan
    std_k4_bneck:             float = math.nan
    max_k4_bneck:             float = math.nan
    frac_nong_bneck:          float = math.nan
    mean_k4_x0hat:            float = math.nan
    mardia_b2p_z:             float = math.nan
    mardia_b2p_z_x0hat:       float = math.nan
    offline_loss_mse:         float = math.nan
    offline_loss_mae:         float = math.nan
    offline_loss_huber:       float = math.nan
    pr_bneck:                 float = math.nan
    effective_rank_bneck:     float = math.nan
    whiteness_bneck:          float = math.nan
    js_gauss_bneck:           float = math.nan
    perturb_gauss_huber:      float = math.nan
    perturb_laplace_huber:    float = math.nan
    perturb_rosenblatt_huber: float = math.nan
    perturb_t3_huber:         float = math.nan


@dataclass
class ExperimentRecord:
    experiment_type: str
    noise_type:      str
    model_name:      str = ""
    label:           str = ""
    config:          Dict[str, Any] = field(default_factory=dict)
    dist:            GaussianityStats = field(default_factory=GaussianityStats)
    loss:            LossStats        = field(default_factory=LossStats)
    optim:           LandscapeStats   = field(default_factory=LandscapeStats)
    traces:          Dict[str, Any]   = field(default_factory=dict)
    extras:          Dict[str, Any]   = field(default_factory=dict)

    def flatten(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "experiment_type": self.experiment_type,
            "noise_type":      self.noise_type,
            "model_name":      self.model_name,
            "label":           self.label,
        }
        for k, v in self.config.items():
            out[f"cfg_{k}"] = v
        for prefix, group in (("dist", self.dist), ("loss", self.loss), ("optim", self.optim)):
            for k, v in asdict(group).items():
                if not (isinstance(v, float) and math.isnan(v)):
                    out[f"{prefix}_{k}"] = v
        for layer_name, layer_metrics in self.traces.items():
            if hasattr(layer_metrics, "__dict__"):
                layer_metrics = asdict(layer_metrics)
            for mk, mv in layer_metrics.items():
                if not (isinstance(mv, float) and math.isnan(mv)):
                    out[f"trace_{layer_name}_{mk}"] = mv
        out.update(self.extras)
        return out


@dataclass
class LatexTableSpec:
    caption:  str
    label:    str
    headers:  List[str]
    col_spec: str
    row_fmt:  Callable[[dict], List[str]]
    group_by: str | None = None
    footer:   str = ""
    size:     str = r"\scriptsize"


# =============================================================================
# 2. Constants — colors, markers, line styles
# =============================================================================

COLORS: Dict[str, str] = {
    "gaussian":          "#3A7EBF",
    "Gaussian":          "#3A7EBF",
    "rosenblatt":        "#E07B39",
    "Rosenblatt":        "#E07B39",
    "full":              "#3A7EBF",
    "no_h2":             "#E07B39",
    "no_h1":             "#55A868",
    "no_skip":           "#C44E52",
    "retrained_no_skip": "#8172B2",
}

MARKERS: Dict[str, str] = {
    "gaussian":   "o",
    "Gaussian":   "o",
    "rosenblatt": "s",
    "Rosenblatt": "s",
}

LINESTYLES: Dict[str, str] = {
    "full":              "-",
    "no_h2":             "--",
    "no_h1":             "-.",
    "no_skip":           ":",
    "retrained_no_skip": "-",
}


# =============================================================================
# 3. Variant label maps
# =============================================================================

OPT_LABELS: Dict[str, str] = {
    "adamw":   "AdamW (baseline)",
    "lion":    "Lion (sign-based, anti-Gaussian)",
    "sgd":     "SGD + momentum (no whitening)",
    # "rmsprop": "RMSprop",
    # "nadam":   "NAdam",
}

NOISE_LABELS: Dict[str, str] = {
    "clean":              "No noise (baseline AdamW)",
    "gaussian":           "Gaussian gradient noise",
    "rosenblatt":         "Rosenblatt gradient noise",
    "rosenblatt_product": "Rosenblatt product noise",
    "laplace":            "Laplace gradient noise",
    # "uniform":            "Uniform gradient noise",
}

LOSS_VARIANTS: Dict[str, str] = {
    "l1":           "L1 (MAE)",
    "l2":           "L2 (MSE)",
    "huber":        "Smooth L1 / Huber (baseline)",
    "quantile":     "Quantile τ=0.50",
    "quantile_0.1": "Quantile τ=0.10",
    "quantile_0.25":"Quantile τ=0.25",
    "quantile_0.5": "Quantile τ=0.50",
    "quantile_0.75":"Quantile τ=0.75",
    "quantile_0.9": "Quantile τ=0.90",
    "elastic":      "Elastic (0.5·L1 + 0.5·L2)",
}

NORM_VARIANTS: Dict[str, str] = {
    "group8": "GroupNorm-8 (baseline)",
    "group4": "GroupNorm-4",
    "group1": "GroupNorm-1 (LayerNorm equiv.)",
    "batch":  "BatchNorm",
    "none":   "No normalisation",
}

ACT_VARIANTS: Dict[str, str] = {
    "silu": "SiLU (baseline)",
    "relu": "ReLU",
    "gelu": "GELU",
    "tanh": "Tanh",
    "mish": "Mish",
}

SKIP_VARIANTS: Dict[str, str] = {
    "full":              "Full Network (Baseline)",
    "no_h2":             "Zeroed $h_2$ (Test-time)",
    "no_h1":             "Zeroed $h_1$ (Test-time)",
    "no_skip":           "Zeroed Both Skips (Test-time)",
    "retrained_no_skip": "Retrained Without Skips",
}

BRIDGE_VARIANTS: Dict[str, str] = {
    "stochastic": "Stochastic Bridge",
    "hybrid":     "Hybrid Bridge",
    "deterministic": "Deterministic Bridge"
}

H_VARIANTS: Dict[float, str] = {
    0.6: "H = 0.6", 
    0.7: "H = 0.7", 
    0.8: "H = 0.8", 
    0.9: "H = 0.9"
}

SIGMA_VARIANTS: Dict[float, str] = {
    4.0:  "σ = 4.0",
    16.0: "σ = 16.0"
}

SIGMA_FN: Dict[str, str] = {
    "multiplicative":  "multiplicative",
    "anisotropic_h":   "anisotropic_h",
    "anisotropic_v":   "anisotropic_v",
    "pca_global":      "pca_global",
    "pca_conditional": "pca_conditional",
    "edge_aware":      "edge_aware",
    "PCA_basis":       "pca_basis",
}

# =============================================================================
# 4. Pipeline-stage label maps
# =============================================================================

STAGE_LABELS_UNET: Dict[str, str] = {
    "input":      r"Input $x_0$",
    "corrupted":  r"Corrupted $x_T$",
    "mid_t05":    r"Mid-gen $x_{0.5}$",
    "bottleneck": r"Bottleneck",
    "x0hat":      r"Output $\hat{x}_0$",
}

STAGE_LABELS_LATENT: Dict[str, str] = {
    "image_input":  r"Image $x_0$",
    "latent_z0":    r"AE latent $z_0$",
    "latent_corr":  r"Corrupted $z_T$",
    "mlp_mid":      r"MLP mid-layer",
    "latent_x0hat": r"Decoded $\hat{x}_0$",
}


# =============================================================================
# 5. Layer keys, labels, derived selections
# =============================================================================

UNET_LAYER_KEYS: List[str] = [
    "init_conv", "down1_0", "down1_1", "pool1",
    "down2_0",   "down2_1", "attn2",   "pool2",
    "mid1",      "attn_mid","mid2",
    "up_res2_0", "up_res2_1","up_attn2",
    "up_res1_0", "up_res1_1","out",
]

LAYER_LABELS: Dict[str, str] = {
    "init_conv":  "init",
    "down1_0":    "d1[0]",
    "down1_1":    "d1[1] h1",
    "pool1":      "pool1",
    "down2_0":    "d2[0]",
    "down2_1":    "d2[1]",
    "attn2":      "attn2 h2",
    "pool2":      "pool2",
    "mid1":       "mid1",
    "attn_mid":   "attn_mid",
    "mid2":       "mid2",
    "up_res2_0":  "up_res2[0]",
    "up_res2_1":  "up_res2[1]",
    "up_attn2":   "up_attn2",
    "up_res1_0":  "up_res1[0]",
    "up_res1_1":  "up_res1[1]",
    "out":        "out",
}

# LaTeX-formatted variant of LAYER_LABELS (used by report tables only)
UNET_LAYER_LABELS_TEX: Dict[str, str] = {
    "init_conv":  r"init",
    "down1_0":    r"down1[0]",
    "down1_1":    r"down1[1] h1",
    "pool1":      r"pool1",
    "down2_0":    r"down2[0]",
    "down2_1":    r"down2[1]",
    "attn2":      r"attn2 h2",
    "pool2":      r"pool2",
    "mid1":       r"mid1",
    "attn_mid":   r"attn\_mid",
    "mid2":       r"mid2 h3",
    "up_res2_0":  r"up\_res2[0]",
    "up_res2_1":  r"up\_res2[1]",
    "up_attn2":   r"up\_attn2 h\_up2",
    "up_res1_0":  r"up\_res1[0]",
    "up_res1_1":  r"up\_res1[1] h\_up1",
    "out":        r"out",
}

DECODER_KEYS: List[str] = [
    "up_res2_0", "up_res2_1", "up_attn2",
    "up_res1_0", "up_res1_1", "out",
]
DECODER_KEYS_K4: List[str] = [f"{k}_k4" for k in DECODER_KEYS]
DECODER_LABELS: List[str] = [LAYER_LABELS[k] for k in DECODER_KEYS]


# =============================================================================
# 6. Experiment registry
# =============================================================================

ExperimentStatus = Literal["runnable", "needs_rerun", "needs_rescope", "not_wired"]


@dataclass(frozen=True)
class ExperimentSpec:
    key:     str
    title:   str
    family:  str
    module:  str
    mode:    str
    status:  ExperimentStatus
    metrics: Tuple[str, ...]
    notes:   str = ""


IMAGE_METRICS: Tuple[str, ...] = ("FID", "fFID", "Accuracy", "SSIM", "LPIPS", "eval_time_s")
GAUSS_METRICS: Tuple[str, ...] = ("kappa3", "kappa4", "mardia_z", "PR", "whiteness", "JS")
OPT_METRICS:   Tuple[str, ...] = ("kappa4", "sharpness", "update_whiteness", "val_loss")


EXPERIMENTS: Tuple[ExperimentSpec, ...] = (
    ExperimentSpec("sigma_geometry", "2a: Noise Geometry Comparison",         "diffusion",   "experiments.visualize_diffusion", "sigma_comparison", "needs_rerun", IMAGE_METRICS),
    ExperimentSpec("bridge",         "2b: Bridge Ablation",                   "diffusion",   "experiments.run_cold_ablation",   "bridge",           "runnable",    IMAGE_METRICS),
    ExperimentSpec("pca_basis",      "2c: PCA vs Pixel Basis",                "diffusion",   "experiments.visualize_diffusion", "pca_basis",        "runnable",    IMAGE_METRICS),
    ExperimentSpec("latent",         "2d: Latent Summary",                    "latent",      "experiments.run_latent",          "latent",           "needs_rerun", IMAGE_METRICS),
    ExperimentSpec("hurst",          "2e: H Ablation",                        "ablation",    "experiments.run_cold_ablation",   "H",                "runnable",    IMAGE_METRICS),
    ExperimentSpec("cold_loss",      "2f: Generation Loss Ablation",          "ablation",    "experiments.run_cold_ablation",   "loss",             "runnable",    IMAGE_METRICS),
    ExperimentSpec("n_steps",        "2g: Sampling Step Ablation",            "ablation",    "experiments.run_cold_ablation",   "steps",            "runnable",    IMAGE_METRICS),
    ExperimentSpec("cfg_scale",      "2h: CFG Scale Ablation",                "ablation",    "experiments.run_cold_ablation",   "cfg_scale",        "runnable",    IMAGE_METRICS),
    ExperimentSpec("alpha",          "Cumulant Gaussianization Probe",        "gaussianity", "rcd.experiments.run_gaussianity", "alpha",            "runnable",    GAUSS_METRICS),
    ExperimentSpec("beta",           "Bottleneck Width vs Gaussianization",   "gaussianity", "rcd.experiments.run_gaussianity", "beta",             "runnable",    GAUSS_METRICS),
    ExperimentSpec("gamma",          "Full Layer-by-Layer Kurtosis Trace",    "gaussianity", "rcd.experiments.run_gaussianity", "gamma",            "runnable",    GAUSS_METRICS),
    ExperimentSpec("delta",          "Latent Perturbation Rigidity Test",     "gaussianity", "rcd.experiments.run_gaussianity", "delta",            "runnable",    ("rigidity",)),
    ExperimentSpec("epsilon",        "Loss Function Ablation",                "ablation",    "rcd.experiments.run_ablation",    "epsilon",          "runnable",    GAUSS_METRICS + ("val_l1",)),
    ExperimentSpec("zeta",           "Normalization Ablation",                "ablation",    "rcd.experiments.run_ablation",    "zeta",             "runnable",    GAUSS_METRICS),
    ExperimentSpec("kappa",          "Activation Function Ablation",          "ablation",    "rcd.experiments.run_ablation",    "kappa",            "runnable",    GAUSS_METRICS),
    ExperimentSpec("mu",             "Skip Connection Ablation",              "ablation",    "rcd.experiments.run_ablation",    "mu",               "runnable",    GAUSS_METRICS + ("val_l1",)),
    ExperimentSpec("theta",          "Time-Conditional Bottleneck κ4",        "ablation",    "rcd.experiments.run_ablation",    "theta",            "runnable",    GAUSS_METRICS),
    ExperimentSpec("omicron",        "Optimiser Comparison",                  "optimizer",   "rcd.experiments.run_optimizer",   "omicron",          "runnable",    OPT_METRICS),
    ExperimentSpec("pi",             "Gradient Noise Distribution",           "optimizer",   "rcd.experiments.run_optimizer",   "pi",               "runnable",    OPT_METRICS),
    ExperimentSpec("rho",            "Rosenblatt-SGLD Landscape Analysis",    "optimizer",   "rcd.experiments.run_optimizer",   "rho",              "runnable",    OPT_METRICS),
    ExperimentSpec("tau",            "Gradient κ₄ Evolution During Training", "optimizer",   "rcd.experiments.run_optimizer",   "tau",              "runnable",    ("kappa4_trace",)),
)


class ExperimentRegistry:
    @classmethod
    def get_unet_keys(cls) -> List[str]:
        return list(UNET_LAYER_KEYS)

    @classmethod
    def get(cls, key: str) -> ExperimentSpec:
        for s in EXPERIMENTS:
            if s.key == key:
                return s
        raise KeyError(f"Unknown experiment: {key!r}")

    @classmethod
    def by_family(cls, family: str) -> Tuple[ExperimentSpec, ...]:
        return tuple(s for s in EXPERIMENTS if s.family == family)