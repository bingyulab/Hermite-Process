from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ExperimentStatus = Literal["runnable", "needs_rerun", "needs_rescope", "not_wired"]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    family: str
    module: str
    mode: str
    status: ExperimentStatus
    metrics: tuple[str, ...]
    notes: str = ""


IMAGE_METRICS = ("FID", "fFID", "Accuracy", "SSIM", "LPIPS", "eval_time_s")
GAUSS_METRICS = ("kappa3", "kappa4", "mardia_z", "PR", "whiteness", "JS")
OPT_METRICS = ("kappa4", "sharpness", "update_whiteness", "val_loss")


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("sigma_geometry", "Experiment 2a: Noise Geometry Comparison",
                   "diffusion", "experiments.visualize_diffusion", "sigma_comparison",
                   "needs_rerun", IMAGE_METRICS,
                   "Rerun edge_aware after empirical E[Sigma^2] fixes."),
    ExperimentSpec("bridge", "Experiment 2b: Bridge Ablation",
                   "diffusion", "experiments.run_cold_ablation", "bridge",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("pca_basis", "PCA vs Pixel Basis",
                   "diffusion", "experiments.visualize_diffusion", "pca_basis",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("latent", "Latent Summary",
                   "latent", "experiments.run_latent", "latent",
                   "needs_rerun", IMAGE_METRICS,
                   "Audit H1/H2 changed latent training semantics."),
    ExperimentSpec("hurst", "H Ablation",
                   "ablation", "experiments.run_cold_ablation", "H",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("cold_loss", "Generation Loss Ablation",
                   "ablation", "experiments.run_cold_ablation", "loss",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("n_steps", "Sampling Step Ablation",
                   "ablation", "experiments.run_cold_ablation", "steps",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("cfg_scale", "CFG Scale Ablation",
                   "ablation", "experiments.run_cold_ablation", "cfg_scale",
                   "runnable", IMAGE_METRICS),
    ExperimentSpec("alpha", "Experiment alpha: Noise Distribution Ablation",
                   "gaussianity", "experiments.run_gaussianity", "alpha",
                   "needs_rerun", GAUSS_METRICS,
                   "Audit H6 changed alpha activation sampling."),
    ExperimentSpec("beta", "Experiment beta: Bottleneck Width",
                   "gaussianity", "experiments.run_gaussianity", "beta",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("gamma", "Experiment gamma: Layer Trace",
                   "gaussianity", "experiments.run_gaussianity", "gamma",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("delta", "Experiment delta: Latent Perturbation Rigidity",
                   "gaussianity", "experiments.run_gaussianity", "delta",
                   "needs_rerun", ("clean_huber", "perturbed_huber")),
    ExperimentSpec("epsilon", "Experiment epsilon: Loss Function Gaussianization",
                   "ablation", "experiments.run_ablation", "epsilon",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("zeta", "Experiment zeta: Normalization Ablation",
                   "ablation", "experiments.run_ablation", "zeta",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("kappa", "Experiment kappa: Activation Function Ablation",
                   "ablation", "experiments.run_ablation", "kappa",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("mu", "Experiment mu: Skip Connection Ablation",
                   "ablation", "experiments.run_ablation", "mu",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("theta", "Experiment theta: Time-Conditional kappa4",
                   "ablation", "experiments.run_ablation", "theta",
                   "needs_rerun", GAUSS_METRICS),
    ExperimentSpec("omicron", "Experiment omicron: Optimizer Comparison",
                   "optimizer", "experiments.run_optimizer", "omicron",
                   "needs_rescope", OPT_METRICS,
                   "Audit M6 says update_k4/update_w1 are under-sampled."),
    ExperimentSpec("pi", "Experiment pi: Gradient Noise Distribution",
                   "optimizer", "experiments.run_optimizer", "pi",
                   "needs_rescope", OPT_METRICS,
                   "Audit H10 renamed Rosenblatt proxy gradient noise."),
    ExperimentSpec("rho", "Experiment rho: Rosenblatt-SGLD Landscape",
                   "optimizer", "experiments.run_optimizer", "rho",
                   "needs_rescope", OPT_METRICS),
    ExperimentSpec("tau", "Experiment tau: Gradient kappa4 Evolution",
                   "optimizer", "experiments.run_optimizer", "tau",
                   "needs_rescope", ("step", "grad_kappa4")),
    ExperimentSpec("sigma_adam", "Experiment sigma: Adam Whitening Visualisation",
                   "optimizer", "experiments.run_optimizer", "sigma",
                   "not_wired", ("update_std", "wasserstein_to_normal"),
                   "Documented in experiment.md but no runner branch is currently wired."),
)


EXPERIMENTS_BY_KEY = {spec.key: spec for spec in EXPERIMENTS}


def runnable_specs() -> tuple[ExperimentSpec, ...]:
    return tuple(spec for spec in EXPERIMENTS if spec.status != "not_wired")
