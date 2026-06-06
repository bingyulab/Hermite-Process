"""
Family runners and CLI dispatch.

The old `run_ablation.py`, `run_gaussianity.py`, `run_optimizer.py`, and
`run_cold_ablation.py` files are replaced by this single entrypoint.
The user selects a family with `--family <name>` and an experiment list
with `--mode <name>` or `--mode all`.

Examples:
    python -m Main --family ablation    --mode all
    python -m Main --family gaussianity --mode beta
    python -m Main --family optimizer   --mode omicron pi
    python -m Main --family cold_ablation --mode sigma_comparison
"""
from __future__ import annotations
import os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from typing import Iterable, Optional


from rcd.data.config import Config
from rcd.evaluation.metrics import ModelEvaluator, precompute_real_imgs
from rcd.experiments.runner import ExperimentRunner
from rcd.experiments import Experiments as E


# =============================================================================
# Family runners
# =============================================================================

class GaussianityRunner(ExperimentRunner):
    family   = "gaussianity"
    run_name = "gaussianity_sweep"
    experiments = {
        "alpha": E.run_experiment_alpha,
        "beta":  E.run_experiment_beta,
        "gamma": E.run_experiment_gamma,
        "delta": E.run_experiment_delta,
    }
    persist_specs = {"alpha": "cumulants", "beta": "beta"}


class AblationRunner(ExperimentRunner):
    family   = "ablation"
    run_name = "ablation_sweep"
    experiments = {
        "epsilon": E.run_experiment_epsilon,
        "zeta":    E.run_experiment_zeta,
        # "kappa":   E.run_experiment_kappa,
        "mu":      E.run_experiment_mu,
        "theta":   E.run_experiment_theta,
    }
    # ε reuses the omicron column layout from the LaTeX spec table.
    persist_specs = {"epsilon": "omicron"}


class OptimizerRunner(ExperimentRunner):
    family   = "optimizer"
    run_name = "optimizer_sweep"
    experiments = {
        "omicron": E.run_experiment_omicron,
        "pi":      E.run_experiment_pi,
        "rho":     E.run_experiment_rho,
        # "tau":     E.run_experiment_tau,
    }
    persist_specs = {"omicron": "omicron"}


class ColdAblationRunner(ExperimentRunner):
    """
    Cold-ablation experiments share an FID/Accuracy/SSIM/LPIPS evaluator
    and a precomputed batch of real images. Both are attached to the runner
    in `_load_data` so that experiments can reuse them across iterations.
    """
    family   = "cold_ablation"
    run_name = "cold_sweep"
    experiments = {
        "sigma_comparison": E.run_experiment_sigma_comparison,
        # "pca_basis":        E.run_experiment_pca_basis,
        "cold_latent":      E.run_experiment_cold_latent,
        "cold_loss":        E.run_experiment_cold_loss,
        "cold_bridge":      E.run_experiment_cold_bridge,
        "cold_h_sweep":     E.run_experiment_cold_h_sweep,
        "n_steps":          E.run_experiment_n_steps,
        "cfg_scale":        E.run_experiment_cfg_scale,
        "generation":       E.run_experiment_generation,
    }

    def _load_data(self, ctx):
        super()._load_data(ctx)

        # Resolve the FID-extractor weights path.
        # On Kaggle: try the read-only data_dir first, fall back to write path.
        _weights_path = ctx.data_dir / "fashion_resnet.pth"
        if not _weights_path.exists():
            _weights_path = str(ctx.base_dir / "fashion_resnet.pth")

        self.evaluator = ModelEvaluator(self.cfg.device, weights_path=_weights_path)
        self.real_imgs = precompute_real_imgs(self.test_ds, self.cfg.n_fid)
        ctx.logger.info(f"Precomputed {len(self.real_imgs)} real images for FID")


# =============================================================================
# CLI dispatch
# =============================================================================

_RUNNERS: dict[str, type[ExperimentRunner]] = {
    "gaussianity":   GaussianityRunner,
    "ablation":      AblationRunner,
    "optimizer":     OptimizerRunner,
    "cold_ablation": ColdAblationRunner,
}


def main(modes: Optional[Iterable[str]] = None) -> None:
    cfg = Config.build_from_cli("RCD Experiments")
    family = getattr(cfg, "family", None) or "ablation"
    if family not in _RUNNERS:
        raise SystemExit(
            f"Unknown family: {family!r}. Valid: {sorted(_RUNNERS)}"
        )
    runner = _RUNNERS[family](cfg)
    runner.run(modes=modes)


if __name__ == "__main__":
    main()