"""
Experiment orchestration base class.

The previous run_*.py files each duplicated the same scaffolding:
seed setup, RunContext wrapping, logging, CSV saves, plot dispatch.
This module supplies one reusable orchestrator. Concrete experiments
implement only their own `run` method.

Design rules:
  * No experiment-specific logic is built into this base class.
  * Configuration variations go through cfg attributes, not subclassing.
  * The runner is responsible for:
        - seed setup (via cfg._setup_environment)
        - data loading (via cfg.dataset)
        - RunContext open/close
        - logging, metric serialisation, manifest dump
        - dispatching to one or more experiment callbacks
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from rcd.data.config import Config
from rcd.data.datasets import _NORM_TF, _get_dataset
from rcd.train.save import RunContext, save_csv, save_latex_table


# Result types accepted by `_persist`: dict-of-dicts, list-of-records, or
# list-of-dicts. The runner normalises them all to row lists for CSV.
RunResult = List[Any]


class ExperimentRunner:
    """
    Base runner. Subclasses register experiment callbacks via `experiments`;
    each callback receives (cfg, ctx, runner) and returns a RunResult.

    Example:
        class GaussianityRunner(ExperimentRunner):
            family = "gaussianity"
            run_name = "gaussianity_sweep"
            experiments = {
                "alpha": run_experiment_alpha,
                "beta":  run_experiment_beta,
                "gamma": run_experiment_gamma,
                "delta": run_experiment_delta,
            }
    """
    family:      str = ""
    run_name:    str = ""
    experiments: Dict[str, Callable[..., RunResult]] = {}
    persist_specs: Dict[str, str] = {}  # mode -> latex spec name

    def __init__(self, cfg: Config) -> None:
        if not self.family or not self.run_name:
            raise ValueError(
                f"{type(self).__name__} must define `family` and `run_name`."
            )
        self.cfg = cfg

    # -------------------------------------------------------------------------
    def run(self, modes: Optional[Iterable[str]] = None) -> Dict[str, RunResult]:
        """
        Run the selected modes. `modes=None` or `modes=["all"]` runs
        every registered experiment.
        """
        active = self._resolve_modes(modes)
        results: Dict[str, RunResult] = {}

        with RunContext(self.cfg, family=self.family, run_name=self.run_name) as ctx:
            ctx.logger.info(f"Dataset: {self.cfg.dataset}  Modes: {','.join(active)}")
            t0 = time.time()

            self._load_data(ctx)

            for mode in active:
                fn = self.experiments[mode]
                ctx.logger.info("=" * 72)
                ctx.logger.info(f"Mode: {mode}")
                rows = fn(self.cfg, ctx, self)
                results[mode] = rows
                self._persist(ctx, mode, rows)

            ctx.logger.info(
                f"Run finished in {time.time() - t0:.1f}s → {ctx.ckpt_dir}"
            )
        return results

    # -------------------------------------------------------------------------
    def _resolve_modes(self, modes: Optional[Iterable[str]]) -> List[str]:
        cli_mode = getattr(self.cfg, "mode", "all")
        if modes is None:
            modes = (cli_mode,)
        modes = list(modes)
        if "all" in modes:
            return list(self.experiments.keys())
        unknown = [m for m in modes if m not in self.experiments]
        if unknown:
            raise KeyError(
                f"Unknown mode(s) {unknown}; valid: {list(self.experiments)}"
            )
        return modes

    # -------------------------------------------------------------------------
    def _load_data(self, ctx: RunContext) -> None:
        """Preload datasets onto runner attributes so callbacks can share them."""
        self.train_ds = _get_dataset(self.cfg.dataset, train=True,  tf=_NORM_TF)
        self.test_ds  = _get_dataset(self.cfg.dataset, train=False, tf=_NORM_TF)
        ctx.logger.info(
            f"Datasets: train={len(self.train_ds)}  test={len(self.test_ds)}"
        )

    # -------------------------------------------------------------------------
    def _persist(self, ctx: RunContext, mode: str, rows: RunResult) -> None:
        if not rows:
            return
        csv_path = ctx.get_path("metric", f"{mode}.csv")
        # Flatten dataclasses if present
        normalised = [
            r.flatten() if hasattr(r, "flatten") else
            (asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r))
            for r in rows
        ]
        save_csv(normalised, csv_path, silent=True)
        ctx.logger.info(f"  CSV → {csv_path.name}")

        spec_name = self.persist_specs.get(mode)
        if spec_name is not None:
            tex_path = ctx.get_path("metric", f"{mode}.tex")
            save_latex_table(normalised, tex_path, spec_name=spec_name)


# =============================================================================
# Convenience: build and run a runner from CLI without boilerplate
# =============================================================================

def run_from_cli(runner_cls: type[ExperimentRunner], description: str = "") -> None:
    cfg = Config.build_from_cli(description or runner_cls.__name__)
    runner = runner_cls(cfg)
    runner.run()