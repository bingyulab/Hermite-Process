from __future__ import annotations
import os
import sys
import yaml
import logging
import datetime
import json
import shutil
from pathlib import Path
from dataclasses import asdict
from typing import Any

from rcd.core.config import Config

class RunContext:
    """
    Deterministically manages the output directory structure for an experiment run.
    Enforces checkpoint, logs, and metric separation.
    """
    def __init__(self, cfg: Config, family: str, run_name: str, base_dir: str | Path | None = None):
        self.cfg = cfg
        self.family = family
        self.run_name = run_name
        self.base_dir = Path(base_dir) if base_dir is not None else Path(cfg.save_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.family / f"{self.run_name}_{timestamp}"
        
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metric_dir = self.run_dir / "metrics"
        self.plot_dir = self.run_dir / "plots"
        self.sample_dir = self.run_dir / "samples"
        self.log_path = self.run_dir / "run.log"
        
        self._logger: logging.Logger | None = None
        self._original_save_dir: Path | None = None

    def __enter__(self) -> RunContext:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        self._original_save_dir = Path(self.cfg.save_dir)
        self.cfg.run_dir = self.run_dir
        self.cfg.ckpt_dir = self.ckpt_dir
        self.cfg.metric_dir = self.metric_dir
        self.cfg.plot_dir = self.plot_dir
        self.cfg.sample_dir = self.sample_dir
        # Backward compatibility: older experiment helpers treat cfg.save_dir
        # as the checkpoint root.
        self.cfg.save_dir = self.ckpt_dir

        # Save config manifest
        cfg_dict = asdict(self.cfg)
        cfg_dict.update({
            "device": str(cfg_dict.get("device", "cpu")),
            "save_dir": str(self.cfg.save_dir),
            "run_dir": str(self.run_dir),
            "checkpoint_dir": str(self.ckpt_dir),
            "metric_dir": str(self.metric_dir),
            "plot_dir": str(self.plot_dir),
            "sample_dir": str(self.sample_dir),
            "family": self.family,
            "run_name": self.run_name,
        })
            
        with open(self.run_dir / "manifest.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
            
        self._setup_logging()
        self.logger.info(f"Started run: {self.run_name} in {self.run_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._organize_artifacts()
        if exc_type is None:
            self.logger.info("Run completed successfully.")
        else:
            self.logger.error(f"Run failed with exception: {exc_val}")
        
        # Clean up logger handlers to avoid duplicate logs in subsequent contexts
        if self._logger:
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)
        if self._original_save_dir is not None:
            self.cfg.save_dir = self._original_save_dir

    def _setup_logging(self) -> None:
        self._logger = logging.getLogger(f"{self.family}.{self.run_name}")
        self._logger.setLevel(logging.INFO)
        # Avoid attaching multiple handlers if logger is reused
        self._logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        
        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("RunContext not entered.")
        return self._logger

    def checkpoint_path(self, *parts: str) -> Path:
        return self.ckpt_dir.joinpath(*parts)

    def metric_path(self, *parts: str) -> Path:
        return self.metric_dir.joinpath(*parts)

    def plot_path(self, *parts: str) -> Path:
        return self.plot_dir.joinpath(*parts)

    def sample_path(self, *parts: str) -> Path:
        return self.sample_dir.joinpath(*parts)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.metric_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def _organize_artifacts(self) -> None:
        """Route legacy files saved under checkpoints/ into artifact folders."""
        if not self.ckpt_dir.exists():
            return
        metric_ext = {".csv", ".json", ".jsonl", ".tex", ".txt"}
        plot_ext = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
        sample_words = ("sample", "samples", "restoration", "grid")

        for path in sorted(self.ckpt_dir.rglob("*")):
            if not path.is_file() or path.suffix == ".pt":
                continue
            rel = path.relative_to(self.ckpt_dir)
            lower = path.name.lower()
            if path.suffix.lower() in metric_ext:
                dest = self.metric_dir / rel
            elif path.suffix.lower() in plot_ext and any(word in lower for word in sample_words):
                dest = self.sample_dir / rel
            elif path.suffix.lower() in plot_ext:
                dest = self.plot_dir / rel
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            shutil.move(str(path), str(dest))
