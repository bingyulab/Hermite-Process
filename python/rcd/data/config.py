"""
Centralised configuration. Single source for all defaults, CLI flags,
measurement budgets, and runtime paths.

Key fixes vs the previous version:
 * Boolean field detection now works under `from __future__ import annotations`.
   The previous version compared `f.type is bool`, which always failed because
   field types are stored as strings under PEP 563.
 * N_MEAS and N_SAMPLES_ALPHA are now configurable attributes, not module
   globals scattered across run scripts.
 * `ckpt_dir`, `metric_dir`, `plot_dir`, `sample_dir`, and `run_dir` are
   declared as Optional so RunContext can assign them without dataclass
   warnings.
 * The optimizer / gaussianity / ablation flags coexist in one Config;
   irrelevant flags are simply ignored per experiment.
"""
from __future__ import annotations

import argparse
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Generator, List, Optional, get_args, get_origin

import numpy as np
import torch

OUT_ROOT = Path("./output/")
_SENTINEL = object()


# -----------------------------------------------------------------------------
# Override context manager — temporary attribute substitution
# -----------------------------------------------------------------------------

@contextmanager
def override(cfg: Any, **kwargs: Any) -> Generator[Any, None, None]:
    original: dict[str, Any] = {}
    for k, v in kwargs.items():
        original[k] = getattr(cfg, k, _SENTINEL)
        setattr(cfg, k, v)
    try:
        yield cfg
    finally:
        for k, v in original.items():
            if v is _SENTINEL:
                try:
                    delattr(cfg, k)
                except AttributeError:
                    pass
            else:
                setattr(cfg, k, v)


# -----------------------------------------------------------------------------
# Unified Config
# -----------------------------------------------------------------------------

@dataclass
class Config:
    # Mode and run selection
    mode:           str   = "all"
    seed:           int   = 42

    # Optimisation hyperparameters
    lr:             float = 2e-4
    epochs:         int   = 30
    ae_epochs:      int   = 20
    ae_lr:          float = 1e-3
    batch_size:     int   = 256

    # Diffusion hyperparameters
    cfg_scale:      float = 2.5
    n_steps:        int   = 50
    n_display:      int   = 8
    sigma_max:      float = 16.0
    base_ch:        int   = 128
    M_eig:          int   = 80
    H:              float = 0.7
    T_MIN:          float = 0.01
    bridge:         str   = "stochastic"

    # Dataset
    dataset:        str   = "FashionMNIST"
    num_classes:    int   = 10
    noise_type:     str   = "rosenblatt"

    # Evaluation
    n_fid:          int   = 10000
    n_ssim:         int   = 200
    k_components:   int   = 64
    no_evaluate:    bool  = False
    no_plot:        bool  = False

    # Measurement budgets (previously undefined module globals)
    n_meas:         int   = 2000
    n_samples_alpha:int   = 2000

    # Per-experiment switches
    baseline:        str  = "multiplicative"
    loss_fn:         str  = "huber"
    bottleneck_key:  str  = "mid2"
    family:          str  = "ablation"     # ablation|gaussianity|optimizer|cold_ablation

    # Lists for ablation sweeps
    bf_list:        List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])
    loss_types:     List[str]   = field(default_factory=lambda: ["l1", "l2", "huber"])
    noise_types:    List[str]   = field(default_factory=lambda: ["rosenblatt", "gaussian"])
    norm_types:     List[str]   = field(default_factory=lambda: ["group8", "group1", "batch"])
    act_fns:        List[str]   = field(default_factory=lambda: ["silu", "gelu", "relu"])
    t_values:       List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    sigma_maxs:     List[float] = field(default_factory=lambda: [4.0, 16.0])
    bridge_types:   List[str]   = field(default_factory=lambda: ["stochastic", "hybrid"])
    h_values:       List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    opt_names:      List[str]   = field(default_factory=lambda: ["adamw", "lion", "sgd"])
    sigma_grid:     List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
    noise_kinds:    List[str]   = field(default_factory=lambda: ["clean", "gaussian", "rosenblatt"])
    cfg_scale_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.5, 4.0])
    n_steps_grid:   List[int]   = field(default_factory=lambda: [5, 10, 20, 50, 100])
    quick:          bool  = False
    # Paths managed by RunContext
    save_dir:       Path  = field(default_factory=lambda: OUT_ROOT)
    run_dir:        Optional[Path] = None
    ckpt_dir:       Optional[Path] = None
    metric_dir:     Optional[Path] = None
    plot_dir:       Optional[Path] = None
    sample_dir:     Optional[Path] = None

    # Runtime device (set in _setup_environment)
    device:         torch.device = field(init=False)

    # -------------------------------------------------------------------------
    @classmethod
    def build_from_cli(cls, description: str = "RCD Experiment Config") -> "Config":
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        excluded = {"device", "save_dir", "run_dir", "ckpt_dir",
                    "metric_dir", "plot_dir", "sample_dir"}

        for f in fields(cls):
            if f.name in excluded:
                continue
            cls._add_field_arg(parser, f)

        parser.add_argument("--save_dir", type=str, default=str(OUT_ROOT),
                            help="Base output/checkpoint directory")

        args, _ = parser.parse_known_args()

        if getattr(args, "quick", False):
            args.epochs = 8
            if args.opt_names is None:
                args.opt_names = ["sgd", "adamw", "lion"]

        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in valid}
        kwargs["save_dir"] = Path(args.save_dir)

        cfg = cls(**kwargs)

        if getattr(args, "epochs", None) is not None and "ae_epochs" not in kwargs:
            cfg.ae_epochs = max(10, cfg.epochs // 2)

        cfg._setup_environment()
        return cfg

    # -------------------------------------------------------------------------
    @staticmethod
    def _add_field_arg(parser: argparse.ArgumentParser, f) -> None:
        """
        Register a dataclass field with argparse. Resolves the actual type
        (works under PEP 563 / `from __future__ import annotations`).
        """
        raw_t = f.type
        resolved = _resolve_type(raw_t)

        if resolved is bool:
            action = "store_false" if f.default is True else "store_true"
            parser.add_argument(f"--{f.name}", action=action,
                                help=f"Toggle {f.name}")
            return

        # List[X] or Optional[List[X]]
        list_inner = _list_inner_type(resolved)
        if list_inner is not None:
            parser.add_argument(f"--{f.name}", nargs="+", type=list_inner,
                                default=f.default,
                                help=f"List of {f.name}")
            return

        # Optional[T] → unwrap
        unwrapped = _optional_inner(resolved) or resolved
        if unwrapped not in (str, int, float):
            unwrapped = str
        parser.add_argument(f"--{f.name}", type=unwrapped, default=f.default,
                            help=f"Set {f.name}")

    # -------------------------------------------------------------------------
    def _setup_environment(self) -> None:
        if os.access(self.save_dir.parent, os.W_OK) or not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


# -----------------------------------------------------------------------------
# Type-resolution helpers (handle PEP 563 string annotations)
# -----------------------------------------------------------------------------

def _resolve_type(t: Any) -> Any:
    """Coerce string annotations to actual types where possible."""
    if isinstance(t, str):
        mapping = {
            "bool": bool, "int": int, "float": float, "str": str,
            "Path": Path,
        }
        if t in mapping:
            return mapping[t]
        # Heuristic for List[...] / Optional[List[...]] strings
        if "list" in t.lower():
            return list
        return str
    return t


def _list_inner_type(t: Any) -> Any:
    """Return the inner type of List[X] / Optional[List[X]], else None."""
    if t is list:
        return str
    origin = get_origin(t)
    if origin is list:
        args = get_args(t)
        return args[0] if args else str
    # Optional[List[X]]
    if origin is type(None) or origin is None:
        return None
    args = get_args(t)
    for a in args:
        if get_origin(a) is list:
            inner = get_args(a)
            return inner[0] if inner else str
    return None


def _optional_inner(t: Any) -> Any:
    """Return T for Optional[T], else None."""
    args = get_args(t)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and len(args) == 2:
        return non_none[0]
    return None