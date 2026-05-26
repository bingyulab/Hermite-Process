"""
Unified checkpoint I/O and model loading.

Replaces the five+ wrappers that previously each did the same thing:
    load_or_train_unet
    load_or_train_latent
    load_or_train_ablation
    load_or_train_for_optimizer
    _load_or_train_flexible_unet

The single public entry point is `load_or_train`.

`load_or_train` does one thing: given a tag, a model factory, and a
trainer callable, it returns a (model, forward_process, extras_tuple)
triple. Whether the model is loaded from disk or trained is decided
solely by checkpoint existence and the `use_pretrained_baseline` flag.

No experiment-specific logic lives in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


_FORMAT = "rcd-v1"


# =============================================================================
# 1. Raw save / load primitives
# =============================================================================

def save_full(
    path:   str | Path,
    model:  nn.Module,
    *,
    ema     = None,
    opt     = None,
    sch     = None,
    scaler  = None,
    epoch:  int = 0,
    extras: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "format": _FORMAT,
        "epoch":  epoch,
        "model":  model.state_dict(),
    }
    if ema    is not None: state["ema"]    = ema.state_dict()
    if opt    is not None: state["opt"]    = opt.state_dict()
    if sch    is not None: state["sch"]    = sch.state_dict()
    if scaler is not None: state["scaler"] = scaler.state_dict()
    if extras is not None: state["extras"] = extras
    torch.save(state, path)


def load_full(
    path:   str | Path,
    model:  nn.Module,
    *,
    ema    = None,
    opt    = None,
    sch    = None,
    scaler = None,
    device = "cpu",
    strict: bool = True,
) -> tuple[int, dict]:
    state = torch.load(path, map_location=device, weights_only=True)
    if "format" not in state or state["format"] != _FORMAT:
        # Pre-formatted (raw state_dict) checkpoint
        model.load_state_dict(state, strict=strict)
        return 0, {}
    model.load_state_dict(state["model"], strict=strict)
    if ema    is not None and "ema"    in state: ema.load_state_dict(state["ema"])
    if opt    is not None and "opt"    in state: opt.load_state_dict(state["opt"])
    if sch    is not None and "sch"    in state: sch.load_state_dict(state["sch"])
    if scaler is not None and "scaler" in state: scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("extras", {})


def find_latest_epoch(directory: str | Path, tag: str) -> tuple[Path, int] | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    best_ep, best_path = -1, None
    for p in directory.glob(f"{tag}_ep*.pt"):
        try:
            ep = int(p.stem.split("_ep")[-1])
        except ValueError:
            continue
        if ep > best_ep:
            best_ep, best_path = ep, p
    return (best_path, best_ep) if best_path is not None else None


# =============================================================================
# 2. Unified loader request
# =============================================================================

@dataclass
class LoadRequest:
    """
    All parameters needed to either load or train a model. Captures the
    behaviour previously scattered across five distinct wrappers.

    Required:
        tag             — unique identifier; used for filenames
        cfg             — Config object (provides device, save_dir, etc.)
        model_factory   — callable that returns a fresh nn.Module
        train_fn        — callable(model, fwd, cfg, ckpt_path, **train_kwargs)
                          returning either nn.Module or (nn.Module, *extras)

    Forward process:
        fwd             — pre-built RosenblattForward, or
        fwd_builder     — callable(cfg) -> RosenblattForward
                          (only one of fwd / fwd_builder may be supplied)

    Storage:
        save_dir        — base directory; checkpoint is `save_dir/subdir/tag_final.pt`
        subdir          — optional subdirectory under save_dir
        ckpt_path       — explicit override for the final-checkpoint path

    Inheritance:
        baseline_path   — if set and `tag_final.pt` does not yet exist, copy
                          weights from baseline_path before either returning
                          or further training. If the baseline checkpoint is
                          missing, training falls back to scratch instead of
                          failing hard.

    Training arguments:
        train_kwargs    — forwarded verbatim to train_fn
    """
    tag:           str
    cfg:           Any
    model_factory: Callable[[], nn.Module]
    train_fn:      Callable[..., Any]

    fwd:           Optional[Any] = None
    fwd_builder:   Optional[Callable[[Any], Any]] = None

    save_dir:      Optional[Path] = None
    subdir:        str            = ""
    ckpt_path:     Optional[Path] = None

    baseline_path: Optional[Path] = None
    train_kwargs:  dict           = field(default_factory=dict)


# =============================================================================
# 3. Single unified entry point
# =============================================================================

def load_or_train(req: LoadRequest) -> tuple[nn.Module, Any, tuple[Any, ...]]:
    """
    Canonical loader/trainer. Returns (model, fwd, extras_tuple).

    extras_tuple contains whatever train_fn returns after the model
    (typically (history, grad_log)). An empty tuple means the model was
    loaded rather than trained.
    """
    cfg = req.cfg
    cfg._setup_environment()

    fwd = _resolve_forward(req)
    ckpt_path = _resolve_ckpt_path(req)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = req.model_factory().to(cfg.device)

    # 1. Inherit-from-baseline path
    if req.baseline_path is not None and not ckpt_path.exists() and req.baseline_path.exists():
        _safe_load(req.baseline_path, model, cfg.device)
        save_full(ckpt_path, model)
        model.eval()
        return model, fwd, ()

    # 2. Plain load
    if ckpt_path.exists():
        _safe_load(ckpt_path, model, cfg.device)
        model.eval()
        return model, fwd, ()

    # 3. Train from scratch (or resume mid-training inside train_fn)
    result = req.train_fn(model, fwd, cfg, ckpt_path, **req.train_kwargs)
    if isinstance(result, tuple):
        trained, *extras = result
    else:
        trained, extras = result, []
    trained.eval()
    return trained, fwd, tuple(extras)


# =============================================================================
# 4. Internal helpers
# =============================================================================

def _resolve_forward(req: LoadRequest) -> Any:
    if (req.fwd is None) == (req.fwd_builder is None):
        raise ValueError(
            "LoadRequest must supply exactly one of `fwd` or `fwd_builder`."
        )
    if req.fwd is not None:
        return req.fwd
    return req.fwd_builder(req.cfg)


def _resolve_ckpt_path(req: LoadRequest) -> Path:
    if req.ckpt_path is not None:
        return Path(req.ckpt_path)
    base = Path(req.save_dir if req.save_dir is not None else req.cfg.save_dir)
    if req.subdir:
        base = base / req.subdir
    return base / f"{req.tag}_final.pt"


def _safe_load(path: Path, model: nn.Module, device) -> None:
    try:
        load_full(path, model, device=device, strict=True)
    except RuntimeError:
        load_full(path, model, device=device, strict=False)