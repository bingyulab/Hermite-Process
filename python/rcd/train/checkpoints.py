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
        save_dir        — base directory; checkpoint is `save_dir/tag_final.pt`
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

    Load priority:
      1. baseline_path (explicit weight-inheritance).
      2. write path  (cfg.save_dir tree)  — already trained in this run.
      3. read path   (cfg.data_dir tree)  — pretrained checkpoint on Kaggle.
      4. Train from scratch; save to write path.
    """
    cfg = req.cfg
    cfg._setup_environment()

    fwd = _resolve_forward(req)
    ckpt_path  = _resolve_ckpt_path(req)          # write location
    read_path  = _resolve_read_path(req)           # read-only location (may be None)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load_or_train] write path: {ckpt_path}")
    if read_path:
        print(f"[load_or_train] read  path: {read_path}")
    if req.baseline_path is not None:
        print(f"[load_or_train] baseline  : {req.baseline_path}")

    model = req.model_factory().to(cfg.device)

    # 1. Inherit-from-baseline
    if req.baseline_path is not None and req.baseline_path.exists():
        print(f"[load_or_train] loading baseline: {req.baseline_path}")
        _safe_load(req.baseline_path, model, cfg.device)
        save_full(ckpt_path, model)
        model.eval()
        return model, fwd, ()

    # 2. Load from write path (already produced in this run)
    if ckpt_path.exists():
        print(f"[load_or_train] loading from write path: {ckpt_path}")
        _safe_load(ckpt_path, model, cfg.device)
        model.eval()
        return model, fwd, ()

    # 3. Load from read-only data_dir (Kaggle pretrained)
    if read_path is not None:
        print(f"[load_or_train] loading from data_dir: {read_path}")
        _safe_load(read_path, model, cfg.device)
        model.eval()
        return model, fwd, ()

    # 4. Train from scratch; save to write path
    print(f"[load_or_train] training from scratch -> {ckpt_path}")
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
    fwd_builder = req.fwd_builder
    assert fwd_builder is not None
    return fwd_builder(req.cfg)


def _resolve_ckpt_path(req: LoadRequest) -> Path:
    """
    Returns the *write* path for a checkpoint.
    Always under save_dir (writable). A separate read-only probe is done
    in load_or_train before deciding to train.
    """
    if req.ckpt_path is not None:
        return Path(req.ckpt_path)
    base = Path(req.save_dir if req.save_dir is not None else req.cfg.save_dir)
    print(f"[resolve_ckpt_path] write base: {base}")
    if req.subdir:
        base = base / req.subdir
    write_path = base / f"{req.tag}_final.pt"
    print(f"[resolve_ckpt_path] write path: {write_path}")
    return write_path


def _resolve_read_path(req: LoadRequest) -> Path | None:
    """
    Returns the *read* path for a checkpoint inside data_dir, or None
    if data_dir == save_dir (off Kaggle) or the file does not exist there.
    """
    cfg = req.cfg
    data_dir = getattr(cfg, "data_dir", None)
    save_dir  = getattr(cfg, "save_dir",  None)
    if data_dir is None or Path(data_dir) == Path(save_dir or ""):
        return None                    # same tree — no separate read location
    base = Path(data_dir)
    if req.save_dir is not None:
        # req.save_dir is the write base; mirror the subdir structure
        try:
            rel = Path(req.save_dir).relative_to(Path(cfg.save_dir))
        except ValueError:
            rel = Path("")
        base = base / rel
    if req.subdir:
        base = base / req.subdir
    candidate = base / f"{req.tag}_final.pt"
    return candidate if candidate.exists() else None

def _safe_load(path: Path, model: nn.Module, device) -> None:
    try:
        load_full(path, model, device=device, strict=True)
    except RuntimeError as e:
        print(f"[load] STRICT LOAD FAILED for {path}: {e}")
        print("[load] falling back to strict=False — weights may be PARTIAL; "
              "results from this checkpoint are suspect.")
        load_full(path, model, device=device, strict=False)