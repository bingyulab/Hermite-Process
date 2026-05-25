"""
Full-state checkpoint save/load.

Each checkpoint stores model, EMA shadow, optimiser, scheduler, AMP scaler,
and current epoch in a single file. Resume is exact: optimiser momentum,
scheduler position, AMP scale, and EMA shadow all carry over.

Backward-compatible loader: `load_full` accepts the legacy bare-state_dict
checkpoints produced by the original training scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


_FORMAT = "rcd-v1"


def save_full(
    path:    str | Path,
    model:   nn.Module,
    *,
    ema      = None,
    opt      = None,
    sch      = None,
    scaler   = None,
    epoch:   int = 0,
    extras:  dict | None = None,
) -> None:
    """Save full training state.

    The caller controls whether `model.state_dict()` is the raw or
    EMA-applied weights. For periodic intermediate checkpoints save raw
    weights (so resume is exact). For the final inference checkpoint save
    EMA-applied weights (so loaders that ignore the `ema` field still get
    the smoothed model).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "format": _FORMAT,
        "epoch":  int(epoch),
        "model":  model.state_dict(),
        "ema":    ema.state_dict()    if ema    is not None else None,
        "opt":    opt.state_dict()    if opt    is not None else None,
        "sch":    sch.state_dict()    if sch    is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extras": extras or {},
    }
    torch.save(state, path)


def load_full(
    path:   str | Path,
    model:  nn.Module,
    *,
    ema     = None,
    opt     = None,
    sch     = None,
    scaler  = None,
    device  = "cpu",
    strict: bool = True,
) -> tuple[int, dict]:
    """Load full training state. Returns (epoch, extras).

    For legacy bare-state_dict files, returns (0, {}). EMA / optimiser state
    cannot be restored from a legacy file; the caller should expect a slight
    EMA discontinuity in that case.
    """
    path = Path(path)
    blob = torch.load(path, map_location=device, weights_only=False)

    if isinstance(blob, dict) and blob.get("format") == _FORMAT:
        info = model.load_state_dict(blob["model"], strict=strict)
        if not strict:
            missing, unexpected = info if isinstance(info, tuple) else ([], [])
            if missing or unexpected:
                print(f"  load_full({path.name}): structural shift found -> "
                      f"missing={len(missing)} unexpected={len(unexpected)}")

        if ema    is not None and blob.get("ema")    is not None:
            # Pass the strict flag here to handle architecture variances safely
            ema.load_state_dict(blob["ema"], device=device, strict=strict)            
        # Optional: Optimization arrays cannot safely map if layer dimensions change
        if opt is not None and blob.get("opt") is not None and strict:
            opt.load_state_dict(blob["opt"])
        if sch is not None and blob.get("sch") is not None and strict:
            sch.load_state_dict(blob["sch"])
        if scaler is not None and blob.get("scaler") is not None:
            scaler.load_state_dict(blob["scaler"])

        return int(blob.get("epoch", 0)), dict(blob.get("extras") or {})

    # Legacy fallback path
    print(f"  load_full({path.name}): legacy checkpoint, optimiser / EMA state skipped.")
    model.load_state_dict(blob, strict=strict)
    return 0, {}


def find_latest_epoch(directory: str | Path, tag: str) -> tuple[Path, int] | None:
    """Return (path, epoch) of the highest tag_epN.pt under `directory`."""
    directory = Path(directory)
    if not directory.exists():
        return None
    pattern = re.compile(rf"^{re.escape(tag)}_ep(\d+)\.pt$")
    best: tuple[Path, int] | None = None
    for p in directory.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if best is None or ep > best[1]:
            best = (p, ep)
    return best