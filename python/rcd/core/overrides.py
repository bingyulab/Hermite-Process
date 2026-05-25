"""
Context-managed Config overrides.

Replaces the in-place `cfg.<attr> = value` mutations scattered through the
experiment runners.  Guarantees restoration even when the wrapped block
raises.  Fixes audit item H9 (run_ablation_n_steps mutating cfg.n_steps
without restoring), and prevents similar leaks for cfg.cfg_scale,
cfg.loss_fn, cfg.bridge, etc.
"""

from __future__ import annotations
from contextlib import contextmanager


_SENTINEL = object()


@contextmanager
def override(cfg, **kwargs):
    """Temporarily set cfg.k = v for each (k, v) in kwargs.  All overrides
    are restored on exit (even on exception).

    Example
    -------
        with override(cfg, n_steps=10, cfg_scale=4.0):
            run_evaluation(cfg)
        # cfg.n_steps and cfg.cfg_scale are back to their previous values.
    """
    original = {}
    for k, v in kwargs.items():
        original[k] = getattr(cfg, k, _SENTINEL)
        setattr(cfg, k, v)
    try:
        yield cfg
    finally:
        for k, v in original.items():
            if v is _SENTINEL:
                # Attribute did not exist before; remove it.
                try:
                    delattr(cfg, k)
                except AttributeError:
                    pass
            else:
                setattr(cfg, k, v)