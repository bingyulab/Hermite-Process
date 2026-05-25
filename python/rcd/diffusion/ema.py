"""
EMA with state_dict / load_state_dict.

Replaces the EMA class in Rosenblatt_cold_diffusion_unified.py. The old class
had no serialisation hooks, so resumed training rebuilt the EMA shadow from
whatever was in `model.state_dict()`. Because checkpoints were saved with the
EMA shadow already applied to the model, the resumed shadow inherited the
EMA-applied values, not the raw weights. This module fixes that by storing
the shadow explicitly and providing state_dict / load_state_dict for use with
the full-state checkpoint format in rcd.checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model  = model
        self.decay  = decay
        self.step   = 0
        self.shadow: dict[str, torch.Tensor] = {
            n: p.data.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def to(self, device: torch.device | str) -> EMA:
        """Move all internal shadow and backup states to the target device."""
        with torch.no_grad():
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
            self.backup = {k: v.to(device) for k, v in self.backup.items()}
        return self
    
    # ── core ────────────────────────────────────────────────────────────────

    def _effective_decay(self) -> float:
        # EDM-style warm-up: linear ramp from 1/10 toward `decay`.
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    def update(self) -> None:
        d = self._effective_decay()
        self.step += 1
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    # Defensive alignment: ensure shadow matches parameter device dynamically
                    if self.shadow[n].device != p.device:
                        self.shadow[n] = self.shadow[n].to(p.device)
                    self.shadow[n].lerp_(p.data, 1.0 - d)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    # ── serialisation ──────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        # Move tensors to CPU for portable storage.
        return {
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
            "step":   int(self.step),
            "decay":  float(self.decay),
        }

    def load_state_dict(
        self, 
        state: dict, 
        device: torch.device | str | None = None, 
        strict: bool = True
    ) -> None:
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = "cpu"                
                
        self.step = int(state.get("step", 0))
        self.decay = float(state.get("decay", self.decay))
        self.backup = {}
        
        loaded_shadow = state["shadow"]
        if strict:
            self.shadow = {k: v.to(device) for k, v in loaded_shadow.items()}
        else:
            # Gracefully adapt to structural differences
            self.shadow = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if n in loaded_shadow:
                        # Parameter exists -> Load checkpoint state
                        self.shadow[n] = loaded_shadow[n].to(device)
                    else:
                        # Parameter is new -> Initialize shadow with current online weights
                        print(f"  [EMA] Initializing missing shadow parameter: {n}")
                        self.shadow[n] = p.data.detach().clone().to(device)