from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch.nn as nn

from rcd.core.config import Config
from rcd.data import _NORM_TF, _get_dataset
from rcd.diffusion import RosenblattForward, build_forward, train_diffusion
from rcd.models import ConditionalUNet


def build_forward_process(
    sigma_fn,
    cfg: Config,
    *,
    noise_type: str | None = None,
    H: float | None = None,
    estimate_eg2: bool = True,
) -> RosenblattForward:
    """Create a configured RosenblattForward for the experiment runners."""
    return build_forward(
        RosenblattForward,
        sigma_fn,
        noise_type=noise_type or cfg.noise_type,
        H=cfg.H if H is None else H,
        M_eig=cfg.M_eig,
        sigma_max=cfg.sigma_max,
        device=cfg.device,
        dataset_name=cfg.dataset,
        get_dataset_fn=_get_dataset,
        transform=_NORM_TF,
        estimate=estimate_eg2,
    )


def train(
    sigma_fn,
    cfg: Config,
    *,
    noise_type: str | None = None,
    H: float | None = None,
    save_dir: str | Path | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    tag: str | None = None,
    loss_type: str | None = None,
) -> tuple[nn.Module, RosenblattForward]:
    """Image-space training entrypoint shared by the organized experiments."""
    noise = noise_type or cfg.noise_type
    hurst = cfg.H if H is None else H
    run_dir = Path(save_dir) if save_dir is not None else Path(cfg.save_dir) / sigma_fn.__name__
    run_dir.mkdir(parents=True, exist_ok=True)

    tag = tag or f"{noise}_{sigma_fn.__name__}_H{hurst}"
    ckpt_path = run_dir / f"{tag}_final.pt"

    fwd = build_forward_process(sigma_fn, cfg, noise_type=noise, H=hurst)
    model = (model_factory or (lambda: ConditionalUNet(num_classes=cfg.num_classes, base_ch=cfg.base_ch)))()
    model = model.to(cfg.device)

    train_ds = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    val_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    model, _history = train_diffusion(
        model,
        fwd,
        train_ds,
        val_ds,
        cfg,
        ckpt_path,
        tag=tag,
        loss_type=loss_type or cfg.loss_fn,
        apply_c_in=True,
    )
    return model, fwd
