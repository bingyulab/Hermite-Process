"""
Dataset utilities only. Result dataclasses and label dictionaries live in
rcd.experiments.registry — do not re-export them from here.
"""
from __future__ import annotations

from typing import Optional

from torchvision import datasets, transforms


_FASHION = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


def class_name(ds_name: str, idx: int) -> str:
    return _FASHION[idx] if "fashion" in ds_name.lower() else f"Digit {idx}"


_NORM_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _get_dataset(
    name:  str,
    train: bool = True,
    tf:    Optional[transforms.Compose] = None,
    root:  str = "./data",
):
    tf = tf or _NORM_TF
    ds_class = datasets.FashionMNIST if "fashion" in name.lower() else datasets.MNIST
    return ds_class(root, train=train, download=True, transform=tf)