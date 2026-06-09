"""
Dataset-level excess kurtosis (kappa4) of FashionMNIST, computed two ways
and reported globally and per class.

Motivation
----------
"kappa4 of the input" is ambiguous for an image dataset because the data
tensor has both a sample axis and a spatial axis. Two reduction orders give
two different objects:

  Way 1 (batch-then-HW):  for each pixel location compute kappa4 across the
        sample axis, then average the per-pixel values over H*W.
        Question: "is pixel (h,w) Gaussian across the dataset, on average?"

  Way 2 (HW-then-batch):  for each image compute kappa4 across its H*W pixels,
        then average over the sample axis.
        Question: "is the spatial intensity histogram of a single image
        Gaussian, on average?"

These are not the same quantity and need not agree in sign.

Both estimators standardise per the axis they reduce over and use the same
variance clamp (1e-8) as rcd.evaluation.gaussianity.compute_marginal_cumulants,
so the numbers are directly comparable to the experiment pipeline.

Usage
-----
    python dataset_kurtosis.py --dataset FashionMNIST --split train \
           --normalize --out dataset_kurtosis.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torchvision import datasets, transforms

VAR_CLAMP = 1e-8


def _load(dataset: str, split: str, normalize: bool, root: str):
    tfs = [transforms.ToTensor()]                     # -> [0,1], shape (1,28,28)
    if normalize:
        tfs.append(transforms.Normalize((0.5,), (0.5,)))   # -> [-1,1]
    tf = transforms.Compose(tfs)
    cls = datasets.FashionMNIST if "fashion" in dataset.lower() else datasets.MNIST
    ds = cls(root, train=(split == "train"), download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=2)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)            # (B,1,28,28)
        ys.append(y)
    return torch.cat(xs, 0).numpy(), torch.cat(ys, 0).numpy()


def _k4_over_axis(X: np.ndarray, axis: int) -> np.ndarray:
    """Excess kurtosis along `axis`, standardising over the same axis."""
    mu = X.mean(axis, keepdims=True)
    Xc = X - mu
    var = np.clip((Xc ** 2).mean(axis, keepdims=True), VAR_CLAMP, None)
    Xs = Xc / np.sqrt(var)
    return (Xs ** 4).mean(axis) - 3.0


def kappa4_two_ways(X: np.ndarray) -> dict:
    """
    X : (N, C, H, W).  Returns global and per-channel kappa4 for both orders.
    For FashionMNIST C == 1, so per-channel == global.
    """
    N, C, H, W = X.shape

    # Way 1: kurtosis over batch (axis 0) per pixel, then average over H*W.
    k4_per_pixel = _k4_over_axis(X, axis=0)                 # (C, H, W)
    way1_channel = k4_per_pixel.reshape(C, -1).mean(1)      # (C,)
    way1_global = float(k4_per_pixel.mean())

    # extra diagnostics for way 1: pixels with near-zero variance inflate the
    # mean (the same mechanism that makes the "input kappa4 = 48" value large).
    pix_var = X.reshape(N, C, -1).var(0).reshape(-1)        # (C*H*W,)
    n_degen = int((pix_var < 1e-3).sum())
    k4_flat = k4_per_pixel.reshape(-1)
    keep = pix_var >= 1e-3
    way1_global_nondegen = float(k4_flat[keep].mean()) if keep.any() else float("nan")

    # Way 2: kurtosis over H*W (per sample, per channel), then average over batch.
    Xs = X.reshape(N, C, H * W)
    k4_per_image = _k4_over_axis(Xs, axis=2)                # (N, C)
    way2_channel = k4_per_image.mean(0)                     # (C,)
    way2_global = float(k4_per_image.mean())

    return {
        "N": N, "C": C,
        "way1_batch_then_hw_global": way1_global,
        "way1_global_drop_degenerate": way1_global_nondegen,
        "way1_n_degenerate_pixels": n_degen,
        "way1_median_per_pixel": float(np.median(k4_flat)),
        "way1_max_per_pixel": float(k4_flat.max()),
        "way2_hw_then_batch_global": way2_global,
        "way2_median_per_image": float(np.median(k4_per_image)),
        "_way1_channel": way1_channel,
        "_way2_channel": way2_channel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FashionMNIST")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--root", default="./data")
    ap.add_argument("--normalize", action="store_true",
                    help="apply Normalize((0.5,),(0.5,)) to match the training pipeline")
    ap.add_argument("--out", default="dataset_kurtosis.csv")
    args = ap.parse_args()

    X, Y = _load(args.dataset, args.split, args.normalize, args.root)
    print(f"loaded {X.shape[0]} images, shape {X.shape[1:]}, "
          f"normalize={args.normalize}, range=[{X.min():.2f},{X.max():.2f}]")

    rows = []

    g = kappa4_two_ways(X)
    print("\n=== GLOBAL ===")
    print(f"  way1 (batch-then-HW)            : {g['way1_batch_then_hw_global']:8.3f}")
    print(f"  way1 dropping degenerate pixels : {g['way1_global_drop_degenerate']:8.3f} "
          f"({g['way1_n_degenerate_pixels']} pixels with var<1e-3, "
          f"max per-pixel kappa4={g['way1_max_per_pixel']:.1f})")
    print(f"  way2 (HW-then-batch)            : {g['way2_hw_then_batch_global']:8.3f}")
    rows.append({"scope": "global", "label": "all", **{k: v for k, v in g.items()
                                                        if not k.startswith("_")}})

    print("\n=== PER CLASS ===")
    for c in sorted(np.unique(Y)):
        gc = kappa4_two_ways(X[Y == c])
        print(f"  class {c}: way1={gc['way1_batch_then_hw_global']:8.3f}  "
              f"way1_nondegen={gc['way1_global_drop_degenerate']:8.3f}  "
              f"way2={gc['way2_hw_then_batch_global']:8.3f}")
        rows.append({"scope": "class", "label": int(c),
                     **{k: v for k, v in gc.items() if not k.startswith("_")}})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()