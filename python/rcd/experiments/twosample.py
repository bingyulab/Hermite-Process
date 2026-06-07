"""
Driver discriminability test (sharpened Stratum-4 question).

Question: does the trained denoiser register which side of the CLT boundary
its driver is on? Operationalised as a two-sample test between the
representations produced under a Rosenblatt driver and a Gaussian driver,
holding the NETWORK FIXED and swapping only the corruption noise on the same
clean images. This isolates the driver (no train+driver confound).

Two estimators (the MMD is the primary test):
  * MMD^2 with an RBF kernel + permutation p-value  — "is there ANY difference"
  * 5-fold linear-discriminator AUC                  — "how linearly separable"

Stages tested:
  corrupted        x_T = x0 + sigma(1) Sigma(x0) eps     (sanity: must separate)
  bottleneck_mean  mid2 spatial-mean   (C)               (pooled view)
  bottleneck_unit  mid2 per-unit       (C*H*W)           (joint view, the question)
  output           x0_hat              (784)

Interpretation:
  AUC -> 0.5 and MMD p > 0.05 at the bottleneck/output  => the network does NOT
  register the driver (the difference present at corruption is gone).
  AUC -> 1.0 / p < 0.05                                  => it DOES register.

Run:
  python -m rcd.experiments.twosample --save_dir output/s42 --seed 42 \
         --noise_types rosenblatt gaussian --H 0.7
(--noise_types lists which TRAINED baselines to use as the fixed network;
 each is probed with both drivers.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rcd.data.config import Config
from rcd.data.datasets import _get_dataset, _NORM_TF
from rcd.train.models import ConditionalUNet
from rcd.train.checkpoints import load_full
from rcd.train.forward import build_forward_process, sigma_multiplicative
from rcd.evaluation.measurement import capture_activations


# ─────────────────────────────────────────────────────────────────────────────
# Two-sample statistics
# ─────────────────────────────────────────────────────────────────────────────

def _pca_standardise(R: np.ndarray, G: np.ndarray, k: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension standardise (so heterogeneous channel scales do not
    dominate) then PCA-reduce to k dims for kernel stability."""
    Z = torch.tensor(np.concatenate([R, G], 0), dtype=torch.float32)
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    k = int(min(k, Zs.shape[1], Zs.shape[0] - 1))
    Zc = Zs - Zs.mean(0)
    _, _, V = torch.pca_lowrank(Zc, q=k, niter=4)
    P = (Zc @ V).numpy()
    return P[: len(R)], P[len(R):]


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    Kxx, Kyy, Kxy = (np.exp(-gamma * _sq_dists(X, X)),
                     np.exp(-gamma * _sq_dists(Y, Y)),
                     np.exp(-gamma * _sq_dists(X, Y)))
    m, n = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean())


def _median_gamma(X: np.ndarray, Y: np.ndarray, cap: int = 500) -> float:
    Z = np.concatenate([X, Y], 0)
    if len(Z) > cap:
        Z = Z[np.random.RandomState(0).choice(len(Z), cap, replace=False)]
    d = _sq_dists(Z, Z)
    med = np.median(d[d > 0])
    return 1.0 / (med + 1e-12)


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 200, seed: int = 0):
    """Returns (MMD^2, p_value). p small => distributions differ."""
    gamma = _median_gamma(X, Y)
    obs = _rbf_mmd2(X, Y, gamma)
    Z = np.concatenate([X, Y], 0)
    m = len(X)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        Zp = Z[rng.permutation(len(Z))]
        null[b] = _rbf_mmd2(Zp[:m], Zp[m:], gamma)
    p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    return obs, float(p)


def cv_auc(X: np.ndarray, Y: np.ndarray) -> float:
    """5-fold linear-discriminator AUC. 0.5 = indistinguishable. NaN if sklearn missing."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        return float("nan")
    Z = np.concatenate([X, Y], 0)
    y = np.concatenate([np.ones(len(X)), np.zeros(len(Y))])
    clf = LogisticRegression(max_iter=2000)
    try:
        return float(cross_val_score(clf, Z, y, cv=5, scoring="roc_auc").mean())
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: fixed network, swap driver
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_driver_pairs(model, fwd_R, fwd_G, loader, device,
                         t_val: float = 1.0, n_max: int = 1000) -> dict:
    """For each clean batch, corrupt with Rosenblatt and with Gaussian (same x0,
    matched variance, matched Sigma) and forward through the SAME model. Capture
    bottleneck and output for each driver. Returns stage -> (feat_R, feat_G)."""
    buf = {k: ([], []) for k in ("corrupted", "bottleneck_mean", "bottleneck_unit", "output")}
    n = 0
    with capture_activations({"mid2": model.mid2}, reduce="none") as st:
        for x0, y in loader:
            if n >= n_max:
                break
            x0, y = x0.to(device), y.to(device)
            B = x0.size(0)
            t = torch.full((B,), t_val, device=device)

            for which, fwd in (("R", fwd_R), ("G", fwd_G)):
                x_T, _, _ = fwd.corrupt(x0, t, y=y)
                c_in = fwd.c_in(t).view(-1, 1, 1, 1)
                st["mid2"].clear()
                pred = model(x_T * c_in, t, y).float()
                acts = st["mid2"].get()           # (B, C, H, W)
                st["mid2"].clear()
                i = 0 if which == "R" else 1
                buf["corrupted"][i].append(x_T.reshape(B, -1).cpu())
                buf["bottleneck_mean"][i].append(
                    (acts.mean(dim=(-2, -1)) if acts.dim() == 4 else acts).cpu())
                buf["bottleneck_unit"][i].append(acts.reshape(B, -1).cpu())
                buf["output"][i].append(pred.reshape(B, -1).cpu())
            n += B

    def cat(L):
        return torch.cat(L, 0)[:n_max].numpy()
    return {stage: (cat(R), cat(G)) for stage, (R, G) in buf.items()}


def run_for_network(cfg, net_tag: str) -> list[dict]:
    """Load the baseline trained under `net_tag`, probe it with both drivers."""
    device = cfg.device
    ckpt = Path(cfg.data_dir) / "checkpoints" / "baseline" / \
        f"{net_tag}_multiplicative_H{cfg.H}_final.pt"
    if not ckpt.exists():
        print(f"[skip] checkpoint not found: {ckpt}")
        return []

    model = ConditionalUNet(num_classes=10, base_ch=cfg.base_ch).to(device)
    load_full(ckpt, model, device=device, strict=False)
    model.eval()

    fwd_R = build_forward_process(sigma_multiplicative(), cfg, noise_type="rosenblatt", H=cfg.H)
    fwd_G = build_forward_process(sigma_multiplicative(), cfg, noise_type="gaussian",   H=cfg.H)

    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    loader  = DataLoader(test_ds, batch_size=min(cfg.batch_size, 128),
                         shuffle=False, num_workers=2)

    n_max = min(getattr(cfg, "n_samples", 1000), 1000)
    pairs = extract_driver_pairs(model, fwd_R, fwd_G, loader, device, n_max=n_max)

    rows = []
    for stage, (R, G) in pairs.items():
        Rp, Gp = _pca_standardise(R, G, k=64)
        mmd2, p = mmd_permutation_test(Rp, Gp, n_perm=200, seed=cfg.seed)
        auc = cv_auc(Rp, Gp)
        rows.append({"net": net_tag, "stage": stage, "n": len(Rp),
                     "mmd2": round(mmd2, 6), "p_value": round(p, 4),
                     "auc": round(auc, 4)})
        print(f"  [{net_tag}] {stage:16s}  MMD^2={mmd2:.5f}  p={p:.4f}  AUC={auc:.4f}")
    return rows


def main():
    cfg = Config.build_from_cli("Driver discriminability (R vs G, fixed net)")
    nets = list(getattr(cfg, "noise_types", ["rosenblatt", "gaussian"]))
    rows = []
    for net_tag in nets:
        print(f"[net = {net_tag}]")
        rows += run_for_network(cfg, net_tag)

    if not rows:
        print("No results (no checkpoints found).")
        return
    out = Path(cfg.save_dir) / "discriminability.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    print("Read: AUC~0.5 and p>0.05 at bottleneck/output => network does not "
          "register the driver. corrupted should separate (sanity).")


if __name__ == "__main__":
    main()