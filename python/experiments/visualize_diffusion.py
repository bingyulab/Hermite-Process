import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from rcd.core.config import Config
from rcd.data import _NORM_TF, _get_dataset, class_name
from rcd.diffusion import (
    RosenblattForward,
    SigmaFn,
    compute_condition_pixel_variance,
    compute_global_pixel_variance,
    sigma_anisotropic,
    sigma_edge_aware,
    sigma_multiplicative,
    sigma_pca_whitened_conditional,
    sigma_pca_whitened_global,
)
from rcd.evaluation import evaluate_model
from rcd.tracker.run_context import RunContext
from experiments.common import train

@torch.no_grad()
def _restoration_grid(model: nn.Module, forward: RosenblattForward,
                       cfg: Config, save_dir: str,
                       tag: str = "", bridge: str = "stochastic") -> None:

    model.eval()
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    found   = {}
    for i in range(len(test_ds)):
        lb = test_ds[i][1]
        if lb not in found:
            found[lb] = i
        if len(found) == 10:
            break

    x0   = torch.stack([test_ds[found[c]][0] for c in range(10)]).to(cfg.device)
    lbl  = torch.arange(10, device=cfg.device)
    null = torch.full_like(lbl, 10)
    xc, _, _ = forward.corrupt(x0, torch.ones(10, device=cfg.device), y=lbl)
    # take random input

    sched  = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    # Which steps to save for display: 0 (corrupted) + n_display equally spaced + final
    save_at = set([0] + [int(round(cfg.n_steps * i / (cfg.n_display - 1)))
                         for i in range(cfg.n_display)])
    save_at.add(cfg.n_steps - 1)   # always include final step
    x_cur  = xc.clone()
    hist   = {}

    for k in range(cfg.n_steps):
        tc   = sched[k].expand(10)
        tn   = sched[k + 1].expand(10)
        c_in = forward.c_in(tc).view(-1, 1, 1, 1)
        if cfg.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x0c = model(x_cur * c_in, tc, lbl).float()
                x0u = model(x_cur * c_in, tc, null).float()
        else:
            x0c = model(x_cur * c_in, tc, lbl)
            x0u = model(x_cur * c_in, tc, null)
        x0h = (x0u + cfg.cfg_scale * (x0c - x0u)).clamp(-1., 1.)
        if k + 1 in save_at:
            hist[k + 1] = x0h.cpu()
        if k < cfg.n_steps - 1:
            if bridge == "stochastic":
                x_cur = forward.recorrupt_stochastic(x0h, tn, y=lbl)
            elif bridge == "hybrid":
                x_cur = forward.recorrupt_hybrid(x_cur, x0h, tc, tn, y=lbl)
            elif bridge == "deterministic":  # deterministic is kept only for explicit ablations.
                x_cur = forward.recorrupt_deterministic(x_cur, x0h, tc, tn)
            else:
                raise ValueError(f"Unknown bridge: {bridge!r}")
        else:
            x_cur = x0h
        
    snap_keys  = sorted(hist.keys())
    snaps      = [hist[k] for k in snap_keys]
    n_cols     = 2 + len(snaps)   # original + corrupted + snapshots

    fig, axes = plt.subplots(10, n_cols, figsize=(2. * n_cols, 14))
    for i in range(10):
        axes[i, 0].imshow((x0[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow((xc[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
        for col, snap in enumerate(snaps):
            axes[i, col + 2].imshow((snap[i, 0] + 1) / 2,
                                    cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(class_name(cfg.dataset, i),
                              fontsize=8, rotation=0, labelpad=40, va="center")
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 1].set_title("Corrupted\nt=1", fontsize=7)
    for col, k in enumerate(snap_keys):
        t_val = 1. - k / cfg.n_steps
        axes[0, col + 2].set_title(f"t={t_val:.2f}\nstep {k}", fontsize=7)
    plt.suptitle(f"Restoration ({cfg.n_steps} steps) — {tag}\n{forward.label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_restoration.png", dpi=120)
    plt.close()

def _sigma_pattern_plot(sigma_fn: SigmaFn, save_dir: str) -> None:
    ds  = _get_dataset("FashionMNIST", train=False, tf=_NORM_TF)
    x0  = ds[0][0].unsqueeze(0)
    with torch.no_grad():
        y_dummy = torch.zeros(1, dtype=torch.long)   # class 0 as representative
        S = (sigma_fn(x0, y_dummy) if getattr(sigma_fn, "needs_label", False) else sigma_fn(x0))[0, 0].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow((x0[0, 0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input image", fontsize=10);  axes[0].axis("off")
    im = axes[1].imshow(S, cmap="hot")
    axes[1].set_title(f"$\\Sigma(\\mathbf{{x}}_0)$\n{sigma_fn.label}", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    plt.tight_layout()
    fp = f"{save_dir}/{sigma_fn.__name__}_pattern.png"
    plt.savefig(fp, dpi=130);  plt.close();plt.show()
    print(f"  Saved {fp}")

def plot_all_sigma_patterns(sigma_fns: list, save_path: str,
                            dataset_name: str = "FashionMNIST",
                            example_classes: list[int] = None) -> None:
    """
    Plot per-pixel Sigma(x0) patterns for all sigma functions in a single figure.

    Layout: rows = classes (if example_classes given, else 1 row with fixed image)
            cols = original + one column per sigma_fn
    """
    if example_classes is None:
        example_classes = [0]   # single row: T-shirt/top

    ds   = _get_dataset(dataset_name, train=False, tf=_NORM_TF)
    # Pick one image per requested class
    found = {}
    for i in range(len(ds)):
        lb = ds[i][1]
        if lb in example_classes and lb not in found:
            found[lb] = ds[i][0].unsqueeze(0)   # (1,1,28,28)
        if len(found) == len(example_classes): break

    n_rows = len(example_classes)
    n_cols = 1 + len(sigma_fns)   # original + one per sigma

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # make 2D

    for row, cls in enumerate(example_classes):
        x0 = found[cls]   # (1,1,28,28)

        # Column 0: original image
        axes[row, 0].imshow((x0[0, 0].numpy() + 1) / 2,
                            cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(class_name(dataset_name, cls),
                                fontsize=9, rotation=0, labelpad=42, va="center")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].axis("off")

        # Columns 1+: sigma patterns
        with torch.no_grad():
            for col, sfn in enumerate(sigma_fns):
                y_dummy = torch.tensor([cls], dtype=torch.long)
                S = (sfn(x0, y_dummy)
                     if getattr(sfn, "needs_label", False)
                     else sfn(x0))
                S_np = S[0, 0].numpy()

                vmax = S_np.max(); vmin = S_np.min()
                im = axes[row, col + 1].imshow(S_np, cmap="hot",
                                               vmin=vmin, vmax=vmax)
                if row == 0:
                    # Show label + E[Sigma^2] above first row
                    eg2 = getattr(sfn, "eg2", float((S**2).mean()))
                    axes[row, col + 1].set_title(
                        f"{sfn.label}\n$E[\\Sigma^2]={eg2:.2f}$",
                        fontsize=8)
                axes[row, col + 1].axis("off")
                plt.colorbar(im, ax=axes[row, col + 1],
                             fraction=0.046, pad=0.04)

    plt.suptitle(r"Per-pixel noise coefficient $\Sigma(\mathbf{x}_0)$",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")

def run_sigma_comparison(cfg: Config) -> list[dict]:
    """
    Core experiment: compare three non-trivial choices of Sigma.

    1. Multiplicative:  Sigma(x0) = diag(g(x0))   
    2. Anisotropic H:   Sigma = A_h_emphasis      
    3. PCA-whitened:    Sigma = C^{-1/2}          
    4. Edge-aware:      Sigma(x0) = diag(|Sobel|) 

    Sigma=I (additive) is NOT included as a comparison target —
    it is the trivial baseline with no geometric structure.
    """

    # Reference real images for FID
    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = torch.stack([test_ds[i][0] for i in range(cfg.n_fid)])
    real_imgs = (real_imgs + 1.0) / 2.0   # [0, 1]

    # Pre-compute PCA whitening from training data
    print("Computing pixel variance for PCA-whitened Sigma...")
    class_vars = compute_condition_pixel_variance(cfg.dataset)   # (1,28,28)
    global_var = compute_global_pixel_variance(cfg.dataset)       # (1,28,28)

    sigma_variants = [
        sigma_multiplicative(),
        sigma_anisotropic(mode="h_emphasis"),
        sigma_anisotropic(mode="v_emphasis"),
        sigma_pca_whitened_conditional(class_vars),
        sigma_pca_whitened_global(global_var),
        sigma_edge_aware(sobel_strength=2.0),
    ]

    results = []

    for sfn in sigma_variants:
        run_dir = f"{cfg.save_dir}/{sfn.__name__}"
        print(f"\n{'='*60}\nExp sigma_comparison: noise={cfg.noise_type}  sigma={sfn.__name__} bridge={cfg.bridge}")
        model, forward = train(sfn, cfg, save_dir=run_dir)

        if not cfg.no_evaluate:
            metrics = evaluate_model(model, forward, real_imgs, test_ds,
                                    cfg, bridge=cfg.bridge)
            results.append({"sigma":     sfn.__name__,
                            "label":     sfn.label,
                            "eg2":       round(forward._eg2, 4),
                            "noise":     cfg.noise_type,
                            "bridge":    cfg.bridge,
                            "FID":       round(metrics['FID'], 2),
                            "fFID":      round(metrics.get('fFID', 0), 2),
                            "Accuracy":  round(metrics['Accuracy'], 2),
                            "SSIM":      round(metrics['SSIM'], 4),
                            "LPIPS":     round(metrics.get('LPIPS', 0), 4),
                            "Eval Time": round(metrics['eval_time_s'], 1)})
            print(f"  {sfn.__name__:25s}  E[Σ²]={forward._eg2:.3f}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")

        if not cfg.no_plot:
            _restoration_grid(model, forward, cfg, run_dir,
                            tag=f"{cfg.noise_type}_{cfg.bridge}_{sfn.__name__}", bridge=cfg.bridge)
            # _sigma_pattern_plot(sfn, run_dir)

    if not cfg.no_plot:
        plot_all_sigma_patterns(
            sigma_variants,
            save_path=f"{cfg.save_dir}/all_sigma_patterns.png",
            example_classes=[0, 1, 7, 9]   # T-shirt, Trouser, Sneaker, Ankle boot
        )
    if not cfg.no_evaluate:
        print("\nSigma comparison FID Summary:")
        for r in results:
            print(f"  noise={r['noise']:10s} bridge={r['bridge']:10s}  sigma={r['sigma']:20s}  FID={r['FID']}   fFID={r['fFID']}  Acc={r['Accuracy']}%  SSIM={r['SSIM']}  LPIPS={r['LPIPS']} Eval Time: {r['Eval Time']:.1f}s")
    return results

def run_exp_pca_basis(
    cfg:          Config,
    bridge:       str   = "stochastic",
) -> dict:
    """
    PCA basis experiment: apply Rosenblatt noise in the top-k PCA directions
    of the pixel covariance, then reconstruct to pixel space.
    
    Answers: is PCA-rotated pixel space a better basis for Rosenblatt noise?
    Mathematically: X_t = x0 + sigma(t) * V_k * diag(lambda_k^{-1/2}) * V_k^T * eps
    where V_k are the top-k eigenvectors of Cov(x0).
    This applies more noise in low-variance PCA directions (rare features)
    and less in high-variance directions (common features).
    """
    # ── 1. Compute PCA from training set ──────────────────────────────────
    print("Computing PCA basis from training set...")
    ds_tr  = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    n_pca  = min(5000, len(ds_tr))
    loader = DataLoader(ds_tr, batch_size=512, shuffle=True, num_workers=2)
    imgs   = []
    for x, _ in loader:
        imgs.append(x.view(x.size(0), -1))     # flatten: (B, 784)
        if sum(i.size(0) for i in imgs) >= n_pca: break
    X = torch.cat(imgs, 0)[:n_pca]             # (n_pca, 784)
    mu = X.mean(0)                              # (784,)
    X_c = X - mu                               # centred
    # SVD on centred data (more stable than eigendecomposition)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    V_k  = Vt[:cfg.k_components].T                 # (784, k)  top-k eigenvectors
    # Variance explained in each direction
    proj = X_c @ V_k                           # (n_pca, k)
    lam_k = proj.var(0).clamp(min=1e-6)        # (k,)  variance per direction
    print(f"  PCA: top-{cfg.k_components} components explain "
          f"{(lam_k.sum() / X_c.var(1).sum() * 100 * cfg.k_components / 784):.1f}% pixel variance")

    # Whitening scale in PCA space: sigma_pca_i = 1/sqrt(lam_i), normalised
    scale_pca = 1. / lam_k.sqrt()             # (k,)
    scale_pca = scale_pca / scale_pca.mean()

    # Move to device
    V_k       = V_k.to(cfg.device)
    mu_d      = mu.to(cfg.device)
    scale_d   = scale_pca.to(cfg.device)

    # ── 2. Build a Sigma-fn that operates in PCA space ────────────────────
    # We implement a custom forward:
    #   corrupt: z_t = x0 + V_k * diag(sigma(t)*scale_pca) * eps_k
    #   where eps_k ~ Z_D^{k}  in the k-dim PCA subspace

    # For FID comparison: also run standard pixel-space
    results = {}
    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real    = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    for nt in ("rosenblatt", "gaussian"):
        results[nt] = {}
        for basis in ("pca", "pixel"):
            print(f"\n{'='*60}\nPCA basis exp: noise={nt}  basis={basis}")
            if basis == "pixel":
                if cfg.baseline == "multiplicative":
                    sfn = sigma_multiplicative()    # standard pixel-space
                    rd  = f"{cfg.save_dir}/multiplicative"
                elif cfg.baseline == "pca_whitened_global":
                    global_var = compute_global_pixel_variance(cfg.dataset) 
                    sfn = sigma_pca_whitened_global(global_var)  # global variance whitening
                    rd  = f"{cfg.save_dir}/pca_whitened_global"
                else:
                    raise ValueError(f"Unknown baseline: {cfg.baseline}")
            else:
                # Anisotropic in PCA basis: back-project scale to pixel space
                # Effective per-pixel scale: A_i = sum_j V_{ij}^2 * scale_j
                A_pixel = (V_k.cpu() ** 2) @ scale_pca.unsqueeze(1)   # (784, 1)
                A_img   = A_pixel.view(1, 28, 28) / A_pixel.mean()
                sfn     = sigma_anisotropic(mode="h_emphasis")         # placeholder
                # Override with PCA-back-projected scale
                _A = A_img.clone()
                def _pca_fn(x0, _A=_A):
                    return _A.to(x0.device).expand_as(x0)
                _pca_fn.__name__    = "pca_basis"
                _pca_fn.label       = rf"PCA basis ($k={cfg.k_components}$)"
                _pca_fn.eg2         = float((_A ** 2).mean())
                _pca_fn.needs_label = False
                sfn = _pca_fn
                rd = f"{cfg.save_dir}/pca_basis"
    
            Path(rd).mkdir(parents=True, exist_ok=True)
            model, fwd = train(sfn, cfg, noise_type=nt, H=cfg.H, save_dir=rd)
            model.eval()

            if not cfg.no_evaluate:
                metrics = evaluate_model(model, fwd, real, test_ds, cfg, bridge=bridge)
                results[nt][basis] = metrics
                print(f"  noise={nt:10s} basis={basis:5s}  FID={metrics['FID']}   fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
            
            if not cfg.no_plot:
                _restoration_grid(model, fwd, cfg, rd, tag=f"{basis}_{nt}", bridge=bridge)
    
    if not cfg.no_evaluate:
        print(f"\nPCA basis summary:")
        for nt in results:
            for basis, metrics in results[nt].items():
                print(f"  noise={nt:10s} basis={basis:5s}  FID={metrics['FID']}   fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
    return results
