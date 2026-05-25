import argparse
import time
from pathlib import Path
import os
import sys
# Add the parent directory (python/) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from rcd.core.config import Config
from rcd.tracker.run_context import RunContext
from rcd.data import _get_dataset, _NORM_TF, class_name
from rcd.diffusion import (
    EMA,
    RosenblattForward,
    generate_latent,
    sample_noise,
    sigma_additive,
)
from rcd.evaluation import (
    FashionFIDWrapper,
    compute_conditional_accuracy,
    compute_fid,
    get_fashion_extractor,
)
from rcd.models import ConvAutoencoder, LatentMLPDenoiser


def run_exp_latent(cfg: Config, ctx: RunContext) -> list[dict]:
    """Gaussian vs Rosenblatt cold diffusion in 64-D latent space."""
    ae_path = Path(ctx.ckpt_dir) / "ae_final.pt"
    ae = ConvAutoencoder().to(cfg.device)
    
    if ae_path.exists():
        ae.load_state_dict(torch.load(ae_path, map_location=cfg.device, weights_only=True))
        ctx.logger.info(f"Loaded AE from {ae_path}")
    else:
        ctx.logger.info("Training Autoencoder")
        ae = train_autoencoder(cfg)
        torch.save(ae.state_dict(), ae_path)
    ae.eval()

    test_ds = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_orig = torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]).to(cfg.device)
    real_re = []
    
    with torch.no_grad():
        for i in range(0, cfg.n_fid, 256):
            r, _ = ae(real_orig[i:i+256])
            real_re.append(r.cpu())
    real_imgs = ((torch.cat(real_re, 0) + 1.) / 2.).clamp(0., 1.)

    results = []
    for nt in ("gaussian", "rosenblatt"):
        for sm in (4., 16.):
            tag = f"{nt}_s{sm}"
            ctx.logger.info(f"\nExp Latent: {tag}")
            
            # Using the old logic but injecting context paths
            # NOTE: Bug H1 fixed in the training module previously
            model, forward = train_latent(ae, cfg, sigma_max=sm, noise_type=nt)
            model.eval()            

            if not cfg.no_evaluate:
                metrics = evaluate_latent_model(model, ae, forward, real_imgs, test_ds, cfg)
                results.append({"noise": nt, "sigma_max": sm, **metrics})
                ctx.logger.info(f"  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}")
            
            _save_latent_samples(model, ae, forward, cfg, tag=tag, save_dir=str(ctx.sample_dir))

    if not cfg.no_evaluate:
        ctx.logger.info("\nLatent summary:")
        for r in results:
            ctx.logger.info(f"  {r}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Latent space experiments")
    parser.add_argument("--dataset",   default="FashionMNIST", choices=["FashionMNIST", "MNIST"])
    parser.add_argument("--epochs",    type=int, default=30)
    args = parser.parse_args()
    
    cfg = Config()
    cfg.dataset = args.dataset
    cfg.epochs = args.epochs
    
    with RunContext(cfg, family="latent", run_name="latent_sweep") as ctx:
        run_exp_latent(cfg, ctx)

if __name__ == "__main__":
    main()
@torch.no_grad()
def evaluate_latent_model(
    model:     LatentMLPDenoiser,
    ae:        ConvAutoencoder,
    forward:   RosenblattForward,
    real_imgs: torch.Tensor,        # [0,1], AE-reconstructed, shape (N,1,28,28)
    test_ds,                        # raw dataset, returns [-1,1]
    cfg:       Config,
) -> dict:
    """
    Unified evaluation for the latent cold diffusion model.
    Mirrors evaluate_model but uses generate_latent and latent-space SSIM.

    real_imgs should be AE-reconstructed real images (not raw pixels),
    so FID measures generation quality relative to what the AE can produce,
    not the gap introduced by AE reconstruction quality.
    """

    model.eval(); ae.eval()
    t0 = time.time()

    # ── 1. Generate n_fid decoded images ──────────────────────────────────
    lbl   = torch.randint(0, 10, (cfg.n_fid,), device=cfg.device)
    fakes = []
    for i in range(0, cfg.n_fid, 200):
        fakes.append(
            generate_latent(model, ae, forward, lbl[i:i+200], cfg).cpu())
    fakes_t = torch.cat(fakes, 0)          # (n_fid, 1, 28, 28), [0,1]

    # ── 2. FID & Fashion-FID ──────────────────────────────────────────────
    fid = compute_fid(real_imgs, fakes_t, device=cfg.device)

    extractor = get_fashion_extractor(cfg.device)
    wrapper = FashionFIDWrapper(extractor).to(cfg.device)
    f_fid = compute_fid(real_imgs, fakes_t, device=cfg.device, wrapper=wrapper)

    # ── 3. Conditional accuracy ───────────────────────────────────────────
    acc = compute_conditional_accuracy(fakes_t, lbl.cpu(), device=cfg.device, extractor=extractor)

    # ── 4. Latent reconstruction SSIM & LPIPS ─────────────────────────────
    # Encode real images → corrupt in latent space → denoise → decode
    # Compare decoded reconstruction with AE-reconstructed originals.
    # This measures how well the latent denoiser reverses the corruption,
    # independently of AE reconstruction quality.
    reals_n1p1 = torch.stack([test_ds[i][0] for i in range(cfg.n_ssim)]).to(cfg.device)
    reals_0to1 = (reals_n1p1 + 1.) / 2.

    # AE-reconstructed originals (the ceiling the latent model can reach)
    ae_recon, _ = ae(reals_n1p1)
    ae_recon_0to1 = ((ae_recon + 1.) / 2.).clamp(0., 1.)

    # Corrupt in latent space at t=1
    D      = ConvAutoencoder.LATENT_DIM
    z0     = ae.encode(reals_n1p1)                          # (n_ssim, 64)
    sig    = forward.sigma_t(torch.ones(cfg.n_ssim, device=cfg.device)).unsqueeze(1)
    eps    = sample_noise(forward.noise_type, (cfg.n_ssim, D),
                          forward.lam_t, forward.M_eig, cfg.device)
    z_T    = z0 + sig * eps

    # Denoise back with the latent model
    real_lbl = torch.tensor([test_ds[i][1] for i in range(cfg.n_ssim)], device=cfg.device)
    null     = torch.full_like(real_lbl, 10)
    sched    = torch.linspace(1., 0., cfg.n_steps + 1, device=cfg.device)
    z        = z_T.clone()

    for k in range(cfg.n_steps):
        tc  = sched[k  ].expand(cfg.n_ssim)
        tn  = sched[k+1].expand(cfg.n_ssim)
        sig = forward.sigma_t(tc).unsqueeze(1)
        cin = (1. + sig**2).pow(-0.5)
        z0c = model(z * cin, tc, real_lbl)
        z0u = model(z * cin, tc, null)
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)
        if k < cfg.n_steps - 1:
            sn  = forward.sigma_t(tn).unsqueeze(1)
            z   = z0h + sn * sample_noise(forward.noise_type, (cfg.n_ssim, D),
                                           forward.lam_t, forward.M_eig, cfg.device)
        else:
            z = z0h

    recon_0to1 = ((ae.decode(z) + 1.) / 2.).clamp(0., 1.)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    ssim_val    = ssim_metric(recon_0to1, ae_recon_0to1).item()
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(cfg.device)
    # LPIPS expects 3 channels
    lpips_val = lpips_metric(recon_0to1.repeat(1, 3, 1, 1), ae_recon_0to1.repeat(1, 3, 1, 1)).item()

    elapsed = time.time() - t0
    return {
        "FID":         round(fid, 2),
        "fFID":        round(f_fid, 2),
        "Accuracy":    round(acc, 2),
        "SSIM":        round(ssim_val, 4),
        "LPIPS":       round(lpips_val, 4),
        "eval_time_s": round(elapsed, 1),
    }

def train_autoencoder(cfg: Config) -> ConvAutoencoder:
    """
    Train a latent-space MLP denoiser.

    Forward process: Z_t = z_0 + sigma(t) * eps,  eps ~ noise_type
    where z_0 = Encoder(x_0) in R^{64}.

    Question 2: does there exist a basis (latent space of autoencoder)
    in which Rosenblatt corruption leads to better results?
    Answer: yes, because (a) the latent distribution is non-Gaussian (richer
    structure), (b) Rosenblatt heavy tails cover the full tail of the latent
    distribution, (c) density theory (Prop. latent-density) remains valid.
    """

    ds  = _get_dataset(cfg.dataset, train=True, tf=_NORM_TF)
    dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                     num_workers=4, pin_memory=True)
    ae  = ConvAutoencoder().to(cfg.device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr)

    for ep in range(cfg.ae_epochs):
        ae.train();  tot = 0
        for x0, _ in dl:
            x0    = x0.to(cfg.device, non_blocking=True)
            r, _  = ae(x0)
            loss  = F.mse_loss(r, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward();  opt.step()
            tot += loss.item() * x0.size(0)
        print(f"  AE ep {ep+1:3d}/{cfg.ae_epochs}  {tot/len(ds):.5f}")

    ckpt_root = Path(getattr(cfg, "ckpt_dir", cfg.save_dir))
    ae_path = ckpt_root / "latent" / "ae_final.pt"
    ae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), ae_path)
    print(f"  AE → {ae_path}")
    return ae

def train_latent(
    ae:           ConvAutoencoder,
    cfg:          Config,
    sigma_max:    float = 4.,
    noise_type:   str   = "rosenblatt",
) -> tuple[LatentMLPDenoiser, RosenblattForward]:
    
    ae.eval()
    tr_dl = DataLoader(_get_dataset(cfg.dataset, train=True,  tf=_NORM_TF),
                       cfg.batch_size, True,  num_workers=4, pin_memory=True,
                       persistent_workers=True)
    va_dl = DataLoader(_get_dataset(cfg.dataset, train=False, tf=_NORM_TF),
                       cfg.batch_size, False, num_workers=4, pin_memory=True,
                       persistent_workers=True)

    D   = ConvAutoencoder.LATENT_DIM
    fwd = RosenblattForward(sigma_additive(), noise_type=noise_type,
                            H=cfg.H, M_eig=cfg.M_eig, sigma_max=sigma_max, device=cfg.device)

    model = LatentMLPDenoiser(latent_dim=D).to(cfg.device)
    tag   = f"lat_{noise_type}_s{sigma_max}"

    # --- ADD MODEL LOADING LOGIC HERE ---
    ckpt_root = Path(getattr(cfg, "ckpt_dir", cfg.save_dir))
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / "latent" / f"{tag}_final.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        print(f"Loading pre-trained Latent model: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
        model.eval()
        return model, fwd
    # ------------------------------------

    ema   = EMA(model, 0.999)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    tag   = f"lat_{noise_type}_s{sigma_max}"

    for ep in range(cfg.epochs):
        t0 = time.time();  model.train();  el = 0
        for x0, lbl in tr_dl:
            x0, lbl = (x0.to(cfg.device, non_blocking=True),
                       lbl.to(cfg.device, non_blocking=True))
            B = x0.size(0)
            with torch.no_grad():
                z0 = ae.encode(x0)
            cf       = torch.rand(B, device=cfg.device) < 0.1
            lbl2     = lbl.clone();  lbl2[cf] = 10
            t        = torch.rand(B, device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
            sig      = fwd.sigma_t(t).unsqueeze(1)          # FIX-1 applied
            eps      = sample_noise(fwd.noise_type, (B, D), fwd.lam_t, cfg.M_eig, cfg.device)
            z_t      = z0 + sig * eps
            cin      = (1.0 + sig ** 2).pow(-0.5)
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(z_t * cin, t, lbl2), z0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step();  ema.update()
            el += loss.item() * B

        el /= len(tr_dl.dataset)
        model.eval();  ema.apply_shadow();  vl = 0
        with torch.no_grad():
            for x0, lbl in va_dl:
                x0, lbl = (x0.to(cfg.device, non_blocking=True),
                           lbl.to(cfg.device, non_blocking=True))
                z0  = ae.encode(x0)
                t   = torch.rand(x0.size(0), device=cfg.device) * (1 - cfg.T_MIN) + cfg.T_MIN
                sig = fwd.sigma_t(t).unsqueeze(1)           # FIX-1 applied
                eps = sample_noise(fwd.noise_type, (x0.size(0), D),
                                   fwd.lam_t, cfg.M_eig, cfg.device)
                z_t = z0 + sig * eps
                cin = (1.0 + sig ** 2).pow(-0.5)
                vl += F.smooth_l1_loss(model(z_t * cin, t, lbl),
                                       z0).item() * z0.size(0)
        vl /= len(va_dl.dataset)
        ema.restore();  sch.step()
        print(f"  [lat] {ep+1:3d}/{cfg.epochs}  tr={el:.5f}  va={vl:.5f}  "
              f"{time.time()-t0:.1f}s")

    ema.apply_shadow()
    torch.save(model.state_dict(), ckpt_path)
    model.eval()
    return model, fwd

@torch.no_grad()
def _save_latent_samples(
    model: LatentMLPDenoiser,
    ae:    ConvAutoencoder,
    fwd:   RosenblattForward,
    cfg:   Config,
    tag:   str = "",
    n_cls: int = 10,
    save_dir: str = ".",
) -> None:
    """Generate n_cls decoded samples (one per class) and save as a grid."""
    model.eval(); ae.eval()
    labels = torch.arange(n_cls, device=cfg.device)
    D      = ConvAutoencoder.LATENT_DIM
    sched  = torch.linspace(1., 0., 50 + 1, device=cfg.device)
    z      = sample_noise(fwd.noise_type, (n_cls, D),
                          fwd.lam_t, fwd.M_eig, cfg.device) * fwd.sigma_max

    null = torch.full_like(labels, 10)
    
    for k in range(50):
        tc = sched[k].expand(n_cls); tn = sched[k+1].expand(n_cls)
        sig = fwd.sigma_t(tc).unsqueeze(1)
        cin = (1. + sig**2).pow(-0.5)
        z0c = model(z * cin, tc, labels)
        z0u = model(z * cin, tc, null)
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)
        if k < 49:
            sn = fwd.sigma_t(tn).unsqueeze(1)
            z  = z0h + sn * sample_noise(fwd.noise_type, (n_cls, D),
                                          fwd.lam_t, fwd.M_eig, cfg.device)
        else:
            z = z0h

    imgs = ((ae.decode(z) + 1.) / 2.).clamp(0., 1.).cpu()  # (10, 1, 28, 28)

    fig, axes = plt.subplots(1, n_cls, figsize=(2. * n_cls, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(class_name(cfg.dataset, i), fontsize=7, rotation=45, ha="right")
        ax.axis("off")
    plt.suptitle(f"Latent samples — {tag}", fontsize=9)
    plt.tight_layout()
    fp = f"{save_dir}/{tag}_samples.png"
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  Saved {fp}")
