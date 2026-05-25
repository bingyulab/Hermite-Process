import torch
import time
from pathlib import Path
import os
import sys
# Add the parent directory (python/) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rcd.core.config import Config
from rcd.core.overrides import override
from rcd.tracker.run_context import RunContext
from rcd.models import ConditionalUNet
from rcd.data import _get_dataset, _NORM_TF
from rcd.evaluation import evaluate_model
from rcd.tracker.checkpoints import load_full
from rcd.diffusion import (
    RosenblattForward,
    estimate_eg2,
    sigma_anisotropic,
    sigma_edge_aware,
    sigma_multiplicative,
    sigma_pca_whitened,
    compute_pixel_variance,
)
from experiments.common import train
from experiments.visualize_diffusion import _restoration_grid


def run_ablation_bridge(
        cfg:        Config,
        noise_type: str = "rosenblatt",
) -> dict:
    """
    Train one model with the stochastic bridge, then evaluate generation
    quality under all three bridge strategies on that same model.
    This isolates the effect of re-corruption from training differences.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    sfn = sigma_multiplicative()
    model, forward = train(sfn, cfg, noise_type=noise_type, H=cfg.H, save_dir=save_dir)
    model.eval()

    results = {}
    for bridge in ("stochastic", "hybrid"):
        print(f"Evaluating bridge strategy: {bridge}")
        if not cfg.no_evaluate:        
            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
            print(f"  bridge={bridge}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}  Eval time={metrics['eval_time_s']:.1f}s")        
            results[bridge] = metrics
        
        if not cfg.no_plot:
            _restoration_grid(model, forward, cfg, save_dir,
                            tag=f"bridge_{bridge}", bridge=bridge)

    if not cfg.no_evaluate:
        print(f"\nBridge ablation summary:")
        for t, m in sorted(results.items()):
            print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    return results

def run_ablation_noise(cfg: Config) -> dict:
    """
    Compare Gaussian vs Rosenblatt noise under the same Sigma (multiplicative)
    and same bridge (stochastic).  Each noise type trains its own model
    because the training distribution differs.

    Key theoretical distinction:
      Gaussian  — score-matching would work (Tweedie's formula holds)
      Rosenblatt — Tweedie fails; cold diffusion is a structural necessity
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    for noise_type in ("gaussian", "rosenblatt"):
        bridge = cfg.bridge
        sfn     = sigma_multiplicative()
        print(f"\n{'='*60}\nNoise ablation: noise_type={noise_type}")
        model, forward = train(sfn, cfg, noise_type=noise_type, H=cfg.H, save_dir=str(save_dir))
        model.eval()

        if not cfg.no_evaluate:
            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
            print(f"  noise={noise_type:<12s}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)}  Eval time={metrics['eval_time_s']:.1f}s")
            results[noise_type] = metrics

        if not cfg.no_plot:
            _restoration_grid(model, forward, cfg, save_dir, tag=f"noise_{noise_type}", bridge=bridge)
    
    if not cfg.no_evaluate:
        print(f"\nNoise ablation summary:")
        for t, m in sorted(results.items()):
            print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    return results

def run_ablation_H(cfg: Config, H_values: list[float] = None) -> dict:
    """
    Ablate the parameter H while matching the noise scale schedule exactly.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    if H_values is None: H_values = [0.6, 0.7, 0.8, 0.9]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    for noise_type in ("gaussian", "rosenblatt"):
        results[noise_type] = {}
        for H_val in H_values:
            print(f"\nExp H ablation: noise={noise_type} H={H_val}")
            # The schedule sigma(t) = sigma_max * t^H is managed inside RosenblattForward.
            # Even for Gaussian, it will use t^H.
            sfn = sigma_multiplicative()
            model, forward = train(sfn, cfg, noise_type=noise_type, H=H_val, save_dir=save_dir)
            model.eval()

            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=cfg.bridge)
            results[noise_type][H_val] = metrics
            
            print(f"  {noise_type:10s} H={H_val}  FID={metrics['FID']}  fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  SSIM={metrics['SSIM']}  LPIPS={metrics.get('LPIPS', 0)} Eval Time: {metrics['eval_time_s']:.1f}s")
            
            _restoration_grid(model, forward, cfg, save_dir,
                              tag=f"{noise_type}_H{H_val}", bridge=cfg.bridge)

    print("\nH ablation summary:")
    for noise_type in results:
        for H_val, m in results[noise_type].items():
            print(f"  {noise_type:10s} H={H_val}  FID={m['FID']}  fFID={m.get('fFID', 0)}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m.get('LPIPS', 0)}  Eval time={m['eval_time_s']:.1f}s")
    return results

def run_ablation_loss(cfg: Config, loss_fns: list[str] = None) -> dict:
    """
    Ablate the loss function used for training the cold diffusion model.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    if loss_fns is None: loss_fns = ["huber", "l1", "l2", "quantile"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    print(f"\n--- Starting Loss Function Ablation ---")
    for noise_type in ("gaussian", "rosenblatt"):
        results[noise_type] = {}
        for loss_name in loss_fns:
            print(f"\nExp Loss ablation: noise={noise_type} loss={loss_name}")

            # Temporarily override the config loss
            cfg.loss_fn = loss_name

            # Train (or load) the model.
            # Huber baseline should load the corresponding pretrained checkpoint.
            sfn = sigma_multiplicative()
            model, forward = train(sfn, cfg, noise_type=noise_type, H=0.7, save_dir=save_dir)
            model.eval()

            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=cfg.bridge)
            results[noise_type][loss_name] = metrics

            print(f"  noise={noise_type:<10s} Loss={loss_name:8s}  FID={metrics['FID']:.2f} fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']:.2f}%  SSIM={metrics['SSIM']:.4f}   LPIPS={metrics.get('LPIPS', 0)}")

            _restoration_grid(model, forward, cfg, save_dir,
                              tag=f"{noise_type}_H0.7_loss_{loss_name}", bridge=cfg.bridge)

    print("\nLoss ablation summary:")
    for noise_type, noise_results in results.items():
        for loss_name, m in noise_results.items():
            print(f"  noise={noise_type:<10s} Loss={loss_name:8s}  FID={m['FID']:.2f} fFID={m.get('fFID', 0)}  Acc={m['Accuracy']:.2f}%  SSIM={m['SSIM']:.4f}   LPIPS={m.get('LPIPS', 0)}")
    
    # Restore the default loss
    cfg.loss_fn = "huber"
    return results

def run_ablation_cfg_scale(cfg: Config, scales: list[float] = None) -> dict:
    """
    Ablate the Classifier-Free Guidance scale at inference time.
    Requires the base model to be trained already.
    """
    save_dir = str(cfg.save_dir / "multiplicative")
    if scales is None: scales = [4.0, 3.0, 2.0, 1.5, 1.0, 0.5]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    print(f"\n--- Starting CFG Scale Ablation ---")
    for noise_type in ("rosenblatt", "gaussian"):
        # Force the config to load the base Huber model for each noise type
        cfg.loss_fn = "huber"
        sfn = sigma_multiplicative()
        model, forward = train(sfn, cfg, noise_type=noise_type, H=0.7, save_dir=save_dir)
        model.eval()

        results[noise_type] = {}
        for scale in scales:
            print(f"\nExp CFG ablation: noise={noise_type} scale={scale}")

            # Override scale for inference
            cfg.cfg_scale = scale

            metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=cfg.bridge)
            results[noise_type][scale] = metrics

            print(f"  noise={noise_type:<10s} CFG={scale:<4.1f}  FID={metrics['FID']:.2f} fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']:.2f}%  SSIM={metrics['SSIM']:.4f}   LPIPS={metrics.get('LPIPS', 0)}")

            _restoration_grid(model, forward, cfg, save_dir,
                              tag=f"{noise_type}_H0.7_cfg_{scale:.1f}", bridge=cfg.bridge)

    print("\nCFG ablation summary:")
    for noise_type, noise_results in results.items():
        for scale, m in noise_results.items():
            print(f"  noise={noise_type:<10s} CFG={scale:<4.1f}  FID={m['FID']:.2f} fFID={m.get('fFID', 0)}  Acc={m['Accuracy']:.2f}%  SSIM={m['SSIM']:.4f}   LPIPS={m.get('LPIPS', 0)}")
    
    # Restore the default cfg_scale
    cfg.cfg_scale = 2.5
    cfg.loss_fn = "huber"
    return results

def run_ablation_n_steps(cfg: Config, steps_list: list[int] = None) -> dict:
    save_dir = str(cfg.save_dir / cfg.baseline)
    if steps_list is None: steps_list = [10, 20, 100]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    results = {}
    with override(cfg, loss_fn="huber"):
        for noise_type in ("gaussian", "rosenblatt"):
            sfn = sigma_multiplicative()
            model, forward = train(sfn, cfg, noise_type=noise_type, H=0.7, save_dir=save_dir)
            model.eval()

            results[noise_type] = {}
            for steps in steps_list:
                print(f"\n--- Exp n_steps ablation: noise={noise_type} n_steps={steps} ---")
                with override(cfg, n_steps=steps):
                    metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge="stochastic")
                    results[noise_type][steps] = metrics

                    print(f"  noise={noise_type:<10s} n_steps={steps:3d}  FID={metrics['FID']:.2f} fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']:.2f}%  SSIM={metrics['SSIM']:.4f}   LPIPS={metrics.get('LPIPS', 0)}  Eval time={metrics['eval_time_s']:.1f}s")

                    if not cfg.no_plot:
                        _restoration_grid(model, forward, cfg, save_dir, tag=f"{noise_type}_nsteps_{steps}", bridge="stochastic")

    print("\nN_Steps Ablation Summary:")
    for noise_type, noise_results in results.items():
        for s, m in noise_results.items():
            print(f"  noise={noise_type:<10s} n_steps={s:3d}  FID={m['FID']:.2f} fFID={m.get('fFID', 0)}  Acc={m['Accuracy']:.2f}%  SSIM={m['SSIM']:.4f}   LPIPS={m.get('LPIPS', 0)}")
    return results

def evaluate_all_models_fid(cfg: Config) -> dict:
    """
    Scan a root directory recursively for completed model checkpoints
    (*_final.pt), load each one, and compute its FID.
    """
    root_path = Path(cfg.save_dir)
    
    if not root_path.exists():
        print(f"Directory {cfg.save_dir} not found.")
        return {}

    test_ds   = _get_dataset(cfg.dataset, train=False, tf=_NORM_TF)
    real_imgs = ((torch.stack([test_ds[i][0] for i in range(cfg.n_fid)]) + 1) / 2).clamp(0, 1)

    model_files = list(root_path.rglob("*_final.pt"))
    print(f"Found {len(model_files)} models in {cfg.save_dir}")
    
    class_vars = None # Lazy load for PCA
    eg2_cache = {}    # Cache E[Sigma^2] to avoid redundant dataloader overhead
    results = {}    

    # --- Initialize time statistics ---
    total_load_time = 0.0
    total_fid_time = 0.0

    for ckpt_path in model_files:
        raw_tag = ckpt_path.stem.replace("_final", "")
        # Differentiate same model filenames in different folders
        tag = f"{ckpt_path.parent.name}/{raw_tag}"
        
        # Skip Autoencoders and Latent models (they don't use ConditionalUNet image backbone)
        if "ae" in raw_tag.lower() or "latent" in raw_tag.lower() or "lat_" in raw_tag.lower():
            print(f"Skipping non-UNet model: {tag}")
            continue
            
        # PCA basis models rely on a dynamic lambda closure for SigmaFn
        if "pca_basis" in raw_tag.lower():
            print(f"Skipping dynamic PCA basis model in general eval: {tag}")
            continue

        # Deduce settings from tag (e.g., rosenblatt_sigma_multiplicative_H0.7)               
        noise_type = "rosenblatt" if "rosenblatt" in raw_tag else "gaussian"
        
        import re
        match_H = re.search(r'H([0-9.]+)', ckpt_path.name)
        H = float(match_H.group(1)) if match_H else 0.7

        if "bridge" in raw_tag:
            bridge = raw_tag.split('_')[1]
        else:
            bridge = cfg.bridge

        # Deduce Sigma Fn
        sfn = sigma_multiplicative()
        if "pca_whitened" in raw_tag:
            if class_vars is None: class_vars = compute_pixel_variance(cfg.dataset)
            sfn = sigma_pca_whitened(class_vars)
        elif "anisotropic" in tag and "h_emphasis" in tag:
            sfn = sigma_anisotropic(mode="h_emphasis")
        elif "anisotropic" in tag and "v_emphasis" in tag:
            sfn = sigma_anisotropic(mode="v_emphasis")
        elif "edge_aware" in tag:
            sfn = sigma_edge_aware()

        print(f"\nEvaluating: {tag}, bridge = {bridge}, H = {H}, noise type = {noise_type}")
        
        # --- Start Timer for Loading Model ---
        t0_load = time.time()
        model = ConditionalUNet(num_classes=10, base_ch=cfg.base_ch).to(cfg.device)
        load_full(ckpt_path, model, device=cfg.device)
        model.eval()

        forward = RosenblattForward(sfn, noise_type=noise_type, H=H, device=cfg.device)
        t_load_elapsed = time.time() - t0_load
        total_load_time += t_load_elapsed
        # --- End Timer for Loading Model ---
        
        sfn_name = sfn.__name__
        if sfn_name not in eg2_cache:
            eg2_cache[sfn_name] = estimate_eg2(sfn, cfg.dataset, _get_dataset, _NORM_TF)
        forward.set_eg2(eg2_cache[sfn_name])

        metrics = evaluate_model(model, forward, real_imgs, test_ds, cfg, bridge=bridge)
        results[tag] = metrics
        total_fid_time += time.time() - t0_load - t_load_elapsed
        print(f"  {tag}: FID={metrics['FID']} fFID={metrics.get('fFID', 0)}  Acc={metrics['Accuracy']}%  "
            f"SSIM={metrics['SSIM']}  LPIPS={metrics['LPIPS']}  ({metrics['eval_time_s']}s)")

    print("\nBatch Evaluation Summary:")
    for t, m in sorted(results.items()):
        print(f"  {t}: FID={m['FID']}  fFID={m['fFID']}  Acc={m['Accuracy']}%  SSIM={m['SSIM']}  LPIPS={m['LPIPS']}")
        
    print(f"\nTime Statistics:")
    print(f"  Total time for loading models: {total_load_time:.2f} seconds")
    print(f"  Total time for generation & FID: {total_fid_time:.2f} seconds")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cold Diffusion Ablations")
    parser.add_argument("--mode", default="all")
    args = parser.parse_args()
    
    cfg = Config()
    cfg.mode = args.mode
    
    with RunContext(cfg, family="ablation", run_name="cold_ablation") as ctx:
        ctx.logger.info("Running cold diffusion ablations")
        if args.mode in ("all", "bridge"):
            run_ablation_bridge(cfg)
        if args.mode in ("all", "H"):
            run_ablation_H(cfg)
        if args.mode in ("all", "noise"):
            run_ablation_noise(cfg)
        if args.mode in ("all", "loss"):
            run_ablation_loss(cfg)
        if args.mode in ("all", "cfg_scale"):
            run_ablation_cfg_scale(cfg)
        if args.mode in ("all", "steps"):
            run_ablation_n_steps(cfg)
            
if __name__ == "__main__":
    main()
