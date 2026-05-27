import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from rcd.data.datasets import _get_dataset, _NORM_TF
from rcd.train.training import generate_samples
from pathlib import Path

def precompute_real_imgs(
    test_ds,
    n_target: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Collect `n_target` real images from `test_ds` in deterministic order.
    Returned tensor has shape (n_target, C, H, W) on CPU. Used by FID-based
    experiments to fix the real-image batch across an entire sweep.
    """
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    chunks = []
    n_done = 0
    for x, _ in loader:
        chunks.append(x)
        n_done += x.size(0)
        if n_done >= n_target:
            break
    return torch.cat(chunks, 0)[:n_target]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Rigidity Testing
# ─────────────────────────────────────────────────────────────────────────────

def rigidity_test(
    model: Any,
    forward: Any,
    test_ds: Any,
    cfg: Any,
    sigma_levels: list[float] | None = None,
    n_batch: int = 64,
) -> dict[str, dict[float, float]]:
    """Latent Perturbation 'Rigidity' Test."""
    if sigma_levels is None:
        sigma_levels = [0.1, 0.3, 0.5, 1.0]

    model.eval()
    device = cfg.device

    # Fetch batch
    x0_batch, y_batch = next(iter(DataLoader(test_ds, batch_size=n_batch, shuffle=True)))
    x0_batch, y_batch = x0_batch.to(device), y_batch.to(device)
    B = x0_batch.size(0)

    # Prepare diffusion context
    t_min = torch.full((B,), cfg.T_MIN, device=device)
    # t_one = torch.ones(B, device=device)
    x_T, _, _ = forward.corrupt(x0_batch, t_min, y=y_batch)
    c_in = forward.c_in(t_min).view(-1, 1, 1, 1)
    t_emb = model.time_mlp(model.t_embed(t_min)) + model.label_emb(y_batch)

    # Encode
    with torch.no_grad():
        h3, h2, h1 = model.encode(x_T * c_in, t_emb)

    bneck_std = h3.std(dim=0, keepdim=True).clamp(min=1e-6)
    bneck_numel = h3.numel() // B

    results = {k: {} for k in ["clean", "gaussian", "laplace", "rosenblatt", "student_t3"]}

    # Baseline
    with torch.no_grad():
        x0h_clean = model.decode(h3, h2, h1, t_emb)
        huber_clean = F.smooth_l1_loss(x0h_clean, x0_batch.float()).item()

    from rcd.train.noise import sample_rosenblatt_proxy, sample_gaussian, sample_laplace
    for σ in sigma_levels:
        results["clean"][σ] = huber_clean
        scale = bneck_std * σ

        # Generate all noises standardized to unit variance, then scale
        noises = {
            "gaussian": sample_gaussian(h3.shape, device) * scale,
            "laplace": sample_laplace(h3.shape, device) * scale,
            "rosenblatt": sample_rosenblatt_proxy(h3.shape, device) * scale,
            "student_t3": tdist.StudentT(df=3.0).sample(h3.shape).to(device) * scale * math.sqrt(1.0 / 3.0)
        }

        # Apply perturbations
        for name, noise in noises.items():
            with torch.no_grad():
                x0h_pert = model.decode(h3 + noise.detach(), h2, h1, t_emb)
                results[name][σ] = F.smooth_l1_loss(x0h_pert, x0_batch.float()).item()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 3. Centralized Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Consolidated stateful evaluation class to prevent redundant initialization."""
    
    def __init__(self, device: torch.device, weights_path: str = f"output/checkpoints/fashion_resnet.pth"):
        self.device = device
        self.weights_path = weights_path

        # 1. Setup Classifiers / Extractors
        self.fashion_extractor = self._build_fashion_extractor().to(device)
        self.fashion_extractor.eval()
        
        # 2. Setup Metrics
        self.fid_standard = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.fid_fashion = FrechetInceptionDistance(
            feature=self._build_fashion_wrapper(), normalize=True
        ).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    def _build_fashion_extractor(self) -> nn.Module:
        class FashionFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = resnet18(num_classes=10)
                self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            def forward(self, x, features_only=False):
                x = self.net.conv1(x)
                x = self.net.bn1(x)
                x = self.net.relu(x)
                x = self.net.maxpool(x)
                x = self.net.layer1(x)
                x = self.net.layer2(x)
                x = self.net.layer3(x)
                x = self.net.layer4(x)
                x = self.net.avgpool(x)
                x = torch.flatten(x, 1)
                if features_only:
                    return x
                return self.net.fc(x)

        extractor = FashionFeatureExtractor().to(self.device)
        w_path = Path(self.weights_path)
        
        # Load weights if they exist, otherwise train a new model
        if w_path.exists():
            print(f"Loading cached FashionMNIST feature extractor from {w_path}...")
            extractor.load_state_dict(torch.load(w_path, map_location=self.device, weights_only=True))
        else:
            print(f"Training FashionMNIST classifier since no weights found at {w_path}...")
            
            
            # Use identical normalization to your generative model pipeline
            train_ds = _get_dataset("FashionMNIST", train=True, tf=_NORM_TF)
            train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
            
            opt = torch.optim.Adam(extractor.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            
            extractor.train()
            with torch.enable_grad():
                for ep in range(3):  # 3 epochs is usually enough to hit ~90% accuracy
                    total_loss = 0
                    for x, y in train_dl:
                        x, y = x.to(self.device, dtype=torch.float32), y.to(self.device)
                        opt.zero_grad()
                        loss = crit(extractor(x), y)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                    print(f"  Extractor Epoch {ep+1}/3 Loss: {total_loss/len(train_dl):.4f}")
            
            # Save the newly trained model for future evaluations
            w_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(extractor.state_dict(), w_path)
            
        extractor.eval()
        return extractor

    def _build_fashion_wrapper(self) -> nn.Module:
        class FashionFIDWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor
                self.num_features = 512
            def forward(self, x):
                if x.shape[1] == 3: x = x.mean(dim=1, keepdim=True)
                return self.extractor((x * 2.0) - 1.0, features_only=True)
        return FashionFIDWrapper(self.fashion_extractor)

    def _format_images(self, imgs: torch.Tensor, target_range: str, force_rgb: bool = False) -> torch.Tensor:
        """Utility to safely handle [0, 1] vs [-1, 1] and grayscale vs RGB."""
        
        # 1. Handle value ranges based on what the metric needs
        if target_range == "0to1" and imgs.min() < 0:
            # Convert [-1, 1] to [0, 1]
            imgs = (imgs + 1.0) / 2.0
        elif target_range == "n1to1" and imgs.min() >= 0:
            # Convert [0, 1] to [-1, 1]
            imgs = (imgs * 2.0) - 1.0
            
        # 2. Handle channels based on what the metric needs
        if force_rgb and imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
            
        return imgs.to(self.device)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, forward: Any, real_imgs: torch.Tensor, test_ds: Any, cfg: Any, bridge: str = "stochastic", tag: str = "default_tag") -> dict:
        t0 = time.time()
        model.eval()

        # Setup Cache Directory
        cache_dir = Path("output/checkpoints/cache/")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{tag}_samples.pt"
        
        # ---------------------------------------------------------
        # 1. LOAD CACHE OR GENERATE SAMPLES
        # ---------------------------------------------------------
        if cache_file.exists():
            print(f"  [Cache Hit] Loading generated samples and features directly from {cache_file.name}...")
            cache = torch.load(cache_file, map_location="cpu", weights_only=True)
            fakes = cache["fakes"]
            labels = cache["labels"].to(self.device)
            recon = cache["recon"]
            real_recon = cache["real_recon"]
        else:
            print(f"  [Cache Miss] Generating samples for {tag}. This will be saved for next time...")
            
            # Generate fake images
            labels = torch.randint(0, cfg.num_classes, (cfg.n_fid,), device=cfg.device)
            fakes = torch.cat([
                generate_samples(model, forward, labels[i:i+200], cfg, bridge=bridge).cpu() 
                for i in range(0, cfg.n_fid, 200)
            ], dim=0)

            # Generate Reconstructions
            n_ssim = min(cfg.n_ssim, len(test_ds))
            real_recon = torch.stack([test_ds[i][0] for i in range(n_ssim)]).to(self.device)
            real_labels = torch.tensor([test_ds[i][1] for i in range(n_ssim)], device=self.device)
            
            x_t, _, _ = forward.corrupt(real_recon, torch.ones(n_ssim, device=self.device), y=real_labels)
            recon = generate_samples(model, forward, real_labels, cfg, bridge=bridge, x_in=x_t).cpu()
            real_recon = real_recon.cpu()

            # Save generations to disk immediately
            torch.save({
                "fakes": fakes,
                "labels": labels.cpu(),
                "recon": recon,
                "real_recon": real_recon
            }, cache_file)

        # ---------------------------------------------------------
        # 2. RUN METRICS (Time taken here is < 1 second now)
        # ---------------------------------------------------------

        # 1. FIDs
        real_0to1_rgb = self._format_images(real_imgs, "0to1", force_rgb=True)
        fake_0to1_rgb = self._format_images(fakes, "0to1", force_rgb=True)
        
        self.fid_standard.reset()
        self.fid_fashion.reset()
        
        batch_size = 50
        for i in range(0, len(real_0to1_rgb), batch_size):
            self.fid_standard.update(real_0to1_rgb[i:i+batch_size], real=True)
            self.fid_fashion.update(real_0to1_rgb[i:i+batch_size], real=True)
            
        for i in range(0, len(fake_0to1_rgb), batch_size):
            self.fid_standard.update(fake_0to1_rgb[i:i+batch_size], real=False)
            self.fid_fashion.update(fake_0to1_rgb[i:i+batch_size], real=False)

        # 2. Accuracy
        fake_n1to1 = self._format_images(fakes, "n1to1")
        correct = sum(
             (self.fashion_extractor(fake_n1to1[i:i+128], features_only=False).argmax(dim=-1) == labels[i:i+128]).sum().item()
            for i in range(0, len(fake_n1to1), 128)
        )

        # 3. SSIM & LPIPS Reconstruction
        
        recon_0to1 = self._format_images(recon, "0to1")
        real_recon_0to1 = self._format_images(real_recon, "0to1")

        return {
            "FID": round(self.fid_standard.compute().item(), 2),
            "fFID": round(self.fid_fashion.compute().item(), 2),
            "Accuracy": round(100.0 * correct / len(fakes), 2),
            "SSIM": round(self.ssim_metric(recon_0to1, real_recon_0to1).item(), 4),
            "LPIPS": round(self.lpips_metric(
                self._format_images(recon_0to1, "0to1", force_rgb=True), 
                self._format_images(real_recon_0to1, "0to1", force_rgb=True)
            ).item(), 4),
            "eval_time_s": round(time.time() - t0, 1),
        }