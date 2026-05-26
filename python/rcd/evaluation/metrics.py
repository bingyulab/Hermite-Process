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

from rcd.train.training import generate_samples
from rcd.train.noise import sample_noise

# Assumed imports from your project
# from your_module import sample_noise, BetaResult, Config, RosenblattForward, ConditionalUNetFlexible

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
    t_one = torch.ones(B, device=device)
    x_T, _, _ = forward.corrupt(x0_batch, t_one, y=y_batch)
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

    # Noise Generators Mapping
    def get_rosenblatt_noise(bneck_numel, B, device, h3_shape):
        if forward.lam_t is None:
            return torch.randn(B, bneck_numel, device=device).view(h3_shape)
        
        ros_device = torch.device("cpu") if device.type == "cuda" else device
        lam_t_cpu = forward.lam_t.to(ros_device)
        chunks = []
        for i in range(0, B, 16):
            n_i = min(16, B - i)
            chunks.append(sample_noise("rosenblatt", (n_i, bneck_numel), lam_t_cpu, forward.M_eig, ros_device))
        
        ros_flat = torch.cat(chunks, dim=0).to(device).view(h3_shape)
        return ros_flat / ros_flat.std(dim=0, keepdim=True).clamp(min=1e-6)

    for σ in sigma_levels:
        results["clean"][σ] = huber_clean
        scale = bneck_std * σ

        # Generate all noises standardized to unit variance, then scale
        noises = {
            "gaussian": torch.randn_like(h3) * scale,
            "laplace": tdist.Laplace(torch.zeros_like(h3), scale / math.sqrt(2.0)).sample(),
            "rosenblatt": get_rosenblatt_noise(bneck_numel, B, device, h3.shape) * scale,
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
    
    def __init__(self, device: torch.device):
        self.device = device
        
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
            def forward(self, x):
                x = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
                x = self.net.layer4(self.net.layer3(self.net.layer2(self.net.layer1(x))))
                return torch.flatten(self.net.avgpool(x), 1)
        return FashionFeatureExtractor()

    def _build_fashion_wrapper(self) -> nn.Module:
        class FashionFIDWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor
                self.num_features = 512
            def forward(self, x):
                if x.shape[1] == 3: x = x.mean(dim=1, keepdim=True)
                return self.extractor((x * 2.0) - 1.0)
        return FashionFIDWrapper(self.fashion_extractor)

    def _format_images(self, imgs: torch.Tensor, target_range: str, force_rgb: bool = False) -> torch.Tensor:
        """Utility to safely handle [0, 1] vs [-1, 1] and grayscale vs RGB."""
        if target_range == "0to1" and imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0
        elif target_range == "n1to1" and imgs.min() >= 0 and imgs.max() <= 1.01:
            imgs = (imgs * 2.0) - 1.0
            
        if force_rgb and imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return imgs.to(self.device)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, forward: Any, real_imgs: torch.Tensor, test_ds: Any, cfg: Any, bridge: str = "stochastic") -> dict:
        t0 = time.time()
        model.eval()

        # Generate fake images
        labels = torch.randint(0, cfg.num_classes, (cfg.n_fid,), device=cfg.device)
        fakes = torch.cat([
            generate_samples(model, forward, labels[i:i+200], cfg, bridge=bridge).cpu() 
            for i in range(0, cfg.n_fid, 200)
        ], dim=0)

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
            (self.fashion_extractor.net(fake_n1to1[i:i+128]).argmax(dim=-1) == labels[i:i+128]).sum().item()
            for i in range(0, len(fake_n1to1), 128)
        )

        # 3. SSIM & LPIPS Reconstruction
        n_ssim = min(cfg.n_ssim, len(test_ds))
        real_recon = torch.stack([test_ds[i][0] for i in range(n_ssim)]).to(self.device)
        real_labels = torch.tensor([test_ds[i][1] for i in range(n_ssim)], device=self.device)
        
        x_t, _, _ = forward.corrupt(real_recon, torch.ones(n_ssim, device=self.device), y=real_labels)
        recon = generate_samples(model, forward, real_labels, cfg, bridge=bridge, x_in=x_t)

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