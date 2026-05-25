import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from rcd.diffusion.sampler import generate_conditional


def get_fashion_extractor(device: torch.device) -> nn.Module:
    class FashionFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = resnet18(num_classes=10)
            self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        def forward(self, x):
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)
            x = self.net.avgpool(x)
            return torch.flatten(x, 1)

    model = FashionFeatureExtractor()
    return model.to(device)


class FashionFIDWrapper(nn.Module):
    """Adapter so torchmetrics FID can use the FashionMNIST feature extractor."""

    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor
        self.extractor.eval()
        self.num_features = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        x = x.to(next(self.extractor.parameters()).device)
        x = (x * 2.0) - 1.0
        return self.extractor(x)

@torch.no_grad()
def compute_conditional_accuracy(fake_imgs: torch.Tensor,
                                 fake_labels: torch.Tensor,
                                 device: torch.device,
                                 batch_size: int = 128,
                                 extractor: nn.Module = None) -> float:
    """Top-1 accuracy of generated images under the FashionMNIST classifier."""
    if extractor is None:
        extractor = get_fashion_extractor(device)
    extractor.eval()
    
    correct = total = 0
    for i in range(0, fake_imgs.size(0), batch_size):
        x = fake_imgs[i:i+batch_size].to(device)
        if x.min() >= 0 and x.max() <= 1.01:
            x = x * 2.0 - 1.0
        
        logits = extractor.net(x)
        y = fake_labels[i:i+batch_size].to(device)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    
    return 100.0 * correct / total

@torch.no_grad()
def compute_fid(real_imgs: torch.Tensor, fake_imgs: torch.Tensor,
                device: torch.device, batch_size: int = 50,
                wrapper: nn.Module = None) -> float:
    """Standard FID via InceptionV3 or custom feature wrapper. Expects [-1, 1] or [0, 1] input."""    
    if wrapper is not None:
        fid = FrechetInceptionDistance(feature=wrapper, normalize=True).to(device)
    else:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    if real_imgs.min() < 0 or real_imgs.max() > 1:
        real_imgs = (real_imgs + 1.0) / 2.0
    if fake_imgs.min() < 0 or fake_imgs.max() > 1:
        fake_imgs = (fake_imgs + 1.0) / 2.0

    if real_imgs.size(1) == 1:
        real_imgs = real_imgs.repeat(1, 3, 1, 1)
    if fake_imgs.size(1) == 1:
        fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

    for i in range(0, real_imgs.size(0), batch_size):
        fid.update(real_imgs[i:i+batch_size].to(device), real=True)
    for i in range(0, fake_imgs.size(0), batch_size):
        fid.update(fake_imgs[i:i+batch_size].to(device), real=False)

    return float(fid.compute().item())


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    forward,
    real_imgs: torch.Tensor,
    test_ds,
    cfg,
    bridge: str = "stochastic",
) -> dict:
    """Evaluate image-space generation with FID, fFID, accuracy, SSIM, and LPIPS."""
    import time

    model.eval()
    t0 = time.time()

    labels = torch.randint(0, cfg.num_classes, (cfg.n_fid,), device=cfg.device)
    fake_batches = []
    for i in range(0, cfg.n_fid, 200):
        fake_batches.append(
            generate_conditional(model, forward, labels[i:i + 200], cfg, bridge=bridge).cpu()
        )
    fakes = torch.cat(fake_batches, dim=0)

    fid = compute_fid(real_imgs, fakes, cfg.device)
    extractor = get_fashion_extractor(cfg.device)
    wrapper = FashionFIDWrapper(extractor).to(cfg.device)
    fashion_fid = compute_fid(real_imgs, fakes, cfg.device, wrapper=wrapper)
    acc = compute_conditional_accuracy(fakes, labels.cpu(), cfg.device, extractor=extractor)

    n_ssim = min(cfg.n_ssim, len(test_ds))
    real_n1p1 = torch.stack([test_ds[i][0] for i in range(n_ssim)]).to(cfg.device)
    real_0to1 = (real_n1p1 + 1.0) / 2.0
    real_labels = torch.tensor([test_ds[i][1] for i in range(n_ssim)], device=cfg.device)
    x_t, _, _ = forward.corrupt(
        real_n1p1,
        torch.ones(n_ssim, device=cfg.device),
        y=real_labels,
    )
    recon = generate_conditional(model, forward, real_labels, cfg, bridge=bridge, x_in=x_t)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    ssim = ssim_metric(recon, real_0to1).item()
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(cfg.device)
    lpips = lpips_metric(
        recon.repeat(1, 3, 1, 1),
        real_0to1.repeat(1, 3, 1, 1),
    ).item()

    return {
        "FID": round(fid, 2),
        "fFID": round(fashion_fid, 2),
        "Accuracy": round(acc, 2),
        "SSIM": round(ssim, 4),
        "LPIPS": round(lpips, 4),
        "eval_time_s": round(time.time() - t0, 1),
    }
