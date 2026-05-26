from __future__ import annotations

from typing import Any, Callable, Dict, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from density_simulation import eigenvalues_LP
from rcd.data.datasets import _get_dataset, _NORM_TF
from rcd.train.noise import sample_noise

# Cache survives the lifetime of the Python process.
_EG2_CACHE: dict[tuple[str, str], float] = {}
SigmaFn = Callable[..., torch.Tensor]


# ─── INTERNAL HELPERS ────────────────────────────────────────────────────────

def _match_and_expand(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Helper to broadcast a spatial scale tensor to match the target's dimension/shape."""
    s = scale.to(target.device)
    while s.dim() < target.dim():
        s = s.unsqueeze(0)
    return s.expand_as(target)


def _set_meta(fn: Callable, name: str, label: str, eg2: float | None, needs_label: bool) -> None:
    """Helper to cleanly attach metadata attributes to sigma functions."""
    fn.__name__ = name
    fn.label = label
    fn.eg2 = eg2
    fn.needs_label = needs_label


# ─── EIGENVALUES & ESTIMATION PATHS ──────────────────────────────────────────

def build_eigenvalues(H: float, M: int, device: torch.device) -> torch.Tensor:
    """Unit-variance LP eigenvalues for Rosenblatt noise."""
    lam = eigenvalues_LP(a=1.0 - H, K=M)
    var = 2.0 * np.sum(lam ** 2)
    if var > 0:
        lam = lam / np.sqrt(var)
    return torch.tensor(lam, dtype=torch.float32, device=device)


def estimate_eg2(sigma_fn: SigmaFn,
                 dataset_name: str,
                 get_dataset_fn: Callable,
                 transform: Any,
                 n_samples: int = 5000,
                 batch_size: int = 512) -> float:
    """Estimate E[ ||Sigma(x0)||_F^2 / numel(x0) ] over n_samples training examples."""
    ds = get_dataset_fn(dataset_name, train=True, tf=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    total = 0.0
    count = 0
    for x0, labels in loader:
        S = sigma_fn(x0, labels) if getattr(sigma_fn, "needs_label", False) else sigma_fn(x0)
        total += (S ** 2).mean().item() * x0.size(0)
        count += x0.size(0)
        if count >= n_samples:
            break
    return total / max(count, 1)


def get_eg2(sigma_fn: SigmaFn,
            dataset_name: str,
            get_dataset_fn: Callable,
            transform: Any,
            n_samples: int = 5000) -> float:
    """Cached version of estimate_eg2."""
    key = (getattr(sigma_fn, "__name__", "anon"), dataset_name)
    if key not in _EG2_CACHE:
        _EG2_CACHE[key] = estimate_eg2(
            sigma_fn, dataset_name, get_dataset_fn, transform, n_samples=n_samples
        )
    return _EG2_CACHE[key]


def clear_eg2_cache() -> None:
    _EG2_CACHE.clear()


# ─── VARIANCE COMPUTATION DATA UTILITIES ─────────────────────────────────────

def compute_pixel_variance(dataset_name: str, conditional: bool = True, n_samples: int = 5000) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Computes per-pixel variance over the dataset.
    If conditional=True, returns a dict mapping class index (0..9) to a (C,H,W) Tensor.
    If conditional=False, returns a dataset-wide global (C,H,W) Tensor.
    """
    ds = _get_dataset(dataset_name, train=True, tf=_NORM_TF)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2)
    
    if not conditional:
        xs: list[torch.Tensor] = []
        seen = 0
        for x, _ in loader:
            xs.append(x)
            seen += x.size(0)
            if seen >= n_samples:
                break
        return torch.cat(xs, dim=0)[:n_samples].var(dim=0)
        
    buckets: dict[int, list[torch.Tensor]] = {}
    for x, y in loader:
        for xi, yi in zip(x, y):
            c = int(yi.item())
            if c not in buckets:
                buckets[c] = []
            if len(buckets[c]) < n_samples:
                buckets[c].append(xi)
        if len(buckets) == 10 and all(len(v) >= n_samples for v in buckets.values()):
            break
            
    return {c: torch.stack(imgs).var(dim=0) for c, imgs in buckets.items()}


# ─── SIGMA FACTORIES (SigmaFn) ───────────────────────────────────────────────

def sigma_additive() -> SigmaFn:
    """Sigma = I (Trivial baseline)."""
    fn = torch.ones_like
    _set_meta(fn, "additive", r"$\Sigma = I$", 1.0, needs_label=False)
    return fn


def sigma_multiplicative() -> SigmaFn:
    """Sigma(x0) = diag(g(x0)), g_i(x) = (1 + |x_i|) / 2."""
    def fn(x0: torch.Tensor) -> torch.Tensor:
        return (1.0 + x0.abs()) / 2.0
    _set_meta(fn, "multiplicative", r"$\Sigma = \mathrm{diag}(g(\mathbf{x}_0))$", 7.0 / 12.0, needs_label=False)
    return fn


def sigma_anisotropic(mode: str = "h_emphasis", img_shape: tuple = (1, 28, 28)) -> SigmaFn:
    """Sigma = A fixed diagonal, precomputed per-pixel scale."""
    C, H_img, W = img_shape
    A = torch.ones(C, H_img, W)
    if mode == "h_emphasis":
        A[:, :, :] = torch.linspace(1., 3., W).view(1, 1, W)
    elif mode == "v_emphasis":
        A[:, :, :] = torch.linspace(1., 3., H_img).view(1, H_img, 1)
    else:
        raise ValueError(f"Unknown anisotropic mode: {mode!r}")
    A /= A.mean()
    _A = A.clone()

    def fn(x0: torch.Tensor) -> torch.Tensor:
        return _match_and_expand(_A, x0)
        
    _set_meta(fn, f"anisotropic_{mode}", rf"$\Sigma = A_{{\rm {mode}}}$", float((_A ** 2).mean().item()), needs_label=False)
    return fn


def sigma_pca_whitened(variance_data: torch.Tensor | dict[int, torch.Tensor]) -> SigmaFn:
    """
    Polymorphic PCA whitening factory. 
    Accepts either global variance tensor or conditional class variance dict.
    """
    if isinstance(variance_data, dict):
        # Class-conditional logic
        _scales: dict[int, torch.Tensor] = {}
        for cls, var in variance_data.items():
            scale = 1. / var.clamp(min=1e-4).sqrt()
            _scales[cls] = scale / scale.mean()
            
        _global = torch.stack(list(_scales.values())).mean(0)
        _global = _global / _global.mean()

        def fn(x0: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
            if y is None:
                return _match_and_expand(_global, x0)
            out = torch.empty_like(x0)
            for b in range(x0.size(0)):
                cls = int(y[b].item())
                s = _scales.get(cls, _global)
                out[b] = _match_and_expand(s, x0[b:b+1]).squeeze(0)
            return out

        eg2 = float(torch.stack([(s**2).mean() for s in _scales.values()]).mean())
        _set_meta(fn, "pca_whitened_conditional", r"$\Sigma=\hat{C}_y^{-1/2}$ (class-cond.)", eg2, needs_label=True)
    else:
        # Global logic
        scale = 1. / variance_data.clamp(min=1e-4).sqrt()
        scale = scale / scale.mean()
        _s = scale.clone()

        def fn(x0: torch.Tensor) -> torch.Tensor:
            return _match_and_expand(_s, x0)

        _set_meta(fn, "pca_whitened_global", r"$\Sigma = \hat{C}^{-1/2}$ (global)", float((scale**2).mean()), needs_label=False)
        
    return fn


def sigma_edge_aware(sobel_strength: float = 2.0) -> SigmaFn:
    """Sigma(x0) = diag(|Sobel(x0)| / mean + base), base=0.5."""
    _sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    _sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    _base, _scale = 0.5, sobel_strength

    def fn(x0: torch.Tensor) -> torch.Tensor:
        dev = x0.device
        sx, sy = _sobel_x.to(dev), _sobel_y.to(dev)
        gx = F.conv2d(x0, sx.expand(x0.shape[1], 1, 3, 3), padding=1, groups=x0.shape[1])
        gy = F.conv2d(x0, sy.expand(x0.shape[1], 1, 3, 3), padding=1, groups=x0.shape[1])
        mag = (gx**2 + gy**2).sqrt()
        m = mag.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-4)
        return _base + _scale * mag / m
        
    _set_meta(fn, "edge_aware", r"$\Sigma = \mathrm{diag}(|\nabla \mathbf{x}_0|)$", None, needs_label=False)
    return fn


# ─── UNIFIED FORWARD PROCESS CLASS ───────────────────────────────────────────

class RosenblattForward:
    """Unified forward process supporting additive and multiplicative modes."""

    def __init__(self,
                 sigma_fn: SigmaFn,
                 noise_type: str = "rosenblatt",
                 H: float = 0.7,
                 M_eig: int = 80,
                 sigma_max: float = 16.0,
                 device: torch.device | str = "cpu") -> None:
        self.sigma_fn = sigma_fn
        self.noise_type = noise_type
        self.H = float(H)
        self.M_eig = M_eig
        self.sigma_max = float(sigma_max)
        self.device = device
        self.name = getattr(sigma_fn, "__name__", "custom")
        self.label = getattr(sigma_fn, "label", self.name)
        eg2_hint = getattr(sigma_fn, "eg2", 1.0)
        if noise_type not in ("gaussian", "rosenblatt"):
            raise ValueError(f"RosenblattForward.noise_type must be 'gaussian' or 'rosenblatt', got {noise_type!r}")
        self._eg2 = 1.0 if eg2_hint is None else float(eg2_hint)
        self.lam_t = build_eigenvalues(H, M_eig, device) if noise_type == "rosenblatt" else None

    def set_eg2(self, eg2: float) -> None:
        """Override E[Sigma(x0)^2] estimated from rcd.train.training data."""
        self._eg2 = float(eg2)

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Scalar sigma(t) = sigma_max * t^H."""
        return self.sigma_max * t.clamp(min=1e-6).pow(self.H)

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        """Input normalisation: 1/sqrt(1 + sigma(t)^2 * E[Sigma^2])."""
        return (1. + self.sigma_t(t)**2 * self._eg2).pow(-0.5)

    def _get_sigma(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Internal router to apply clean signature-checking execution."""
        if getattr(self.sigma_fn, "needs_label", False):
            return self.sigma_fn(x, y)
        return self.sigma_fn(x)

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_t, eps, sig_scalar) where x_t = x0 + sig * Sigma(x0) * eps."""
        t = t.to(x0.device, dtype=torch.float32)
        sig = self.sigma_t(t).view(-1, *([1] * (x0.dim() - 1)))
        eps = sample_noise(self.noise_type, x0.shape, self.lam_t, self.M_eig, x0.device)
        S = self._get_sigma(x0, y)
        return x0 + sig * S * eps, eps, sig    

    def recorrupt_stochastic(self, x0_hat: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """FIXED stochastic bridge re-corruption."""
        x_next, _, _ = self.corrupt(x0_hat, t_next, y=y)
        return x_next

    def recorrupt_deterministic(self, x_cur: torch.Tensor, x0_hat: torch.Tensor, t_cur: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """BROKEN for Rosenblatt (cumulant amplification). Kept for legacy ablation."""
        sc = self.sigma_t(t_cur).view(-1, *([1] * (x_cur.dim() - 1)))
        sn = self.sigma_t(t_next).view(-1, *([1] * (x_cur.dim() - 1)))
        return x0_hat + (sn / sc.clamp(min=1e-5)) * (x_cur - x0_hat)

    def recorrupt_hybrid(self, x_cur: torch.Tensor, x0_hat: torch.Tensor, t_cur: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Hybrid bridge interpolation via eta schedule."""
        s_cur = self.sigma_t(t_cur).view(-1, *([1] * (x_cur.dim() - 1)))
        s_next = self.sigma_t(t_next).view(-1, *([1] * (x_cur.dim() - 1)))
        r = (s_next / s_cur.clamp(min=1e-5)).clamp(0., 1.)
        eta = r.pow(self.H / (self.H + 1.))
        
        S_hat = self._get_sigma(x0_hat, y)
        eps_hat = (x_cur - x0_hat) / (s_cur * S_hat).clamp(min=1e-5)
        eps_new = sample_noise(self.noise_type, x0_hat.shape, self.lam_t, self.M_eig, x0_hat.device)
        mix = (1. - eta ** 2).clamp(min=0.).sqrt() * eps_hat + eta * eps_new
        return x0_hat + s_next * S_hat * mix


# ─── OUTER CONST-BUILDERS ───────────────────────────────────────────────────

def build_forward(forward_cls: Any,
                  sigma_fn: SigmaFn,
                  *,
                  noise_type: str,
                  H: float,
                  M_eig: int,
                  sigma_max: float,
                  device: Any,
                  dataset_name: str,
                  get_dataset_fn: Callable,
                  transform: Any,
                  estimate: bool = True) -> Any:
    """Construct a RosenblattForward instance with empirical E[Sigma^2]."""
    fwd = forward_cls(sigma_fn, noise_type=noise_type, H=H, M_eig=M_eig, sigma_max=sigma_max, device=device)
    if estimate:
        eg2 = get_eg2(sigma_fn, dataset_name, get_dataset_fn, transform)
        fwd.set_eg2(eg2)
    return fwd


def build_forward_process(sigma_fn: SigmaFn, cfg: Any, *, noise_type: str | None = None, H: float | None = None, estimate_eg2: bool = True) -> RosenblattForward:
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