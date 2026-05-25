import torch
from pathlib import Path
from dataclasses import dataclass


OUT_ROOT = Path("./output/")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    mode:         str   = "all" 
    cfg_scale:    float = 2.5               # Classifier-Free Guidance scale
    n_steps:      int   = 50
    n_display:    int   = 8                 # Number of intermediate steps to save for display (excluding final step)
    sigma_max:    float = 16.0              # Maximum noise level at t=1.0
    base_ch:      int   = 128               # UNet base channels (DO NOT set to batch_size)
    M_eig:        int   = 80                # Number of eigenvalues for LP expansion
    lr:           float = 2e-4
    epochs:       int   = 30
    ae_epochs:    int   = 20
    ae_lr:        float = 1e-3
    batch_size:   int   = 256
    dataset:      str   = "FashionMNIST"
    num_classes:  int   = 10
    noise_type:   str   = "rosenblatt"
    n_fid:        int   = 10000             # Number of samples for FID evaluation
    H:            float = 0.7               # Hurst index
    T_MIN:        float = 0.01
    bridge:       str   = "stochastic"      # "stochastic" | "Hybrid" | "deterministic"
    device:       torch.device = get_device()
    save_dir:     Path  = OUT_ROOT
    n_ssim:       int   = 200               # Number of samples for SSIM evaluation
    k_components: int   = 64                # Number of PCA components for exp_pca_basis
    no_evaluate:  bool  = False
    no_plot:      bool  = False
    baseline:     str   = "multiplicative"  # "multiplicative" | "anisotropic_h_emphasis" | "anisotropic_v_emphasis" | "pca_whitened_conditional" | "pca_whitened_global" | "edge_aware"
    loss_fn:      str   = "huber"           # "huber" | "l2" | "l1" | "quantile"
    seed:         int   = 42

def resolve_argparse_type(t):
    # Handle both actual types and string representations
    if t == int or t == 'int':
        return int
    elif t == float or t == 'float':
        return float
    elif t == str or t == 'str':
        return str
    elif t == bool or t == 'bool':
        return None  # handled separately
    elif t == Path or t == 'Path':
        return Path
    else:
        return str  # fallback
