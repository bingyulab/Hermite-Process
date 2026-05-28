# Hermite-Process

A collection of code and experiments for simulating and analysing Hermite (Rosenblatt-type) processes and for studying diffusion models that generate non-Gaussian signals.

This repository contains simulation utilities, experiment scripts, data pipelines, and figure-generation utilities used in the accompanying write-up.

## Highlights
- Fast simulation of Hermite / Rosenblatt processes.
- Density estimation and comparisons (LP / wavelet / Donsker methods).
- Diffusion-model experiments and ablations (multiplicative noise, optimizer ablations, PCA variants).
- Reproducible scripts to regenerate figures from the paper.

## Quick start
Requirements: Python 3.8+ and the packages listed in `requirements.txt`.

1. create and activate a virtual environment (recommended):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing figures
- Figure generation and marginal comparisons live under `python/` (see `marginal.py` and `density_simulation.py`).
- Diffusion experiments and model checkpoints are stored under `output/diffusion/` (baselines are in `output/diffusion/multiplicative`).
- To regenerate paper figures, run the relevant script and point `--save_dir` to an empty folder or an existing experiment folder.

Example:

```bash
# generate marginals and comparison plots
python python/marginal.py --mode run_all

# run the optimizer ablation experiment
python python/Experiment_Optimizer.py --mode ablation
```

```
salloc -N 2 -n 2 --exclusive 
module purge
module load ai/PyTorch/2.3.0-foss-2023b


```

## Project layout

Top-level layout ( important folders):

```
python/            # experiment and simulation code
latex/             # paper source
imgs/              # figures (and tikz sources)
output/            # experiment outputs and model checkpoints
data/              # datasets (e.g. FashionMNIST)
```

The code layout is:
```
python/
├── rcd/
│   ├── data/
│   │   └── config.py        (Global hyperparameters and dataclass struct)
│   │   └── datasets.py      (Dataset loaders, normalisations)
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── ema.py           (Exponential Moving Average tools)
│   │   ├── forward.py       (RosenblattForward & PCA basis logic)
│   │   ├── noise.py         (Rosenblatt, Generalized, Additive noises)
│   │   ├── sampler.py       (Diffusion samplers and latent generators)
│   │   └── training.py      (Defines training loops and MSE wrappers)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification.py(FashionFeatureExtractor logic)
│   │   └── metrics.py       (FID, conditional_accuracy, evaluating models)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py   (Latent AE models)
│   │   ├── latent.py        (MLP Denoisers for latents)
│   │   ├── layers.py        (Common Unet ops, AdaGN blocks)
│   │   └── unet.py          (ConditionalUNets for all modalities)
│   └── tracker/
│       ├── __init__.py
│       └── run_context.py   (The new determinisic saving logic)
│
├── experiments/
│   ├── run_ablation.py       (L1/L2, Activation, Norm ablations)
│   ├── run_cold_ablation.py  (Rosenblatt Noise, Bridge, H, CFG scale experiments)
│   ├── run_gaussianity.py    (K3 skew bottleneck testing)
│   ├── run_latent.py         (64-D Latent space generation)
│   ├── run_optimizer.py      (Optimizer & gradient studies)
│   └── visualize_diffusion.py(All sigma plots, grids, noise paths comparisons)
```

Key locations:
- Baseline diffusion checkpoints: `output/diffusion/multiplicative`
- Figure outputs: `imgs/` and `output/experiments`

## Notes & tips
- Use `--save_dir output/diffusion` for diffusion experiments; pretrained baselines are found under `output/diffusion/multiplicative`.
- If you need more memory for large simulations, reduce `--n_fid` or run on a machine with more RAM / GPU.

## Citation
If you use this code in your research, please cite the project or contact the author for the correct citation.

## License & contact
This repository is available under the terms in `LICENSE`.
Questions / issues: open an issue or contact the maintainer.

---
Reference: https://github.com/markveillette/rosenblatt