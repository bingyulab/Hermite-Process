# Hermite-Process and Cold-Diffusion Experiments
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
- Figure generation and marginal comparisons live under `python/` (see `marginal.py`,`density_simulation.py`, `path_simulation.py` ,etc.).
- Diffusion experiments and model checkpoints are stored under `output/` with multiple seed runs.

To run the whole experiment suite and regenerate all figures, use the following command:

```bash
bash ./run.sh

# On a HPC cluster, 
salloc -N 2 -n 2 --exclusive 
module purge
module load ai/PyTorch/2.3.0-foss-2023b

sbatch luncher.sh # luncher_multiGPU.sh

squeue -j 5446980
scontrol show job 5446980
```

If you want to run specific experiments, use the following commands:

```bash
# run the optimizer ablation experiment
python -m Main --family gaussianity --mode all 

python -m Main --family cold_ablation --mode "cold_latent generation" 

python -m Main --family ablation --mode "epsilon zeta mu theta" --noise_types rosenblatt --seed $s --save_dir "$D"
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
├── density_simulation.py
├── logs
│   ├── main_exp_xxxxxx.err
├── luncher_multiGPU.sh
├── luncher.sh
├── Main.py
├── output
│   ├── s42
├── path_simulation.py
├── plot.py
├── rcd
│   ├── data
│   │   ├── config.py
│   │   ├── datasets.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   ├── evaluation
│   │   ├── gaussianity.py
│   │   ├── __init__.py
│   │   ├── measurement.py
│   │   ├── metrics.py
│   │   └── __pycache__
│   ├── experiments
│   │   ├── Experiments.py
│   │   ├── __init__.py
│   │   ├── _plot.py
│   │   ├── registry.py
│   │   ├── runner.py
│   │   └── twosample.py
│   └── train
│       ├── checkpoints.py
│       ├── forward.py
│       ├── __init__.py
│       ├── models.py
│       ├── noise.py
│       ├── optim.py
│       ├── plotting.py
│       ├── save.py
│       └── training.py
├── report_seeds.py
└── run.sh
```


## Citation
If you use this code in your research, please cite the project or contact the author for the correct citation.

## License & contact
This repository is available under the terms in `LICENSE`.
Questions / issues: open an issue or contact the maintainer.

---
Reference: https://github.com/markveillette/rosenblatt