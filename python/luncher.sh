#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH -c 7
#SBATCH --job-name=main_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activate environment
source ~/.bashrc
conda activate YOUR_ENV_NAME

# Check GPU
nvidia-smi

# Optional: force one GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Run tasks sequentially
python -m Main --family ablation       --mode all
python -m Main --family gaussianity    --mode all
python -m Main --family optimizer      --mode all
python -m Main --family cold_ablation  --mode all