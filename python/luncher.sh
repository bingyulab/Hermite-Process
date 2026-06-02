#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH -c 7
#SBATCH --job-name=main_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1. Clear environment and load the correct cluster module
source ~/.bashrc
module purge
module load ai/PyTorch/2.3.0-foss-2023b

# 2. Create and activate a clean virtual environment
# This inherits the cluster's PyTorch but keeps new installs self-contained
if [ ! -d "$HOME/rcd_env" ]; then
    python -m venv --system-site-packages "$HOME/rcd_env"
fi
source "$HOME/rcd_env/bin/activate"

# 3. FORCE Python to completely ignore your corrupted ~/.local site-packages
export PYTHONNOUSERSITE=1

# 4. Install dependencies cleanly inside the virtual environment
pip install --upgrade pip
pip install -r ../requirements.txt

# Check GPU
nvidia-smi

# Force one GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Run tasks sequentially
python -m Main --family ablation       --mode all
python -m Main --family gaussianity    --mode all
python -m Main --family optimizer      --mode all
python -m Main --family cold_ablation  --mode all