#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 7
#SBATCH --job-name=main_parallel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/.bashrc
module load ai/PyTorch/2.3.0-foss-2023b

pip install -r ../requirements.txt
# pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.18.1

# Check GPU
nvidia-smi

srun --exclusive -N1 -n1 bash -c '
export CUDA_VISIBLE_DEVICES=0
python -m Main --family ablation --mode all
' &

srun --exclusive -N1 -n1 bash -c '
export CUDA_VISIBLE_DEVICES=1
python -m Main --family gaussianity --mode all
' &

srun --exclusive -N1 -n1 bash -c '
export CUDA_VISIBLE_DEVICES=2
python -m Main --family optimizer --mode all
' &

srun --exclusive -N1 -n1 bash -c '
export CUDA_VISIBLE_DEVICES=3
python -m Main --family cold_ablation --mode all
' &

wait