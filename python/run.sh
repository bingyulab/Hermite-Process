#!/usr/bin/env bash
set -euo pipefail

# === Check how many GPUs are available ===
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    # Default to 0 if nvidia-smi is not found
    NUM_GPUS=0
fi

echo "Number of available GPUs: $NUM_GPUS"

# Function to run the pipeline for a single seed
run_seed() {
  local s=$1
  local gpu_id=$2
  
  if [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "Starting seed $s on GPU $gpu_id..."
  else
    echo "Starting seed $s (No GPU assigned)..."
  fi

  local D="output/s$s"  

  # baselines (R+G) + Gaussianization probes: alpha (equivalence), beta (unstable),
  # gamma/delta (free ride)
  python -m Main --family gaussianity --mode all \
      --noise_types rosenblatt gaussian --seed $s --save_dir "$D"
  
  # R-vs-G FID head-to-head (the equivalence claim), reuses the baselines above
  python -m Main --family cold_ablation --mode "cold_latent generation" \
      --noise_types rosenblatt gaussian --seed $s --save_dir "$D"

  # small-effect ablations (epsilon tiny, zeta near-null, mu moderate, theta)
  python -m Main --family ablation --mode "epsilon zeta mu theta" \
      --noise_types rosenblatt --seed $s --save_dir "$D"

  # discriminability test (fixed-net, R vs G driver) on the per-seed baselines
  python -m rcd.experiments.twosample \
      --noise_types rosenblatt gaussian --seed $s --save_dir "$D" --H 0.7
}

# === 3 seeds: nulls / small effects / unstable / R-vs-G head-to-head ===
SEEDS=(44 43 42)
i=0

for s in "${SEEDS[@]}"; do
  if [ "$NUM_GPUS" -gt 1 ]; then
    # Distribute over available GPUs and run in parallel
    gpu_id=$((i % NUM_GPUS))
    run_seed "$s" "$gpu_id" &
    
    # Check if we've filled all available GPUs for this batch
    if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ]; then
      echo "GPUs filled. Waiting for the current batch to finish..."
      wait
    fi
  else
    # Run sequentially if only 1 (or 0) GPUs are found
    run_seed "$s" 0
  fi
  i=$((i + 1))
done

# Wait for any remaining background tasks to finish (e.g., the last seed if not perfectly divisible)
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Waiting for all parallel seed processes to complete..."
  wait
  echo "All seed runs completed!"
fi

# === 1 seed (seed 42): large qualitative effects, direction obvious ===
echo "Running remaining seed 42 qualitative effects..."
if [ "$NUM_GPUS" -gt 0 ]; then
  # Assign to the first GPU
  export CUDA_VISIBLE_DEVICES=0
fi

python -m Main --family cold_ablation \
    --mode "sigma_comparison cold_h_sweep n_steps cfg_scale cold_bridge cold_loss" \
    --noise_types rosenblatt --seed 42 --save_dir output/s42

# === Reports ===
echo "Generating reports..."
python report.py

# === figures (aggregates across whatever seeds are present) ===
echo "Generating figures..."
python plots.py --results_dir output --out_dir output/figures