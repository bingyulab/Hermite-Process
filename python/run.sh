#!/usr/bin/env bash
set -euo pipefail

# === 3 seeds: nulls / small effects / unstable / R-vs-G head-to-head ===
for s in 42 43 44; do
  D=output/s$s

  # baselines (R+G) + Gaussianization probes: alpha (equivalence), beta (unstable),
  # gamma/delta (free ride)
  python -m Main --family gaussianity --mode all \
      --noise_types rosenblatt gaussian --seed $s --save_dir $D

  # R-vs-G FID head-to-head (the equivalence claim), reuses the baselines above
  python -m Main --family cold_ablation --mode "cold_latent generation" \
      --noise_types rosenblatt gaussian --seed $s --save_dir $D

  # small-effect ablations (epsilon tiny, zeta near-null, mu moderate, theta)
  python -m Main --family ablation --mode "epsilon zeta mu theta" \
      --noise_types rosenblatt --seed $s --save_dir $D

  # discriminability test (fixed-net, R vs G driver) on the per-seed baselines
  python -m rcd.experiments.twosample \
      --noise_types rosenblatt gaussian --seed $s --save_dir $D --H 0.7
done

# === 1 seed (seed 42): large qualitative effects, direction obvious ===
python -m Main --family cold_ablation \
    --mode "sigma_comparison cold_h_sweep n_steps cfg_scale cold_bridge cold_loss" \
    --noise_types rosenblatt --seed 42 --save_dir output/s42

# === Reports ===
python report.py
# === figures (aggregates across whatever seeds are present) ===
python plots.py --results_dir output --out_dir output/figures