for s in 43 44 42; do
    # Cold: both only where it is the FID head-to-head
    python -m Main --family cold_ablation --mode "sigma_comparison cold_latent generation" --noise_types rosenblatt gaussian --seed $s --save_dir output/s$s
    python -m Main --family cold_ablation --mode "cold_h_sweep n_steps cfg_scale cold_bridge cold_loss" --noise_types rosenblatt --seed $s --save_dir output/s$s
    # Gaussianity: needs the Gaussian reference (alpha/theta cheap)
    python -m Main --family gaussianity  --mode all --noise_types rosenblatt gaussian --seed $s --save_dir output/s$s
    # Optimizer + ablations: Rosenblatt only
    python -m Main --family optimizer    --mode all --noise_types rosenblatt --seed $s --save_dir output/s$s
    python -m Main --family ablation     --mode "epsilon zeta mu theta" --noise_types rosenblatt --seed $s --save_dir output/s$s
done