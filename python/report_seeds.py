"""
report_seeds.py
===============
Load per-seed CSVs from output/s{42,43,44}/ and print μ ± σ summary tables.

Usage
-----
    python report_seeds.py                        # default: output/s42  s43  s44
    python report_seeds.py --roots output/s42 output/s43 output/s44
    python report_seeds.py --roots /path/a /path/b /path/c --latex

Output
------
One section per experiment.  Each row is one (noise_type, condition) key.
Columns printed as   μ ± σ  over the seeds that actually have the file.
A missing file for one seed is silently skipped; a warning is printed.
--latex writes a .tex fragment next to this script.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mu_pm_sigma(values: np.ndarray, decimals: int = 3) -> str:
    """Format an array of scalar values as  μ ± σ  string."""
    if len(values) == 0:
        return "—"
    mu  = float(np.mean(values))
    sig = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    fmt = f".{decimals}f"
    return f"{mu:{fmt}} ± {sig:{fmt}}"


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return None


def _collect(roots: List[Path], rel: str) -> List[pd.DataFrame]:
    """Return one DataFrame per seed that has rel, printing a warning for missing."""
    frames = []
    for root in roots:
        p = root / rel
        df = _load_csv(p)
        if df is None:
            warnings.warn(f"Missing: {p}")
        else:
            df["_seed"] = root.name          # tag for debugging
            frames.append(df)
    return frames


def _numeric_cols(df: pd.DataFrame, exclude: Tuple[str, ...] = ()) -> List[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if c not in exclude and not c.startswith("_")]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-experiment aggregators
#     Each returns a dict:  group_key -> {metric_name -> np.ndarray of seed values}
# ─────────────────────────────────────────────────────────────────────────────

def _agg_by_key(
    frames:      List[pd.DataFrame],
    group_cols:  List[str],
    value_cols:  List[str],
) -> Dict[Tuple, Dict[str, np.ndarray]]:
    """
    For each unique combination of group_cols, collect value_cols
    across seeds.  Returns {group_key_tuple: {col: array_of_seed_means}}.
    """
    # Per-seed: compute the mean of value_cols within each group.
    per_seed: List[pd.DataFrame] = []
    for df in frames:
        cols_present = [c for c in group_cols + value_cols if c in df.columns]
        sub = df[cols_present].copy()
        group_present = [c for c in group_cols if c in sub.columns]
        val_present   = [c for c in value_cols  if c in sub.columns]
        if not group_present or not val_present:
            continue
        agg = sub.groupby(group_present, sort=False)[val_present].mean().reset_index()
        per_seed.append(agg)

    if not per_seed:
        return {}

    # Merge across seeds.
    combined = pd.concat(per_seed, ignore_index=True)
    group_present = [c for c in group_cols if c in combined.columns]
    val_present   = [c for c in value_cols  if c in combined.columns]

    result: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for key, grp in combined.groupby(group_present, sort=False):
        k = key if isinstance(key, tuple) else (key,)
        result[k] = {col: grp[col].dropna().values for col in val_present}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Printers
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─"

def _header(title: str) -> None:
    w = 100
    print(f"\n{_SEP * w}")
    print(f"  {title}")
    print(_SEP * w)


def _print_table(
    agg:         Dict[Tuple, Dict[str, np.ndarray]],
    group_names: List[str],
    metrics:     List[str],
    decimals:    int = 3,
    n_seeds:     int = 3,
) -> None:
    if not agg:
        print("  (no data)")
        return

    # Build header
    key_w   = max(40, max(len("/".join(str(x) for x in k)) for k in agg) + 2)
    col_w   = max(20, 14 + 2 * decimals)
    hdr_key = "/".join(group_names)
    header  = f"  {hdr_key:<{key_w}}" + "".join(f"  {m:>{col_w}}" for m in metrics)
    print(header)
    print("  " + _SEP * (len(header) - 2))

    for key, cols in agg.items():
        row_key = "/".join(str(x) for x in key)
        row = f"  {row_key:<{key_w}}"
        for m in metrics:
            vals = cols.get(m, np.array([]))
            cell = _mu_pm_sigma(vals, decimals)
            row += f"  {cell:>{col_w}}"
        print(row)

    n_actual = max((len(v) for c in agg.values() for v in c.values()), default=0)
    print(f"\n  Seeds used: {n_actual} / {n_seeds}")


def _latex_table(
    agg:         Dict[Tuple, Dict[str, np.ndarray]],
    group_names: List[str],
    metrics:     List[str],
    caption:     str,
    label:       str,
    decimals:    int = 3,
) -> str:
    """Return a LaTeX tabular string."""
    n_cols = len(group_names) + len(metrics)
    col_spec = "l" * len(group_names) + " " + "r" * len(metrics)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(group_names + [f"${m}$" for m in metrics]) + r" \\",
        r"\midrule",
    ]
    for key, cols in agg.items():
        cells = [str(x) for x in key]
        for m in metrics:
            vals = cols.get(m, np.array([]))
            cells.append(_mu_pm_sigma(vals, decimals))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-experiment report functions
# ─────────────────────────────────────────────────────────────────────────────

def report_alpha(roots, latex=False):
    """α — pipeline-stage cumulants.  Grouping: noise_type × label (stage)."""
    _header("α  —  Cumulant Gaussianization probe  (pipeline stages)")
    frames = _collect(roots, "metrics/gaussianity/alpha.csv")
    metrics = ["dist_k4", "dist_k3", "dist_pr", "dist_mardia_z", "dist_frac_nong"]
    agg = _agg_by_key(frames, ["noise_type", "label"], metrics)
    _print_table(agg, ["noise_type", "stage"], metrics, decimals=3, n_seeds=len(roots))
    return agg


def report_beta(roots, latex=False):
    """β — bottleneck width vs Gaussianization.  Grouping: noise_type × bf."""
    _header("β  —  Bottleneck width vs Gaussianization")
    frames = _collect(roots, "metrics/gaussianity/beta.csv")
    metrics = [
        "mean_k4_bneck", "mean_k4_input", "mean_k4_x0hat",
        "pr_bneck", "whiteness_bneck", "mardia_b2p_z",
        "offline_loss_huber", "perturb_gauss_huber", "perturb_rosenblatt_huber",
    ]
    agg = _agg_by_key(frames, ["noise_type", "bottleneck_factor"], metrics)
    _print_table(agg, ["noise_type", "bf"], metrics, decimals=3, n_seeds=len(roots))
    return agg


def report_gamma(roots, latex=False):
    """γ — full layer-by-layer κ4 trace.  Grouping: noise_type × layer_key."""
    _header("γ  —  Layer-by-layer κ4 trace")
    frames = _collect(roots, "metrics/gaussianity/gamma.csv")
    metrics = ["mean_k4", "std_k4", "pr", "whiteness", "mardia_b2p_z"]
    agg = _agg_by_key(frames, ["noise_type", "layer_key"], metrics)
    _print_table(agg, ["noise_type", "layer"], metrics, decimals=3, n_seeds=len(roots))
    return agg


def report_delta(roots, latex=False):
    """δ — latent rigidity.  Grouping: noise_type × perturbation × sigma."""
    _header("δ  —  Latent perturbation rigidity")
    frames = _collect(roots, "metrics/gaussianity/delta.csv")
    metrics = ["huber_loss"]
    agg = _agg_by_key(frames, ["noise_type", "perturbation", "sigma"], metrics)
    _print_table(agg, ["noise_type", "perturbation", "σ"], metrics, decimals=4,
                 n_seeds=len(roots))
    return agg


def report_epsilon(roots, latex=False):
    """ε — loss-function ablation.  Grouping: noise_type × loss_type."""
    _header("ε  —  Loss-function ablation")
    frames = _collect(roots, "metrics/ablation/epsilon.csv")
    metrics = ["dist_k4", "dist_pr", "dist_mardia_z", "loss_l1", "loss_huber"]
    agg = _agg_by_key(frames, ["noise_type", "cfg_loss_type"], metrics)
    _print_table(agg, ["noise_type", "loss_type"], metrics, decimals=4,
                 n_seeds=len(roots))
    return agg


def report_zeta(roots, latex=False):
    """ζ — normalization ablation.  Grouping: noise_type × norm_type."""
    _header("ζ  —  Normalisation ablation")
    frames = _collect(roots, "metrics/ablation/zeta.csv")
    metrics = ["dist_k4", "dist_pr", "dist_mardia_z", "loss_l1"]
    agg = _agg_by_key(frames, ["noise_type", "cfg_norm_type"], metrics)
    _print_table(agg, ["noise_type", "norm_type"], metrics, decimals=4,
                 n_seeds=len(roots))
    return agg


def report_mu(roots, latex=False):
    """μ — skip-connection ablation.  Grouping: noise_type × variant."""
    _header("μ  —  Skip-connection ablation")
    frames = _collect(roots, "metrics/ablation/mu.csv")
    metrics = ["dist_k4", "dist_pr", "dist_mardia_z", "loss_l1"]
    agg = _agg_by_key(frames, ["noise_type", "cfg_variant"], metrics)
    _print_table(agg, ["noise_type", "variant"], metrics, decimals=4,
                 n_seeds=len(roots))
    return agg


def report_theta(roots, latex=False):
    """θ — time-conditional κ4.  Grouping: noise_type × t_value."""
    _header("θ  —  Time-conditional bottleneck κ4")
    frames = _collect(roots, "metrics/ablation/theta.csv")
    metrics = ["dist_k4", "dist_mardia_z"]
    agg = _agg_by_key(frames, ["noise_type", "cfg_t_value"], metrics)
    _print_table(agg, ["noise_type", "t_value"], metrics, decimals=3,
                 n_seeds=len(roots))
    return agg


def report_cold_latent(roots, latex=False):
    """cold_latent — latent-space FID.  Grouping: noise_type × sigma_max."""
    _header("cold_latent  —  Latent-space generation quality (FID / Accuracy / SSIM)")
    frames = _collect(roots, "metrics/cold_ablation/cold_latent.csv")
    # extras are stored flat after ExperimentRecord.flatten()
    metrics = ["FID", "fFID", "Accuracy", "SSIM", "LPIPS"]
    agg = _agg_by_key(frames, ["noise_type", "cfg_sigma_max"], metrics)
    _print_table(agg, ["noise_type", "σ_max"], metrics, decimals=2,
                 n_seeds=len(roots))
    return agg


def report_twosample(roots, latex=False):
    """
    twosample — discriminability test (R vs G).
    Schema unknown; we aggregate every numeric column by noise_type.
    """
    _header("twosample  —  Two-sample discriminability test (R vs G)")
    # twosample may land in several places depending on implementation
    frames: List[pd.DataFrame] = []
    candidates = [
        "twosample.csv",
        "metrics/twosample.csv",
        "metrics/gaussianity/twosample.csv",
        "metrics/cold_ablation/twosample.csv",
    ]
    for root in roots:
        df = None
        for c in candidates:
            df = _load_csv(root / c)
            if df is not None:
                break
        if df is None:
            warnings.warn(f"twosample.csv not found under {root}")
        else:
            df["_seed"] = root.name
            frames.append(df)

    if not frames:
        print("  (no twosample data found — skipping)")
        return {}

    # Auto-detect grouping column and numeric metrics
    sample = frames[0]
    group_col = "noise_type" if "noise_type" in sample.columns else sample.columns[0]
    exclude = ("_seed", group_col)
    metrics = _numeric_cols(sample, exclude=exclude)
    if not metrics:
        print("  (no numeric columns detected in twosample.csv)")
        return {}

    agg = _agg_by_key(frames, [group_col], metrics)
    _print_table(agg, [group_col], metrics, decimals=4, n_seeds=len(roots))
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LaTeX output
# ─────────────────────────────────────────────────────────────────────────────

def _write_latex(
    all_agg:   Dict[str, Dict],
    out_path:  Path,
) -> None:
    """Write a .tex file with one table per experiment."""
    specs = {
        "alpha":        (["noise_type", "stage"],
                         ["dist_k4", "dist_k3", "dist_pr", "dist_mardia_z"],
                         "α: Pipeline-stage cumulants (μ ± σ, 3 seeds)",
                         "tab:alpha_seeds"),
        "beta":         (["noise_type", "bf"],
                         ["mean_k4_bneck", "pr_bneck", "offline_loss_huber"],
                         "β: Bottleneck width vs Gaussianization (μ ± σ, 3 seeds)",
                         "tab:beta_seeds"),
        "gamma":        (["noise_type", "layer"],
                         ["mean_k4", "pr", "whiteness"],
                         "γ: Layer-by-layer κ4 trace (μ ± σ, 3 seeds)",
                         "tab:gamma_seeds"),
        "delta":        (["noise_type", "perturbation", "σ"],
                         ["huber_loss"],
                         "δ: Latent rigidity (μ ± σ, 3 seeds)",
                         "tab:delta_seeds"),
        "epsilon":      (["noise_type", "loss_type"],
                         ["dist_k4", "dist_pr", "loss_l1"],
                         "ε: Loss-function ablation (μ ± σ, 3 seeds)",
                         "tab:epsilon_seeds"),
        "zeta":         (["noise_type", "norm_type"],
                         ["dist_k4", "dist_pr", "loss_l1"],
                         "ζ: Normalisation ablation (μ ± σ, 3 seeds)",
                         "tab:zeta_seeds"),
        "mu":           (["noise_type", "variant"],
                         ["dist_k4", "dist_pr", "loss_l1"],
                         "μ: Skip-connection ablation (μ ± σ, 3 seeds)",
                         "tab:mu_seeds"),
        "theta":        (["noise_type", "t_value"],
                         ["dist_k4", "dist_mardia_z"],
                         "θ: Time-conditional κ4 (μ ± σ, 3 seeds)",
                         "tab:theta_seeds"),
        "cold_latent":  (["noise_type", "σ_max"],
                         ["FID", "fFID", "Accuracy", "SSIM", "LPIPS"],
                         "cold\\_latent: Latent FID (μ ± σ, 3 seeds)",
                         "tab:cold_latent_seeds"),
        "twosample":    (None, None,
                         "twosample: Discriminability (μ ± σ, 3 seeds)",
                         "tab:twosample_seeds"),
    }

    blocks = []
    for key, agg in all_agg.items():
        if not agg or key not in specs:
            continue
        group_names, metrics, caption, label = specs[key]
        if group_names is None:
            # Auto-detect for twosample
            sample_key = next(iter(agg))
            group_names = [f"g{i}" for i in range(len(sample_key))]
            metrics = list(next(iter(agg.values())).keys())
        blocks.append(_latex_table(agg, group_names, metrics, caption, label))

    out_path.write_text("\n\n".join(blocks))
    print(f"\n  LaTeX written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report μ ± σ across seeds from RCD experiment CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--roots", nargs="+",
        default=["output/s42", "output/s43", "output/s44"],
        help="Root output directories, one per seed.",
    )
    p.add_argument(
        "--latex", action="store_true",
        help="Write a .tex summary alongside this script.",
    )
    p.add_argument(
        "--experiments", nargs="+",
        default=["alpha", "beta", "gamma", "delta",
                 "epsilon", "zeta", "mu", "theta",
                 "cold_latent", "twosample"],
        help="Which experiments to include.",
    )
    return p.parse_args()


_REPORTERS = {
    "alpha":       report_alpha,
    "beta":        report_beta,
    "gamma":       report_gamma,
    "delta":       report_delta,
    "epsilon":     report_epsilon,
    "zeta":        report_zeta,
    "mu":          report_mu,
    "theta":       report_theta,
    "cold_latent": report_cold_latent,
    "twosample":   report_twosample,
}


def main() -> None:
    args = parse_args()
    roots = [Path(r) for r in args.roots]

    print(f"\nSeeds:  {[r.name for r in roots]}")
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(f"WARNING: these root directories do not exist: {missing}")

    all_agg: Dict[str, Dict] = {}
    for name in args.experiments:
        if name not in _REPORTERS:
            print(f"\nWARNING: unknown experiment '{name}', skipping.")
            continue
        try:
            all_agg[name] = _REPORTERS[name](roots, latex=args.latex)
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback; traceback.print_exc()

    if args.latex:
        out_path = Path(__file__).with_suffix(".tex")
        _write_latex(all_agg, out_path)

    print(f"\n{'─' * 100}")
    print("Done.")


if __name__ == "__main__":
    main()