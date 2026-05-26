#!/usr/bin/env python3
"""
Robust Experiment Utilities
===========================
Unified configuration-driven exporters for CSV, LaTeX, and console tables.
"""

from __future__ import annotations

import csv
import math
import sys
import json
import yaml
import shutil
import logging
import datetime
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Callable

import torch
import matplotlib.pyplot as plt

from rcd.data.config import Config
from rcd.experiments.registry import LatexTableSpec


# ─────────────────────────────────────────────────────────────────────────────
# 1. Run Context
# ─────────────────────────────────────────────────────────────────────────────

class RunContext:
    """
    Deterministically manages the output directory structure for an experiment run.
    Enforces checkpoint, logs, and metric separation.
    """
    def __init__(self, cfg: Config, family: str, run_name: str, base_dir: str | Path | None = None):
        self.cfg = cfg
        self.family = family
        self.run_name = run_name
        self.base_dir = Path(base_dir) if base_dir is not None else Path(cfg.save_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / self.family
        
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metric_dir = self.run_dir / "metrics"
        self.plot_dir = self.run_dir / "plots"
        self.sample_dir = self.run_dir / "samples"
        self.log_path = self.run_dir / f"run_{self.run_name}_{timestamp}.log"
        
        self._logger: logging.Logger | None = None
        self._original_save_dir: Path | None = None

    def __enter__(self) -> RunContext:
        for d in (self.ckpt_dir, self.metric_dir, self.plot_dir, self.sample_dir):
            d.mkdir(parents=True, exist_ok=True)
            
        self._original_save_dir = Path(self.cfg.save_dir)
        self.cfg.run_dir = self.run_dir
        self.cfg.ckpt_dir = self.ckpt_dir
        self.cfg.metric_dir = self.metric_dir
        self.cfg.plot_dir = self.plot_dir
        self.cfg.sample_dir = self.sample_dir
        self.cfg.save_dir = self.ckpt_dir  # Backward compatibility

        # Save config manifest
        cfg_dict = asdict(self.cfg) if is_dataclass(self.cfg) else dict(self.cfg)
        cfg_dict.update({
            "device": str(cfg_dict.get("device", "cpu")),
            "save_dir": str(self.cfg.save_dir),
            "run_dir": str(self.run_dir),
            "family": self.family,
            "run_name": self.run_name,
        })
            
        with open(self.run_dir / "manifest.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
            
        self._setup_logging()
        self.logger.info(f"Started run: {self.run_name} in {self.run_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._organize_artifacts()
        if exc_type is None:
            self.logger.info("Run completed successfully.")
        else:
            self.logger.error(f"Run failed with exception: {exc_val}")
        
        if self._logger:
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)
        if self._original_save_dir is not None:
            self.cfg.save_dir = self._original_save_dir

    def _setup_logging(self) -> None:
        self._logger = logging.getLogger(f"{self.family}.{self.run_name}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        for handler in (logging.FileHandler(self.log_path), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("RunContext not entered.")
        return self._logger

    def get_path(self, category: str, *parts: str) -> Path:
        dirs = {"ckpt": self.ckpt_dir, "metric": self.metric_dir, "plot": self.plot_dir, "sample": self.sample_dir}
        return dirs[category].joinpath(*parts)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.get_path("metric", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def _organize_artifacts(self) -> None:
        if not self.ckpt_dir.exists(): return
        ext_map = {
            ".csv": self.metric_dir, ".json": self.metric_dir, ".jsonl": self.metric_dir, ".tex": self.metric_dir, ".txt": self.metric_dir,
            ".pdf": self.plot_dir, ".png": self.plot_dir, ".jpg": self.plot_dir, ".jpeg": self.plot_dir, ".svg": self.plot_dir
        }
        sample_words = ("sample", "samples", "restoration", "grid")

        for path in sorted(self.ckpt_dir.rglob("*")):
            if not path.is_file() or path.suffix == ".pt": continue
            
            dest_dir = ext_map.get(path.suffix.lower())
            if not dest_dir: continue
            
            if dest_dir == self.plot_dir and any(w in path.name.lower() for w in sample_words):
                dest_dir = self.sample_dir
                
            dest = dest_dir / path.relative_to(self.ckpt_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Universal Data Formatters & Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 3, sign: bool = False) -> str:
    """Formats floats consistently, handling NaNs."""
    if math.isnan(v) or v is None:
        return "—"
    fmt_str = f"{{:{'+' if sign else ''}.{decimals}f}}"
    return fmt_str.format(v)

def _extract_row_data(row: Any) -> dict:
    """Normalizes a dataclass or object into a dictionary."""
    if is_dataclass(row): return asdict(row)
    if isinstance(row, dict): return row
    return vars(row)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Robust CSV Exporter
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(data: Iterable[Any] | dict, path: Path, float_decimals: int = 5, trace_key: str = "layer_key", silent: bool = False) -> None:
    """
    A unified function to save ANY list of dataclasses, dicts, OR nested traces to CSV.
    """
    if not data: return
    
    # Handle nested trace dictionaries (e.g. from Experiment γ)
    if isinstance(data, dict):
        rows_list = []
        for i, (key, sub_dict) in enumerate(data.items()):
            row = {trace_key: key, "depth": i}
            row.update(sub_dict)
            rows_list.append(row)
    else:
        rows_list = [_extract_row_data(r) for r in data]

    fields = list(rows_list[0].keys())

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_list:
            w.writerow({k: round(v, float_decimals) if isinstance(v, float) else v for k, v in row.items()})
    
    if not silent:
        print(f"  → CSV saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuration-Driven LaTeX Exporter
# ─────────────────────────────────────────────────────────────────────────────

LATEX_SPECS = {
    "cumulants": LatexTableSpec(
        caption=r"Cumulants $\kappa_3$, $\kappa_4$, Participation Ratio (PR), effective rank (ER), covariance whiteness, and Mardia kurtosis at each activation stage.",
        label="tab:cumulants",
        col_spec="ll r r r r r r r r",
        headers=["Model", "Stage", r"$\overline{|\kappa_3|}$", r"$\overline{\kappa_4}$", r"$\%|\kappa_4|{>}0.5$", "PR", "ER", "White", r"$b_{2,p}$", r"$b_{2,p}^*$"],
        group_by="model",
        row_fmt=lambda r: [str(r.get('model', '')), str(r.get('stage', '')), _fmt(r.get('mean_abs_k3', 0),3), _fmt(r.get('mean_k4', 0),3,True), fr"{r.get('frac_nong', 0)*100:.1f}\%", _fmt(r.get('pr', 0),1), _fmt(r.get('effective_rank', 0),1), _fmt(r.get('whiteness', 0),3), _fmt(r.get('mardia_b2p', 0),1), _fmt(r.get('mardia_b2p_exp', 0),1)],
        footer=r"\multicolumn{10}{l}{\footnotesize ER = effective rank; White = \|C-\mathrm{diag}(C)\|_F/\|C\|_F$.}"
    ),
    "beta": LatexTableSpec(
        caption=r"Experiment $\beta$: bottleneck width vs Gaussianization. PR = Participation Ratio; Rig = Huber loss after perturbation.",
        label="tab:beta",
        col_spec="ll r r r r r r r r r",
        headers=["Noise", r"$\alpha_{\rm bf}$", "$C$", r"$\kappa_4^{\rm in}$", r"$\kappa_4^{\rm bn}$", r"$\kappa_4^{\rm out}$", "PR", "White", "Mardia-$Z$", r"Rig-$\mathcal{N}$", "Rig-Ros"],
        group_by="noise_type",
        row_fmt=lambda r: [str(r.get('noise_type', '')), _fmt(r.get('bottleneck_factor', 0),2), str(r.get('bneck_ch', 0)), _fmt(r.get('mean_k4_input', 0),3,True), _fmt(r.get('mean_k4_bneck', 0),3,True), _fmt(r.get('mean_k4_x0hat', 0),3,True), _fmt(r.get('pr_bneck', 0),1), _fmt(r.get('whiteness_bneck', 0),3), _fmt(r.get('mardia_b2p_z', 0),2,True), _fmt(r.get('perturb_gauss_huber', 0),4), _fmt(r.get('perturb_rosenblatt_huber', 0),4)]
    ),
    "omicron": LatexTableSpec(
        caption=r"Experiment~$\omicron$: optimiser comparison.",
        label="tab:omicron",
        col_spec="ll rr rr rr",
        headers=["Noise", "Optimiser", r"$\bar{\kappa}_4^{\rm bn}$", "PR", "Mardia-$Z$", "$L_1$", "Sharp", r"$\kappa_4^{\rm upd}$"],
        group_by="noise_type",
        row_fmt=lambda r: [str(r.get('noise_type', '')), str(r.get('label', '')).split('(')[0].strip(), _fmt(r.get('bn_kappa4', 0),3,True), _fmt(r.get('bn_pr', 0),1), _fmt(r.get('bn_mardia_z', 0),2,True), _fmt(r.get('val_l1', 0),4), _fmt(r.get('sharpness', 0),4), _fmt(r.get('update_k4', 0),3,True)]
    )
}



def save_latex_table(rows: Iterable[Any], path: Path, spec_name: str) -> None:
    """Unified engine to generate LaTeX tables based on predefined specifications."""
    if spec_name not in LATEX_SPECS:
        raise ValueError(f"Unknown LaTeX spec: {spec_name}")
    
    spec = LATEX_SPECS[spec_name]
    lines = [
        r"\begin{table}[ht]", r"\centering", spec.size,
        f"\\caption{{{spec.caption}}}", f"\\label{{{spec.label}}}",
        f"\\begin{{tabular}}{{{spec.col_spec}}}", r"\toprule",
        " & ".join(spec.headers) + r" \\", r"\midrule"
    ]
    
    prev_group = None
    for r_raw in rows:
        r_dict = _extract_row_data(r_raw)
        
        # Handle midrule grouping
        if spec.group_by and prev_group is not None and r_dict.get(spec.group_by) != prev_group:
            lines.append(r"\midrule")
        prev_group = r_dict.get(spec.group_by)
        
        lines.append(" & ".join(spec.row_fmt(r_dict)) + r" \\")
        
    lines.append(r"\bottomrule")
    if spec.footer: lines.append(spec.footer)
    lines.extend([r"\end{tabular}", r"\end{table}"])
    
    path.write_text("\n".join(lines))
    print(f"  → LaTeX saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Generic Console Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_table(rows: Iterable[Any], title: str, cols: list[tuple[str, str, Callable]], group_by: str | None = None) -> None:
    """
    Prints a dynamically formatted console table.
    cols: list of (Header, format_string_for_header, row_extractor_callable)
    """
    print(f"\n── {title} " + "─" * (80 - len(title) - 4))
    
    # Build header
    hdr = "  ".join(f"{{:>{fmt}}}".format(name).replace('>', fmt[0] if fmt[0] in '<>' else '>') 
                    for name, fmt, _ in cols)
    print(hdr)
    print("─" * len(hdr))
    
    prev_group = None
    for r_raw in rows:
        r = _extract_row_data(r_raw)
        if group_by and prev_group is not None and r.get(group_by) != prev_group:
            print()
        prev_group = r.get(group_by)
        
        row_str = "  ".join(f"{{:{fmt}}}".format(ext(r)) for _, fmt, ext in cols)
        print(row_str)
    print("─" * len(hdr))


# ─────────────────────────────────────────────────────────────────────────────
# 6. PyTorch Image Exporter
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_latent_samples(model, ae, fwd, cfg, sample_noise_fn, tag: str = "", n_cls: int = 10, save_dir: str = ".") -> None:
    """
    Generates decoded samples (one per class) and saves as a grid image.
    """
    model.eval(); ae.eval()
    labels = torch.arange(n_cls, device=cfg.device)
    D = getattr(ae, 'LATENT_DIM', 64)
    sched = torch.linspace(1., 0., 50 + 1, device=cfg.device)
    z = sample_noise_fn(fwd.noise_type, (n_cls, D), fwd.lam_t, fwd.M_eig, cfg.device) * fwd.sigma_max

    null = torch.full_like(labels, 10)
    
    for k in range(50):
        tc, tn = sched[k].expand(n_cls), sched[k+1].expand(n_cls)
        sig = fwd.sigma_t(tc).unsqueeze(1)
        cin = (1. + sig**2).pow(-0.5)
        
        z0c = model(z * cin, tc, labels)
        z0u = model(z * cin, tc, null)
        z0h = z0u + cfg.cfg_scale * (z0c - z0u)
        
        if k < 49:
            sn = fwd.sigma_t(tn).unsqueeze(1)
            z = z0h + sn * sample_noise_fn(fwd.noise_type, (n_cls, D), fwd.lam_t, fwd.M_eig, cfg.device)
        else:
            z = z0h

    imgs = ((ae.decode(z) + 1.) / 2.).clamp(0., 1.).cpu()

    fig, axes = plt.subplots(1, n_cls, figsize=(2. * n_cls, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(i), fontsize=7, rotation=45, ha="right")
        ax.axis("off")
        
    plt.suptitle(f"Latent samples — {tag}", fontsize=9)
    plt.tight_layout()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fp = f"{save_dir}/{tag}_samples.png"
    plt.savefig(fp, dpi=130)
    plt.close()
    print(f"  Saved {fp}")


def print_cumulant_table(rows: list) -> None:
    hdr = f"  {'Model':12s}  {'Stage':22s}  {'N':5s}  {'D':5s}  {'κ4':>7s}  {'Z':>7s}"
    print("\n" + "─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))
    prev = None
    for r in rows:
        if prev is not None and r.model_name != prev:
            print()
        prev = r.model_name
        print(f"  {r.model_name:12s}  {r.label:22s}"
              f"  {getattr(r,'N',0):5d}  {getattr(r,'D',0):5d}"
              f"  {getattr(r,'mean_k4',float('nan')):+7.3f}"
              f"  {getattr(r,'mardia_b2p_z',float('nan')):+7.2f}")
    print("─" * len(hdr))


def save_cumulant_csv(rows: list, path: Path) -> None:
    save_csv([r.flatten() for r in rows], path)

