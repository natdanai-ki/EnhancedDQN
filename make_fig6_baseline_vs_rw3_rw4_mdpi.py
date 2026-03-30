#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Figure 6.3: Baseline vs RW3 and RW4.

Supports two input modes:
1) --results_summary reports/results_summary.csv
2) --experiments_dir E:/Enhanced_DQN-RW-Experiment/experiments

In experiments directory mode, the script searches:
- experiments/RW3/seed*/summary.txt
- experiments/RW4/seed*/summary.txt

If summary.txt is unavailable, it tries to infer annual energy by summing energy-like columns
from CSV files inside each seed folder.

Outputs:
- annual_energy_baseline_vs_rw3_rw4.png/pdf
- table2_annual_energy_summary.csv
- rw3_seed_energy_details.csv
- rw4_seed_energy_details.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_mdpi_style(font_family: str = "Times New Roman") -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 10,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "savefig.pad_inches": 0.03,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, formats: List[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if fmt:
            out_path = out_dir / f"{stem}.{fmt}"
            if fmt in {"pdf", "eps", "svg"}:
                fig.savefig(out_path, bbox_inches="tight")
            else:
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def _format_sd_for_table(x: float) -> str:
    if pd.isna(x) or abs(float(x)) < 1e-9:
        return "0.00"
    return f"{float(x):.6f}"


# -------------------------
# Mode 1: results_summary.csv
# -------------------------

def read_results_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    for col in ["rw", "run_dir"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["seed", "baseline_energy", "ai_energy", "episodes_logged"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "rw" not in df.columns:
        raise ValueError("results_summary.csv missing required column: 'rw'")

    df = df[df["rw"].str.startswith("RW", na=False)].copy()

    if "run_dir" in df.columns:
        df = df[~df["run_dir"].str.contains(r"test|debug", case=False, na=False)].copy()

    if "seed" in df.columns:
        sort_cols = [c for c in ["rw", "seed", "episodes_logged"] if c in df.columns]
        df = df.sort_values(sort_cols).drop_duplicates(subset=["rw", "seed"], keep="last")

    for col in ["baseline_energy", "ai_energy"]:
        if col not in df.columns:
            raise ValueError(f"results_summary.csv missing required column: '{col}'")

    return df


def stats_for_rw_from_results_summary(df: pd.DataFrame, rw: str) -> Tuple[float, pd.DataFrame, Dict[str, float]]:
    sub = df[df["rw"] == rw].copy()
    if sub.empty:
        raise ValueError(f"{rw} not found in results_summary.csv")

    baseline_energy = float(sub["baseline_energy"].dropna().astype(float).mean())
    seed_df = sub[[c for c in ["seed", "baseline_energy", "ai_energy", "run_dir"] if c in sub.columns]].copy()
    if "seed" not in seed_df.columns:
        seed_df.insert(0, "seed", range(len(seed_df)))
    seed_df["seed"] = seed_df["seed"].astype(int)
    seed_df = seed_df.sort_values("seed").reset_index(drop=True)

    energies = seed_df["ai_energy"].astype(float).to_numpy()
    stats = {
        "rw": rw,
        "n": int(len(energies)),
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies, ddof=1)) if len(energies) >= 2 else 0.0,
    }
    return baseline_energy, seed_df, stats


# -------------------------
# Mode 2: experiments/RW*/seed folders
# -------------------------

def parse_summary_txt(summary_path: Path) -> Tuple[Optional[float], Optional[float]]:
    txt = summary_path.read_text(encoding="utf-8", errors="ignore")
    base = None
    ai = None
    patterns_base = [
        r"Baseline\s*\(.*?\)\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Baseline\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    patterns_ai = [
        r"AI\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)",
        r"RL\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for p in patterns_base:
        m = re.search(p, txt, flags=re.I)
        if m:
            base = float(m.group(1))
            break
    for p in patterns_ai:
        m = re.search(p, txt, flags=re.I)
        if m:
            ai = float(m.group(1))
            break
    return base, ai


ENERGY_CANDIDATES = [
    "energy_kwh", "energy", "energy_used", "power_kwh", "cooling_energy_kwh",
    "step_energy_kwh", "hourly_energy_kwh", "kwh"
]


def sum_energy_from_csv(csv_path: Path) -> Optional[float]:
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in ENERGY_CANDIDATES:
        if cand in cols:
            real_col = cols[cand]
            try:
                full = pd.read_csv(csv_path, usecols=[real_col])
                return float(pd.to_numeric(full[real_col], errors="coerce").fillna(0).sum())
            except Exception:
                return None
    return None


def infer_seed_id(path: Path) -> Optional[int]:
    for part in [path.name] + [p.name for p in path.parents]:
        m = re.fullmatch(r"seed[_-]?(\d+)", part, flags=re.I)
        if m:
            return int(m.group(1))
    return None


def find_seed_records_for_rw(rw_dir: Path) -> pd.DataFrame:
    records = []
    seed_dirs = [p for p in rw_dir.rglob("*") if p.is_dir() and re.fullmatch(r"seed[_-]?\d+", p.name, flags=re.I)]
    for seed_dir in sorted(seed_dirs):
        seed_id = infer_seed_id(seed_dir)
        baseline_energy = None
        ai_energy = None
        evidence = None

        summary_files = list(seed_dir.rglob("summary.txt"))
        for sf in summary_files:
            base, ai = parse_summary_txt(sf)
            if base is not None and ai is not None:
                baseline_energy = base
                ai_energy = ai
                evidence = str(sf)
                break

        if baseline_energy is None or ai_energy is None:
            csv_files = list(seed_dir.rglob("*.csv"))
            ai_candidates = []
            base_candidates = []
            for cf in csv_files:
                val = sum_energy_from_csv(cf)
                if val is None:
                    continue
                low = cf.name.lower()
                if "baseline" in low:
                    base_candidates.append((cf, val))
                elif any(k in low for k in ["ai", "agent", "rl", "dqn", "yearly_results"]):
                    ai_candidates.append((cf, val))
            if base_candidates and ai_candidates:
                baseline_energy = base_candidates[0][1]
                ai_energy = ai_candidates[0][1]
                evidence = f"{base_candidates[0][0]} | {ai_candidates[0][0]}"

        if seed_id is not None and baseline_energy is not None and ai_energy is not None:
            records.append(
                {
                    "seed": seed_id,
                    "baseline_energy": baseline_energy,
                    "ai_energy": ai_energy,
                    "source": evidence,
                }
            )

    if not records:
        raise ValueError(
            f"No valid seed results found under: {rw_dir}\n"
            "Expected files like seed0/.../summary.txt containing Baseline Energy and AI Energy."
        )

    return pd.DataFrame(records).sort_values("seed").drop_duplicates(subset=["seed"], keep="last").reset_index(drop=True)


def stats_for_rw_from_experiments(experiments_dir: Path, rw: str) -> Tuple[float, pd.DataFrame, Dict[str, float]]:
    rw_dir = experiments_dir / rw
    if not rw_dir.exists():
        raise ValueError(f"{rw_dir} not found")

    seed_df = find_seed_records_for_rw(rw_dir)
    baseline_energy = float(seed_df["baseline_energy"].astype(float).mean())
    energies = seed_df["ai_energy"].astype(float).to_numpy()
    stats = {
        "rw": rw,
        "n": int(len(energies)),
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies, ddof=1)) if len(energies) >= 2 else 0.0,
    }
    return baseline_energy, seed_df, stats


# -------------------------
# Combined helpers
# -------------------------

def load_rw_pair_from_results_summary(results_summary: Path, rws: List[str]) -> Tuple[float, Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    df = read_results_summary(results_summary)
    baseline_candidates = []
    seed_dfs: Dict[str, pd.DataFrame] = {}
    stats_map: Dict[str, Dict[str, float]] = {}
    for rw in rws:
        baseline_energy, seed_df, stats = stats_for_rw_from_results_summary(df, rw)
        baseline_candidates.append(baseline_energy)
        seed_dfs[rw] = seed_df
        stats_map[rw] = stats
    baseline_energy = float(np.mean(baseline_candidates))
    return baseline_energy, seed_dfs, stats_map


def load_rw_pair_from_experiments(experiments_dir: Path, rws: List[str]) -> Tuple[float, Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    baseline_candidates = []
    seed_dfs: Dict[str, pd.DataFrame] = {}
    stats_map: Dict[str, Dict[str, float]] = {}
    for rw in rws:
        baseline_energy, seed_df, stats = stats_for_rw_from_experiments(experiments_dir, rw)
        baseline_candidates.append(baseline_energy)
        seed_dfs[rw] = seed_df
        stats_map[rw] = stats
    baseline_energy = float(np.mean(baseline_candidates))
    return baseline_energy, seed_dfs, stats_map


# -------------------------
# Plotting + tables
# -------------------------

def fig_baseline_vs_rw3_rw4(baseline_energy: float, stats_map: Dict[str, Dict[str, float]], out_dir: Path, formats: List[str], dpi: int) -> None:
    labels = ["Baseline (25°C)"]
    values = [baseline_energy]
    errs = [0.0]
    colors = ["#9e9e9e"]

    for rw, color in [("RW3", "#1f77b4"), ("RW4", "#ff7f0e")]:
        if rw in stats_map:
            labels.append(f"Enhanced DQN ({rw})\n(mean ± std, n={stats_map[rw]['n']})")
            values.append(stats_map[rw]["mean"])
            errs.append(stats_map[rw]["std"])
            colors.append(color)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars = ax.bar(labels, values, yerr=errs, capsize=6, color=colors, edgecolor="black", linewidth=0.8)

    ylim_top = max(np.array(values) + np.array(errs)) * 1.16
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Annual Energy (kWh)")
    ax.set_xlabel("")

    for bar, val, err in zip(bars, values, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + ylim_top * 0.015,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    save_fig(fig, out_dir, "annual_energy_baseline_vs_rw3_rw4", formats, dpi)
    plt.close(fig)


def write_detail_csv(seed_df: pd.DataFrame, rw: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = seed_df.copy()
    detail["saving_kwh"] = detail["baseline_energy"] - detail["ai_energy"]
    detail["saving_pct"] = (detail["saving_kwh"] / detail["baseline_energy"]) * 100.0
    path = out_dir / f"{rw.lower()}_seed_energy_details.csv"
    detail.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_summary_table(baseline_energy: float, stats_map: Dict[str, Dict[str, float]], out_dir: Path) -> Path:
    rows = [
        {
            "Control": "Baseline (25°C)",
            "Annual Energy (kWh)": f"{baseline_energy:.2f}",
            "Std (kWh)": "0.00",
            "n (seeds)": "-",
            "Saving (kWh)": "",
            "Saving (%)": "",
        }
    ]

    for rw in ["RW3", "RW4"]:
        if rw in stats_map:
            mean = stats_map[rw]["mean"]
            std = stats_map[rw]["std"]
            rows.append(
                {
                    "Control": f"Enhanced DQN ({rw})",
                    "Annual Energy (kWh)": f"{mean:.2f}",
                    "Std (kWh)": _format_sd_for_table(std),
                    "n (seeds)": int(stats_map[rw]["n"]),
                    "Saving (kWh)": f"{(baseline_energy - mean):.2f}",
                    "Saving (%)": f"{((baseline_energy - mean) / baseline_energy * 100.0):.6f}",
                }
            )

    summary = pd.DataFrame(rows)
    path = out_dir / "table2_annual_energy_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Figure 6.3 using Baseline vs RW3 and RW4.")
    ap.add_argument("--results_summary", help="Path to reports/results_summary.csv")
    ap.add_argument("--experiments_dir", help="Path to experiments root containing RW3 and RW4")
    ap.add_argument("--out", default="figures_mdpi", help="Output folder")
    ap.add_argument("--dpi", type=int, default=600, help="DPI for PNG export")
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated formats")
    ap.add_argument("--font_family", default="Times New Roman", help="Preferred font family")
    args = ap.parse_args()

    if not args.results_summary and not args.experiments_dir:
        raise SystemExit("Please provide either --results_summary or --experiments_dir")

    set_mdpi_style(args.font_family)
    out_dir = Path(args.out)
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    rws = ["RW3", "RW4"]

    if args.results_summary:
        baseline_energy, seed_dfs, stats_map = load_rw_pair_from_results_summary(Path(args.results_summary), rws)
    else:
        baseline_energy, seed_dfs, stats_map = load_rw_pair_from_experiments(Path(args.experiments_dir), rws)

    fig_baseline_vs_rw3_rw4(baseline_energy, stats_map, out_dir, formats, args.dpi)

    detail_paths = []
    for rw in rws:
        if rw in seed_dfs:
            detail_paths.append(write_detail_csv(seed_dfs[rw], rw, out_dir))

    summary_path = write_summary_table(baseline_energy, stats_map, out_dir)

    print("Figure 6.3 generated: Baseline vs RW3 and RW4")
    print(f"- Baseline mean: {baseline_energy:.2f} kWh")
    for rw in rws:
        s = stats_map.get(rw)
        if s:
            saving_kwh = baseline_energy - s["mean"]
            saving_pct = saving_kwh / baseline_energy * 100.0
            print(f"- {rw} mean ± std: {s['mean']:.2f} ± {s['std']:.2f} kWh (n={s['n']}) | Saving: {saving_kwh:.2f} kWh ({saving_pct:.2f}%)")
    print(f"- Figure output: {out_dir.resolve()}")
    for p in detail_paths:
        print(f"- Detail: {p.resolve()}")
    print(f"- Summary table: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
