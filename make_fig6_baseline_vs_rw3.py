#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MDPI-ready Figure 6 generator (Split-type AC annual energy)

Generates:
  (A) annual_energy_baseline_vs_rw3.(png/pdf): Baseline vs RW3 (no error bars)
  (B) annual_energy_baseline_vs_rw1_rw4.(png/pdf): Baseline vs RW1–RW4 (mean ± std across seeds)
  (C) table2_annual_energy_summary.csv: compact annual energy summary table

Expected input: reports/results_summary.csv produced by your pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Styling helpers
# -------------------------

def set_mdpi_style(font_family: str = "DejaVu Sans") -> None:
    """A conservative, MDPI-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str, formats: List[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


# -------------------------
# Data parsing
# -------------------------

def read_results_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Strip BOM + whitespace
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # Normalize key columns
    for col in ["rw", "run_dir"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Best-effort numeric conversion
    for col in [
        "seed",
        "baseline_energy",
        "ai_energy",
        "episodes_logged",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop obviously non-RW rows (keep rows where rw starts with 'RW')
    if "rw" not in df.columns:
        raise ValueError("results_summary.csv missing required column: 'rw'")
    df = df[df["rw"].str.startswith("RW", na=False)].copy()

    # Remove test/debug runs if present
    if "run_dir" in df.columns:
        df = df[~df["run_dir"].str.contains(r"test|debug", case=False, na=False)].copy()

    # If duplicates exist for same (rw, seed), keep the most-complete one (largest episodes_logged)
    if "seed" in df.columns:
        if "episodes_logged" in df.columns:
            df = df.sort_values(["rw", "seed", "episodes_logged"], ascending=[True, True, True])
        else:
            df = df.sort_values(["rw", "seed"], ascending=[True, True])
        df = df.drop_duplicates(subset=["rw", "seed"], keep="last").copy()

    # Sanity: ensure energy columns exist
    for col in ["baseline_energy", "ai_energy"]:
        if col not in df.columns:
            raise ValueError(f"results_summary.csv missing required column: '{col}'")

    return df


def compute_baseline_energy(df: pd.DataFrame) -> float:
    baseline_vals = df["baseline_energy"].dropna().astype(float)
    if baseline_vals.empty:
        raise ValueError("baseline_energy column is empty; cannot compute baseline.")
    return float(baseline_vals.mean())


def compute_rw_stats(df: pd.DataFrame, rw: str) -> Dict[str, float]:
    sub = df[df["rw"] == rw].copy()
    energies = sub["ai_energy"].dropna().astype(float).values
    if energies.size == 0:
        raise ValueError(f"No ai_energy values found for {rw}.")

    n = int(energies.size)
    mean = float(np.mean(energies))
    std = float(np.std(energies, ddof=1)) if n >= 2 else 0.0

    return {"rw": rw, "n": n, "mean": mean, "std": std}


def compute_all_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for rw in ["RW1", "RW2", "RW3", "RW4"]:
        if (df["rw"] == rw).any():
            stats[rw] = compute_rw_stats(df, rw)
    if "RW3" not in stats:
        raise ValueError("RW3 not found in results_summary.csv (after filtering).")
    return stats


# -------------------------
# Plotting
# -------------------------

def annotate_bars(ax: plt.Axes, bars, values: List[float], y_offsets: List[float] | None = None, fmt: str = "{:.2f}"):
    if y_offsets is None:
        y_offsets = [0.0] * len(values)
    for bar, val, off in zip(bars, values, y_offsets):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + off,
            fmt.format(val),
            ha="center",
            va="bottom",
        )


def fig_baseline_vs_rw3(baseline_energy: float, rw3: Dict[str, float], out_dir: Path, formats: List[str], dpi: int) -> Tuple[float, float]:
    labels = ["Baseline (25°C)", "Enhanced DQN (RW3)\n(mean over seeds)"]
    values = [baseline_energy, rw3["mean"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        labels,
        values,
        color=["#9e9e9e", "#1f77b4"],
        edgecolor="black",
        linewidth=0.8,
    )

    # Place numbers outside (above) bars with 2 decimals
    ylim_top = max(values) * 1.12
    ax.set_ylim(0, ylim_top)
    offset = ylim_top * 0.015
    annotate_bars(ax, bars, values, y_offsets=[offset, offset], fmt="{:.2f}")

    # ax.set_title("Annual Energy Consumption: Baseline vs Enhanced DQN (RW3)")
    ax.set_ylabel("Annual Energy (kWh)")

    save_fig(fig, out_dir, "annual_energy_baseline_vs_rw3", formats, dpi)
    plt.close(fig)

    saving_kwh = baseline_energy - rw3["mean"]
    saving_pct = (saving_kwh / baseline_energy) * 100.0
    return saving_kwh, saving_pct


def fig_baseline_vs_rw1_rw4(baseline_energy: float, stats: Dict[str, Dict[str, float]], out_dir: Path, formats: List[str], dpi: int) -> None:
    labels = ["Baseline (25°C)"]
    means = [baseline_energy]
    stds = [0.0]

    for rw in ["RW1", "RW2", "RW3", "RW4"]:
        if rw in stats:
            labels.append(rw)
            means.append(stats[rw]["mean"])
            stds.append(stats[rw]["std"])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Baseline gray; RW colors from tab10
    colors = ["#9e9e9e"] + [plt.cm.tab10(i) for i in range(1, len(labels))]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.set_title("Annual Energy Consumption: Baseline vs Reward Weight Configurations")
    ax.set_ylabel("Annual Energy (kWh)")

    # Ensure annotations do not overlap error bars
    ylim_top = (max(np.array(means) + np.array(stds))) * 1.15
    ax.set_ylim(0, ylim_top)
    base_offset = ylim_top * 0.012
    y_offsets = [base_offset + s for s in stds]
    annotate_bars(ax, bars, means, y_offsets=y_offsets, fmt="{:.2f}")

    save_fig(fig, out_dir, "annual_energy_baseline_vs_rw1_rw4", formats, dpi)
    plt.close(fig)


def write_table2(baseline_energy: float, stats: Dict[str, Dict[str, float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(
        {
            "Control": "Baseline (25°C)",
            "Annual Energy (kWh)": baseline_energy,
            "Std (kWh)": 0.0,
            "n (seeds)": "-",
            "Saving (kWh)": np.nan,
            "Saving (%)": np.nan,
        }
    )

    for rw in ["RW1", "RW2", "RW3", "RW4"]:
        if rw not in stats:
            continue
        mean = stats[rw]["mean"]
        std = stats[rw]["std"]
        n = int(stats[rw]["n"])
        saving_kwh = baseline_energy - mean
        saving_pct = (saving_kwh / baseline_energy) * 100.0
        rows.append(
            {
                "Control": f"Enhanced DQN ({rw})",
                "Annual Energy (kWh)": mean,
                "Std (kWh)": std,
                "n (seeds)": n,
                "Saving (kWh)": saving_kwh,
                "Saving (%)": saving_pct,
            }
        )

    t2 = pd.DataFrame(rows)
    # Format for readability in CSV consumers (Word/Excel): keep numeric columns numeric, not strings
    out_path = out_dir / "table2_annual_energy_summary.csv"
    t2.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MDPI-ready Figure 6 (annual energy) plots + Table 2.")
    ap.add_argument("--results_summary", required=True, help="Path to reports/results_summary.csv")
    ap.add_argument("--out", default="figures_mdpi", help="Output folder (will be created if missing)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG export")
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated formats, e.g., png,pdf")
    ap.add_argument("--font_family", default="DejaVu Sans", help="Font family (optional)")
    args = ap.parse_args()

    set_mdpi_style(args.font_family)

    rs_path = Path(args.results_summary)
    out_dir = Path(args.out)
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]

    df = read_results_summary(rs_path)
    baseline_energy = compute_baseline_energy(df)
    stats = compute_all_stats(df)

    # Figure A: baseline vs RW3 (no error bars)
    saving_kwh, saving_pct = fig_baseline_vs_rw3(baseline_energy, stats["RW3"], out_dir, formats, args.dpi)

    # Figure B: baseline vs RW1–RW4 (mean ± std)
    fig_baseline_vs_rw1_rw4(baseline_energy, stats, out_dir, formats, args.dpi)

    # Table 2
    t2_path = write_table2(baseline_energy, stats, out_dir)

    # Console summary
    print("Figure 6 generated:")
    print(f"- Baseline (25°C): {baseline_energy:.2f} kWh")
    print(f"- Enhanced DQN (RW3) mean: {stats['RW3']['mean']:.2f} kWh (n={stats['RW3']['n']} seeds)")
    print(f"- Saving: {saving_kwh:.2f} kWh ({saving_pct:.2f}%)")
    print(f"- Output directory: {out_dir.resolve()}")
    print(f"- Table saved: {t2_path.resolve()}")


if __name__ == "__main__":
    main()
