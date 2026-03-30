#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make Figure 7 style plots: Daily energy vs PM10 (dual-axis) for one or multiple RW configs.

Inputs:
- experiments directory containing RW1..RW4/seed0..seed4
- yearly_results.csv per seed directory (produced by run_365day_sim.py)
  Expected columns (hourly): day_of_year, energy, pm10

Outputs:
- figure7_daily_energy_vs_pm10_<RW>.(png/pdf)

Design goals (MDPI-ready):
- No title inside the figure (use caption in manuscript instead)
- Clear axis labels and units
- Legend placed at upper-right without covering curves
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def _find_yearly_results_csv(seed_dir: Path) -> Path | None:
    p = seed_dir / "yearly_results.csv"
    if p.exists():
        return p

    candidates = list(seed_dir.rglob("yearly_results.csv"))
    if candidates:
        candidates.sort(key=lambda x: (len(x.parts), str(x)))
        return candidates[0]

    candidates = list(seed_dir.rglob("*yearly*results*.csv"))
    if candidates:
        candidates.sort(key=lambda x: (len(x.parts), str(x)))
        return candidates[0]

    return None


def _daily_series_from_yearly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean_cols(df)

    for col in ["day_of_year", "energy", "pm10"]:
        if col not in df.columns:
            raise ValueError(
                f"{path} missing column '{col}'. Found columns: {list(df.columns)}"
            )

    daily = (
        df.groupby("day_of_year", as_index=False)
          .agg(daily_energy_kwh=("energy", "sum"),
               pm10=("pm10", "mean"))
          .sort_values("day_of_year")
          .reset_index(drop=True)
    )
    return daily


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def _set_month_xticks(ax, days: np.ndarray):
    month_starts = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    month_ends = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    month_mid = (month_starts + month_ends) / 2.0
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if len(days) == 0:
        ax.set_xticks(month_mid)
        ax.set_xticklabels(month_labels)
        return

    dmin = float(np.nanmin(days))
    dmax = float(np.nanmax(days))
    mask = (month_mid >= dmin) & (month_mid <= dmax)
    ax.set_xticks(month_mid[mask])
    ax.set_xticklabels([m for m, keep in zip(month_labels, mask) if keep])


def _discover_seed_dirs(experiments_dir: Path, rw: str, include_test: bool = False) -> list[Path]:
    rw_dir = experiments_dir / rw
    if not rw_dir.exists():
        return []

    seed_dirs = []
    for p in rw_dir.rglob("*"):
        if not p.is_dir():
            continue
        if re.fullmatch(r"seed[_-]?\d+", p.name, flags=re.IGNORECASE):
            if not include_test and "test" in str(p).lower():
                continue
            seed_dirs.append(p)

    seed_dirs = sorted(seed_dirs, key=lambda x: (len(x.parts), str(x)))

    # deduplicate by seed folder name, prefer shallower path
    best = {}
    for p in seed_dirs:
        key = p.name.lower()
        if key not in best:
            best[key] = p
    return [best[k] for k in sorted(best.keys(), key=lambda s: int(re.findall(r'\d+', s)[0]))]


def plot_one_rw(
    rw: str,
    seed_run_dirs: list[Path],
    out_dir: Path,
    dpi: int,
    formats: list[str],
    ma_window: int,
    figsize: tuple[float, float],
    font_family: str,
    base_fontsize: int,
    strict: bool,
    x_axis: str = "month",
):
    seed_dailies = []
    used_seeds = 0

    for run_dir in seed_run_dirs:
        yr = _find_yearly_results_csv(run_dir)
        if yr is None:
            msg = f"[WARN] Missing yearly_results.csv under: {run_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        daily = _daily_series_from_yearly(yr)
        used_seeds += 1
        seed_dailies.append(daily)

    if used_seeds == 0:
        msg = f"[WARN] No usable seeds for {rw}. Skipping."
        if strict:
            raise RuntimeError(msg)
        print(msg)
        return

    days = seed_dailies[0]["day_of_year"].to_numpy()
    energy_stack = np.vstack([d["daily_energy_kwh"].to_numpy() for d in seed_dailies])
    pm10_stack = np.vstack([d["pm10"].to_numpy() for d in seed_dailies])

    energy_mean = energy_stack.mean(axis=0)
    pm10_mean = pm10_stack.mean(axis=0)

    energy_ma = _moving_average(energy_mean, ma_window)
    pm10_ma = _moving_average(pm10_mean, ma_window)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family, "Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"] = base_fontsize

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    energy_color = "#1f77b4"
    pm10_color = "#ff7f0e"

    l1, = ax1.plot(
        days,
        energy_ma,
        color=energy_color,
        linewidth=2.4,
        label=f"Daily energy ({rw}, MA{ma_window})",
    )
    l2, = ax2.plot(
        days,
        pm10_ma,
        color=pm10_color,
        linestyle="--",
        linewidth=2.2,
        label=f"PM10 (MA{ma_window})",
    )

    if str(x_axis).lower() == "month":
        _set_month_xticks(ax1, days)
        ax1.set_xlabel("Month", fontsize=10, fontweight="bold")
    else:
        ax1.set_xlabel("Day of year", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Daily energy consumption (kWh)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("PM10 concentration (µg/m³)", fontsize=10, fontweight="bold")

    ax1.tick_params(axis="both", labelsize=8)
    ax2.tick_params(axis="both", labelsize=8)
    ax1.grid(True, alpha=0.3)

    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(
        lines, labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=8
    )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure7_daily_energy_vs_pm10_{rw}"
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {stem} (n={used_seeds} seed(s)) to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", required=True, help="Path to experiments root, e.g. E:/Enhanced_DQN-RW-Experiment/experiments")
    ap.add_argument("--out", default="figures_mdpi", help="Output folder")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--formats", default="png,pdf", help="Comma-separated, e.g., png,pdf")
    ap.add_argument("--ma_window", type=int, default=7)
    ap.add_argument(
        "--x_axis",
        default="month",
        choices=["month", "day"],
        help="X-axis labeling: 'month' (Jan–Dec ticks) or 'day' (day-of-year)",
    )
    ap.add_argument("--rws", default="RW1,RW2,RW3,RW4", help="Comma-separated list, e.g., RW1,RW2,RW3,RW4")
    ap.add_argument("--include_test", action="store_true",
                    help="Include folders containing 'test' (default: excluded)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any required yearly_results.csv is missing (default: skip with warning)")
    ap.add_argument("--figwidth", type=float, default=8.0)
    ap.add_argument("--figheight", type=float, default=3.8)
    ap.add_argument("--font_family", default="Arial")
    ap.add_argument("--base_fontsize", type=int, default=8)
    args = ap.parse_args()

    experiments_dir = Path(args.experiments_dir).resolve()
    out_dir = Path(args.out)
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    rws = [x.strip() for x in args.rws.split(",") if x.strip()]

    for rw in rws:
        seed_dirs = _discover_seed_dirs(experiments_dir, rw, include_test=args.include_test)
        if not seed_dirs:
            print(f"[WARN] RW '{rw}' not found or no seed folders found under {experiments_dir}.")
            continue

        plot_one_rw(
            rw=rw,
            seed_run_dirs=seed_dirs,
            out_dir=out_dir,
            dpi=args.dpi,
            formats=formats,
            ma_window=args.ma_window,
            figsize=(args.figwidth, args.figheight),
            font_family=args.font_family,
            base_fontsize=args.base_fontsize,
            strict=args.strict,
            x_axis=args.x_axis,
        )


if __name__ == "__main__":
    main()
