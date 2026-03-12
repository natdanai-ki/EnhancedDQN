
"""
mdpi_plot_builder.py
Create manuscript-ready figures (PNG 300dpi + PDF vector) for RW experiments.

Inputs supported:
- results_summary.csv (aggregates per run: rw, seed, baseline_energy, ai_energy, saving_pct, ...)
- experiments folder containing per-run files:
    experiments/RW*/seed*/train_log.csv
    experiments/RW*/seed*/yearly_results.csv (optional but recommended)
    experiments/RW*/seed*/summary.txt (optional)

Figures produced:
1) Learning curve per RW (MA window) with seed lines + mean ± std.
2) Annual AI energy vs Baseline (bar mean±std + baseline horizontal line).
3) Annual saving% (bar mean±std).
4) Monthly AI energy profile per RW (mean±std across seeds) if yearly_results.csv exists.

No seaborn is used. Colors are left to Matplotlib defaults unless you modify them.

Example (Windows Anaconda Prompt):
    conda activate hvacrl
    chcp 65001
    python src\\mdpi_plot_builder.py --experiments experiments --results_summary reports\\results_summary.csv --out figures_mdpi

You can change fonts/sizes from CLI flags.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def infer_rw_seed(path: Path) -> Tuple[Optional[str], Optional[int]]:
    """Infer RW and seed from a path like .../experiments/RW3/seed0/train_log.csv"""
    rw = None
    seed = None
    for part in path.parts:
        if re.fullmatch(r"RW\d+", part, flags=re.IGNORECASE):
            rw = part.upper()
        if re.fullmatch(r"seed\d+", part, flags=re.IGNORECASE):
            seed = int(re.findall(r"\d+", part)[0])
    return rw, seed


def moving_avg(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return moving average and index positions (for episode alignment)."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr, np.arange(len(arr))
    ma = np.convolve(arr, np.ones(window) / window, mode="valid")
    idx = np.arange(window - 1, len(arr))
    return ma, idx


def day_to_month(day: int) -> int:
    # cumulative days (non-leap)
    edges = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    d0 = max(0, min(364, day - 1))
    return int(np.searchsorted(edges[1:], d0, side="right"))  # 0..11


MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# -----------------------------
# Discovery: find run folders
# -----------------------------
def discover_runs(experiments_root: Path) -> pd.DataFrame:
    train_logs = list(experiments_root.rglob("train_log.csv"))
    rows = []
    for tl in train_logs:
        rw, seed = infer_rw_seed(tl)
        if rw is None or seed is None:
            continue
        run_dir = tl.parent
        rows.append({
            "rw": rw,
            "seed": seed,
            "run_dir": str(run_dir),
            "train_log": str(tl),
            "yearly_results": str(run_dir / "yearly_results.csv") if (run_dir / "yearly_results.csv").exists() else None,
            "summary_txt": str(run_dir / "summary.txt") if (run_dir / "summary.txt").exists() else None,
            "config_json": str(run_dir / "config.json") if (run_dir / "config.json").exists() else None,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["rw","seed"]).reset_index(drop=True)


# -----------------------------
# Plotting functions
# -----------------------------
def apply_style(font_family: str, base_fontsize: int, title_size: int, label_size: int, tick_size: int, legend_size: int):
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })


def save_fig(fig, out_base: Path, formats: List[str], dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == "png":
            fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
        elif fmt == "pdf":
            fig.savefig(out_base.with_suffix(".pdf"))
        else:
            raise ValueError(f"Unsupported format: {fmt}")


def plot_learning_curves(runs_df: pd.DataFrame, out_dir: Path, ma_window: int, formats: List[str], dpi: int,
                         figwidth: float, figheight: float):
    out = []
    for rw in sorted(runs_df["rw"].unique()):
        sub = runs_df[runs_df["rw"] == rw].copy()
        cols = []
        for _, row in sub.iterrows():
            df = pd.read_csv(row["train_log"])
            df = df.sort_values("episode")
            ep = df["episode"].to_numpy()
            r = df["reward"].astype(float).to_numpy()

            r_ma, idx = moving_avg(r, ma_window)
            ep_ma = ep[idx]
            cols.append(pd.Series(r_ma, index=ep_ma, name=f"seed{int(row['seed'])}"))

        if not cols:
            continue

        mat = pd.concat(cols, axis=1, join="inner").astype(float)
        x = mat.index.to_numpy(dtype=float)
        mean = mat.mean(axis=1).to_numpy(dtype=float)
        std = mat.std(axis=1, ddof=1).to_numpy(dtype=float) if mat.shape[1] > 1 else np.zeros_like(mean)

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)

        for c in mat.columns:
            ax.plot(x, mat[c].to_numpy(dtype=float), linewidth=1, label=c)

        ax.plot(x, mean, linewidth=2, label="mean")
        if mat.shape[1] > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)

        ax.set_title(f"Learning Curve (Reward) — {rw} (MA{ma_window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Smoothed Total Reward")
        ax.legend()

        out_base = out_dir / f"learning_curve_{rw}_MA{ma_window}_mean_std"
        save_fig(fig, out_base, formats, dpi)
        plt.close(fig)
        out.append(out_base)
    return out




def plot_annual_bars(
    results_summary_csv: Path,
    out_dir: Path,
    formats: List[str],
    dpi: int,
    figwidth: float,
    figheight: float,
    mode: str = "monthly_avg",
) -> None:
    """
    Figure: Baseline vs RW1–RW4 energy (bar chart).

    - Adds a Baseline bar on x-axis.
    - Bars have distinct colors.
    - Error bars are shown ONLY when n_seeds >= 2 (std across seeds).
    - Bar labels show ONLY the energy value (no %), as requested.
    - mode:
        - "annual": use annual kWh
        - "monthly_avg": use (annual kWh)/12 to make differences more readable
    """
    df = pd.read_csv(results_summary_csv)

    # ---- Flexible column detection (to tolerate small schema differences)
    def _find_col(candidates: List[str]) -> Optional[str]:
        lower_map = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().strip()
            if key in lower_map:
                return lower_map[key]
        # try fuzzy contains
        for cand in candidates:
            key = cand.lower().strip()
            for c in df.columns:
                if key in c.lower().strip():
                    return c
        return None

    rw_col = _find_col(["rw", "rw_id", "reward_weight", "reward_weights", "rw_config"])
    ai_col = _find_col(["ai_energy", "annual_ai_energy", "total_ai_energy", "energy_ai"])
    baseline_col = _find_col(["baseline_energy", "annual_baseline_energy", "total_baseline_energy", "energy_baseline"])
    seed_col = _find_col(["seed", "random_seed"])

    if rw_col is None or ai_col is None or baseline_col is None:
        raise ValueError(
            "results_summary.csv schema not recognized. "
            f"Need columns for RW, ai_energy, baseline_energy. Found: {list(df.columns)}"
        )

    # numeric parsing + clean
    df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
    df[baseline_col] = pd.to_numeric(df[baseline_col], errors="coerce")

    # Baseline (deterministic, so std=0)
    baseline_vals = df[baseline_col].dropna().values
    if baseline_vals.size == 0:
        raise ValueError(
            "baseline_energy column exists but has no numeric values. "
            "Please ensure you have run baseline simulation and updated results_summary.csv."
        )
    baseline_annual = float(np.nanmean(baseline_vals))

    # AI per RW (mean/std across seeds)
    g = df.dropna(subset=[ai_col]).groupby(rw_col)[ai_col]
    rw_means = g.mean().to_dict()
    rw_stds = g.std(ddof=1).to_dict()
    rw_ns = g.count().to_dict()

    # Desired RW order
    rw_order = ["RW1", "RW2", "RW3", "RW4"]
    present_rws = [rw for rw in rw_order if rw in rw_means]

    # Build plot vectors (Baseline + RW1..RW4)
    labels = ["Baseline (25°C)"] + present_rws
    annual_means = [baseline_annual] + [float(rw_means[rw]) for rw in present_rws]
    annual_stds = [0.0] + [float(rw_stds.get(rw, np.nan)) for rw in present_rws]
    ns = [1] + [int(rw_ns.get(rw, 0)) for rw in present_rws]

    # Convert to monthly average if requested
    if mode.lower() in ["monthly", "monthly_avg", "avg_monthly", "monthlymean"]:
        means = [v / 12.0 for v in annual_means]
        stds = [v / 12.0 if np.isfinite(v) else np.nan for v in annual_stds]
        ylabel = "Average Monthly Energy (kWh/month)"
        fname = "annual_energy_ai_vs_baseline"  # keep filename for compatibility
        value_fmt = "{:,.1f}"
    else:
        means = annual_means
        stds = annual_stds
        ylabel = "Annual Energy (kWh)"
        fname = "annual_energy_ai_vs_baseline"
        value_fmt = "{:,.0f}"

    
    # ------------------------------------------------------------------
    # Export numerical summary for manuscript text/table (no error bars shown).
    # This CSV reports mean±std across seeds (std is NaN when only 1 seed).
    # ------------------------------------------------------------------
    try:
        baseline_annual = float(annual_means[0])
        stats_rows = []
        # number of seeds: baseline is deterministic (set to 1), RWx from group counts
        seed_counts = [1] + [rw_counts.get(rw, 0) for rw in present_rws]
        for lbl, a_mean, a_std, nseed in zip(labels, annual_means, annual_stds, seed_counts):
            a_mean = float(a_mean) if pd.notna(a_mean) else float('nan')
            a_std = float(a_std) if pd.notna(a_std) else float('nan')
            if lbl.lower().startswith("baseline"):
                saving_kwh = 0.0
                saving_pct = 0.0
            else:
                saving_kwh = baseline_annual - a_mean
                saving_pct = (saving_kwh / baseline_annual * 100.0) if baseline_annual else float('nan')
            stats_rows.append({
                "Control": lbl,
                "n_seeds": int(nseed) if nseed is not None else 0,
                "annual_mean_kwh": a_mean,
                "annual_std_kwh": a_std,
                "monthly_mean_kwh": a_mean / 12.0 if pd.notna(a_mean) else float('nan'),
                "monthly_std_kwh": a_std / 12.0 if pd.notna(a_std) else float('nan'),
                "saving_kwh": saving_kwh,
                "saving_percent": saving_pct,
            })
        df_stats = pd.DataFrame(stats_rows)
        df_stats.to_csv(Path(out_dir) / "annual_energy_stats.csv", index=False)
    except Exception as _e:
        # Non-fatal: plotting should still proceed even if CSV export fails
        pass

# Drop any non-finite means (to avoid NaN axis limits)
    keep_idx = [i for i, v in enumerate(means) if np.isfinite(v)]
    labels = [labels[i] for i in keep_idx]
    means = [means[i] for i in keep_idx]
    stds = [stds[i] for i in keep_idx]
    ns = [ns[i] for i in keep_idx]

    # Colors (distinct per bar)
    # Baseline = gray; RW1..RW4 = tab colors
    base_color = "0.6"
    tab = plt.get_cmap("tab10").colors
    rw_color_map = {"RW1": tab[0], "RW2": tab[1], "RW3": tab[2], "RW4": tab[3]}
    colors = [base_color] + [rw_color_map.get(lbl, tab[0]) for lbl in labels[1:]]

    # ---- Plot
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    x = np.arange(len(labels))

    bars = ax.bar(
        x, means,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )

    # Value labels (energy only; no %)
    for b, v in zip(bars, means):
        if not np.isfinite(v):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() * 1.01,
            value_fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_title("Energy Consumption: Baseline vs Enhanced DQN (RW1–RW4)")
    ax.set_xlabel("Control Strategy")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Robust y-limits
    finite_max = max([m + (s if (n >= 2 and np.isfinite(s)) else 0.0) for m, s, n in zip(means, stds, ns) if np.isfinite(m)], default=1.0)
    finite_min = min([m - (s if (n >= 2 and np.isfinite(s)) else 0.0) for m, s, n in zip(means, stds, ns) if np.isfinite(m)], default=0.0)
    if not np.isfinite(finite_min) or not np.isfinite(finite_max) or finite_max <= 0:
        finite_min, finite_max = 0.0, 1.0

    pad = (finite_max - finite_min) * 0.15 if finite_max > finite_min else finite_max * 0.15
    ax.set_ylim(max(0.0, finite_min - pad), finite_max + pad)

    fig.tight_layout()

    # Save outputs (e.g., PNG 300 dpi and/or vector PDF) using the shared helper.
    # Provide the base path without extension.
    save_fig(fig, out_dir / fname, formats=formats, dpi=dpi)

    plt.close(fig)

def plot_monthly_profiles(runs_df: pd.DataFrame, out_dir: Path, formats: List[str], dpi: int,
                          figwidth: float, figheight: float):
    rows = []
    for _, r in runs_df.iterrows():
        yr = r.get("yearly_results", None)
        if yr is None or (isinstance(yr, float) and np.isnan(yr)):
            continue
        yr_path = Path(str(yr))
        if not yr_path.exists():
            continue
        dfy = pd.read_csv(yr_path)
        if "day_of_year" not in dfy.columns or "energy" not in dfy.columns:
            continue
        day = dfy["day_of_year"].astype(int).clip(1, 365).to_numpy()
        energy = dfy["energy"].astype(float).to_numpy()
        months = np.array([day_to_month(int(d)) for d in day], dtype=int)
        for m in range(12):
            rows.append({
                "rw": r["rw"],
                "seed": int(r["seed"]),
                "month": m,
                "month_name": MONTH_NAMES[m],
                "energy_sum": float(energy[months == m].sum()),
            })

    if not rows:
        return []

    monthly = pd.DataFrame(rows)
    agg = monthly.groupby(["rw","month","month_name"], as_index=False).agg(
        mean_energy=("energy_sum", "mean"),
        std_energy=("energy_sum", "std"),
        n=("seed", "nunique"),
    ).sort_values(["rw","month"])

    outputs = []
    for rw in sorted(agg["rw"].unique()):
        sub = agg[agg["rw"] == rw]
        x = np.arange(len(sub))
        y = sub["mean_energy"].to_numpy(dtype=float)
        yerr = sub["std_energy"].fillna(0.0).to_numpy(dtype=float)

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=2, marker="o")
        if sub["n"].max() > 1:
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["month_name"].tolist())
        ax.set_title(f"Monthly Energy Profile (AI) — {rw} (mean±std)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Energy (kWh-equivalent)")

        out_base = out_dir / f"monthly_energy_profile_{rw}"
        save_fig(fig, out_base, formats, dpi)
        plt.close(fig)
        outputs.append(out_base)

    return outputs


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments", help="Path to experiments root")
    parser.add_argument("--results_summary", required=True, help="Path to results_summary.csv")
    parser.add_argument("--out", default="figures_mdpi", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (PDF is vector)")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated formats: png,pdf")

    parser.add_argument("--energy_bar_mode", choices=["annual", "monthly_avg"], default="monthly_avg",
                        help="Units for annual_energy_ai_vs_baseline: annual or monthly_avg (annual/12)")


    # style knobs
    parser.add_argument("--font_family", default="Arial", help="Font family (e.g., Arial, Times New Roman)")
    parser.add_argument("--base_fontsize", type=int, default=10)
    parser.add_argument("--title_size", type=int, default=11)
    parser.add_argument("--label_size", type=int, default=10)
    parser.add_argument("--tick_size", type=int, default=9)
    parser.add_argument("--legend_size", type=int, default=9)

    # figure knobs
    parser.add_argument("--figwidth", type=float, default=6.5, help="Figure width in inches")
    parser.add_argument("--figheight", type=float, default=4.0, help="Figure height in inches")
    parser.add_argument("--annual_bar_mode", type=str, default="monthly_avg", choices=["annual","monthly_avg"], help="Energy bar mode: annual (kWh/year) or monthly_avg (annual/12)")

    # learning curve knobs
    parser.add_argument("--ma_window", type=int, default=50, help="Moving average window for learning curve")

    args = parser.parse_args()

    experiments_root = Path(args.experiments)
    results_summary_csv = Path(args.results_summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    apply_style(args.font_family, args.base_fontsize, args.title_size, args.label_size, args.tick_size, args.legend_size)

    runs_df = discover_runs(experiments_root)
    if len(runs_df) == 0:
        raise SystemExit(f"No runs found under: {experiments_root.resolve()} (expected train_log.csv files)")

    # 1) Learning curves
    plot_learning_curves(
        runs_df, out_dir, args.ma_window, formats, args.dpi,
        args.figwidth, args.figheight
    )

    # 2) Annual bars
    plot_annual_bars(results_summary_csv, out_dir, formats, args.dpi, args.figwidth, args.figheight, args.annual_bar_mode)

    # 3) Monthly profiles (only if yearly_results.csv exists)
    plot_monthly_profiles(
        runs_df, out_dir, formats, args.dpi,
        args.figwidth, args.figheight
    )

    print(f"Done. Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
