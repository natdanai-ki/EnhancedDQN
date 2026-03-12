# plot_final_comparison.py (MDPI-ready)
# Usage example:
#   python plot_final_comparison.py --experiments experiments_lander --out figures_mdpi --dpi 600 --formats png,pdf --ma_window 50 --last_n 100
#
# Expected structure (flexible):
#   experiments_lander/
#     DQN/seed0/train_log.csv
#     DQN/seed1/train_log.csv
#     DoubleDQN/seed0/train_log.csv
#     DuelingDQN/seed0/train_log.csv
#     EnhancedDQN/seed0/train_log.csv
#
# If your folders are different, use --map "Label=FolderName" multiple times.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REWARD_COL_CANDIDATES = [
    "reward", "episode_reward", "ep_reward", "total_reward", "return", "episodic_return"
]
EP_COL_CANDIDATES = ["episode", "Episode", "ep", "timestep", "step"]

def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def read_train_log(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Strip BOM / whitespace from headers
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    rcol = _pick_col(df, REWARD_COL_CANDIDATES)
    if rcol is None:
        raise ValueError(f"Cannot find reward column in {csv_path}. Columns={list(df.columns)}")

    ecol = _pick_col(df, EP_COL_CANDIDATES)
    if ecol is None:
        episodes = np.arange(len(df), dtype=int)
    else:
        episodes = df[ecol].to_numpy()

    rewards = df[rcol].to_numpy(dtype=float)
    return episodes, rewards

def moving_average(x: np.ndarray, window: int):
    if window <= 1:
        return x
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()

def stack_to_min_len(arrays):
    min_len = min(len(a) for a in arrays)
    A = np.stack([a[:min_len] for a in arrays], axis=0)
    return A, min_len

def find_logs(exp_root: Path, folder: str):
    base = exp_root / folder
    if not base.exists():
        return []

    # Prefer seed*/train_log.csv
    logs = sorted(base.glob("seed*/train_log.csv"))
    if logs:
        return logs

    # Fallback: any train_log.csv under this folder (one level deeper)
    logs = sorted(base.rglob("train_log.csv"))
    return logs

def save_fig(fig, out_path: Path, dpi: int, formats):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_path.with_suffix(f".{fmt}")
        if fmt.lower() in ["pdf", "eps", "svg"]:
            fig.savefig(p, bbox_inches="tight")
        else:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")

def plot_learning_curves(all_runs, out_dir: Path, dpi: int, formats, ma_window: int, solved_threshold: float | None):
    # MDPI-friendly rcParams
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()

    summary_rows = []

    for label, runs in all_runs.items():
        # runs: list of (episodes, rewards)
        smoothed = []
        ep_ref = None
        for ep, r in runs:
            ep_ref = ep if ep_ref is None else ep_ref
            smoothed.append(moving_average(r, ma_window))

        A, min_len = stack_to_min_len(smoothed)
        mean = A.mean(axis=0)
        std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
        ep = ep_ref[:min_len] if ep_ref is not None else np.arange(min_len)

        ax.plot(ep, mean, linewidth=2.2, label=f"{label}")
        ax.fill_between(ep, mean - std, mean + std, alpha=0.18)

        # steady-state stats (last 100 episodes on RAW rewards, not MA)
        last_n = min(100, min(len(r) for _, r in runs))
        pooled = np.concatenate([r[-last_n:] for _, r in runs], axis=0)
        summary_rows.append({
            "Algorithm": label,
            "Seeds(n)": len(runs),
            "LastN": last_n,
            "Mean(lastN)": float(np.mean(pooled)),
            "Std(lastN)": float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0,
            "Median(lastN)": float(np.median(pooled)),
            "IQR(lastN)": float(np.percentile(pooled, 75) - np.percentile(pooled, 25)),
        })

    if solved_threshold is not None:
        ax.axhline(y=solved_threshold, linestyle="--", linewidth=1.4, label=f"Solved Threshold ({solved_threshold:g})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Episode Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)

    save_fig(fig, out_dir / "chapter4_learning_curves", dpi, formats)
    plt.close(fig)

    # Export summary table
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "chapter4_performance_table.csv", index=False, encoding="utf-8-sig")

def plot_reward_distribution(all_runs, out_dir: Path, dpi: int, formats, last_n: int):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labels = []
    data = []

    for label, runs in all_runs.items():
        n = min(last_n, min(len(r) for _, r in runs))
        pooled = np.concatenate([r[-n:] for _, r in runs], axis=0)
        labels.append(label)
        data.append(pooled)

    fig = plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
        widths=0.6
    )

    # light styling (no aggressive colors)
    for box in bp["boxes"]:
        box.set_alpha(0.55)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Reward Score")
    ax.grid(True, axis="y", alpha=0.25)

    save_fig(fig, out_dir / "chapter4_reward_distribution", dpi, formats)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, required=True, help="Root folder containing algorithm subfolders")
    parser.add_argument("--out", type=str, default="figures_mdpi", help="Output folder")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for PNG export (MDPI prefers high-res)")
    parser.add_argument("--formats", type=str, default="png,pdf", help="Comma-separated formats: png,pdf")
    parser.add_argument("--ma_window", type=int, default=50, help="Moving average window for learning curves")
    parser.add_argument("--last_n", type=int, default=100, help="Last N episodes for reward distribution")
    parser.add_argument("--solved_threshold", type=float, default=200.0, help="Solved threshold line (set negative to disable)")
    parser.add_argument("--map", action="append", default=[], help='Optional mapping "Label=FolderName" (repeatable)')
    args = parser.parse_args()

    exp_root = Path(args.experiments)
    out_dir = Path(args.out)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    # Default folder map (edit if your names differ)
    folder_map = {
        "DQN": "DQN",
        "Double DQN": "DoubleDQN",
        "Dueling DQN": "DuelingDQN",
        "Enhanced DQN (Ours)": "EnhancedDQN",
    }
    # override/add via --map
    for m in args.map:
        if "=" in m:
            k, v = m.split("=", 1)
            folder_map[k.strip()] = v.strip()

    all_runs = {}
    for label, folder in folder_map.items():
        logs = find_logs(exp_root, folder)
        if not logs:
            continue
        runs = []
        for p in logs:
            ep, r = read_train_log(p)
            runs.append((ep, r))
        all_runs[label] = runs

    if not all_runs:
        raise RuntimeError(f"No train_log.csv found under {exp_root}. Check folder names or use --map.")

    solved = None if args.solved_threshold < 0 else args.solved_threshold

    plot_learning_curves(
        all_runs=all_runs,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        ma_window=args.ma_window,
        solved_threshold=solved
    )
    plot_reward_distribution(
        all_runs=all_runs,
        out_dir=out_dir,
        dpi=args.dpi,
        formats=formats,
        last_n=args.last_n
    )

    print("Done. Outputs saved to:", out_dir.resolve())
    print("Also exported:", (out_dir / "chapter4_performance_table.csv").resolve())

if __name__ == "__main__":
    main()
