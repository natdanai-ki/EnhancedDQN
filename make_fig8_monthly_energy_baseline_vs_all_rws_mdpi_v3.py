import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

def day_to_month(day):
    edges = np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
    d0 = max(0, min(364, int(day)-1))
    return int(np.searchsorted(edges[1:], d0, side="right"))

def load_monthly_from_yearly(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if "day_of_year" not in df.columns:
        raise ValueError(f"{csv_path} missing 'day_of_year' column")

    if "energy" in df.columns:
        energy_col = "energy"
    elif "energy_kwh" in df.columns:
        energy_col = "energy_kwh"
    else:
        raise ValueError(f"{csv_path} missing energy column")

    df["month_idx"] = df["day_of_year"].apply(day_to_month)
    monthly = df.groupby("month_idx")[energy_col].sum()
    monthly = monthly.reindex(range(12), fill_value=0.0).to_numpy(dtype=float)
    return monthly

def find_baseline_monthly(root: Path) -> np.ndarray:
    candidates = [
        root / "experiments" / "baseline25" / "baseline_yearly_results.csv",
        root / "experiments" / "baseline25" / "yearly_results.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_monthly_from_yearly(p)

    for p in sorted((root / "experiments").rglob("*baseline*yearly*results*.csv")):
        return load_monthly_from_yearly(p)

    raise FileNotFoundError("Could not find baseline_yearly_results.csv under experiments/baseline25")

def find_rw_seed_files(rw_dir: Path):
    seed_files = []
    if not rw_dir.exists():
        return seed_files

    seed_dirs = sorted(
        [p for p in rw_dir.iterdir() if p.is_dir() and re.fullmatch(r"seed\d+", p.name, flags=re.I)],
        key=lambda p: int(re.findall(r"\d+", p.name)[0])
    )

    for seed_dir in seed_dirs:
        candidates = [
            seed_dir / "yearly_results.csv",
            seed_dir / "out" / "yearly_results.csv",
        ]
        found = None
        for p in candidates:
            if p.exists():
                found = p
                break
        if found is None:
            deep = sorted(seed_dir.rglob("yearly_results.csv"), key=lambda x: (len(x.parts), str(x)))
            if deep:
                found = deep[0]
        if found is not None:
            seed_id = int(re.findall(r"\d+", seed_dir.name)[0])
            seed_files.append((seed_id, found))
    return seed_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--out", default="figures_mdpi")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--export_only", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_monthly = find_baseline_monthly(root)

    rws = ["RW1", "RW2", "RW3", "RW4"]
    rw_mean = {}
    rw_std = {}
    rw_counts = {}
    monthly_detail_rows = []

    for rw in rws:
        rw_dir = root / "experiments" / rw
        seed_files = find_rw_seed_files(rw_dir)

        seed_monthlies = []
        used_seed_ids = []

        for seed_id, path in seed_files:
            try:
                monthly = load_monthly_from_yearly(path)
                seed_monthlies.append(monthly)
                used_seed_ids.append(seed_id)

                for month_idx, value in enumerate(monthly):
                    monthly_detail_rows.append({
                        "Series": rw,
                        "Seed": seed_id,
                        "MonthIndex": month_idx + 1,
                        "Month": MONTH_NAMES[month_idx],
                        "MonthlyEnergy_kWh": float(value),
                        "SourceFile": str(path),
                    })
            except Exception as e:
                print(f"[WARN] Skip {rw} seed{seed_id}: {e}")

        if seed_monthlies:
            arr = np.array(seed_monthlies, dtype=float)
            rw_mean[rw] = arr.mean(axis=0)
            rw_std[rw] = arr.std(axis=0, ddof=1) if arr.shape[0] >= 2 else np.zeros(12)
            rw_counts[rw] = arr.shape[0]
            print(f"[OK] {rw}: using {arr.shape[0]} seed(s) -> {used_seed_ids}")
        else:
            print(f"[WARN] {rw}: no usable yearly_results.csv found")

    # Export detail CSV: every seed, every month
    detail_df = pd.DataFrame(monthly_detail_rows)
    detail_path = out_dir / "monthly_energy_seed_details.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # Export plot values CSV: exactly what the line plot uses
    rows = []
    for i, month in enumerate(MONTH_NAMES):
        row = {
            "MonthIndex": i + 1,
            "Month": month,
            "Baseline_25C": float(baseline_monthly[i]),
        }
        for rw in rws:
            row[f"{rw}_Mean"] = float(rw_mean[rw][i]) if rw in rw_mean else np.nan
            row[f"{rw}_Std"] = float(rw_std[rw][i]) if rw in rw_std else np.nan
            row[f"{rw}_Lower"] = float(rw_mean[rw][i] - rw_std[rw][i]) if rw in rw_mean else np.nan
            row[f"{rw}_Upper"] = float(rw_mean[rw][i] + rw_std[rw][i]) if rw in rw_mean else np.nan
            row[f"{rw}_NSeeds"] = int(rw_counts[rw]) if rw in rw_counts else 0
        rows.append(row)

    plot_df = pd.DataFrame(rows)
    plot_path = out_dir / "monthly_energy_plot_values.csv"
    plot_df.to_csv(plot_path, index=False, encoding="utf-8-sig")

    if args.export_only:
        print("CSV export complete.")
        print("Plot values:", plot_path.resolve())
        print("Seed details:", detail_path.resolve())
        return

    plt.figure(figsize=(6.5, 4))
    x = np.arange(12)

    plt.plot(
        x, baseline_monthly,
        linestyle="--",
        linewidth=2.5,
        label="Baseline (25°C)"
    )

    colors = {
        "RW1": "#ff7f0e",
        "RW2": "#2ca02c",
        "RW3": "#d62728",
        "RW4": "#9467bd"
    }

    for rw in rws:
        if rw in rw_mean:
            lw = 3 if rw == "RW3" else 2
            plt.plot(
                x, rw_mean[rw],
                linewidth=lw,
                color=colors[rw],
                label=rw
            )

    plt.xticks(x, MONTH_NAMES)
    plt.xlabel("Month")
    plt.ylabel("Monthly Energy (kWh)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    base = out_dir / "monthly_energy_baseline_vs_all_rws"
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    for fmt in formats:
        if fmt == "png":
            plt.savefig(base.with_suffix(".png"), dpi=args.dpi)
        elif fmt == "pdf":
            plt.savefig(base.with_suffix(".pdf"))

    plt.close()
    print("Figure saved to:", out_dir.resolve())
    print("Plot values:", plot_path.resolve())
    print("Seed details:", detail_path.resolve())

if __name__ == "__main__":
    main()
