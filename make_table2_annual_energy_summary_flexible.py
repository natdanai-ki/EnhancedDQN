#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


NEAR_ZERO_EPS = 1e-9


def parse_summary_txt(summary_path: Path) -> Tuple[Optional[float], Optional[float]]:
    txt = summary_path.read_text(encoding="utf-8", errors="ignore")
    base = None
    ai = None
    m1 = re.search(r"Baseline\s*\(.*?\)\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)", txt, flags=re.I)
    m2 = re.search(r"AI\s*Energy:\s*([0-9]+(?:\.[0-9]+)?)", txt, flags=re.I)
    if m1:
        base = float(m1.group(1))
    if m2:
        ai = float(m2.group(1))
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

        for sf in seed_dir.rglob("summary.txt"):
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
                elif any(k in low for k in ["ai", "agent", "rl", "dqn"]):
                    ai_candidates.append((cf, val))
            if base_candidates and ai_candidates:
                baseline_energy = base_candidates[0][1]
                ai_energy = ai_candidates[0][1]
                evidence = f"{base_candidates[0][0]} | {ai_candidates[0][0]}"

        if seed_id is not None and baseline_energy is not None and ai_energy is not None:
            records.append({
                "rw": rw_dir.name,
                "seed": seed_id,
                "baseline_energy": baseline_energy,
                "ai_energy": ai_energy,
                "source": evidence,
            })

    if not records:
        return pd.DataFrame(columns=["rw", "seed", "baseline_energy", "ai_energy", "source"])

    return (
        pd.DataFrame(records)
        .sort_values(["rw", "seed"])
        .drop_duplicates(subset=["rw", "seed"], keep="last")
        .reset_index(drop=True)
    )


def clean_std_value(value: float) -> float:
    """Convert tiny floating-point noise to exactly 0.0 for CSV export."""
    if pd.isna(value):
        return value
    return 0.0 if abs(float(value)) < NEAR_ZERO_EPS else float(value)


def build_summary(experiments_dir: Path, include_rws: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    for rw in include_rws:
        rw_dir = experiments_dir / rw
        if rw_dir.exists() and rw_dir.is_dir():
            df = find_seed_records_for_rw(rw_dir)
            if not df.empty:
                details.append(df)

    if not details:
        raise ValueError(f"No valid RW seed results found under: {experiments_dir}")

    detail_df = pd.concat(details, ignore_index=True)
    summary_rows = []

    baseline_all = pd.to_numeric(detail_df["baseline_energy"], errors="coerce").dropna()
    baseline_mean_all = float(baseline_all.mean())
    baseline_std_all = float(baseline_all.std(ddof=1)) if len(baseline_all) >= 2 else 0.0
    baseline_std_all = clean_std_value(baseline_std_all)
    summary_rows.append({
        "Control": "Baseline (all available seeds)",
        "Annual Energy (kWh)": baseline_mean_all,
        "Std (kWh)": baseline_std_all,
        "n (seeds)": int(len(baseline_all)),
        "Saving (kWh)": np.nan,
        "Saving (%)": np.nan,
    })

    for rw in include_rws:
        sub = detail_df[detail_df["rw"] == rw].copy()
        if sub.empty:
            continue
        base_vals = pd.to_numeric(sub["baseline_energy"], errors="coerce").dropna()
        ai_vals = pd.to_numeric(sub["ai_energy"], errors="coerce").dropna()
        if ai_vals.empty:
            continue
        baseline_mean = float(base_vals.mean())
        ai_mean = float(ai_vals.mean())
        ai_std = float(ai_vals.std(ddof=1)) if len(ai_vals) >= 2 else 0.0
        ai_std = clean_std_value(ai_std)
        n = int(len(ai_vals))
        saving_kwh = baseline_mean - ai_mean
        saving_pct = (saving_kwh / baseline_mean * 100.0) if baseline_mean else np.nan
        summary_rows.append({
            "Control": f"Enhanced DQN ({rw})",
            "Annual Energy (kWh)": ai_mean,
            "Std (kWh)": ai_std,
            "n (seeds)": n,
            "Saving (kWh)": saving_kwh,
            "Saving (%)": saving_pct,
        })

    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def format_summary_for_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns so CSV stores plain decimals, not scientific notation."""
    df = summary_df.copy()
    for col in ["Annual Energy (kWh)", "Saving (kWh)", "Saving (%)"]:
        df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")

    def _fmt_std(x: float) -> str:
        if pd.isna(x):
            return ""
        x = clean_std_value(float(x))
        if x == 0.0:
            return "0.00"
        return f"{x:.6f}"

    df["Std (kWh)"] = df["Std (kWh)"].map(_fmt_std)
    df["n (seeds)"] = df["n (seeds)"].astype("Int64")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build annual energy summary using all available seeds per RW.")
    ap.add_argument("--experiments_dir", required=True, help="Path to experiments folder containing RW1..RW4")
    ap.add_argument("--out", default="figures_mdpi", help="Output folder")
    ap.add_argument("--rws", default="RW1,RW2,RW3,RW4", help="Comma-separated RW folders to include")
    args = ap.parse_args()

    experiments_dir = Path(args.experiments_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_rws = [x.strip() for x in str(args.rws).split(",") if x.strip()]

    detail_df, summary_df = build_summary(experiments_dir, include_rws)
    summary_df_csv = format_summary_for_csv(summary_df)

    detail_path = out_dir / "annual_energy_seed_details_all_rw.csv"
    summary_path = out_dir / "table2_annual_energy_summary.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df_csv.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Done")
    print(f"- Detail:  {detail_path.resolve()}")
    print(f"- Summary: {summary_path.resolve()}")
    print(summary_df_csv.to_string(index=False))


if __name__ == "__main__":
    main()
