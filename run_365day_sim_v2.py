import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from split_type_env_v1 import SplitTypeEnv
from EnhancedDQN import EnhancedDQNAgent


def _fixed_action_for_temp(fixed_temp_c: int):
    """Map fixed temperature (18..31C) to temp_idx (0..13)."""
    temp_idx = int(np.clip(fixed_temp_c - 18, 0, 13))
    # action = (vane, temp_idx, fan, mode)
    # Note: in current env, vane/mode are not used; temp_idx & fan affect cooling.
    return (0, temp_idx, 1, 0)


def simulate_baseline(env: SplitTypeEnv, fixed_temp_c: int = 25, record: bool = True):
    """
    Baseline policy: fixed setpoint temp (default 25C) for all steps.
    Returns:
      total_energy (float),
      records (list[dict]) if record else []
    """
    state, _ = env.reset(seed=123)
    terminated = False
    total_energy = 0.0
    records = []

    action = _fixed_action_for_temp(fixed_temp_c)

    while not terminated:
        next_state, reward, terminated, _, info = env.step(action)
        total_energy += float(info["energy"])

        if record:
            records.append(
                {
                    "day_of_year": float(info["day_of_year"]),
                    "hour": float(info["hour"]),
                    "energy": float(info["energy"]),
                    "temp": float(info["temp"]),
                    "filter": float(info["filter"]),
                    "comfort_target": float(info["comfort_target"]),
                    "pm10": float(info.get("pm10", 0.0)),
                    "comfort_penalty": float(info["comfort_penalty"]),
                    "maintenance_penalty": float(info["maintenance_penalty"]),
                }
            )

        state = next_state

    return total_energy, records


def simulate_ai(env: SplitTypeEnv, model_path: str, action_dims):
    state_dim = env.observation_space.shape[0]
    total_actions = int(np.prod(action_dims))

    agent = EnhancedDQNAgent(state_dim, total_actions)
    ckpt = torch.load(model_path, map_location=agent.device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        agent.q_net.load_state_dict(ckpt["model_state_dict"])
    else:
        agent.q_net.load_state_dict(ckpt)

    state, _ = env.reset(seed=999)
    terminated = False
    total_energy = 0.0
    records = []

    while not terminated:
        action_idx = agent.select_action(state, train=False)
        action = np.unravel_index(action_idx, action_dims)

        next_state, reward, terminated, _, info = env.step(action)
        total_energy += float(info["energy"])

        records.append(
            {
                "day_of_year": float(info["day_of_year"]),
                "hour": float(info["hour"]),
                "energy": float(info["energy"]),
                "temp": float(info["temp"]),
                "filter": float(info["filter"]),
                "comfort_target": float(info["comfort_target"]),
                "pm10": float(info.get("pm10", 0.0)),
                "comfort_penalty": float(info["comfort_penalty"]),
                "maintenance_penalty": float(info["maintenance_penalty"]),
            }
        )
        state = next_state

    return total_energy, records


def main():
    parser = argparse.ArgumentParser()

    # Mode control:
    # - both: run baseline + AI (default)
    # - baseline: baseline only (no model required)
    # - ai: AI only (requires model; baseline_energy not computed)
    parser.add_argument("--mode", choices=["both", "baseline", "ai"], default="both")

    parser.add_argument("--model", default="", help="Path to model .pth (required for mode=ai/both)")
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--weather", default="data/processed/chiangmai_365d_hourly.csv")
    parser.add_argument("--baseline_temp", type=int, default=25)

    # Output names (optional)
    parser.add_argument("--ai_csv", default="yearly_results.csv", help="AI output CSV name (mode=ai/both)")
    parser.add_argument("--baseline_csv", default="baseline_yearly_results.csv", help="Baseline output CSV name (mode=baseline/both)")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Envs
    env_base = SplitTypeEnv(args.weather)
    env_ai = SplitTypeEnv(args.weather)

    action_dims = env_ai.action_space.nvec

    baseline_energy = None
    ai_energy = None
    saving_pct = None

    # Baseline
    baseline_records = []
    if args.mode in ["both", "baseline"]:
        baseline_energy, baseline_records = simulate_baseline(env_base, fixed_temp_c=args.baseline_temp, record=True)
        baseline_csv = out_dir / args.baseline_csv
        pd.DataFrame(baseline_records).to_csv(baseline_csv, index=False)
        print(f"📄 Baseline CSV: {baseline_csv.resolve()}")

    # AI
    ai_records = []
    if args.mode in ["both", "ai"]:
        if not args.model:
            raise SystemExit("ERROR: --model is required for --mode ai or both")
        ai_energy, ai_records = simulate_ai(env_ai, args.model, action_dims)
        ai_csv = out_dir / args.ai_csv
        pd.DataFrame(ai_records).to_csv(ai_csv, index=False)
        print(f"📄 AI CSV:       {ai_csv.resolve()}")

    # Saving
    if args.mode == "both" and baseline_energy is not None and ai_energy is not None:
        saving_pct = (baseline_energy - ai_energy) / baseline_energy * 100.0

    # Summary
    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        if baseline_energy is not None:
            f.write(f"Baseline (fixed {args.baseline_temp}C) Energy: {baseline_energy:.4f}\n")
        if ai_energy is not None:
            f.write(f"AI Energy: {ai_energy:.4f}\n")
        if saving_pct is not None:
            f.write(f"Saving Percentage: {saving_pct:.2f}%\n")

    print("=" * 70)
    print("✅ 365-day simulation done")
    print(f"🧾 Summary:     {summary_txt.resolve()}")
    if baseline_energy is not None:
        print(f"⚡ Baseline Energy: {baseline_energy:.2f}")
    if ai_energy is not None:
        print(f"⚡ AI Energy:       {ai_energy:.2f}")
    if saving_pct is not None:
        print(f"✅ Saving:          {saving_pct:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
