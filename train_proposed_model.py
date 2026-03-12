import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from split_type_env_v1 import SplitTypeEnv
from EnhancedDQN import EnhancedDQNAgent, AgentConfig


RW_CONFIGS = {
    "RW1": {"w_energy": 1.0,  "w_maint": 1.0, "w_comfort": 1.0},
    "RW2": {"w_energy": 4.0,  "w_maint": 2.0, "w_comfort": 1.0},
    "RW3": {"w_energy": 6.0,  "w_maint": 2.0, "w_comfort": 0.8},
    "RW4": {"w_energy": 15.0, "w_maint": 5.0, "w_comfort": 1.0},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rw", required=True, choices=RW_CONFIGS.keys())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--target_update", type=int, default=50)  # episodes
    parser.add_argument("--out", required=True)
    parser.add_argument("--weather", default="data/processed/chiangmai_365d_hourly.csv")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if not os.path.exists(args.weather):
        raise FileNotFoundError(f"Weather file not found: {args.weather}")

    weights = RW_CONFIGS[args.rw]
    env = SplitTypeEnv(args.weather, reward_weights=weights)

    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    total_actions = int(np.prod(action_dims))  # 420

    agent = EnhancedDQNAgent(state_dim, total_actions, config=AgentConfig())

    LOG_FILE = out_dir / "train_log.csv"
    MODEL_FILE = out_dir / "model_best.pth"
    CONFIG_FILE = out_dir / "config.json"

    config = {
        "rw": args.rw,
        "seed": args.seed,
        "episodes": args.episodes,
        "target_update": args.target_update,
        "reward_weights": weights,
        "weather_file": args.weather,
        "action_dims": [int(x) for x in action_dims],
        "device": str(agent.device),
        "agent": {
            "lr": agent.cfg.lr,
            "gamma": agent.cfg.gamma,
            "batch_size": agent.cfg.batch_size,
            "replay_size": agent.cfg.replay_size,
            "eps_start": agent.cfg.eps_start,
            "eps_end": agent.cfg.eps_end,
            "eps_decay_steps": agent.cfg.eps_decay_steps,
        },
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    history = []
    best_avg_reward_50 = -float("inf")

    start = time.time()
    print("=" * 70)
    print(f"🚀 TRAIN | {args.rw} | seed={args.seed} | episodes={args.episodes}")
    print(f"🧠 Device: {agent.device} | Actions: {total_actions} | State: {state_dim}")
    print(f"📁 Out: {out_dir.resolve()}")
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        terminated = False

        total_reward = 0.0
        sum_energy = 0.0
        sum_comfort = 0.0
        sum_maint = 0.0
        sum_loss = 0.0
        loss_count = 0

        while not terminated:
            action_idx = agent.select_action(state, train=True)
            action = np.unravel_index(action_idx, action_dims)

            next_state, reward, terminated, _, info = env.step(action)

            agent.store_transition(state, action_idx, reward, next_state, terminated)
            loss = agent.update()
            if loss is not None:
                sum_loss += loss
                loss_count += 1

            state = next_state
            total_reward += float(reward)

            sum_energy += float(info.get("energy", 0.0))
            sum_comfort += float(info.get("comfort_penalty", 0.0))
            sum_maint += float(info.get("maintenance_penalty", 0.0))

        avg_loss = (sum_loss / loss_count) if loss_count else 0.0

        history.append(
            {
                "episode": ep,
                "reward": total_reward,
                "energy_sum": sum_energy,
                "comfort_sum": sum_comfort,
                "maint_sum": sum_maint,
                "avg_loss": avg_loss,
                "epsilon": agent.epsilon(),
            }
        )

        if ep % args.target_update == 0:
            agent.update_target()

        if ep % 50 == 0:
            recent = history[-50:]
            avg_r = float(np.mean([h["reward"] for h in recent]))
            pd.DataFrame(history).to_csv(LOG_FILE, index=False)

            elapsed_hr = (time.time() - start) / 3600.0
            print(f"Ep {ep:4d} | AvgR(50)={avg_r:10.2f} | eps={agent.epsilon():.3f} | {elapsed_hr:.2f} hr")

            if avg_r > best_avg_reward_50:
                best_avg_reward_50 = avg_r
                torch.save(
                    {
                        "model_state_dict": agent.q_net.state_dict(),
                        "meta": config,
                        "best_avg_reward_50": float(best_avg_reward_50),
                    },
                    MODEL_FILE,
                )

    pd.DataFrame(history).to_csv(LOG_FILE, index=False)

    total_hr = (time.time() - start) / 3600.0
    print("=" * 70)
    print("✅ TRAIN DONE")
    print(f"⏱️ Total time: {total_hr:.2f} hr")
    print(f"💾 Model: {MODEL_FILE.resolve()}")
    print(f"📊 Log:   {LOG_FILE.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
