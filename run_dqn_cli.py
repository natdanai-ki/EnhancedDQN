import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
from core_agent import DQNAgent


def run_dqn(env_name, episodes, seed, output_path):
    env = gym.make(env_name)
    np.random.seed(seed)
    env.reset(seed=seed)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    rewards, losses = [], []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward, total_loss, done = 0, 0, False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            state = next_state
            total_reward += reward
            total_loss += loss

        rewards.append(total_reward)
        losses.append(total_loss)

        if (episode + 1) % 100 == 0:
            mean_r = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | Mean Reward(last100): {mean_r:.2f}")

    df = pd.DataFrame({"episode": range(1, episodes + 1), "reward": rewards, "loss": losses})
    df.to_csv(output_path, index=False)
    print(f"✅ Training complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="dqn_output.csv")
    args = parser.parse_args()

    run_dqn(args.env, args.episodes, args.seed, args.out)
