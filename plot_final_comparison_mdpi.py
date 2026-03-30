import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ===============================
# MDPI-Ready Global Style
# ===============================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.labelsize": 8,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 600
})

sns.set_style("whitegrid")
sns.set_context("paper")

# ===============================
# Data Loader
# ===============================

def load_processed_data(algo_prefix):
    patterns = [
        f"{algo_prefix}_seed*.csv",
        f"data_raw/{algo_prefix}_seed*.csv",
        f"../01_Baselines_Study/01_Baselines_Study/data_raw/{algo_prefix}_seed*.csv",
        f"../02_Proposed_Enhanced_DQN/{algo_prefix}_seed*.csv",
        f"../02_Proposed_Enhanced_DQN/data_results/{algo_prefix}_seed*.csv"
    ]

    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))

    all_files = sorted(list(set(all_files)))

    if not all_files:
        print(f"⚠️ No files found for {algo_prefix}")
        return None

    rewards = []
    min_length = 2000

    for filename in all_files:
        df = pd.read_csv(filename)
        if "reward" in df.columns:
            data = df["reward"].values
            if len(data) >= min_length:
                rewards.append(data[:min_length])

    if not rewards:
        return None

    return np.array(rewards)

# ===============================
# Main Plot Function
# ===============================

def generate_results_mdpi():

    algorithms = ['DQN', 'Double DQN', 'Dueling DQN', 'Enhanced DQN']
    prefixes = ['dqn', 'ddqn', 'dueling', 'enhanced']

    colors = {
        'DQN': '#C0392B',
        'Double DQN': '#2980B9',
        'Dueling DQN': '#D68910',
        'Enhanced DQN': '#1E8449'
    }

    summary_stats = []
    window_size = 50

    # ==================================
    # Figure 6.1 Learning Curves
    # ==================================

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for algo, prefix in zip(algorithms, prefixes):

        data = load_processed_data(prefix)
        if data is None:
            continue

        mean_seed = np.mean(data, axis=0)
        std_seed = np.std(data, axis=0)

        smooth_mean = pd.Series(mean_seed).rolling(window=window_size, min_periods=1).mean()
        smooth_std = pd.Series(std_seed).rolling(window=window_size, min_periods=1).mean()

        episodes = np.arange(1, len(smooth_mean) + 1)

        lw = 2.5 if "Enhanced" in algo else 1.5

        ax.plot(episodes, smooth_mean,
                label=algo,
                color=colors[algo],
                linewidth=lw)

        ax.fill_between(episodes,
                        smooth_mean - smooth_std,
                        smooth_mean + smooth_std,
                        color=colors[algo],
                        alpha=0.15)

        final_100_mean = np.mean(mean_seed[-100:])
        final_100_std = np.std(mean_seed[-100:])
        summary_stats.append({
            "Algorithm": algo,
            "Final Mean Reward": round(final_100_mean, 2),
            "Std (Last100)": round(final_100_std, 2)
        })

    ax.axhline(200, linestyle="--", linewidth=1.0, color="black", alpha=0.6)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean Episodic Return")
    ax.set_xlim(0, 2000)

    ax.legend(frameon=True)
    plt.tight_layout()

    plt.savefig("Figure_6_1_Learning_Curves.png", dpi=600)
    plt.savefig("Figure_6_1_Learning_Curves.pdf")

    print("✅ Saved: Figure_6_1_Learning_Curves")

    # ==================================
    # Figure 6.2 Reward Distribution
    # ==================================

    box_data = []

    for algo, prefix in zip(algorithms, prefixes):
        data = load_processed_data(prefix)
        if data is None:
            continue

        last100 = data[:, -100:].flatten()

        for val in last100:
            box_data.append({
                "Algorithm": algo,
                "Reward": val
            })

    if box_data:
        df_box = pd.DataFrame(box_data)

        fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

        sns.boxplot(
            x="Algorithm",
            y="Reward",
            data=df_box,
            palette=colors,
            width=0.6,
            fliersize=2,
            linewidth=1,
            ax=ax2
        )

        ax2.set_xlabel("")
        ax2.set_ylabel("Episodic Return")

        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig("Figure_6_2_Reward_Distribution.png", dpi=600)
        plt.savefig("Figure_6_2_Reward_Distribution.pdf")

        print("✅ Saved: Figure_6_2_Reward_Distribution")

    # ==================================
    # Table Export
    # ==================================

    if summary_stats:
        df_table = pd.DataFrame(summary_stats)
        df_table.to_csv("Table_6_1_Performance_Summary.csv", index=False)
        print("✅ Saved: Table_6_1_Performance_Summary")

if __name__ == "__main__":
    generate_results_mdpi()
