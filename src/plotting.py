import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(log_root):
    """
    Generate plots from the monitor CSV files in the specified log directory.
    """
    if not os.path.exists(log_root):
        raise FileNotFoundError(f"Log root directory {log_root} does not exist.")
    
    timestamp = sorted(os.listdir(log_root))[-1]  # use latest timestamp folder
    if not timestamp:
        raise RuntimeError("No timestamp directories found in the log root.")
    
    log_dir = os.path.join(log_root, timestamp)
    # Load all monitor CSVs (one for each parallel environment)
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} does not exist.")
    
    monitor_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("monitor") and f.endswith(".csv")
    ]

    dfs = []
    for idx, file in enumerate(monitor_files):
        try:
            df = pd.read_csv(file, skiprows=1)  # skip Gym Monitor metadata
            df["env_id"] = idx
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not dfs:
        raise RuntimeError("No monitor.csv files found.")

    df = pd.concat(dfs, ignore_index=True)
    # Sort by time for all parallel environments
    df = df.sort_values(by="t").reset_index(drop=True)

    # Rolling reward for smoother visualization
    df["rolling_reward"] = df["r"].rolling(window=50).mean()

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Rolling Reward over Environment Time
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="t", y="rolling_reward")
    plt.title("Rolling Reward over Time (window=50)")
    plt.xlabel("Environment Time (s)")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "rolling_reward.png"))
    plt.close()

    # Rolling Episode Length Over Time
    plt.figure(figsize=(10, 5))
    df["rolling_length"] = df["l"].rolling(window=50).mean()
    sns.lineplot(data=df, x="t", y="rolling_length")
    plt.title("Rolling Episode Length Over Time (window=50)")
    plt.xlabel("Environment Time (s)")
    plt.ylabel("Average Episode Length")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "rolling_episode_length_over_time_smoothed.png"))
    plt.close()

    # Reward vs Episode Length
    plt.figure(figsize=(8, 5))
    plt.scatter(df["l"], df["r"], alpha=0.5, c="teal")
    plt.title("Reward vs Episode Length")
    plt.xlabel("Episode Length")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "reward_vs_length.png"))
    plt.close()

    print(f"Plots saved in: {log_dir}")


if __name__ == "__main__":
    log_root = "logs/parking_policy"
    try:
        generate_plots(log_root)
    except Exception as e:
        print(f"Error generating plots: {e}")
        raise
    else:
        print("Plots generated successfully.")
