import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from new_parking_env_multi import ParkingEnv  # multi-vehicle env with 2 controlled vehicles
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime
import imageio
import matplotlib.pyplot as plt
import os
import csv

# Training and recording parameters
TRAIN = True
RECORD = False
RECORD_LIMIT = 300  # Frames to record in the video
EPISODE_RECORD_LIMIT = 10

# Environment Parameters
PARKED_CARS = 5
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
COLLISION_PENALTY = -10
REWARD_WEIGHTS = [100, 100, 5, 5, 1, 1, 100, 100, 5, 5, 1, 1]  # 2 vehicles x 6 features
ACTION_FREQUENCY = 5
SIMULATION_FREQUENCY = 15

config = {
    "screen_width": SCREEN_WIDTH,
    "screen_height": SCREEN_HEIGHT,
    "vehicles_count": PARKED_CARS,
    "policy_frequency": ACTION_FREQUENCY,
    "simulation_frequency": SIMULATION_FREQUENCY,
    "reward_weights": REWARD_WEIGHTS,
    "collision_reward": COLLISION_PENALTY,
    "other_vehicles_type": "parked",
    "controlled_vehicles": 2,
}

def make_custom_env():
    def _init():
        env = ParkingEnv(config=config)
        return env
    return _init

def plot_and_save(rewards, lengths, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    episodes = list(range(1, len(rewards) + 1))

    plt.figure()
    plt.plot(episodes, rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episode")
    plt.savefig(os.path.join(output_dir, "reward_vs_episode.png"))
    plt.close()

    plt.figure()
    plt.plot(episodes, lengths, label="Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Length vs Episode")
    plt.savefig(os.path.join(output_dir, "length_vs_episode.png"))
    plt.close()

    # Save to CSV
    csv_path = os.path.join(output_dir, "episode_stats.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Length"])
        for i, (r, l) in enumerate(zip(rewards, lengths), 1):
            writer.writerow([i, r, l])

if __name__ == "__main__":
    n_cpu = 4
    batch_size = 64
    n_steps = batch_size * 30
    timesteps = 50000

    if TRAIN:
        env = DummyVecEnv([make_custom_env()])

        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=1e-3,
            n_epochs=5,
            gamma=0.95,
            device="cpu",
            ent_coef=0.05
        )

        episode_rewards = []
        episode_lengths = []

        obs = env.reset()
        for episode in range(1, timesteps // n_steps + 1):
            total_reward = 0
            total_length = 0
            done = False

            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                total_length += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(total_length)

            if episode % 20 == 0:
                plot_and_save(episode_rewards, episode_lengths)

            model.learn(total_timesteps=n_steps, reset_num_timesteps=False)

        model.save("parking_policy/model")
        del model

    else:
        model = PPO.load("parking_policy/model")
        env = ParkingEnv(config=config, render_mode="rgb_array")

        env.reset()
        env.render()
        frames = []
        gif_filename = f"parking_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        t = 0
        e = 0

        episode_rewards = []
        episode_lengths = []

        while True and t < RECORD_LIMIT and e < EPISODE_RECORD_LIMIT:
            e += 1
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done and t < RECORD_LIMIT:
                t += 1
                action, _states = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                print(f"Step: {t}, Reward: {reward}")
                frame = env.render()
                if RECORD:
                    frames.append(frame)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if e % 20 == 0:
                plot_and_save(episode_rewards, episode_lengths)

        if RECORD:
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"GIF saved to {gif_filename}")
