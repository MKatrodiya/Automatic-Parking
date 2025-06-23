import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from new_parking_env_multi import ParkingEnv  # multi-vehicle env with 2 controlled vehicles
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
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
#REWARD_WEIGHTS = [100, 100, 5, 5, 1, 1, 100, 100, 5, 5, 1, 1]  # 2 vehicles x 6 features
REWARD_WEIGHTS = [100, 100, 5, 5, 1, 1] * 2
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

class EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=100, output_dir="plots", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.output_dir = output_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            self.episode_count += 1
            obs = self.eval_env.reset()
            done = [False]
            total_reward = 0
            total_length = 0
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward[0]
                total_length += 1

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(total_length)

            if self.episode_count % self.eval_freq == 0:
                self._save_plot_and_csv()
                model_path = os.path.join(self.output_dir, f"checkpoint_ep{self.episode_count}.zip")
                self.model.save(model_path)
                if self.verbose:
                    print(f"Checkpoint saved to {model_path}")

        return True

    def _save_plot_and_csv(self):
        episodes = list(range(1, len(self.episode_rewards) + 1))
        plt.figure()
        plt.plot(episodes, self.episode_rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward vs Episode")
        plt.savefig(os.path.join(self.output_dir, "reward_vs_episode.png"))
        plt.close()

        plt.figure()
        plt.plot(episodes, self.episode_lengths, label="Length")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Length vs Episode")
        plt.savefig(os.path.join(self.output_dir, "length_vs_episode.png"))
        plt.close()

        with open(os.path.join(self.output_dir, "episode_stats.csv"), mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Length"])
            for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_lengths), 1):
                writer.writerow([i, r, l])

if __name__ == "__main__":
    n_cpu = 4
    batch_size = 64
    n_steps = batch_size * 30
    timesteps = 2000000

    if TRAIN:
        env = DummyVecEnv([make_custom_env()])
        eval_env = DummyVecEnv([make_custom_env()])
        callback = EvalAndCheckpointCallback(eval_env, eval_freq=100, output_dir="plots", verbose=1)

        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=1e-3,
            n_epochs=10,
            gamma=0.95,
            device="cpu",
            #ent_coef=0.05,
            ent_coef=0.01,         # Lower entropy to reduce random exploration
            clip_range=0.2         #  Explicit PPO clip range
        )

        model.learn(total_timesteps=timesteps, callback=callback)
        model.save("parking_policy/model")

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

            if e % 100 == 0:
                episodes = list(range(1, len(episode_rewards) + 1))
                plt.figure()
                plt.plot(episodes, episode_rewards, label="Reward")
                plt.xlabel("Episode")
                plt.ylabel("Total Reward")
                plt.title("Reward vs Episode")
                plt.savefig("reward_vs_episode.png")
                plt.close()

                plt.figure()
                plt.plot(episodes, episode_lengths, label="Length")
                plt.xlabel("Episode")
                plt.ylabel("Length")
                plt.title("Length vs Episode")
                plt.savefig("length_vs_episode.png")
                plt.close()

                with open("episode_stats.csv", mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Episode", "Reward", "Length"])
                    for i, (r, l) in enumerate(zip(episode_rewards, episode_lengths), 1):
                        writer.writerow([i, r, l])

        if RECORD:
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"GIF saved to {gif_filename}")
