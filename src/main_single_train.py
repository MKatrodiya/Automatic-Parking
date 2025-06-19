# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:08:42 2025

@author: lijia
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from new_parking_env import ParkingEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import datetime
import imageio

# new parking environment
from new_parking_env import ParkingEnv

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
REWARD_WEIGHTS = [100, 100, 5, 5, 1, 1]
ACTION_FREQUENCY = 3
SIMULATION_FREQUENCY = 15

config = {
    "screen_width": SCREEN_WIDTH,
    "screen_height": SCREEN_HEIGHT,
    "vehicles_count": PARKED_CARS,
    "policy_frequency": ACTION_FREQUENCY,
    "simulation_frequency": SIMULATION_FREQUENCY,
    "reward_weights": REWARD_WEIGHTS,
    "collision_reward": COLLISION_PENALTY
}

def make_custom_env():
    def _init():
        env = ParkingEnv(config=config)
        return env
    return _init

if __name__ == "__main__":
    n_cpu = 4
    batch_size = 64
    n_steps = batch_size * 30
    timesteps = 200000

    if TRAIN:
        env = SubprocVecEnv([make_custom_env() for _ in range(n_cpu)])
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=batch_size, n_steps=n_steps,
                    learning_rate=1e-3, n_epochs=5, gamma=0.95, device="cpu", ent_coef=0.05)
        model.learn(total_timesteps=timesteps)
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
        while True and t < RECORD_LIMIT and e < EPISODE_RECORD_LIMIT:
            e += 1
            obs, info = env.reset()
            done = False
            while not done and t < RECORD_LIMIT:
                t += 1
                action, _states = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Step: {t}, Reward: {reward}")
                frame = env.render()
                if RECORD:
                    frames.append(frame)

        if RECORD:
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"GIF saved to {gif_filename}")