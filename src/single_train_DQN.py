import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from highway_env.envs import ParkingEnv
import imageio

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import datetime

# Training and recording parameters
TRAIN = True
RECORD = False
RECORD_LIMIT = 100  # Frames to record in the video
EPISODE_RECORD_LIMIT = 1

# Environment Parameters
PARKED_CARS = 0
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
COLLISION_PENALTY = -10
REWARD_WEIGHTS = [100, 100, 5, 5, 1, 1]

# Environment Parameters
config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": REWARD_WEIGHTS,
        "normalize": True,
    },
    'other_vehicles_type': 'parked',
    'screen_width': SCREEN_WIDTH,
    'screen_height': SCREEN_HEIGHT,
    'vehicles_count': PARKED_CARS,
    "action": {
        "type": "DiscreteMetaAction"  # Set action space to discrete
    },
    # 'reward_weights': REWARD_WEIGHTS
}

if __name__ == "__main__":
    batch_size = 64
    train_timesteps = 2000000

    if TRAIN:
        env = make_vec_env("parking-v0", n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs={'config': config})
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            batch_size=batch_size,
            learning_rate=1e-3,
            gamma=0.8,
            buffer_size=50000,
            learning_starts=1000,
            target_update_interval=500,
            train_freq=4,
        )
        model.learn(total_timesteps=train_timesteps)
        model.save("parking_policy/dqn_model")
        del model

    else:
        rm = "rgb_array"
        model = DQN.load("parking_policy/dqn_model")
        env = gym.make("parking-v0", config=config, render_mode=rm)

        env.reset()
        env.render()
        frames = []
        gif_filename = f"parking_run_dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        t = 0
        e = 0
        while True and t < RECORD_LIMIT and e < EPISODE_RECORD_LIMIT:
            e += 1
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done and t < RECORD_LIMIT:
                t += 1
                action, _states = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                print(f"Step: {t}, Total Reward: {reward}")
                frame = env.render()
                if RECORD:
                    frames.append(frame)
        if RECORD:
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"GIF saved to {gif_filename}")
