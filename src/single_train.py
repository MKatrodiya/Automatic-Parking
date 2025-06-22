import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from highway_env.envs import ParkingEnv
import torch
from custom_environment import custom_parking_env
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import datetime
import csv
import os


# Training and recording parameters
TRAIN = False
RECORD = True
RECORD_LIMIT = 1000 # Frames to record in the video
EPISODE_RECORD_LIMIT = 100
previous_model_path = 'parking_policy/model' # Path to a previously trained model to continue

# Environment Parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
COLLISION_REWARD = -5
REWARD_WEIGHTS = [1.0, 0.3, 0.0, 0.0, 0.00, 0.00]
STEERING_RANGE = np.deg2rad(45)
ACTION_FREQUENCY = 5 #(Hz) How many times per second an action is taken
SIMULATION_FREQUENCY = 15 #(Hz) How many times per second the simulation is updated, for physics and rendering
REWARD_THRESHOLD = 0.12
ADD_WALLS = True
static_vehicles = list(range(28))  # Static vehicles are parked cars, can be a list of lane indices or vehicle objects
free_spaces = [3, 4, 9, 12, 15, 18, 22]
static_vehicles = [v for v in static_vehicles if v not in free_spaces]
# filled_spaces_n = 12 # Must be <= 21 
# static_vehicles = list(np.random.choice(static_vehicles, filled_spaces_n, replace=False))
# print(f"Static vehicles (parked cars) selected: {static_vehicles}")
PARKED_CARS = len(static_vehicles) # Number of parked cars based on the static vehicles list

# Environment Parameters
config = {
    'comment': 'New reward function',
    'other_vehicles_type': 'parked',
    'screen_width': SCREEN_WIDTH,
    'screen_height': SCREEN_HEIGHT,
    'vehicles_count': PARKED_CARS,
    'policy_frequency': ACTION_FREQUENCY,
    'simulation_frequency': SIMULATION_FREQUENCY,
    'reward_weights': REWARD_WEIGHTS,
    'success_goal_reward': REWARD_THRESHOLD,
    'collision_reward': COLLISION_REWARD,
    'steering_range': STEERING_RANGE,
    'add_walls': ADD_WALLS,
    'static_vehicles': static_vehicles
}


def linear_schedule(initial_value):
    def schedule(progress_remaining: float):
        return progress_remaining * initial_value
    return schedule

if __name__ == "__main__":
    n_envs = 8
    batch_size = 256
    n_steps = 256
    timesteps = 3e6
    learning_rate = 1e-3
    n_epochs = 10
    gamma = 0.95
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.Tanh
    )
    target_kl = 0.03


    #env = gym.make("parking-v0")

    if TRAIN:
        env = make_vec_env(lambda: custom_parking_env.CustomParkingEnv(config=config), n_envs = n_envs, vec_env_cls=SubprocVecEnv)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=batch_size, n_steps=n_steps,
                learning_rate=learning_rate, n_epochs=n_epochs, gamma=gamma, target_kl=target_kl, device="cpu", policy_kwargs=policy_kwargs,
                tensorboard_log=f"logs/parking_policy/{timestamp}/")
        
        # Save the config to a CSV file
        os.makedirs(f"logs/parking_policy/{timestamp}", exist_ok=True)
        with open(f"logs/parking_policy/{timestamp}/params.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(config.items())
               
        # if previous_model_path:
        #     # Load a previously trained model to continue training
        #     model = PPO.load(previous_model_path, env=env)
        #     print(f"Loaded model from {previous_model_path}")

        # Train the model
        model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=timestamp)
        model.save("parking_policy/model")
        model.save(f"logs/parking_policy/{timestamp}/model")
        del model

    else:        
        def evaluate_model(model, env, num_episodes=10, render=False, record_limit=100, deterministic=True):
            rewards = []
            successes = 0
            for ep in range(num_episodes):
                obs, info = env.reset()
                done = False
                total_reward = 0
                t = 0
                frames = []
                while not done and t < record_limit:
                    t += 1
                    action, _states = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    total_reward += reward
                    if render:
                        frame = env.render()
                        frames.append(frame)
                rewards.append(total_reward)

                if info["is_success"]:
                    successes += 1
                if render and frames:
                    gif_filename = f"../res/parking_eval_ep{ep+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    path = os.path.dirname(gif_filename)
                    os.makedirs(path, exist_ok=True)
                    imageio.mimsave(gif_filename, frames, fps=30)
                    print(f"Episode {ep+1} GIF saved to {gif_filename}")
                avg_reward = sum(rewards) / len(rewards)
            success_rate = successes / num_episodes * 100
            print(f"Average reward over {num_episodes} episodes: {avg_reward}")
            print(f"Success rate: {success_rate:.2f}% ({successes}/{num_episodes})")

        # Load the trained model
        # parking_obstacles.register_env()
        rm = "rgb_array"
        model = PPO.load("parking_policy/model")
        model.tensorboard_log = None
        env = gym.make("CustomParking-v0", config=config, render_mode=rm)
        
        # print(model.policy)

        evaluate_model(model, env, num_episodes=EPISODE_RECORD_LIMIT, render=RECORD, record_limit=RECORD_LIMIT, deterministic=True)
