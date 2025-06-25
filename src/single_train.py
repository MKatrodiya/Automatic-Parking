import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import datetime
import csv
import os
from custom_environment import custom_parking_env
from evaluation import evaluate_model
from plotting import generate_plots


# Training and recording parameters
TRAIN = False # Set to True to train the model, False to evaluate
RECORD = False # Set to True to record the GIF during the evaluation
CONTINUE_TRAINING = False # Set to True to continue training from a previously saved model
RECORD_LIMIT = 1000 # Frames to record in the GIF
EPISODE_RECORD_LIMIT = 10 # Episodes to record in the GIF
previous_model_path = '../res/parking_policy/model' # Path to a previously trained model to continue

# Environment Parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
COLLISION_REWARD = -5 # Reward for collisions, negative value
REWARD_WEIGHTS = [1, 0.3, 0.00, 0.00, 0.02, 0.02] # Weights for the reward function components
# REWARD_WEIGHTS = [3, 3, 0.01, 0.01, 0.2, 0.2]
STEERING_RANGE = np.deg2rad(45) # Steering range in radians
ACTION_FREQUENCY = 5 #(Hz) How many times per second an action is taken
SIMULATION_FREQUENCY = 15 #(Hz) How many times per second the simulation is updated, for physics and rendering
REWARD_THRESHOLD = 0.12 # Reward threshold for success, positive value
ADD_WALLS = True # Whether to add walls to the environment
STATIC_VEHICLES = list(range(28))  # Static vehicles are parked cars, can be a list of lane indices or vehicle objects
free_spaces = [3, 4, 9, 12, 15, 18, 22]
STATIC_VEHICLES = [v for v in STATIC_VEHICLES if v not in free_spaces]
# filled_spaces_n = 12 # Must be <= 21
# static_vehicles = list(np.random.choice(static_vehicles, filled_spaces_n, replace=False))
PARKED_CARS = len(STATIC_VEHICLES) # Number of parked cars based on the static vehicles list

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
    'static_vehicles': STATIC_VEHICLES
}

def linear_schedule(initial_value):
    """ Create a linear schedule for the learning rate.
    The learning rate will decrease linearly from `initial_value` to 0 over the course
    of training.
    Args:
        initial_value: The initial value of the learning rate.
    Returns:
        function: A function that takes a progress remaining (between 0 and 1)"""
    def schedule(progress_remaining: float):
        return progress_remaining * initial_value
    return schedule

def make_env(config, timestamp, rank=0):
    """
    Create an environment with monitor wrapper to log training data.
    Args:
        config (dict): Configuration for the environment.
        timestamp (str): Timestamp for logging.
        rank (int): Environment instance rank.
    Returns:
        function: A function that initializes the environment.
    """
    os.makedirs(f"../res/logs/parking_policy/{timestamp}", exist_ok=True)
    def _init():
        env = custom_parking_env.CustomParkingEnv(config=config)
        monitor_path = f"../res/logs/parking_policy/{timestamp}/monitor_{rank}.csv"
        return Monitor(env, filename=monitor_path)
    return _init

if __name__ == "__main__":
    n_envs = 8 # Number of parallel environments
    batch_size = 64 # Batch size for training
    n_steps = batch_size * 30  # Number of steps to run in each environment before updating the model
    timesteps = 20000  # Total timesteps for training 
    
    if TRAIN:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env = SubprocVecEnv([make_env(config, timestamp, rank=i) for i in range(n_envs)])
        model = PPO("MultiInputPolicy", env, verbose=1, batch_size=batch_size, n_steps=n_steps,
                learning_rate=1e-3, n_epochs=5, gamma=0.95, device="cpu", ent_coef=0.005, 
                tensorboard_log=f"../res/logs/parking_policy/{timestamp}/")
        
        # Save the config to a CSV file
        os.makedirs(f"../res/logs/parking_policy/{timestamp}", exist_ok=True)
        with open(f"../res/logs/parking_policy/{timestamp}/params.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(config.items())
               
        if CONTINUE_TRAINING and previous_model_path:
            # Load a previously trained model to continue training
            model = PPO.load(previous_model_path, env=env)
            print(f"Loaded model from {previous_model_path}")

        # Train the model
        model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=timestamp)
        model.save("../res/parking_policy/model")
        model.save(f"../res/logs/parking_policy/{timestamp}/model")
        del model

        generate_plots("../res/logs/parking_policy") # Generate plots from the training logs

    else:        
        # Load the trained model
        rm = "rgb_array"
        model = PPO.load("../res/parking_policy/model")
        model.tensorboard_log = None
        env = gym.make("CustomParking-v0", config=config, render_mode=rm)
        print(model.policy)
        deterministic = True
        evaluate_model(model, env, num_episodes=1000, render=RECORD, record_limit=RECORD_LIMIT, deterministic=deterministic)
        exit()

        env.reset()
        
        env.render()
        frames = []
        gif_filename = f"../res/parking_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        t = 0
        e = 0
        while True and t < RECORD_LIMIT and e < EPISODE_RECORD_LIMIT:
            e += 1
            obs, info= env.reset()
            done = False
            total_reward = 0
            
            while not done and t < RECORD_LIMIT:
                t += 1
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                # print(f"Step: {t}, Total Reward: {reward}")
                frame = env.render()
                if RECORD:
                    frames.append(frame)
        if RECORD:
            imageio.mimsave(gif_filename, frames, fps=SIMULATION_FREQUENCY)
            print(f"GIF saved to {gif_filename}")
