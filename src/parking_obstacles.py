

from __future__ import annotations


import numpy as np
import gymnasium as gym
from gymnasium import Env

from highway_env.envs.common.observation import KinematicsGoalObservation
from highway_env.envs.parking_env import ParkingEnv
from highway_env.envs.common.action import Action, ActionType, action_factory

class CustomParkingEnv(ParkingEnv):
    """
    Custom Parking Environment with parked vehicles as obstacles.
    """
    def __init__(self, config=None, render_mode: str | None = None) :
        super().__init__(config, render_mode=render_mode)


    def define_spaces(self):
        super().define_spaces()
        print("Defining spaces for CustomParkingEnv")
        self.observation_type = KinematicsWithParkedObs(self)
        #self.action_type = action_factory(self, self.config["action"])
        #self.observation_space = self.observation_type.space()
        #self.action_space = self.action_type.space()

class KinematicsWithParkedObs(KinematicsGoalObservation):
    def __init__(self, env: Env, **kwargs: dict):
        scales = [100, 100, 5, 5, 1, 1]
        super().__init__(env, scales, **kwargs)

    def space(self):
        super_space = super().space()
        return super_space
        # 5 for ego, 5*4 for nearby vehicles, 5*3 for parked vehicles (example: 3 parked)
        # Adjust the number of parked vehicles as needed
        num_parked = len(getattr(self.env, "parked_vehicles", []))
        
        obs_dim = 5 + 5 * self.env.config["vehicles_count"] + 5 * num_parked
        return gym.spaces.Box(low=-1e3, high=1e3, shape=(obs_dim,), dtype=np.float32)

    def observe(self):
        obs = super().observe()
        vs = [(vehicle.position[0], vehicle.position[1]) for vehicle in getattr(self.env.road, "vehicles", [])]
        obs["parked_vehicles"] = np.array(vs, dtype=np.float32)
        return obs

def register_env():
    import gymnasium as gym
    from gymnasium.envs.registration import register

    register(
        id="customparking-v0",
        entry_point="parking_obstacles:CustomParkingEnv",
    )

if __name__ == "__main__":
    register_env()
    config = {
        'other_vehicles_type': 'parked',
        'vehicles_count': 4,
        'policy_frequency': 5,
        'simulation_frequency': 15,
        # 'reward_weights': REWARD_WEIGHTS
    }
    env = gym.make("customparking-v0", render_mode='rgb_array', config=config)
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
    env.close()