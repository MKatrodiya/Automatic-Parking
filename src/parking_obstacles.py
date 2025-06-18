

from __future__ import annotations


import numpy as np
import gymnasium as gym
from gymnasium import Env

from highway_env.envs.common.observation import ObservationType
from highway_env.envs.parking_env import ParkingEnv

class CustomParkingEnv(ParkingEnv):
    """
    Custom Parking Environment with parked vehicles as obstacles.
    """
    def __init__(self, config=None, render_mode: str | None = None) :
        super().__init__(config, render_mode=render_mode)
        # self.observation_space = KinematicsWithParkedObs().space()


class KinematicsWithParkedObs(ObservationType):
    def space(self):
        num_parked = 10
        obs_dim = 6 + 2 * num_parked
        low = np.array([-1e3] * obs_dim)
        high = np.array([1e3] * obs_dim)
        return self.np_random.uniform(low=low, high=high).shape

    def observe(self):
        ego = self.env.controlled_vehicles[0]
        obs = [ego.position[0], ego.position[1],
               ego.velocity[0], ego.velocity[1],
               np.cos(ego.heading), np.sin(ego.heading)]

        parked = [v for v in self.env.road.vehicles if v not in self.env.controlled_vehicles]
        for v in parked[:10]:
            obs += [v.position[0], v.position[1]]
        while len(obs) < 6 + 2 * 10:
            obs += [0.0, 0.0]

        return {
            "observation": np.array(obs, dtype=np.float32),
            "desired_goal": ego.goal.position if ego.goal else np.zeros(2),
            "achieved_goal": np.array(ego.position, dtype=np.float32)
        }
    

def register_env():
    import gymnasium as gym
    from gymnasium.envs.registration import register

    register(
        id="customparking-v0",
        entry_point="parking_obstacles:CustomParkingEnv",
    )

if __name__ == "__main__":
    register_env()
    env = gym.make("customparking-v0", render_mode='rgb_array')
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
    env.close()