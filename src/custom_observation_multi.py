# custom_observation_multi.py
import numpy as np
import gymnasium as gym

class CustomParkingObservation:
    """Single-agent observation that includes two controlled vehicles and N parked vehicles."""

    def __init__(self, env, n_parked=None):
        self.env = env
        self.n = n_parked

    def space(self):
        return gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12 + 2 * self.n,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6 * 2,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6 * 2,), dtype=np.float32),
        })

    def observe(self):
        assert len(self.env.controlled_vehicles) >= 2, "Need two controlled vehicles"

        ego_1 = self.env.controlled_vehicles[0]
        ego_2 = self.env.controlled_vehicles[1]

        def encode_ego(ego):
            return [
                ego.position[0], ego.position[1],
                ego.velocity[0], ego.velocity[1],
                np.cos(ego.heading), np.sin(ego.heading)
            ]

        ego_obs_1 = encode_ego(ego_1)
        ego_obs_2 = encode_ego(ego_2)

        # collect parked vehicle positions
        parked = []
        cnt = 0
        for v in self.env.road.vehicles:
            if v in self.env.controlled_vehicles:
                continue
            parked.extend([v.position[0], v.position[1]])
            cnt += 1
            if cnt >= self.n:
                break

        while cnt < self.n:
            parked.extend([0.0, 0.0])
            cnt += 1

        # achieved goal: both ego states
        achieved_goal = np.array(ego_obs_1 + ego_obs_2, dtype=np.float32)

        # desired goal: each goal's position + zero velocity + heading
        def encode_goal(ego):
            if hasattr(ego, "goal") and ego.goal is not None:
                if isinstance(ego.goal, dict):
                    goal_pos = ego.goal.get("position", [0.0, 0.0])
                    goal_heading = ego.goal.get("heading", 0.0)
                else:
                    goal_pos = ego.goal.position
                    goal_heading = getattr(ego.goal, "heading", 0.0)
            else:
                goal_pos = [0.0, 0.0]
                goal_heading = 0.0
            return [
                goal_pos[0], goal_pos[1], 0.0, 0.0,
                np.cos(goal_heading), np.sin(goal_heading)
            ]

        desired_goal = np.array(encode_goal(ego_1) + encode_goal(ego_2), dtype=np.float32)

        observation = np.array(ego_obs_1 + ego_obs_2 + parked, dtype=np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }
