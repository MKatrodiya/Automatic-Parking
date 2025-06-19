import numpy as np
import gymnasium as gym

class CustomParkingObservation:
    """Observation including ego + up to N parked vehicles positions, goal-based."""

    def __init__(self, env, n_parked=None):
        self.env = env
        self.n = n_parked

    def space(self):
        return gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 2 * self.n,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

    def observe(self):
        ego = self.env.controlled_vehicles[0]
        obs = [
            ego.position[0], ego.position[1],
            ego.velocity[0], ego.velocity[1],
            np.cos(ego.heading), np.sin(ego.heading)
        ]

        # collect parked vehicle positions
        parked = []
        cnt = 0
        for v in self.env.road.vehicles:
            if v is ego:
                continue
            parked.extend([v.position[0], v.position[1]])
            cnt += 1
            if cnt >= self.n:
                break

        # pad if fewer than self.n
        while cnt < self.n:
            parked.extend([0.0, 0.0])
            cnt += 1

        data = np.array(obs + parked, dtype=np.float32)
        achieved_goal = np.array(obs[:6], dtype=np.float32)
        # desired goal: goal 的位置 + 速度为0 + heading方向
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

        desired_goal = np.array([
            goal_pos[0], goal_pos[1], 0.0, 0.0, np.cos(goal_heading), np.sin(goal_heading)
        ], dtype=np.float32)


        return {
            "observation": data,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }
