# new_parking_env_multi.py
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import ContinuousAction
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from custom_observation_multi import CustomParkingObservation  # your updated file
from types import SimpleNamespace

class ParkingEnv(AbstractEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config=config, render_mode=render_mode)
        self.define_spaces()

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "action": {"type": "ContinuousAction"},
            "vehicles_count": 5,
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02, 1, 0.3, 0, 0, 0.02, 0.02],
            "collision_reward": -5,
            "success_goal_reward": 0.1,
            "duration": 100,
            "add_walls": True,
            "controlled_vehicles": 2,
        })
        return config

    def define_spaces(self):
        self.action_type = ContinuousAction(self)
        self.action_space = self.action_type.space()
        self.observation_type = CustomParkingObservation(self, n_parked=self.config["vehicles_count"])
        self.observation_space = self.observation_type.space()

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 14) -> None:
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset + length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset - length], width=width, line_types=lt))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        empty_spots = list(self.road.network.lanes_dict().keys())
        self.controlled_vehicles = []

        # Controlled vehicles
        for i in range(self.config["controlled_vehicles"]):
            x0 = (i - self.config["controlled_vehicles"] // 2) * 10
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0], 2 * np.pi * self.np_random.uniform(), 0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal for each controlled vehicle
        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            goal_pos = lane.position(lane.length / 2, 0)
            goal_heading = lane.heading
            vehicle.goal = SimpleNamespace(
                position=goal_pos,
                velocity=np.array([0.0, 0.0]),
                heading=goal_heading,
                heading_cos=np.cos(goal_heading),
                heading_sin=np.sin(goal_heading),
                collidable=False
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # Parked vehicles
        for _ in range(self.config["vehicles_count"]):
            if not empty_spots:
                break
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, 4, speed=0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
                self.road.objects.append(obstacle)

    """def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        reward += self.config["collision_reward"] * sum(v.crashed for v in self.controlled_vehicles)
        return reward"""
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        # smooth
        diff = achieved_goal - desired_goal
        weighted_sq_diff = np.dot(diff ** 2, np.array(self.config["reward_weights"]))
        return -0.0001 * weighted_sq_diff
    
    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        reward += self.config["collision_reward"] * sum(v.crashed for v in self.controlled_vehicles)
        return reward





    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})

class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 5})

