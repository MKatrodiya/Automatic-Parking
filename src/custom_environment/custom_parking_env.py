import numpy as np
from gymnasium.envs.registration import register

from highway_env.envs import parking_env
from highway_env.vehicle.controller import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle

class CustomParkingEnv(parking_env.ParkingEnv):
    """Custom Parking Environment with custom observation and action handling."""

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Other vehicles
        locations = self.config["static_vehicles"]
        valid_spots = [3, 4, 9, 12, 15, 18, 22]
        goal_lane_index = empty_spots[int(np.random.choice(valid_spots))]
        # print(goal_lane_index)
        goal_lane = self.road.network.get_lane(goal_lane_index)
        filled_spaces_n = 12 # Must be <= 21 
        locations = list(np.random.choice(locations, filled_spaces_n, replace=False))
        locations = sorted(locations, reverse=True)
        # print(locations)
        for i in range(filled_spaces_n):
            if not empty_spots:
                continue
            lane_index = empty_spots[locations[i]]
            v = Vehicle.make_on_lane(self.road, lane_index, 4, speed=0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = (i - self.config["controlled_vehicles"] // 2) * 10
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0], 2 * np.pi * self.np_random.uniform(), 0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            #vehicle_spot_index = empty_spots.index(vehicle.lane_index)
            #pp(vehicle_spot_index)

            # empty_spots.remove(vehicle.lane_index)

        lane = goal_lane
        # Goal
        for vehicle in self.controlled_vehicles:
            # lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            # lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            # print("goal index: ",lane_index)
            # empty_spots.remove(lane_index)



        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
 
register(
    id="CustomParking-v0",
    entry_point="custom_environment.custom_parking_env:CustomParkingEnv",
)
