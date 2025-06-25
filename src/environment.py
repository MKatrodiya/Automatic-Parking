import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
import time
from custom_environment import custom_parking_env

# vehicle to be controlled by the agent is called ego-vehicle
config = {
    "other_vehicles_type": "highway_env.vehicle.kinematics.KinematicsVehicle", # type of other vehicles in the environment
    "screen_width": 600, # width of the rendering window
    "screen_height": 600, # height of the rendering window
    "centering_position": [0.5, 0.5],
    "scaling": 5.5,
    "show_trajectories": False, # show trajectories of this vehicle
    "render_agent": True,
    "policy_frequency": 1,
    "vehicles_count": 10, # number of other vehicles in the environment
    "action": {
        # action: (throttle (acceleration), steering angle)
        # acceleration is in m/s^2, steering is in radians
        "type": "ContinuousAction",
    }
}

env = gymnasium.make("parking-v0", render_mode='rgb_array', config=config)
obs, reward = env.reset()
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Observation:", obs)
print("Reward:", reward)

for _ in range(10):
    action = env.action_space.sample() # Random action from the action space
    # An observation is a dictionary:
    # {
    #     "observation": np.array,  # The state of the environment
    #     "achieved_goal": np.array, # The achieved goal of the agent
    #     "desired_goal": np.array,  # The desired goal of the agent
    # }

    observation = obs["observation"]
    # The observation features are: ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h']
    # x, y: position of the ego vehicle
    # vx, vy: velocity components in x and y directions
    # cos_h, sin_h: cosine and sine of the heading angle (orientation)

    print("vehicle's current position (x, y):", observation[0:2])
    print("vehicle's current velocity (vx, vy):", observation[2:4])
    print("vehicle's heading (cos_h, sin_h):", observation[4:6])

    # To move the ego-car to the left or right, we need to change the steering angle.
    # To accelerate or decelerate, we change the acceleration value.
    # For the 'parking-v0' environment with 'ContinuousAction', the action is a 2D array:
    # [acceleration, steering]
    #   acceleration: positive to speed up, negative to slow down
    #   steering: negative to steer left, positive to steer right (in radians)

    # Car moves left or right relative to its current heading. 
    # Current heading is determined by the cosine and sine of the heading angle.
    # Example: Move left with acceleration
    # action = [1, -0.5]  # accelerate and steer left

    # Example: Move right with acceleration
    # action = [0.5, 0.2]   # accelerate and steer right

    print("Action:", action)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    time.sleep(2) # Pause for a few seconds to visualize the action effect
env.close()

# plt.imshow(env.render())
# plt.show()

pprint.pprint(env.unwrapped.config)