import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint
# %matplotlib inline

env = gymnasium.make('parking-v0', render_mode='rgb_array')
obs, reward = env.reset()
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Observation:", obs)
print("Reward:", reward)

# for _ in range(3):
#     action = env.unwrapped.action_type.actions_indexes["IDLE"]
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

# plt.imshow(env.render())
# plt.show()

pprint.pprint(env.unwrapped.config)