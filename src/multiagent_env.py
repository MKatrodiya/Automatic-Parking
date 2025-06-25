import highway_env
import gymnasium as gym
import imageio

num_agents = 6

config = {
    'controlled_vehicles': num_agents,
    'action': {
        'type': "MultiAgentAction",
        'action_config': {
            "type": "ContinuousAction",
        }
    }
}

env = gym.make('parking-v0', render_mode='rgb_array', config=config)
obs, reward = env.reset()

frames = []
num_steps = 100  # Set how many frames you want in your GIF

for _ in range(num_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # Returns an RGB array
    frames.append(frame)

env.close()

# Save frames as GIF
imageio.mimsave('parking_agents.gif', frames, fps=15)
