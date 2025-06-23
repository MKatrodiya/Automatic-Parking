import datetime
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
from highway_env.envs import ParkingEnv

def evaluate_model(model, env, num_episodes=10, render=False, record_limit=100, deterministic=True):
            rewards = []
            successes = 0
            for ep in range(num_episodes):
                obs, info = env.reset()
                done = False
                total_reward = 0
                t = 0
                frames = []
                while not done and t < record_limit:
                    t += 1
                    action, _states = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    total_reward += reward
                    if render:
                        frame = env.render()
                        frames.append(frame)
                rewards.append(total_reward)

                if info["is_success"]:
                    successes += 1
                if render and frames:
                    gif_filename = f"../res/parking_eval_ep{ep+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    imageio.mimsave(gif_filename, frames, fps=30)
                    print(f"Episode {ep+1} GIF saved to {gif_filename}")
                avg_reward = sum(rewards) / len(rewards)
            success_rate = successes / num_episodes * 100
            print(f"Average reward over {num_episodes} episodes: {avg_reward}")
            print(f"Success rate: {success_rate:.2f}% ({successes}/{num_episodes})")

if __name__ == "__main__":
    
    # Load the trained model
    model = PPO.load("parking_policy/model")
    env = gym.make("parking-v0", config={"some_config_key": "some_config_value"}, render_mode="rgb_array")

    # Evaluate the model
    evaluate_model(model, env, num_episodes=10, render=True, record_limit=100)