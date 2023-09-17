# This script is used to check gymnasium works in the Docker environment
import IPython
import gymnasium as gym

env = gym.make("Ant-v4", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
