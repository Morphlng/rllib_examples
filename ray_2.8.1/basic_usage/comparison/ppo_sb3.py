"""
Author: Morphlng
Date: 2024-02-25 13:38:58
LastEditTime: 2024-02-29 21:52:05
LastEditors: Morphlng
Description: Example script on how to train, save, load, and test a stable baselines 2 agent
            Code taken and adjusted from SB2 docs:
            https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html
FilePath: /rllib_examples/ray_2.8.1/basic_usage/comparison/ppo_sb3.py
"""

import gymnasium as gym
from stable_baselines3 import PPO

# settings used for both stable baselines and rllib
env_name = "CartPole-v1"
train_steps = 100000
learning_rate = 1e-3
save_dir = "saved_models"

save_path = f"{save_dir}/sb_model_{train_steps}steps"
env = gym.make(env_name)

# training and saving
model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=1)
model.learn(total_timesteps=train_steps)
model.save(save_path)
print(f"Trained model saved at {save_path}")

# delete and load model (just for illustration)
del model, env
model = PPO.load(save_path)
print(f"Agent loaded from saved model at {save_path}")

# inference
env = gym.make(env_name, render_mode="human")
obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Cart pole ended after {i} steps.")
        break
