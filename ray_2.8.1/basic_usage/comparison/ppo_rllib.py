"""
Author: Morphlng
Date: 2024-02-25 13:38:21
LastEditTime: 2024-02-29 21:50:30
LastEditors: Morphlng
Description: Example script on how to train, save, load, and test an RLlib agent.
            Equivalent script with stable baselines: ppo_sb2.py.
            Demonstrates transition from stable_baselines to Ray RLlib.
FilePath: /rllib_examples/ray_2.8.1/basic_usage/ppo_rllib.py
"""

import os

import gymnasium as gym
import ray.rllib.algorithms.ppo as ppo
from ray import train, tune

# settings used for both stable baselines and rllib
env_name = "CartPole-v1"
train_steps = 100000
learning_rate = 1e-3
save_dir = os.path.abspath("./saved_models")

# training and saving
analysis = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"timesteps_total": train_steps},
        local_dir=save_dir,
        checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True,
        ),
    ),
    param_space={"env": env_name, "lr": learning_rate},
).fit()
# retrieve the checkpoint path
checkpoint_path = analysis.get_best_result().get_best_checkpoint(
    metric="episode_reward_mean", mode="max"
)
print(f"Trained model saved at {checkpoint_path}")

# load and restore model
agent = ppo.PPO(env=env_name)
agent.restore(checkpoint_path)
print(f"Agent loaded from saved model at {checkpoint_path}")

# inference
env = gym.make(env_name, render_mode="human")
obs, info = env.reset()
for i in range(1000):
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Cart pole ended after {i} steps.")
        break
