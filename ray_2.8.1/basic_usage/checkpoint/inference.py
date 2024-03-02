"""
Author: Morphlng
Date: 2024-02-28 22:37:33
LastEditTime: 2024-03-01 15:35:19
LastEditors: Morphlng
Description: Demonstrate how to load a trained agent and perform inference.
FilePath: /rllib_examples/ray_2.8.1/basic_usage/checkpoint/inference.py
"""
from __future__ import annotations

import argparse

import gymnasium as gym
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.policy import local_policy_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--rl_module", action="store_true")
    parser.add_argument("--no_explore", action="store_false")
    args = parser.parse_args()

    policy = Policy.from_checkpoint(args.checkpoint)
    if isinstance(policy, dict):
        policy = policy.get("default_policy", next(iter(policy.values())))

    env = gym.make(args.env, render_mode="human")
    if "ALE" in args.env:
        env = wrap_deepmind(env)

    for i in range(args.epoch):
        obs, info = env.reset()
        reward_sum = 0
        terminated = truncated = False
        while not terminated and not truncated:
            if args.rl_module:
                # Warning: compute_single_action will not load filters (E.g. MeanStdFilter)
                # Thus you may not get the same performance as the original training.
                action, state, info = policy.compute_single_action(
                    obs, explore=args.no_explore
                )
            else:
                # Rl-module does not compatible with this function
                policy_outputs = local_policy_inference(
                    policy, "env", "agent", obs, explore=args.no_explore
                )
                action, state, info = policy_outputs[0]

            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            env.render()
        print(f"Episode {i} finished with reward {reward_sum}!")

    env.close()
