"""
Author: Morphlng
Date: 2024-02-28 22:37:33
LastEditTime: 2024-03-01 15:35:33
LastEditors: Morphlng
Description: Demonstrate how to load a raw modelv2 model and perform inference.
FilePath: /rllib_examples/ray_2.8.1/basic_usage/checkpoint/load_modelv2_torch_model.py
"""

import argparse
import os

import gymnasium as gym
import torch


def select_action_index(output):
    logits = output["action_dist_inputs"]
    return torch.argmax(logits).item()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env", type=str, default="CartPole-v1")
    argparser.add_argument(
        "--model", type=str, default=os.path.join(__file__, "./model/modelv2/model.pt")
    )
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device)
    model.eval()

    env = gym.make(args.env, render_mode="human")
    for i in range(10):
        obs, info = env.reset()
        reward_sum = 0
        terminated = truncated = False
        while not terminated and not truncated:
            output = model({"obs": torch.from_numpy(obs).unsqueeze(0).to(device)})
            obs, reward, terminated, truncated, info = env.step(
                select_action_index(output)
            )
            reward_sum += reward
            env.render()
        print(f"Episode {i} finished with reward {reward_sum}!")
