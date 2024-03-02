"""
Author: Morphlng
Date: 2024-02-28 22:37:33
LastEditTime: 2024-03-01 15:37:48
LastEditors: Morphlng
Description: Collect offline data during training.
FilePath: /rllib_examples/ray_2.8.1/basic_usage/offline/collect_offline_data.py
"""

from __future__ import annotations

import argparse
import importlib
import os

import ray.rllib.algorithms as algos
from ray import train, tune


def import_algo(name: str) -> tuple[type[algos.Algorithm], type[algos.AlgorithmConfig]]:
    """Try import algorithm from `ray.rllib.algorithms.name`

    Args:
        name (str): name of the algorithm

    Returns:
        tuple[Algorithm, AlgorithmConfig]: algorithm and its config
    """

    algo_path = os.path.dirname(algos.__file__)
    module_path = os.path.join(algo_path, name.lower())
    if not os.path.exists(module_path):
        raise AttributeError(f"Algorithm {name} not found!")
    else:
        algo_module = importlib.import_module(f"ray.rllib.algorithms.{name.lower()}")
        name = name.upper()
        algo = getattr(algo_module, name)
        config = getattr(algo_module, f"{name}Config")
        return algo, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--max_file_size", type=int, default=5000000)
    parser.add_argument("--max_steps", type=int, default=100000)

    # Note: Recording and rendering in this example
    # should work for both local_mode=True|False.
    args = parser.parse_args()
    algo, algoConfig = import_algo(args.algo)

    os.makedirs(args.output, exist_ok=True)

    # Example config switching on rendering.
    config = (
        algoConfig()
        .environment(args.env)
        .framework(args.framework)
        .offline_data(output=args.output, output_max_file_size=args.max_file_size)
        .resources(num_gpus=1)
    )

    stop = {
        "timesteps_total": args.max_steps,
    }

    result = tune.Tuner(
        algo,
        param_space=config.to_dict(),
        run_config=train.RunConfig(stop=stop),
    ).fit()

    print(result.get_best_result())
