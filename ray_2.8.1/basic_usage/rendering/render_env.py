"""
Author: Morphlng
Date: 2024-02-28 22:37:33
LastEditTime: 2024-03-01 15:36:58
LastEditors: Morphlng
Description: classic_control environment rendering example. (RLlib cannot properly render these environment)
FilePath: /rllib_examples/ray_2.8.1/basic_usage/rendering/render_env.py
"""
from __future__ import annotations

import argparse
import importlib
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import ray.rllib.algorithms as algos
from ray import train, tune
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env


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


class RenderEnv(gym.Env):
    def __init__(self, configs: dict = {}):
        """This environment wraps a gym environment and renders it. The reason why we need this is because
        RLlib used `gymnasium.envs.classic_control.rendering` to render the environment, which is not available

        Args:
            configs (dict, optional): _description_. Defaults to {}.
        """

        self.env_name = configs.get("env", "CartPole-v1")
        self.render_mode = configs.get("render_mode", None)

        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        self.env.metadata["render_fps"] = 30

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        res = self.env.step(action)
        if self.render_mode is not None:
            self.render()
        return res

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


MultiAgentRenderEnv = make_multi_agent(RenderEnv)

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
    parser.add_argument("--multi-agent", action="store_true")
    parser.add_argument("--stop-iters", type=int, default=50)
    parser.add_argument("--stop-timesteps", type=int, default=1000000)
    parser.add_argument("--stop-reward", type=float, default=200.0)
    args = parser.parse_args()

    register_env(
        "multi_agent_env",
        lambda config: (
            config.update({"num_agents": 4, "env": args.env}),
            MultiAgentRenderEnv(config),
        )[-1],
    )
    register_env(
        "render_env",
        lambda config: (config.update({"env": args.env}), RenderEnv(config))[-1],
    )

    algo, algoConfig = import_algo(args.algo)

    # Example config switching on rendering.
    config = (
        algoConfig()
        .environment(
            "multi_agent_env" if args.multi_agent else "render_env",
        )
        .framework(args.framework)
        # Use a vectorized env with 2 sub-envs.
        .rollouts(num_envs_per_worker=2, num_rollout_workers=3)
        .evaluation(
            # Evaluate once per training iteration.
            evaluation_interval=min(20, args.stop_iters // 2),
            # Run evaluation on (at least) two episodes
            evaluation_duration=2,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            evaluation_num_workers=1,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            evaluation_config=algoConfig.overrides(
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                env_config={"render_mode": "human"},
            ),
        )
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .resources(num_gpus=1, num_gpus_per_worker=0, num_cpus_per_worker=2)
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "sampler_results/episode_reward_mean": args.stop_reward,
    }

    result = tune.Tuner(
        algo,
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            name=f"{args.algo}-{"ma" if args.multi_agent else "single"}-{args.env}",
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
            stop=stop,
        ),
    ).fit()

    print(result.get_best_result())
