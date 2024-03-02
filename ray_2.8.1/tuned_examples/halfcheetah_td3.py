"""
Author: Morphlng
Date: 2024-03-01 15:40:35
LastEditTime: 2024-03-02 09:18:35
LastEditors: Morphlng
Description: Example of TD3 on HalfCheetah. Ported from https://github.com/ray-project/ray/blob/ray-2.8.1/rllib/tuned_examples/td3/mujoco-td3.yaml
FilePath: /rllib_examples/ray_2.8.1/tuned_examples/halfcheetah_td3.py
"""

from ray import train, tune
from ray.rllib.algorithms.td3 import TD3Config

config = (
    TD3Config()
    .framework("torch")
    .environment("HalfCheetah-v2")
    .rollouts(
        num_envs_per_worker=2,
        num_rollout_workers=3,
    )
    .exploration(exploration_config={"random_timesteps": 10000})
    .evaluation(evaluation_interval=10, evaluation_duration=10)
    .resources(num_gpus=1, num_cpus_per_worker=4, num_gpus_per_worker=0)
)


stop = {
    "timesteps_total": 1000000,
}

tune.Tuner(
    "TD3",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        name="halfcheetah_td3",
        storage_path="/mnt/nfs_share",
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=10, checkpoint_at_end=True
        ),
        stop=stop,
    ),
).fit()
