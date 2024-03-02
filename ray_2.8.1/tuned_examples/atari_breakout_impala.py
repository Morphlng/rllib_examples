"""
Author: Morphlng
Date: 2024-03-01 15:40:35
LastEditTime: 2024-03-02 09:21:51
LastEditors: Morphlng
Description: Example of IMPALA on Atari Breakout. Ported from https://github.com/ray-project/ray/blob/ray-2.8.1/rllib/tuned_examples/impala/atari-impala.yaml.
FilePath: /rllib_examples/ray_2.8.1/tuned_examples/atari_breakout_impala.py
"""

from ray import air, tune
from ray.rllib.algorithms.impala import ImpalaConfig

config = (
    ImpalaConfig()
    .training(
        train_batch_size=4000, lr_schedule=[[0, 0.0005], [20000000, 0.000000000001]]
    )
    .framework("torch")
    .environment(
        "ALE/Breakout-v5",
        env_config={
            "frameskip": 1,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        clip_rewards=True,
    )
    .rollouts(num_envs_per_worker=5, num_rollout_workers=3, rollout_fragment_length=50)
    .exploration(exploration_config={"random_timesteps": 10000})
    .evaluation(evaluation_interval=10, evaluation_duration=10)
    .resources(num_gpus=1, num_cpus_per_worker=2, num_gpus_per_worker=0.5)
)


stop = {
    "sampler_results/episode_reward_mean": 400,
}

tune.Tuner(
    "IMPALA",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        name="IMPALA_atari_breakout",
        storage_path="/mnt/nfs_share",
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=10, checkpoint_at_end=True
        ),
        stop=stop,
    ),
).fit()
