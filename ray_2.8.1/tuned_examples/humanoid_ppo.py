"""
Author: Morphlng
Date: 2024-03-01 15:40:35
LastEditTime: 2024-03-02 09:17:15
LastEditors: Morphlng
Description: Example of PPO with GAE on Humanoid. Ported from https://github.com/ray-project/ray/blob/ray-2.8.1/rllib/tuned_examples/ppo/humanoid-ppo-gae.yaml
FilePath: /rllib_examples/ray_2.8.1/tuned_examples/humanoid_ppo.py
"""

from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .training(
        gamma=0.995,
        lr=0.0001,
        lambda_=0.95,
        clip_param=0.2,
        kl_coeff=1.0,
        num_sgd_iter=20,
        sgd_minibatch_size=32768,
        train_batch_size=320000,
        model={"free_log_std": True},
        _enable_learner_api=False,
    )
    .framework("torch")
    .environment("Humanoid-v4")
    .rollouts(
        num_envs_per_worker=2,
        num_rollout_workers=3,
        batch_mode="complete_episodes",
        observation_filter="MeanStdFilter",
    )
    .evaluation(evaluation_interval=10, evaluation_duration=1)
    .resources(num_gpus=1, num_cpus_per_worker=4)
    .rl_module(_enable_rl_module_api=False)
)


stop = {
    "sampler_results/episode_reward_mean": 6000,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        name="humanoid_ppo",
        storage_path="/mnt/nfs_share",
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=5, checkpoint_at_end=True
        ),
        stop=stop,
    ),
).fit()
