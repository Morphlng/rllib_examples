"""
Author: Morphlng
Date: 2024-03-01 15:40:35
LastEditTime: 2024-03-02 09:20:12
LastEditors: Morphlng
Description: Example of APPO on Multi-Agent CartPole. Ported from https://github.com/ray-project/ray/blob/ray-2.8.1/rllib/tuned_examples/appo/multi-agent-cartpole-appo.yaml
FilePath: /rllib_examples/ray_2.8.1/tuned_examples/ma_cartpole_appo.py
"""

from ray import train, tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.registry import register_env

register_env("env", lambda cfg: MultiAgentCartPole(config=cfg))


config = (
    APPOConfig()
    .training(
        num_sgd_iter=1,
        vf_loss_coeff=0.005,
        vtrace=True,
        model={
            "fcnet_hiddens": [32],
            "fcnet_activation": "linear",
            "vf_share_layers": True,
        },
    )
    .environment("env", env_config={"num_agents": 4})
    .rollouts(
        num_envs_per_worker=5,
        num_rollout_workers=4,
        observation_filter="MeanStdFilter",
    )
    .resources(num_gpus=1, num_cpus_per_worker=4, num_gpus_per_worker=0.2)
    .multi_agent(
        policies=["p0", "p1", "p2", "p3"],
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: f"p{agent_id}"),
    )
)

stop = {
    "sampler_results/episode_reward_mean": 600,  # 600 / 4 (==num_agents) = 150
    "timesteps_total": 1000000,
}

tune.Tuner(
    "APPO",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        name="MA_CartPole_APPO",
        storage_path="/mnt/nfs_share",
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=5, checkpoint_at_end=True
        ),
        stop=stop,
    ),
).fit()
