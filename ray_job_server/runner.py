import logging
import os

from ray import train, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy import Policy
from ray.tune.registry import get_trainable_cls

from customization import (
    CustomCallback,
    create_checkpoint_loading_callback,
    report_progress,
)


class ExperimentRunner:
    def __init__(self, config: dict):
        # TODO: Config dataclass
        self.config = config
        self.task_id = config["task_id"]
        self.task_name = config["task_name"]
        self.task_type = config["task_type"]
        self.task_stage = config["task_stage"]
        self.task_config = config["task_config"]
        self.run_config = config.get("run_config", {"storage": "/mnt/ray/ray_results"})
        self.resume = config.get("resume", False)

    def run(self):
        if self.resume:
            return self.resume_job()
        elif self.task_stage == "pretrain":
            return self.run_pretrain()
        elif self.task_stage == "evolve":
            return self.run_evolve()
        elif self.task_stage == "transfer":
            return self.run_transfer()
        elif self.task_stage == "inference":
            return self.run_inference()
        else:
            raise ValueError(f"Invalid task_stage: {self.task_stage}")

    def get_algo_config(self, stage_config: dict):
        hyperparams: dict = stage_config["hyperparams"]
        base_model_config: dict = stage_config.get("base_model", {})
        resource_config: dict = stage_config.get("resources", {})
        algorithm: str = self.task_config["pretrain"]["algorithm"]

        trainable = get_trainable_cls(algorithm)
        default_config: AlgorithmConfig = trainable.get_default_config()
        if base_model_config:
            ckpt_dir = base_model_config["checkpoint_dir"]
            callback = create_checkpoint_loading_callback(ckpt_dir)
        else:
            callback = CustomCallback

        algo_config = (
            default_config.framework("torch")
            .training(model={"custom_model": None, "custom_model_config": {}})
            .environment(self.task_type)
            .debugging(log_level="ERROR")
            .callbacks(callback)
            .rollouts(num_rollout_workers=resource_config.get("num_workers", 1))
            .resources(num_gpus=resource_config.get("num_gpus", 0))
            .training(
                lr=hyperparams.get("lr", 1e-4),
                train_batch_size=hyperparams.get("train_batch_size", 256),
                gamma=hyperparams.get("gamma", 0.99),
            )
        )

        return algo_config

    def get_tuner(self, algo_config: AlgorithmConfig, stage_config: dict):
        checkpoint_config = stage_config["checkpoint"]
        stop_config = stage_config["stop"]

        return tune.Tuner(
            algo_config.algo_class,
            param_space=algo_config.to_dict(),
            run_config=train.RunConfig(
                name=self.task_id,
                storage_path=self.run_config["storage"],
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=checkpoint_config.get("num_to_keep", 1),
                    checkpoint_score_attribute=checkpoint_config.get(
                        "checkpoint_score_attribute", "episode_reward_mean"
                    ),
                    checkpoint_score_order=checkpoint_config.get(
                        "checkpoint_score_order", "max"
                    ),
                    checkpoint_at_end=checkpoint_config.get("checkpoint_at_end", True),
                    checkpoint_frequency=checkpoint_config.get(
                        "checkpoint_frequency", 50
                    ),
                ),
                stop=stop_config,
                log_to_file=True,
            ),
        )

    def run_pretrain(self):
        pretrain_config = self.task_config["pretrain"]
        dataset_config = pretrain_config["dataset"]
        data_path = dataset_config["data_path"]

        algo_config = self.get_algo_config(pretrain_config)
        algo_config = algo_config.offline_data(input_=data_path)

        policy_config = pretrain_config["pretrain_policy"]["config"]
        if hasattr(algo_config, "replay_buffer_config"):
            algo_config.replay_buffer_config.update(
                {
                    "capacity": policy_config.get("capacity", 50000),
                }
            )

        tuner = self.get_tuner(algo_config, pretrain_config)
        return tuner.fit()

    def run_evolve(self):
        evolve_config = self.task_config["evolve"]
        algo_config = self.get_algo_config(evolve_config)
        tuner = self.get_tuner(algo_config, evolve_config)
        return tuner.fit()

    def run_transfer(self):
        transfer_config = self.task_config["transfer"]
        algo_config = self.get_algo_config(transfer_config)
        tuner = self.get_tuner(algo_config, transfer_config)
        return tuner.fit()

    def run_inference(self):
        from datetime import datetime

        import numpy as np
        from ray.tune.registry import ENV_CREATOR, _global_registry

        logger = logging.getLogger("ray")

        inference_config = self.task_config["inference"]
        env_config = inference_config["env_config"]
        metrics = inference_config["metrics"]
        params = inference_config["params"]

        agent_config = inference_config["base_model"]
        checkpoint_dir = agent_config["checkpoint_dir"]
        policy_name = agent_config.get("policy_name", "default_policy")

        policies = Policy.from_checkpoint(checkpoint_dir)
        policy = policies[policy_name] if isinstance(policies, dict) else policies
        logger.info(f"Loaded policy '{policy_name}' from '{checkpoint_dir}'")

        try:
            env_creator = _global_registry.get(ENV_CREATOR, self.task_type)
            env = env_creator(env_config)
        except ValueError as e:
            logger.error(f"Environment {self.task_type} not registered.")
            raise ValueError(f"Environment {self.task_type} not registered.") from e

        start_time = datetime.now().isoformat()
        reward_data = []
        step_data = []
        metric_data = []
        for i in range(params["num_episodes"]):
            obs, _ = env.reset()
            state = policy.get_initial_state()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action, state, _ = policy.compute_single_action(
                    obs, state, explore=params["explore"]
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1

            reward_data.append(episode_reward)
            step_data.append(steps)

            # TODO: Add metrics
            trial_data = {
                "iter": i + 1,
                "reward": np.mean(reward_data),
                "episode_reward_max": np.max(reward_data),
                "episode_reward_min": np.min(reward_data),
                "episode_len_mean": np.mean(step_data),
            }
            report_progress(self.task_name, "RUNNING", start_time, trial_data)

        report_progress(self.task_name, "TERMINATED", start_time, trial_data)
        logger.info(f"Finished inference for {params['num_episodes']} episodes.")

    def resume_job(self):
        root = self.run_config["storage"]
        path = os.path.join(os.path.expanduser(root), self.task_id)
        algorithm = self.task_config["pretrain"]["algorithm"]
        tuner = tune.Tuner.restore(path, get_trainable_cls(algorithm))
        return tuner.fit()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    if args.config.endswith(".json"):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)

    runner = ExperimentRunner(config)
    results = runner.run()
