import os

from ray import train, tune
from ray.rllib.algorithms.dqn import DQNConfig

from customization.callbacks import CustomCallback

directory = os.path.abspath(os.path.dirname(__file__))
save_to = os.path.join(directory, "data")


algo_config: DQNConfig = (
    DQNConfig()
    .framework("torch")
    .environment("LocalPathPlanning")
    .callbacks(CustomCallback)
    .offline_data(output=save_to, output_compress_columns=[])
    .debugging(log_level="ERROR")
    .training(
        lr=1e-5,
        train_batch_size=256,
    )
)

tuner = tune.Tuner(
    "DQN",
    param_space=algo_config.to_dict(),
    run_config=train.RunConfig(
        name="uav2d",
        storage_path=None,
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            checkpoint_at_end=True,
        ),
        stop={"training_iteration": 100},
    ),
)

tuner.fit()