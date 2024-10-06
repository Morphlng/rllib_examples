import os

from ray import train, tune
from ray.rllib.algorithms.dqn import DQNConfig

from customization.callbacks import CustomCallback

directory = os.path.abspath(os.getcwd())
save_to = os.path.join(directory, "data")


algo_config: DQNConfig = (
    DQNConfig()
    .framework("torch")
    .environment("uav_path_planning")
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
        name="uav_path_planning_data_collect",
        checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
        stop={"timesteps_total": 1000000},
    ),
)

tuner.fit()
