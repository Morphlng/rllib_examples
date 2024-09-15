from __future__ import annotations

import logging

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy


class CustomCallback(DefaultCallbacks):
    """Base class for checkpoint saving callbacks.

    Note: This callback must be used so that `Result.from_path()` can be used to load the checkpoint.
    """

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        if algorithm._storage:
            algorithm._storage.current_checkpoint_index += 1
            result["checkpoint_dir_name"] = algorithm._storage.checkpoint_dir_name
            algorithm._storage.current_checkpoint_index -= 1


def create_checkpoint_loading_callback(checkpoint_dir: str, base_cls=CustomCallback):
    """Create a callback class that loads a checkpoint before training."""

    class CheckpointLoadingCallback(base_cls):
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
            super().on_algorithm_init(algorithm=algorithm, **kwargs)

            policies = Policy.from_checkpoint(checkpoint_dir)
            algorithm.set_weights(
                {pid: policy.get_weights() for pid, policy in policies.items()}
                if isinstance(policies, dict)
                else {"default_policy": policies.get_weights()}
            )
            self.logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    return CheckpointLoadingCallback
