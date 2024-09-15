from ray.tune.registry import register_env
from customization.uav2d import UAV2D
from customization.progress_report import report_progress
from customization.callbacks import CustomCallback, create_checkpoint_loading_callback

register_env("LocalPathPlanning", UAV2D)
