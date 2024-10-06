from ray.tune.registry import register_env

from customization.callbacks import CustomCallback, checkpoint_callback
from customization.progress_report import report_progress


def global_path_planning(config: dict):
    from uav_3d.env import Uav3DEnv
    from uav_3d.utils import deep_update

    env_config = {
        "simulator": {
            "map_size": (100, 100, 20),
            "uav_size": (1, 1, 1),
            "uav_view_range": 20,
            "num_buildings": 15,
            "building_min_size": 4,
            "building_max_size": 8,
            "building_min_height": 5,
            "building_max_height": 16,
            "num_dynamic_obstacles": 0,
            "min_goal_distance": 5,
        },
        "observation": {"type": "relative"},
        "action": {"type": "default"},
        "reward": {"type": "potential_field"},
        "env": {"max_steps": 300},
    }
    deep_update(env_config, config, True)
    return Uav3DEnv(env_config)


def local_path_planning(config: dict):
    from uav_3d.env import Uav3DEnv
    from uav_3d.utils import deep_update

    env_config = {
        "simulator": {
            "map_size": (50, 50, 15),
            "uav_size": (1, 1, 1),
            "uav_view_range": 15,
            "num_buildings": 6,
            "building_min_size": 4,
            "building_max_size": 6,
            "building_min_height": 4,
            "building_max_height": 10,
            "num_dynamic_obstacles": 0,
            "min_goal_distance": 3,
        },
        "observation": {"type": "lpp"},
        "action": {"type": "lpp"},
        "reward": {"type": "lpp"},
        "env": {"max_steps": 200},
    }
    deep_update(env_config, config, True)
    return Uav3DEnv(env_config)


register_env("GlobalPathPlanning", global_path_planning)
register_env("LocalPathPlanning", local_path_planning)
