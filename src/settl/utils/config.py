from typing import Dict, Union

import os
import yaml

__all__ = [
    "requirements",
    "load_config",
    "config_dict_to_str"
]

requirements = [
    "TRAIN",  # Dataloader for train dataset
    "EVAL",  # Dataloader for validation dataset
    "EPOCHS",  # Number of epochs
    "LR",  # Initial learning rate
    "OPTIMIZER",  # Optimizer
    "SCHEDULER",  # Learning rate scheduler
    "WARMUP",  # Warmup wrapper for scheduler
    "CRITERION",  # Loss function
]


def load_config(
        path: Union[str, bytes, os.PathLike],
) -> Dict:
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def config_dict_to_str(
        config: Dict,
) -> str:
    config_str = []

    for key, value in config.items():
        if isinstance(value, dict):
            item_config = \
                f"{key}: {value['TYPE']}(" \
                f"{', '.join([f'{k}={v}' for k, v in value['ARGS'].items()])}" \
                f")"
        else:
            item_config = f"{key}: {value}"

        config_str.append(f"{item_config}")

    return "\n".join(config_str)
