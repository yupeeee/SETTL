from typing import Dict

import torch.nn as nn

__all__ = [
    "load_criterion",
]


def load_criterion(
        config: Dict,
) -> nn.modules.loss._Loss:
    criterion = getattr(nn, config["CRITERION"])

    return criterion()
