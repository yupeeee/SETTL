from typing import Dict

from torch import optim
import torch.nn as nn

__all__ = [
    "load_optimizer",
]


def load_optimizer(
        config: Dict,
        model: nn.Module,
) -> optim.Optimizer:
    optimizer = getattr(optim, config["OPTIMIZER"]["TYPE"])

    return optimizer(
        params=model.parameters(),
        lr=config["LR"],
        **config["OPTIMIZER"]["ARGS"],
    )
