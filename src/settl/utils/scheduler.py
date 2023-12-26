from typing import Dict

from torch.optim import lr_scheduler, Optimizer

__all__ = [
    "load_scheduler",
]


def load_scheduler(
        config: Dict,
        optimizer: Optimizer,
) -> lr_scheduler.LRScheduler:
    scheduler = getattr(lr_scheduler, config["SCHEDULER"]["TYPE"])

    return scheduler(
        optimizer=optimizer,
        **config["SCHEDULER"]["ARGS"]
    )
