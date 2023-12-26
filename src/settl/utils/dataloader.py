from typing import Dict

from torch.utils.data import DataLoader

__all__ = [
    "load_train_dataloader",
    "load_val_dataloader",
]


def load_train_dataloader(
        config: Dict,
        train_dataset,
) -> DataLoader:
    assert config["TRAIN"]["TYPE"] == "DataLoader", \
        f"Configuration error: TYPE of TRAIN must be DataLoader, got {config['TRAIN']['TYPE']}"

    return DataLoader(
        dataset=train_dataset,
        **config["TRAIN"]["ARGS"],
    )


def load_val_dataloader(
        config: Dict,
        val_dataset,
) -> DataLoader:
    assert config["EVAL"]["TYPE"] == "DataLoader", \
        f"Configuration error: TYPE of EVAL must be DataLoader, got {config['EVAL']['TYPE']}"

    return DataLoader(
        dataset=val_dataset,
        **config["TRAIN"]["ARGS"],
    )
