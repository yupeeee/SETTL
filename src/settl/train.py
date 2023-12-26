from typing import Union

from datetime import datetime
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .utils import *

__all__ = [
    "Trainer",
]


class Trainer:
    def __init__(
            self,
            config_path: Union[str, bytes, os.PathLike],
            train_dataset,
            val_dataset,
            model: nn.Module,
            log_dir: str,
            dataset_name: str,
            model_name: str,
            weights_save_period: int = 10,
            use_cuda: bool = False,
    ) -> None:
        self.config = load_config(config_path)

        self.train_dataloader = load_train_dataloader(self.config, train_dataset)
        self.val_dataloader = load_val_dataloader(self.config, val_dataset)
        self.batch_size = self.train_dataloader.batch_size
        self.model = model
        self.epochs = self.config["EPOCHS"]
        self.init_lr = self.config["LR"]
        self.optimizer = load_optimizer(self.config, self.model)
        self.scheduler = load_scheduler(self.config, self.optimizer)
        # self.warmup_scheduler: TODO
        self.criterion = load_criterion(self.config)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.weights_save_period = weights_save_period
        self.ID = f"{self.dataset_name}-" \
                  f"{self.model_name}-" \
                  f"{datetime.now().strftime('%y%m%d%H%M%S')}"
        self.log_dir = os.path.join(log_dir, self.ID)
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def run(self, ) -> None:
        self.model = self.model.to(self.device)

        best_val_acc = 0.

        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            self.writer.add_scalar(
                tag="LR",
                scalar_value=self.optimizer.param_groups[0]['lr'],
                global_step=epoch,
            )

            self.train(epoch)
            val_acc = self.eval(epoch)

            if best_val_acc < val_acc:
                print(f"Saving best weights to "
                      f"{os.path.join(self.log_dir, 'weights')}... (EPOCH {epoch})")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.log_dir, "weights", f"{self.ID}-best.pt"),
                )

            if not epoch % self.weights_save_period:
                print(f"Saving weights to "
                      f"{os.path.join(self.log_dir, 'weights')}... (EPOCH {epoch})")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.log_dir, "weights", f"{self.ID}-epoch_{epoch}.pt"),
                )

    def train(self, epoch: int, ) -> None:
        start = time.time()
        self.model.train()

        train_loss = 0.
        train_top1_acc = 0.
        train_top5_acc = 0.

        for batch_idx, (data, targets) in enumerate(self.train_dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(data)

            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()

            print(f"[EPOCH {epoch}/{self.epochs}] "
                  f"{batch_idx * self.batch_size + len(data)}/{len(self.train_dataloader.dataset)}\t"
                  f"LR: {self.optimizer.param_groups[0]['lr']:.4f}\t"
                  f"LOSS: {loss.item():.4f}")

            # loss update
            train_loss += loss.item()

            # acc update
            _, top1_preds = outputs.max(1)
            _, top5_preds = outputs.topk(k=5, dim=-1)

            train_top1_acc += float(top1_preds.eq(targets).sum().detach().to("cpu"))
            for k in range(5):
                train_top5_acc += float(top5_preds[:, k].eq(targets).sum().detach().to("cpu"))

        finish = time.time()

        train_loss = train_loss / len(self.train_dataloader)
        train_top1_acc = train_top1_acc / len(self.train_dataloader.dataset)
        train_top5_acc = train_top5_acc / len(self.train_dataloader.dataset)

        print(f"[EPOCH {epoch}/{self.epochs}]\n"
              f"TRAIN LOSS: {train_loss:.4f}\n"
              f"TRAIN ACC@1: {train_top1_acc * 100:.4f}%\n"
              f"TRAIN ACC@5: {train_top5_acc * 100:.4f}%")

        self.writer.add_scalar(
            tag="TRAIN/time",
            scalar_value=finish - start,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="TRAIN/avg_loss",
            scalar_value=train_loss,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="TRAIN/acc@1",
            scalar_value=train_top1_acc,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="TRAIN/acc@5",
            scalar_value=train_top5_acc,
            global_step=epoch,
        )

        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]  # removing '.'

            self.writer.add_histogram(f"PARAMS/{layer}/{attr}", param, epoch)

    def eval(self, epoch: int, ) -> float:
        start = time.time()
        self.model.eval()

        val_loss = 0.
        val_top1_acc = 0.
        val_top5_acc = 0.

        for batch_idx, (data, targets) in enumerate(self.val_dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            outputs = self.model(data)

            loss = self.criterion(outputs, targets)

            # loss update
            val_loss += loss.item()

            # acc update
            _, top1_preds = outputs.max(1)
            _, top5_preds = outputs.topk(k=5, dim=-1)

            val_top1_acc += float(top1_preds.eq(targets).sum().detach().to("cpu"))
            for k in range(5):
                val_top5_acc += float(top5_preds[:, k].eq(targets).sum().detach().to("cpu"))

        finish = time.time()

        val_loss = val_loss / len(self.val_dataloader)
        val_top1_acc = val_top1_acc / len(self.val_dataloader.dataset)
        val_top5_acc = val_top5_acc / len(self.val_dataloader.dataset)

        print(f"EVAL LOSS: {val_loss:.4f}\n"
              f"EVAL ACC@1: {val_top1_acc * 100:.4f}%\n"
              f"EVAL ACC@5: {val_top5_acc * 100:.4f}%\n")

        self.writer.add_scalar(
            tag="EVAL/time",
            scalar_value=finish - start,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="EVAL/avg_loss",
            scalar_value=val_loss,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="EVAL/acc@1",
            scalar_value=val_top1_acc,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="EVAL/acc@5",
            scalar_value=val_top5_acc,
            global_step=epoch,
        )

        return val_top1_acc
