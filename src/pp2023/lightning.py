from typing import Any, Callable

import math
import omegaconf as oc
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import Tensor
import torch.nn as nn
import torch.distributions as td
import os

from torch.optim.optimizer import Optimizer

from .distribution import DistributionalForecast
from .cli.base import build_dataloaders_from_config


class FromConfigDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        loader, _, _ = build_dataloaders_from_config(self.cfg)
        return loader

    def val_dataloader(self):
        _, loader, _ = build_dataloaders_from_config(self.cfg)
        return loader

    def test_dataloader(self):
        _, _, loader = build_dataloaders_from_config(self.cfg)
        return loader


class PP2023Module(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        distribution_strategy: Callable[[torch.Tensor], DistributionalForecast],
        optimizer=None,
        scheduler=None,
        scheduler_interval="epoch",
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    ):
        super().__init__()
        self.model = model
        self.distribution_strat = distribution_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval

        self.min_crps = float("inf")
        self.validation_step_crpss = []
        self.validation_step_counts = []

    def forward(self, batch):
        print("forward")
        new_batch = {}
        for k in batch:
            if batch[k].dtype == torch.double:
                new_batch[k] = batch[k].to(torch.float32)
            else:
                new_batch[k] = batch[k]

        print("calling model")
        return self.model(new_batch)

    def make_missing_obs_mask(self, batch):
        target = batch["target"]
        mask = ~(torch.isnan(target).any(dim=-1))
        return mask

    def make_prediction(self, batch, mask):
        params = self.make_parameters(batch, mask)
        predicted_distribution = self.distribution_strat.from_tensor(params)
        return predicted_distribution

    def make_parameters(self, batch, mask):
        nwp_base = self.distribution_strat.nwp_base(batch)
        correction = self.forward(batch)

        prediction = nwp_base + correction
        masked_prediction = prediction[mask]

        return masked_prediction

    def compute_loss(
        self,
        predicted_distribution,
        target,
        log_step=False,
        logging_prefix="Train",
    ):
        print("Compute loss", os.getenv("LOCAL_RANK"))
        loss = predicted_distribution.loss(target)
        mean_loss = loss.mean()

        if False:
            self.log(
                f"{logging_prefix}/Loss/All_step",
                mean_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                rank_zero_only=True,
            )

        self.log(
            f"{logging_prefix}/Loss/All",
            mean_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # self.log(
        #     f"{logging_prefix}/Loss/t2m",
        #     loss[..., 0].mean(),
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     f"{logging_prefix}/Loss/si10",
        #     loss[..., 1].mean(),
        #     on_epoch=True,
        #     sync_dist=True,
        # )

        return mean_loss

    def log_crps(
        self,
        predicted_distribution,
        target,
        log_epoch=True,
        prefix="Train",
    ):
        print("Log CRPS", os.getenv("LOCAL_RANK"))
        crpss = predicted_distribution.crps(target)

        print("Calling Log CRPS", os.getenv("LOCAL_RANK"))
        self.log(
            f"{prefix}/CRPS/All",
            crpss.mean(),
            on_epoch=log_epoch,
            prog_bar=True,
            sync_dist=True,
        )
        print("Done Calling Log CRPS", os.getenv("LOCAL_RANK"))
        # self.log(
        #     f"{prefix}/CRPS/t2m",
        #     crpss[..., 0].mean(),
        #     on_epoch=log_epoch,
        #     sync_dist=True,
        # )
        # self.log(
        #     f"{prefix}/CRPS/si10",
        #     crpss[..., 1].mean(),
        #     on_epoch=log_epoch,
        #     sync_dist=True,
        # )

        return crpss.mean()

    def on_before_backward(self, loss: Tensor) -> None:
        print("Before backward", os.getenv("LOCAL_RANK"))

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        print("Before optimizer step", os.getenv("LOCAL_RANK"))

    def on_train_epoch_start(self) -> None:
        print("On train epoch start", os.getenv("LOCAL_RANK"))

    def optimizer_step(self, *args, **kwargs) -> None:
        print("After optimizer step", os.getenv("LOCAL_RANK"))
        return super().optimizer_step(*args, **kwargs)

    def training_step(self, batch):
        print("Training step", os.getenv("LOCAL_RANK"))
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(predicted_distribution, masked_target, log_step=False)

        self.log_crps(predicted_distribution, masked_target)
        print("After log_crps", os.getenv("LOCAL_RANK"))

        return loss

    def validation_step(self, batch, batch_idx):
        print("Val step", os.getenv("LOCAL_RANK"))
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(
            predicted_distribution, masked_target, logging_prefix="Val"
        )

        crps = self.log_crps(predicted_distribution, masked_target, prefix="Val")

        self.validation_step_crpss.append(crps.mean().detach())
        self.validation_step_counts.append(mask.sum().detach())

    def on_train_epoch_end(self) -> None:
        print("Train epoch end", os.getenv("LOCAL_RANK"))
        self.validation_step_counts = []
        self.validation_step_crpss = []

    def predict_step(self, batch, batch_idx):
        batch_size = batch["features"].shape[0]

        predicted_distribution = self.make_parameters(
            batch,
            torch.ones(
                batch_size,
                dtype=torch.bool,
                device=batch["forecast"].device,
            ),
        )

        return {
            "forecast_time": batch["forecast_time"],
            "step_idx": batch["step_idx"],
            "prediction": predicted_distribution,
        }

    def on_train_epoch_end(self) -> None:
        print("Train batch end", os.getenv("LOCAL_RANK"))
        gathered_crps = self.all_gather(self.validation_step_crpss)
        gathered_counts = self.all_gather(self.validation_step_counts)

        print("GATHERED BEFORE CAT", gathered_crps)

        gathered_crps_pt = torch.stack(gathered_crps)
        gathered_counts_pt = torch.stack(gathered_counts)

        print("GATHERED", os.getenv("LOCAL_RANK"), gathered_crps_pt.shape)
        print("GATHERED", os.getenv("LOCAL_RANK"), gathered_crps_pt)
        print("GATHERED_COUNTS", os.getenv("LOCAL_RANK"), gathered_counts_pt)

        pytorch_crps_epoch = (
            gathered_counts_pt * gathered_crps_pt
        ).sum() / gathered_counts_pt.sum()

        print("pytorch_crps", os.getenv("LOCAL_RANK"), pytorch_crps_epoch)

        if (pytorch_crps_epoch < self.min_crps).item():
            self.log("min_crps", pytorch_crps_epoch, rank_zero_only=True)
            self.min_crps = pytorch_crps_epoch

        self.validation_step_crpss = []
        self.validation_step_counts = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss.compute()
        self.log(
            "Test/Loss/All", test_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.test_loss.reset()

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "Val/Loss/All",
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }


def flatten_config(cfg: oc.OmegaConf):
    return pd.json_normalize(
        oc.OmegaConf.to_container(cfg, resolve=True), sep="."
    ).to_dict(orient="records")[0]


class LogHyperparametersCallback(pl.Callback):
    """Callback that logs hyperparameters on training begin. This is better than
    logging the hyperparameters directly ourselves, because if the validation sanity
    check fails, nothing will be logged, and our logged outputs will be cleaner."""

    def __init__(self, hyperparameters: oc.OmegaConf):
        self.hparams = hyperparameters

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        to_log = flatten_config(self.hparams)

        if "backtest" in self.hparams:
            full_cfg = flatten_config(self.hparams)
            backtest_cfg = {
                k: full_cfg[k] for k in full_cfg if k.startswith("backtest")
            }
            to_log.update(backtest_cfg)

        pl_module.logger.log_hyperparams(to_log)
