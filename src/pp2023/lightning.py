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
        variable_idx=None,
    ):
        super().__init__()
        self.model = model
        self.distribution_strat = distribution_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.variable_idx = None

        self.min_crps = float("inf")
        self.validation_step_crpss = []
        self.validation_step_t2m_crpss = []
        self.validation_step_si10_crpss = []
        self.validation_step_counts = []

    def forward(self, batch):
        new_batch = {}
        for k in batch:
            if batch[k].dtype == torch.double:
                new_batch[k] = batch[k].to(torch.float32)
            else:
                new_batch[k] = batch[k]

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
        loss = predicted_distribution.loss(target)
        mean_loss = loss.mean()

        if False:
            self.log(
                f"{logging_prefix}/Loss/All_step",
                mean_loss,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

        self.log(
            f"{logging_prefix}/Loss/All",
            mean_loss,
            on_epoch=True,
        )

        return mean_loss

    def log_crps(
        self,
        predicted_distribution,
        target,
        log_epoch=True,
        prefix="Train",
        aggregate_per_variable=False,
    ):
        crpss = predicted_distribution.crps(target)

        self.log(
            f"{prefix}/CRPS/All",
            crpss.mean(),
            on_epoch=log_epoch,
            sync_dist=True,
        )

        # There is an option to target only one variable in the distribution, so
        # we need to check if the crps actually contains multiple columns.
        if aggregate_per_variable and crpss.shape[-1] > 1:
            self.validation_step_t2m_crpss.append(crpss[..., 0].mean())
            self.validation_step_si10_crpss.append(crpss[..., 1].mean())

        return crpss.mean()

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            if batch_idx % 20 == 0:
                print(".", end="", flush=True)

        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(predicted_distribution, masked_target, log_step=False)

        self.log_crps(predicted_distribution, masked_target)

        return loss

    def on_validation_start(self) -> None:
        if self.trainer.is_global_zero:
            print("Validating...")

    def validation_step(self, batch, batch_idx):
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(
            predicted_distribution, masked_target, logging_prefix="Val"
        )

        crps = self.log_crps(
            predicted_distribution,
            masked_target,
            prefix="Val",
            aggregate_per_variable=True,
        )

        self.validation_step_crpss.append(crps.mean().detach())
        self.validation_step_counts.append(mask.sum().detach())

    def on_train_epoch_end(self) -> None:
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
            "step_ns": batch["step_ns"],
            "prediction": predicted_distribution,
        }

    def on_train_epoch_end(self) -> None:
        gathered_crps = self.all_gather(self.validation_step_crpss)
        gathered_t2m_crps = self.all_gather(self.validation_step_t2m_crpss)
        gathered_si10_crps = self.all_gather(self.validation_step_si10_crpss)
        gathered_counts = self.all_gather(self.validation_step_counts)

        gathered_crps_pt = torch.stack(gathered_crps)
        gathered_counts_pt = torch.stack(gathered_counts)

        pytorch_crps_epoch = (
            gathered_counts_pt * gathered_crps_pt
        ).sum() / gathered_counts_pt.sum()

        print(pytorch_crps_epoch.item())

        if (pytorch_crps_epoch < self.min_crps).item():
            self.log("min_crps", pytorch_crps_epoch, rank_zero_only=True)
            self.min_crps = pytorch_crps_epoch

        if len(gathered_t2m_crps) > 0:
            """If gathered_t2m_crps is length 0, it means we did not gather statistics
            per variable, so don't try to log them."""
            gathered_crps_t2m_pt = torch.stack(gathered_t2m_crps)
            gathered_crps_si10_pt = torch.stack(gathered_si10_crps)

            pytorch_t2m_epoch = (
                gathered_counts_pt * gathered_crps_t2m_pt
            ).sum() / gathered_counts_pt.sum()

            pytorch_si10_epoch = (
                gathered_counts_pt * gathered_crps_si10_pt
            ).sum() / gathered_counts_pt.sum()

            self.log("Val/CRPS/t2m", pytorch_t2m_epoch, rank_zero_only=True)
            self.log("Val/CRPS/si10", pytorch_si10_epoch, rank_zero_only=True)

    def on_train_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            print(f"Epoch: {self.current_epoch}")
        self.validation_step_crpss = []
        self.validation_step_counts = []
        self.validation_step_t2m_crpss = []
        self.validation_step_si10_crpss = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss.compute()
        self.log("Test/Loss/All", test_loss, on_epoch=True, sync_dist=True)
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
