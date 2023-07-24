from typing import Callable

import math
import omegaconf as oc
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as td

from .distribution import DistributionalForecast


class PP2023Module(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        distribution_strategy: Callable[[torch.Tensor], DistributionalForecast],
        optimizer,
        scheduler,
        scheduler_interval="epoch",
    ):
        super().__init__()
        self.model = model
        self.distribution_strat = distribution_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval

        self.min_crps = float("inf")
        self.validation_step_outputs = []

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
        nwp_base = self.distribution_strat.nwp_base(batch)
        correction = self.forward(batch)

        prediction = nwp_base + correction

        masked_prediction = prediction[mask]
        predicted_distribution = self.distribution_strat.from_tensor(masked_prediction)
        return predicted_distribution

    def compute_loss(
        self,
        predicted_distribution,
        target,
        log_step=False,
        log_epoch=True,
        logging_prefix="Train",
    ):
        loss = predicted_distribution.loss(target)
        mean_loss = loss.mean()
        self.log(
            f"{logging_prefix}/Loss/All",
            mean_loss,
            on_step=log_step,
            on_epoch=log_epoch,
            prog_bar=True,
        )

        self.log(
            f"{logging_prefix}/Loss/t2m",
            loss[..., 0].mean(),
            on_step=log_step,
            on_epoch=log_epoch,
        )
        self.log(
            f"{logging_prefix}/Loss/si10",
            loss[..., 1].mean(),
            on_step=log_step,
            on_epoch=log_epoch,
        )

        return mean_loss

    def log_crps(
        self,
        predicted_distribution,
        target,
        log_step=False,
        log_epoch=True,
        prefix="Train",
    ):
        crpss = predicted_distribution.crps(target)
        self.log(f"{prefix}/CRPS/All", crpss.mean(), on_epoch=log_epoch, prog_bar=True)
        self.log(
            f"{prefix}/CRPS/t2m",
            crpss[..., 0].mean(),
            on_epoch=log_epoch,
            on_step=log_step,
        )
        self.log(
            f"{prefix}/CRPS/si10",
            crpss[..., 1].mean(),
            on_epoch=log_epoch,
            on_step=log_step,
        )

        return crpss.mean()

    def training_step(self, batch):
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(predicted_distribution, masked_target, log_step=True)

        self.log_crps(predicted_distribution, masked_target)

        return loss

    def validation_step(self, batch, batch_idx):
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]
        predicted_distribution = self.make_prediction(batch, mask)

        loss = self.compute_loss(
            predicted_distribution, masked_target, logging_prefix="Val"
        )

        crps = self.log_crps(predicted_distribution, masked_target, prefix="Val")

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "crps": crps.mean().detach(),
                "count": mask.sum().detach(),
            }
        )

    def on_train_epoch_end(self) -> None:
        sum_counts = 0
        sum_crps = 0

        for r in self.validation_step_outputs:
            count = r["count"]
            sum_crps += count * r["crps"]
            sum_counts += count

        crps_epoch = sum_crps / sum_counts

        if (crps_epoch < self.min_crps).item():
            self.log("min_crps", crps_epoch)
            self.min_crps = crps_epoch

        self.validation_step_outputs = []

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss.compute()
        self.log("Test/Loss/All", test_loss, on_epoch=True, prog_bar=True)
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
