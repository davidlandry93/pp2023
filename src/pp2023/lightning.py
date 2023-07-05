from typing import Callable

import math
import omegaconf as oc
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.distributions as td

from .model.distribution import DistributionalForecast


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

    def forward(self, batch):
        new_batch = {}
        for k in batch:
            if batch[k].dtype == torch.double:
                new_batch[k] = batch[k].float()
            else:
                new_batch[k] = batch[k]

        return self.model(new_batch)

    def training_step(self, batch):
        nwp_base = self.distribution_strat.nwp_base(batch)
        correction = self.forward(batch)
        prediction = nwp_base + correction

        target = batch["target"]
        mask = ~(torch.isnan(target).any(dim=-1))

        masked_target = target[mask]
        masked_prediction = prediction[mask]

        predicted_distribution = self.distribution_strat(masked_prediction)

        log_prob_loss = -predicted_distribution.log_prob(masked_target)
        mean_log_prob_loss = log_prob_loss.mean()
        self.log(
            "Train/Loss/All",
            mean_log_prob_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        crpss = predicted_distribution.crps(masked_target)
        self.log("Train/CRPS/All", crpss.mean(), on_epoch=True, prog_bar=True)
        self.log("Train/CRPS/t2m", crpss[..., 0].mean(), on_epoch=True, on_step=False)
        self.log("Train/CRPS/si10", crpss[..., 1].mean(), on_epoch=True, on_step=False)

        self.log(
            "Train/Loss/t2m",
            log_prob_loss[..., 0].mean(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "Train/Loss/si10",
            log_prob_loss[..., 1].mean(),
            on_epoch=True,
            on_step=False,
        )

        return mean_log_prob_loss

    def validation_step(self, batch, batch_idx):
        breakpoint()
        nwp_base = self.distribution_strat.nwp_base(batch)
        correction = self.forward(batch)
        prediction = nwp_base + correction

        target = batch["target"]
        mask = ~(torch.isnan(target).any(dim=-1))

        masked_target = target[mask]
        masked_prediction = prediction[mask]

        predicted_distribution = self.distribution_strat.from_features(
            masked_prediction
        )

        log_prob_loss = -predicted_distribution.log_prob(masked_target)
        mean_log_prob_loss = log_prob_loss.mean()
        self.log(
            "Val/Loss/All",
            mean_log_prob_loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )

        self.log(
            "Val/Loss/t2m",
            log_prob_loss[..., 0].mean(),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log(
            "Val/Loss/si10",
            log_prob_loss[..., 1].mean(),
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )

        crpss = predicted_distribution.crps(masked_target)
        self.log(
            "Val/CRPS/All", crpss.mean(), on_epoch=True, prog_bar=True, on_step=False
        )
        self.log("Val/CRPS/t2m", crpss[..., 0].mean(), on_epoch=True, on_step=False)
        self.log("Val/CRPS/si10", crpss[..., 1].mean(), on_epoch=True, on_step=False)

        return mean_log_prob_loss

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
