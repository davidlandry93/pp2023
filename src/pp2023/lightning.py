from typing import Any, Callable

import omegaconf as oc
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn


from .distribution.mapping import PP2023_DistributionMapping
from .cli.base import build_datasets_from_config, build_dataloader_from_dataset


class FromConfigDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        if self.train_dataset is None:
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = build_datasets_from_config(self.cfg)

    def train_dataloader(self):
        return build_dataloader_from_dataset(self.train_dataset, self.cfg, shuffle=True)

    def val_dataloader(self):
        return build_dataloader_from_dataset(self.val_dataset, self.cfg, shuffle=False)

    def test_dataloader(self):
        return build_dataloader_from_dataset(self.test_dataset, self.cfg, shuffle=False)

    def predict_dataloader(self):
        return build_dataloader_from_dataset(self.test_dataset, self.cfg, shuffle=False)


class PP2023Module(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        distribution_mapping: PP2023_DistributionMapping,
        optimizer=None,
        scheduler=None,
        scheduler_interval="epoch",
        variable_idx=None,
    ):
        super().__init__()
        self.model = model
        self.distribution_map = distribution_mapping
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.variable_idx = variable_idx

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

        if self.variable_idx is not None:
            mask = ~(torch.isnan(target[..., self.variable_idx]))
        else:
            mask = ~(torch.isnan(target).any(dim=-1))

        return mask

    def compute_loss(
        self,
        predicted_distribution,
        target,
        logging_prefix="Train",
        log_step=False,
    ):
        loss = predicted_distribution.loss(target)
        mean_loss = loss.mean()

        self.log(
            f"{logging_prefix}/Loss/All",
            mean_loss,
            on_epoch=True,
            on_step=log_step,
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
        if aggregate_per_variable and crpss.shape[-1] > 1 and len(crpss.shape) > 1:
            self.validation_step_t2m_crpss.append(crpss[..., 0].mean())
            self.validation_step_si10_crpss.append(crpss[..., 1].mean())

        return crpss.mean()

    def make_distribution(self, batch):
        model_output = self.forward(batch)

        mask = self.make_missing_obs_mask(batch)

        masked_target = batch["target"][mask]
        masked_forecast = batch["forecast"].transpose(1, 2)[mask]
        masked_model_output = model_output[mask]

        std_prior = batch["std_prior"][mask].unsqueeze(-1)

        if self.variable_idx is not None:
            masked_target = masked_target[..., [self.variable_idx]]
            masked_forecast = masked_forecast[..., [self.variable_idx]]

        predicted_distribution = self.distribution_map.make_distribution(
            masked_forecast, masked_model_output, std_prior=None
        )

        return predicted_distribution

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            if batch_idx % 20 == 0:
                print(".", end="", flush=True)

        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]

        if self.variable_idx is not None:
            masked_target = masked_target[..., [self.variable_idx]]

        predicted_distribution = self.make_distribution(batch)

        loss = self.compute_loss(predicted_distribution, masked_target, log_step=False)

        self.log_crps(predicted_distribution, masked_target)

        return loss

    def on_validation_start(self) -> None:
        if self.trainer.is_global_zero:
            print("Validating...")

    def validation_step(self, batch, batch_idx):
        mask = self.make_missing_obs_mask(batch)
        masked_target = batch["target"][mask]

        if self.variable_idx is not None:
            masked_target = masked_target[..., [self.variable_idx]]

        predicted_distribution = self.make_distribution(batch)

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
        predicted_distribution = self.make_distribution(batch)
        distribution_dict = predicted_distribution.to_dict()

        mask = self.make_missing_obs_mask(batch)

        batch_size, n_stations = batch["target"].shape[0:2]

        parameters = distribution_dict["parameters"]

        forecast_time = (
            batch["forecast_time"].unsqueeze(-1).expand(batch_size, n_stations)[mask]
        )
        step_idx = batch["step_idx"].unsqueeze(-1).expand(batch_size, n_stations)[mask]
        step_ns = batch["step_ns"].unsqueeze(-1).expand(batch_size, n_stations)[mask]

        station = (
            torch.arange(0, n_stations, device=step_idx.device)
            .unsqueeze(0)
            .expand(batch_size, n_stations)[mask]
        )

        return {
            "forecast_time": forecast_time,
            "step_idx": step_idx,
            "step_ns": step_ns,
            "station": station,
            "distribution_type": distribution_dict["distribution_type"],
            "parameters": parameters,
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
