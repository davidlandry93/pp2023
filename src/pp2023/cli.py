import hydra
import mlflow
import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .lightning import PP2023Module, LogHyperparametersCallback


def build_dataloaders(cfg):
    train_dataset, val_dataset, test_dataset = hydra.utils.instantiate(
        cfg.ex.dataset.maker,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.ex.batch_size,
        num_workers=cfg.num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader


def build_model(cfg, n_steps, n_stations, n_features):
    model = hydra.utils.instantiate(
        cfg.ex.model, in_features=n_features, n_steps=n_steps, n_stations=n_stations
    )

    return model


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def train_cli(cfg):
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(cfg)

    model = build_model(
        cfg,
        cfg.ex.dataset.n_steps,
        cfg.ex.dataset.n_stations,
        cfg.ex.dataset.n_features,
    )

    optimizer = hydra.utils.instantiate(cfg.ex.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.ex.scheduler, optimizer)
    lightning_module = PP2023Module(model, optimizer, scheduler)

    tags = {
        "cwd": os.getcwd(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLUM_ARRAY_TASK_ID", ""),
    }

    logger = pl.loggers.mlflow.MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        run_name=cfg.logging.mlflow.run_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
        tags=tags,
    )

    checkpoint_callback = (
        ModelCheckpoint(monitor="Val/CRPS/All", auto_insert_metric_name=False),
    )
    callbacks = [
        LogHyperparametersCallback(cfg.ex),
        LearningRateMonitor(),
        checkpoint_callback,
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        log_every_n_steps=cfg.ex.log_every_n_steps,
        logger=logger,
        max_epochs=cfg.ex.get("max_epochs", None),
        callbacks=callbacks,
    )

    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.log_artifact(checkpoint_callback.best_model_path)
