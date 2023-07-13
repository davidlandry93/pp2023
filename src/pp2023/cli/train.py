import hydra
import mlflow
import os
import torch
import torch.utils.data

import pytorch_lightning as pl
import pytorch_lightning.loggers.mlflow
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from .base import build_model_from_config, build_dataloaders_from_config
from ..lightning import PP2023Module, LogHyperparametersCallback


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def train_cli(cfg):
    steps_per_epoch = cfg.ex.dataset.train_size // cfg.ex.batch_size

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow_client = mlflow.MlflowClient()
    experiments = mlflow_client.search_experiments()
    experiments = [
        ex for ex in experiments if ex.name == cfg.logging.mlflow.experiment_name
    ]

    if len(experiments) == 0:
        experiment_id = mlflow_client.create_experiment(
            name=cfg.logging.mlflow.experiment_name
        )
    else:
        [experiment] = experiments
        experiment_id = experiment.experiment_id

    if len(experiments) != 1:
        raise RuntimeError("Could not retrieve experiment")

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders_from_config(
        cfg
    )

    model = build_model_from_config(cfg)
    distribution_strat = hydra.utils.instantiate(cfg.ex.distribution.strategy)
    optimizer = hydra.utils.instantiate(cfg.ex.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(
        cfg.ex.scheduler.instance, optimizer, steps_per_epoch=steps_per_epoch
    )
    lightning_module = PP2023Module(
        model,
        distribution_strat,
        optimizer,
        scheduler,
        scheduler_interval=cfg.ex.scheduler.interval,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="Val/CRPS/All", auto_insert_metric_name=False
    )
    callbacks = [
        LogHyperparametersCallback(cfg.ex),
        LearningRateMonitor(),
        checkpoint_callback,
        EarlyStopping(
            monitor="Val/CRPS/All",
            min_delta=1e-4,
            patience=cfg.ex.early_stopping_patience,
        ),
    ]

    tags = {
        "cwd": os.getcwd(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLUM_ARRAY_TASK_ID", ""),
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        tags=tags,
        run_name=cfg.logging.mlflow.run_name,
    ) as mlflow_run:
        os.symlink(".hydra", "hydra")
        mlflow.log_artifact("hydra")

        logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
            experiment_name=cfg.logging.mlflow.experiment_name,
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            run_id=mlflow_run.info.run_id,
        )

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

        if checkpoint_callback.best_model_path:
            os.symlink(checkpoint_callback.best_model_path, "best_checkpoint.ckpt")
            mlflow.log_artifact("best_checkpoint.ckpt")

        if trainer.state.status == "finished" and cfg.logging.mlflow.save_model:
            if "model_name" in cfg.logging.mlflow and cfg.logging.mlflow.model_name:
                mlflow.register_model(
                    f"runs:/{mlflow_run.info.run_id}", cfg.logging.mlflow.model_name
                )
