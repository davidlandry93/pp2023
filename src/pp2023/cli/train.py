import hydra
import logging
import mlflow
import os
import torch
import torch.utils.data
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import pytorch_lightning.loggers.mlflow
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from .base import build_model_from_config, build_dataloaders_from_config
from ..lightning import PP2023Module, LogHyperparametersCallback


logger = logging.getLogger(__name__)


class CheckpointArtifactCallback(Callback):
    def __init__(self, best_checkpoint_callback):
        self.best_checkpoint_callback = best_checkpoint_callback
        self.previous_best = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print("On train epoch end checkpoint callback", os.getenv("LOCAL_RANK"))
        if trainer.is_global_zero:
            if (
                self.best_checkpoint_callback.best_model_path
                and self.best_checkpoint_callback.best_model_path != self.previous_best
            ):
                logger.debug("Logging new best chestkpoint to MLFlow")
                self.previous_best = self.best_checkpoint_callback.best_model_path

                if os.path.islink("best_checkpoint.ckpt"):
                    os.unlink("best_checkpoint.ckpt")

                os.symlink(
                    self.best_checkpoint_callback.best_model_path,
                    "best_checkpoint.ckpt",
                )
                mlflow.log_artifact("best_checkpoint.ckpt")


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def train_cli(cfg):
    node_rank = os.getenv("NODE_RANK", None)
    local_rank = os.getenv("LOCAL_RANK", None)

    is_main_process = node_rank is None or (node_rank == 0 and local_rank == 0)

    logger.info(f"Working from: {os.getcwd()}")
    if "pre" in cfg and cfg["pre"] is not None:
        for k in cfg["pre"]:
            cmd = cfg["pre"][k]
            cmd_str = " ".join(cmd)
            logger.info(f"Running pre-command {k}.")
            logger.info(f"External command: {cmd_str}")
            subprocess.run(cfg["pre"][k], check=True, capture_output=True)

    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)

    # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html.
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow_client = mlflow.MlflowClient()
    experiments = mlflow_client.search_experiments()
    experiments = [ex for ex in experiments if ex.name == cfg.mlflow.experiment_name]

    if len(experiments) == 0:
        experiment_id = mlflow_client.create_experiment(name=cfg.mlflow.experiment_name)
    else:
        [experiment] = experiments
        experiment_id = experiment.experiment_id

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders_from_config(
        cfg
    )

    # steps_per_epoch = 0
    # for _ in train_dataloader:
    #     steps_per_epoch += 1

    steps_per_epoch = len(train_dataloader)

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

    callbacks = [
        LogHyperparametersCallback(cfg.ex),
        LearningRateMonitor(),
        EarlyStopping(
            monitor="Val/CRPS/All",
            min_delta=1e-4,
            patience=cfg.ex.early_stopping_patience,
        ),
    ]

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        monitor="Val/CRPS/All",
        auto_insert_metric_name=False,
        save_last=True,
    )

    callbacks.append(best_checkpoint_callback)

    n_parameters = sum(p.numel() for p in model.parameters())

    tags = {
        "cwd": os.getcwd(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLUM_ARRAY_TASK_ID", ""),
        "n_parameters": n_parameters,
    }

    if is_main_process:
        mlflow_run = mlflow.start_run(
            experiment_id=experiment_id,
            tags=tags,
            run_name=cfg.mlflow.run_name,
        )
        mlflow_logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_id=mlflow_run.info.run_id,
            artifact_location=cfg.mlflow.artifact_location,
        )
    else:
        mlflow_run = None
        mlflow_logger = None

    try:
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="ddp",
            log_every_n_steps=cfg.ex.log_every_n_steps,
            logger=mlflow_logger,
            max_epochs=cfg.ex.get("max_epochs", None),
            callbacks=callbacks,
        )

        if is_main_process:
            os.symlink(".hydra", "hydra")
            mlflow.log_artifact("hydra")

        trainer.fit(
            lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if is_main_process:
            if trainer.state.status == "finished" and cfg.mlflow.save_model:
                if "model_name" in cfg.mlflow and cfg.mlflow.model_name:
                    mlflow.register_model(
                        f"runs:/{mlflow_run.info.run_id}", cfg.mlflow.model_name
                    )

            mlflow.log_artifact("last.ckpt")
    finally:
        if is_main_process:
            mlflow.end_run()
