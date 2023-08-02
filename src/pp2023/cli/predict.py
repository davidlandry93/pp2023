import argparse
import hydra
import logging
import mlflow
import numpy as np
import omegaconf as oc
import os
import pandas as pd
import pathlib
import pytorch_lightning as pl
import torch
import tqdm
import sys
import xarray as xr
import yaml
import urllib.parse

from .base import build_model_from_config, build_dataloaders_from_config
from ..lightning import PP2023Module

logger = logging.getLogger(__name__)


def artifact_path_of_run(run_id: str, mlflow_tracking_uri=None) -> pathlib.Path:
    client = mlflow.MlflowClient(mlflow_tracking_uri)
    mlflow_run = client.get_run(run_id)

    artifact_uri = mlflow_run.info.artifact_uri
    parsed_uri = urllib.parse.urlparse(artifact_uri)

    if parsed_uri.scheme != "file":
        raise ValueError()

    artifact_path = pathlib.Path(parsed_uri.path)
    return artifact_path


def load_cfg_from_run(artifact_path: pathlib.Path) -> oc.DictConfig:
    overrides_file = artifact_path / "hydra" / "overrides.yaml"
    with overrides_file.open() as f:
        overrides = yaml.safe_load(f)

    with hydra.initialize_config_module("pp2023.conf", version_base="1.3"):
        cfg = hydra.compose("train", [*overrides, "ex.dataset.maker.cache=False"])

    return cfg


def predict(run_id, mlflow_tracking_uri=None, test_set=False):
    artifact_path = artifact_path_of_run(
        run_id, mlflow_tracking_uri=mlflow_tracking_uri
    )
    cfg = load_cfg_from_run(artifact_path)

    _, val_loader, test_loader = build_dataloaders_from_config(cfg)

    model = build_model_from_config(cfg)
    distribution_strategy = hydra.utils.instantiate(cfg.ex.distribution.strategy)
    module = PP2023Module.load_from_checkpoint(
        artifact_path / "best_checkpoint.ckpt",
        model=model,
        distribution_strategy=distribution_strategy,
        map_location="cpu",
    )

    dataloader = test_loader if test_set else val_loader

    trainer = pl.Trainer(
        accelerator="auto",
    )

    predictions = trainer.predict(module, dataloader, return_predictions=True)

    to_return = {}
    for k in predictions[0]:
        to_return[k] = torch.cat([p[k] for p in predictions]).cpu()

    return to_return


def get_run_id_of_model_name(model_name, mlflow_tracking_uri=None):
    client = mlflow.MlflowClient(mlflow_tracking_uri)
    client.get_model_version()

    raise NotImplementedError()


def predict_cli():
    parser = argparse.ArgumentParser()
    run_name_group = parser.add_mutually_exclusive_group(required=True)
    run_name_group.add_argument("--model-name", type=str, default=None)
    run_name_group.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", None)
    )
    # parser.add_argument(
    #     "--dataset-path",
    #     required=True,
    #     type=pathlib.Path,
    #     help=(
    #         "Path containing a collection of pytorch records that will be used for "
    #         "prediction."
    #     ),
    # )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "If true, predict on the test set. If false, predict on the val set. "
            "Defaults to false."
        ),
    )
    parser.add_argument(
        "--save-to",
        type=pathlib.Path,
        default="./predictions.pt",
    )
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level="INFO")

    if args.run_id is None:
        run_id = get_run_id_of_model_name(
            args.model_name, mlflow_tracking_uri=args.mlflow_tracking_uri
        )
    else:
        run_id = args.run_id

    preds = predict(run_id)
    torch.save(preds, args.save_to)
