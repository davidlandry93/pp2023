import argparse
import hydra
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import pathlib
import torch
import tqdm
import sys
import xarray as xr
import yaml

from .base import build_model_from_config, build_dataloaders_from_config


def predict_cli():
    parser = argparse.ArgumentParser()
    run_name_group = parser.add_mutually_exclusive_group(required=True)
    run_name_group.add_argument("--model-name", type=str, default=None)
    run_name_group.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", None)
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "If true, predict on the test set. If false, predict on the val set. "
            "Defaults to false."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level="INFO")
