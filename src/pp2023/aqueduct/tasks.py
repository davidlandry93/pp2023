from typing import Optional

import hydra
import mlflow
import numpy as np
import os
import pandas as pd
import pathlib
import torch
import tqdm
import urllib.parse
import xarray as xr
import yaml

import aqueduct as aq

from eddie.robust2023 import StationList
from eddie.ens10_metar.tasks import RescaleStatistics, rescale_strategy_of_var

from ..cli.base import build_model_from_config, build_dataloaders_from_config

from ..cli.predict import predict


def get_run_id_from_model_name(model_name: str) -> str:
    [model_info] = mlflow.search_model_versions(
        filter_string=f"name='{model_name}'",
        order_by=["attribute.version_number DESC"],
        max_results=1,
    )

    if not model_info or model_info.run_id is None:
        tracking_uri = mlflow.get_tracking_uri()
        raise KeyError(
            f"Could not find model with name {model_name} in MLFlow server at {tracking_uri}"
        )

    run_id = model_info.run_id
    return run_id


def get_artifact_path_from_run_id(run_id: str) -> pathlib.Path:
    run_info = mlflow.get_run(run_id)

    artifact_uri = run_info.info.artifact_uri
    if artifact_uri is None:
        raise RuntimeError("Model does not have an artifact uri")

    parsed_uri = urllib.parse.urlparse(artifact_uri)

    if parsed_uri.scheme != "file":
        raise ValueError()

    artifact_path = pathlib.Path(parsed_uri.path)

    artifact_path = pathlib.Path(artifact_uri)
    return artifact_path


def interpret_tensor_for_variable(
    tensor: np.array, forecast_time: np.array, step_idx: np.array
) -> xr.DataArray:
    tensor_xr = (
        xr.DataArray(
            tensor,
            dims=["batch", "station", "parameter"],
            coords={
                "batch": pd.MultiIndex.from_arrays(
                    (forecast_time, step_idx), names=("forecast_time", "step")
                ),
            },
        )
        .unstack("batch")
        .assign_coords(step=[pd.to_timedelta(x, unit="days") for x in range(3)])
    )

    return tensor_xr


def interpret_predictions(
    predictions: list[dict[str, torch.Tensor]], stations: xr.DataArray
) -> xr.Dataset:
    forecast_time = predictions["forecast_time"].numpy().astype("datetime64[ns]")
    step_idx = predictions["step_idx"].numpy()

    t2m = interpret_tensor_for_variable(
        predictions["prediction"][:, :, 0], forecast_time, step_idx
    )
    si10 = interpret_tensor_for_variable(
        predictions["prediction"][:, :, 1], forecast_time, step_idx
    )

    return (
        xr.Dataset({"t2m": t2m, "si10": si10})
        .sortby("forecast_time")
        .transpose("forecast_time", "step", "station", "parameter")
        .assign_coords(station=stations)
    )


def make_prediction(
    run_id: str, test_set: bool = False
) -> list[dict[str, torch.Tensor]]:
    artifact_path = get_artifact_path_from_run_id(run_id)

    with (artifact_path / "hydra" / "overrides.yaml").open() as overrides_file:
        overrides = yaml.safe_load(overrides_file)

    with hydra.initialize_config_module(
        "pp2023.conf", job_name="infer", version_base="1.3"
    ):
        cfg = hydra.compose("train", overrides=overrides)

    lightning_checkpoint = torch.load(
        artifact_path / "best_checkpoint.ckpt", map_location="cpu"
    )
    lightning_state_dict = lightning_checkpoint["state_dict"]

    state_dict = {}
    for k in lightning_state_dict:
        # Remove the "model." predict that lightning added.
        new_key = ".".join(k.split(".")[1:])
        state_dict[new_key] = lightning_state_dict[k]

    model = build_model_from_config(cfg)
    model.load_state_dict(state_dict)

    _, val_loader, test_loader = build_dataloaders_from_config(cfg)

    loader = test_loader if test_set else val_loader

    predictions = []
    for b in tqdm.tqdm(loader):
        with torch.no_grad():
            pred = model(b)

        predictions.append(
            {
                "forecast_time": b["forecast_time"],
                "model_predictions": pred,
                "step_idx": b["step_idx"],
            }
        )

    return predictions


def rescale_predictions_ensemble(
    prediction: xr.Dataset, statistics: xr.Dataset
) -> xr.Dataset:
    t2m = prediction.t2m * statistics.std_obs_t2m + statistics.mean_obs_t2m
    log_si10 = (
        prediction.si10 * statistics.log_std_obs_si10 + statistics.log_mean_obs_si10
    )
    si10 = np.expm1(log_si10)

    return xr.Dataset({"t2m": t2m, "si10": si10})


def rescale_predictions(prediction: xr.Dataset, statistics: xr.Dataset) -> xr.Dataset:
    t2m_std = prediction.t2m.sel(parameter="std") * statistics.std_t2m
    t2m_mean = (
        prediction.t2m.sel(parameter="mean") * statistics.std_t2m + statistics.mean_t2m
    )
    t2m = xr.concat([t2m_mean, t2m_std], dim="parameter")

    t2m = prediction.t2m * statistics.std_t2m + statistics.mean_t2m

    si10 = prediction.si10 * statistics.si10_log_std + statistics.si10_log_mean

    si10_mean_from_model = prediction.si10.sel(parameter="mean")
    si10_std_from_model = prediction.si10.sel(parameter="std")

    si10_log_mean = (
        si10_mean_from_model * statistics.log_std_si10 + statistics.log_mean_si10
    )
    si10_log_std = si10_std_from_model * statistics.log_std_si10

    si10_std = (si10_log_std / (si10_log_mean)).assign_coords(parameter="std")
    si10_mean = (np.exp(si10_log_mean + 0.5 * (si10_std) ** 2) - 1).assign_coords(
        parameter="mean"
    )

    si10 = xr.concat([si10_mean, si10_std], dim="parameter")

    return xr.Dataset({"t2m": t2m, "si10": si10})


class ModelPredictions(aq.Task):
    def __init__(
        self,
        station_set: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.model_name = model_name
        self.run_id = run_id
        self.station_set = station_set

    def requirements(self):
        return RescaleStatistics(self.station_set)

    def run(self, reqs: tuple[xr.Dataset]) -> xr.Dataset:
        rescale_statistics = reqs

        run_id = (
            get_run_id_from_model_name(self.model_name)
            if self.model_name
            else self.run_id
        )
        # predictions = make_prediction(run_id)
        predictions = predict(run_id)
        predictions_xr = interpret_predictions(predictions, rescale_statistics.station)
        rescaled_predictions = rescale_predictions_ensemble(
            predictions_xr, rescale_statistics
        )

        return rescaled_predictions
