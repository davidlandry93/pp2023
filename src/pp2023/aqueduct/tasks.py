from typing import Optional

import hydra
import mlflow
import numpy as np
import os
import pandas as pd
import pathlib
import torch
import tqdm
import xarray as xr
import yaml

import aqueduct as aq

from eddie.robust2023 import StationList
from eddie.ens10_metar.tasks import RescaleStatistics, rescale_strategy_of_var

from ..cli.base import build_model_from_config, build_dataloaders_from_config


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
                "parameter": ["mean", "std"],
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
    datasets = []
    for p in predictions:
        prediction_tensor = p["model_predictions"].numpy()
        forecast_time = p["forecast_time"].numpy().astype("datetime64[ns]")
        step_idx = p["step_idx"].numpy()
        t2m = interpret_tensor_for_variable(
            prediction_tensor[:, :, 0, :],
            forecast_time=forecast_time,
            step_idx=step_idx,
        )
        si10 = interpret_tensor_for_variable(
            prediction_tensor[:, :, 1, :],
            forecast_time=forecast_time,
            step_idx=step_idx,
        )

        datasets.append(xr.Dataset({"t2m": t2m, "si10": si10}))

    return (
        xr.concat(datasets, dim="forecast_time")
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


def rescale_predictions(prediction: xr.Dataset, statistics: xr.Dataset) -> xr.Dataset:
    t2m_std = prediction.t2m.sel(parameter="std") * statistics.std_t2m
    t2m_mean = (
        prediction.t2m.sel(parameter="mean") * statistics.std_t2m + statistics.mean_t2m
    )
    t2m = xr.concat([t2m_mean, t2m_std], dim="parameter")

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

    # Suppose si10 follows distribution X = N(mean, std).
    # We did our correction on Y = log(X + 1) to "normalize" the distribution.
    # So the std we have here is the std of Y.
    # To map it back to the std of X we need use the formula
    # std_y = std_x / (mean + 1).
    # It makes sense if you think of the derivative.
    # Once we have that, we can map the mean back. See how this post uses the delta
    # method to make a reasonable approximation:
    # https://stats.stackexchange.com/questions/93082/if-x-is-normally-distributed-can-logx-also-be-normally-distributed

    return xr.Dataset({"t2m": t2m, "si10": si10})


class ModelPredictions(aq.Task):
    def __init__(
        self,
        station_set: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI"),
    ):
        self.model_name = model_name
        self.run_id = run_id
        self.station_set = station_set
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def requirements(self):
        return RescaleStatistics(self.station_set)

    def run(self, reqs: tuple[xr.Dataset]) -> xr.Dataset:
        rescale_statistics = reqs

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        run_id = (
            get_run_id_from_model_name(self.model_name)
            if self.model_name
            else self.run_id
        )
        predictions = make_prediction(run_id)
        predictions_xr = interpret_predictions(predictions, rescale_statistics.station)
        rescaled_predictions = rescale_predictions(predictions_xr, rescale_statistics)

        return rescaled_predictions
