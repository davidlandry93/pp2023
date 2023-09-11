from typing import Optional

import hydra
import mlflow
import numpy as np
import omegaconf as oc
import os
import pandas as pd
import pathlib
import torch
import tqdm
import urllib.parse
import xarray as xr
import yaml

import pytorch_lightning as pl

import aqueduct as aq

from eddie.robust2023 import StationList
from eddie.ens10_metar.tasks import RescaleStatistics, rescale_strategy_of_var
from eddie.pp2023.smc01 import SMC01_RescaleStatistics

from ..lightning import PP2023Module
from ..cli.base import build_model_from_config, build_dataloaders_from_config
from ..distribution.mapping import PP2023_DistributionMapping


def get_run_object(run_id: str, mlflow_tracking_uri=None):
    client = mlflow.MlflowClient(mlflow_tracking_uri)
    mlflow_run = client.get_run(run_id)

    return mlflow_run


def artifact_path_of_run(mlflow_run) -> pathlib.Path:
    artifact_uri = mlflow_run.info.artifact_uri
    parsed_uri = urllib.parse.urlparse(artifact_uri)

    if parsed_uri.scheme == "file":
        artifact_path = pathlib.Path(parsed_uri.path)
    else:
        artifact_path = pathlib.Path(artifact_uri)

    return artifact_path


def load_cfg_from_run(artifact_path: pathlib.Path, overrides=None) -> oc.DictConfig:
    overrides_file = artifact_path / "hydra" / "overrides.yaml"
    with overrides_file.open() as f:
        run_overrides = yaml.safe_load(f)

    hydra_file = artifact_path / "hydra" / "hydra.yaml"

    if overrides is None:
        overrides = []

    merged_overrides = [*run_overrides, "ex.dataset.maker.cache=False", *overrides]
    print(merged_overrides)

    with hydra.initialize_config_module("pp2023.conf", version_base="1.3"):
        cfg = hydra.compose("train", merged_overrides)

    print(cfg)

    return cfg


def get_model_from_id(mlflow_run, overrides=None, map_location=""):
    artifact_path = artifact_path_of_run(mlflow_run)
    cfg = load_cfg_from_run(artifact_path, overrides=overrides)

    model = build_model_from_config(cfg)
    distribution_mapping = hydra.utils.instantiate(cfg.ex.distribution.mapping)
    module = PP2023Module.load_from_checkpoint(
        artifact_path / "best_checkpoint.ckpt",
        model=model,
        distribution_mapping=distribution_mapping,
        map_location="cpu",
        variable_idx=cfg.ex.variable_idx,
    )

    return cfg, module


def predict(run_id, mlflow_tracking_uri=None, test_set=False, overrides=None):
    mlflow_run = get_run_object(run_id, mlflow_tracking_uri)
    cfg, module = get_model_from_id(mlflow_run, overrides=overrides)

    _, val_loader, test_loader = build_dataloaders_from_config(cfg)
    dataloader = test_loader if test_set else val_loader

    trainer = pl.Trainer(
        accelerator="auto",
    )

    predictions = trainer.predict(module, dataloader, return_predictions=True)

    to_return = {}
    for k in predictions[0]:
        if k != "distribution_type":
            to_return[k] = torch.cat([p[k] for p in predictions]).cpu()

    to_return["distribution_type"] = predictions[0]["distribution_type"]

    return to_return, cfg


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
    tensor: np.array,
    forecast_time: np.array,
    step_idx: np.array,
    station_idx: np.array,
) -> xr.DataArray:
    tensor_xr = xr.DataArray(
        tensor,
        dims=["batch", "parameter"],
        coords={
            "batch": pd.MultiIndex.from_arrays(
                (forecast_time, step_idx, station_idx),
                names=("forecast_time", "step", "station"),
            ),
        },
    )

    tensor_xr = tensor_xr.unstack("batch")

    return tensor_xr


def interpret_predictions(
    predictions: list[dict[str, torch.Tensor]],
    stations: xr.DataArray,
) -> xr.Dataset:
    forecast_time = predictions["forecast_time"].numpy().astype("datetime64[ns]")

    step_coord_np = predictions["step_ns"].numpy().astype("timedelta64[ns]")

    station_idx_np = predictions["station"].numpy()

    t2m = interpret_tensor_for_variable(
        predictions["parameters"][:, 0],
        forecast_time,
        step_coord_np,
        station_idx_np,
    )

    data_arrays = {"t2m": t2m}

    if predictions["parameters"].shape[1] > 1:
        """We made a prediction for wind as well"""
        si10 = interpret_tensor_for_variable(
            predictions["parameters"][:, 1],
            forecast_time,
            step_coord_np,
            station_idx_np,
        )
        data_arrays["si10"] = si10

    return (
        xr.Dataset(data_arrays)
        .sortby("forecast_time")
        .transpose("forecast_time", "step", "station", "parameter")
        .reindex(station=list(range(len(stations))))
        .assign_coords(station=stations)
    )


def rescale_predictions_ensemble(
    prediction: xr.Dataset, statistics: xr.Dataset
) -> xr.Dataset:
    t2m = prediction.t2m * statistics.std_obs_t2m + statistics.mean_obs_t2m

    output_vars = {"t2m": t2m}

    # Unnecessary since we changed the wind rescale strategy to log only
    # log_si10 = (
    #     prediction.si10 * statistics.log_std_obs_si10 + statistics.log_mean_obs_si10
    # )

    if "si10" in prediction:
        si10 = np.clip(np.exp(prediction.si10), a_min=0.0, a_max=None)
        output_vars["si10"] = si10

    return xr.Dataset(output_vars)


# def rescale_normal_parameters(
#     prediction: xr.Dataset, statistics: xr.Dataset
# ) -> xr.Dataset:
#     t2m_mu_rescaled = prediction.t2m.isel(parameter=0)
#     log_t2m_sigma_rescaled = prediction.t2m.isel(parameter=1)
#     t2m_sigma_rescaled = np.exp(log_t2m_sigma_rescaled)

#     t2m_mu = t2m_mu_rescaled * statistics.std_obs_t2m + statistics.mean_obs_t2m
#     t2m_sigma = t2m_sigma_rescaled * statistics.std_obs_t2m

#     t2m = np.stack((t2m_mu, t2m_sigma), axis=-1)
#     t2m_xr = xr.DataArray(t2m, coords=prediction.coords)
#     output_vars = {"t2m": t2m_xr}

#     if "si10" in prediction:
#         log_si10_mu = prediction.si10.isel(parameter=0)
#         log_si10_sigma = np.exp(prediction.si10.isel(parameter=1))

#         """See https://stats.stackexchange.com/questions/93082/if-x-is-normally-distributed-can-logx-also-be-normally-distributed for wind"""
#         si10_sigma = log_si10_sigma / log_si10_mu
#         si10_mu = np.exp(log_si10_mu + 0.5 * log_si10_sigma**2)

#         si10 = np.stack((si10_mu, si10_sigma), axis=-1)

#         si10_xr = xr.DataArray(si10, coords=prediction.coords)
#         output_vars["si10"] = si10_xr

#     output_xr = xr.Dataset(output_vars, coords=prediction.coords)

#     return output_xr


def rescale_predictions(
    predictions_xr: xr.Dataset, prediction_type: str, rescale_statistics: xr.Dataset
) -> xr.Dataset:
    match prediction_type:
        case "normal":
            return rescale_normal_predictions(predictions_xr, rescale_statistics)
        case "quantile":
            return rescale_predictions_ensemble(predictions_xr, rescale_statistics)
        case "deterministic":
            return rescale_deterministic_predictions(predictions_xr, rescale_statistics)
        case _:
            raise ValueError()


def rescale_normal_predictions(
    parameters: xr.DataArray, rescale_statistics: xr.DataArray
) -> xr.DataArray:
    parameters = parameters.assign_coords(parameter=["loc", "scale"])

    t2m_loc = parameters.t2m.sel(parameter="loc")
    t2m_loc_rescaled = (
        t2m_loc * rescale_statistics.std_obs_t2m + rescale_statistics.mean_obs_t2m
    )

    t2m_std = parameters.t2m.sel(parameter="scale")
    t2m_scale_rescaled = t2m_std * rescale_statistics.std_obs_t2m

    t2m = xr.combine_nested(
        [t2m_loc_rescaled, t2m_scale_rescaled], concat_dim="parameter"
    )

    return xr.Dataset({"t2m": t2m}, coords=t2m.coords)


def rescale_deterministic_predictions(
    parameters, rescale_statistics: xr.Dataset
) -> xr.Dataset:
    pred = rescale_predictions_ensemble(parameters, rescale_statistics)
    return pred


class ModelPredictions(aq.Task):
    def __init__(
        self,
        station_set: str,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        overrides: Optional[list[str]] = None,
        test_set: bool = False,
    ):
        self.model_name = model_name
        self.run_id = run_id
        self.station_set = station_set
        self.overrides = overrides
        self.step_coord = [pd.to_timedelta(x, unit="days") for x in range(3)]
        self.test_set = test_set

    def requirements(self):
        return RescaleStatistics(self.station_set)

    def run(self, reqs: tuple[xr.Dataset]) -> xr.Dataset:
        rescale_statistics = reqs

        run_id = (
            get_run_id_from_model_name(self.model_name)
            if self.model_name
            else self.run_id
        )

        self.run_id = run_id
        predictions, cfg = predict(
            run_id, overrides=self.overrides, test_set=self.test_set
        )

        predictions_xr = interpret_predictions(predictions, rescale_statistics.station)
        rescaled_predictions = rescale_predictions(
            predictions_xr, predictions["distribution_type"], rescale_statistics
        )

        return rescaled_predictions

    def artifact(self):
        set_name = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(
            f"pp2023/predictions/ens10/{self.run_id}_{set_name}.nc"
        )


class SMC01_ModelPredictions(ModelPredictions):
    def __init__(
        self,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        overrides: Optional[list[str]] = None,
        test_set: bool = False,
    ):
        self.model_name = model_name
        self.run_id = run_id
        self.overrides = overrides
        self.step_coord = [pd.to_timedelta(3 * x, unit="hours") for x in range(81)]
        self.test_set = test_set

    def requirements(self):
        return SMC01_RescaleStatistics()

    def artifact(self):
        set_name = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(
            f"pp2023/predictions/smc/{self.run_id}_{set_name}.nc"
        )
