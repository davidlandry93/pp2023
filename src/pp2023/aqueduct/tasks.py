from typing import Optional, Any
from aqueduct.artifact import ArtifactSpec
from aqueduct.task_tree import TaskTree

import hydra
import math
import mlflow
import numpy as np
import omegaconf as oc
import pandas as pd
import pathlib
import scipy.stats
import torch
import urllib.parse
import xarray as xr
import yaml

import pytorch_lightning as pl

import aqueduct as aq

from eddie.ens10_metar.tasks import (
    RescaleStatistics,
    ValTestObs,
)
from eddie.pp2023.smc01 import (
    SMC01_RescaleStatistics,
    SMC01_ValTestObs,
    SMC01_TrainObs,
    SMC01_QualityControlMask,
)

from ..lightning import PP2023Module, FromConfigDataModule
from ..cli.base import build_model_from_config, build_dataloaders_from_config


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

    merged_overrides = [*run_overrides, *overrides]

    with hydra.initialize_config_module("pp2023.conf", version_base="1.3"):
        cfg = hydra.compose("train", merged_overrides)

    return cfg


def get_cfg_from_run(mlflow_run, overrides=None):
    artifact_path = artifact_path_of_run(mlflow_run)
    return load_cfg_from_run(artifact_path, overrides=overrides)


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

    datamodule = FromConfigDataModule(cfg)
    datamodule.setup()

    trainer = pl.Trainer(
        accelerator="auto",
    )

    if test_set:
        dataset = datamodule.test_dataloader()
    else:
        dataset = datamodule.val_dataloader()

    predictions = trainer.predict(module, dataset, return_predictions=True)

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
        dataset_mappings = {
            "gdps_hdf_24h": "gdps",
            "gdps_hdf_24h_onestep": "gdps",
            "gdps_hdf_24h_no6": "gdps",
            "ens10_hdf": "ens10",
        }

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

        mlflow_run = get_run_object(run_id)

        rescaled_predictions = rescaled_predictions.assign_coords(
            distribution=mlflow_run.data.params["distribution_name"],
            dataset=dataset_mappings[mlflow_run.data.params["dataset_name"]],
            model=mlflow_run.data.params["model_name"],
            n_members=mlflow_run.data.params["dataset.n_members"],
            step_feature=mlflow_run.data.params.get("model.use_step_feature", None),
            step_embedding=mlflow_run.data.params.get("model.use_step_embedding", None),
            space_features=mlflow_run.data.params.get(
                "model.use_spatial_features", None
            ),
            station_embedding=mlflow_run.data.params.get(
                "model.use_station_embedding", None
            ),
            step_partition=mlflow_run.data.params.get("dataset.step_hour_label", False),
            share_step=mlflow_run.data.params.get("model.share_step", None),
            share_station=mlflow_run.data.params.get("model.share_station", None),
            run_id=run_id,
            validation_score=mlflow_run.data.metrics["min_crps"],
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


class NWPModelPredictions(aq.Task):
    def __init__(self, dataset="gdps", test_set=True, variant="debiased"):
        self.dataset = dataset
        self.test_set = test_set
        self.variant = variant

    def requirements(self):
        val_test_obs_req = (
            SMC01_ValTestObs()
            if self.dataset == "gdps"
            else ValTestObs(station_set="ens10_stations_smc.csv")
        )
        requirements = [val_test_obs_req]
        if self.variant == "climato":
            if self.dataset != "gdps":
                raise RuntimeError()

            requirements.append(MonthlyClimatologyModel(test_set=self.test_set))

        return requirements

    def run(self, requirements):
        val_test_artifacts, *rest = requirements
        val_artifact, test_artifact = val_test_artifacts.artifacts

        preds_artifact = test_artifact if self.test_set else val_artifact
        preds = xr.open_dataset(preds_artifact.path)

        if self.dataset == "gdps":
            preds = preds.isel(step=slice(0, 82, 8)).expand_dims(dim="number")

        if self.variant == "raw":
            model_name = "raw_nwp"
            distribution_name = "quantile"
            preds = preds.rename({"number": "parameter"})
        elif self.variant == "debiased":
            model_name = "debiased"
            distribution_name = "quantile"

            obs = preds[["obs_t2m"]].rename({"obs_t2m": "t2m"})
            bias = (obs - preds[["t2m"]]).mean(dim=["forecast_time", "number"])

            preds = (preds + bias).rename({"number": "parameter"})
        elif self.variant == "climato":
            model_name = "climato"
            distribution_name = "normal"
            preds = rest[0]
        else:
            raise KeyError("Unhandled model variant.")

        return (
            preds[["t2m"]]
            .assign_coords(
                dataset=self.dataset, distribution=distribution_name, model=model_name
            )
            .transpose("forecast_time", "step", "station", "parameter")
        )


class NWPModelPredictionsDispatch(aq.Task):
    """This task is a wrapper around NWPModelPredictions that allows us to dispatch.
    Eventually, all of the dispatching logic should be moved into this task.
    """

    def __init__(self, dataset="gdps", test_set=True, variant="debiased"):
        self.dataset = dataset
        self.test_set = test_set
        self.variant = variant

    def requirements(self):
        if self.variant == "naive":
            return NaiveNWPModelPredictions(test_set=self.test_set)
        else:
            return NWPModelPredictions(
                dataset=self.dataset, test_set=self.test_set, variant=self.variant
            )

    def run(self, requirements):
        return requirements


class NaiveNWPModelPredictions(aq.Task):
    """NWP Model prediction, augmented with a naive uncertainty distribution made from
    typical errors inside the train set."""

    def __init__(self, test_set=True):
        self.test_set = test_set

    def requirements(self) -> TaskTree:
        return [SMC01_TrainObs(), SMC01_ValTestObs()]

    def run(self, reqs) -> xr.Dataset:
        train_obs_artifact, val_test_obs_artifacts = reqs

        train_obs = xr.open_dataset(train_obs_artifact.path)
        val_obs_path = (
            val_test_obs_artifacts.artifacts[1].path
            if self.test_set
            else val_test_obs_artifacts.artifacts[0].path
        )
        val_obs = xr.open_dataset(val_obs_path)

        # Compute uncertainty estimate from the train set.
        errors = train_obs.obs_t2m - train_obs.t2m

        by_forecast_hour_mean = []
        by_forecast_hour = []
        for forecast_hour in [0, 12]:
            group = errors.where(errors.forecast_time.dt.hour == forecast_hour).groupby(
                "forecast_time.month"
            )

            forecast_hour_mean = group.mean(dim="forecast_time").assign_coords(
                forecast_hour=forecast_hour
            )
            forecast_hour_std = group.std(dim="forecast_time")
            forecast_hour_std = forecast_hour_std.assign_coords(
                forecast_hour=forecast_hour
            )
            by_forecast_hour_mean.append(forecast_hour_mean)
            by_forecast_hour.append(forecast_hour_std)

        error_mean = xr.concat(by_forecast_hour_mean, dim="forecast_hour")
        error_std = xr.concat(by_forecast_hour, dim="forecast_hour")

        forecast_bias = error_mean.sel(
            month=val_obs.forecast_time.dt.month,
            forecast_hour=val_obs.forecast_time.dt.hour,
        )
        forecast_mean = val_obs.t2m + forecast_bias

        forecast_std = error_std.sel(
            month=val_obs.forecast_time.dt.month,
            forecast_hour=val_obs.forecast_time.dt.hour,
        )

        naive_forecast = xr.concat(
            [forecast_mean, forecast_std], dim="parameter"
        ).assign_coords(
            parameter=["loc", "scale"], distribution="normal", dataset="gdps"
        )

        naive_forecast = naive_forecast.isel(step=slice(0, 82, 8))

        return xr.Dataset({"t2m": naive_forecast}).assign_coords(model="naive")

    def artifact(self):
        set_name = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(f"pp2023/predictions/nwp/naive_{set_name}.nc")


class MonthlyClimatologyModel(aq.Task):
    def __init__(self, test_set=True):
        self.test_set = test_set

    def requirements(self):
        return [SMC01_TrainObs(), SMC01_ValTestObs()]

    def run(self, requirements):
        train_artifact, val_test_artifact = requirements
        val_artifact, test_artifact = val_test_artifact.artifacts

        train_obs = xr.open_dataset(train_artifact.path).sel(
            step=[pd.to_timedelta(x, unit="D") for x in range(0, 11)]
        )

        obs_artifact = test_artifact if self.test_set else val_artifact
        obs = xr.open_dataset(obs_artifact.path)

        hours = []
        for forecast_hour in [0, 12]:
            months = []
            for forecast_month in range(1, 13):
                mean = train_obs.where(
                    (train_obs.forecast_time.dt.hour == forecast_hour)
                    & (train_obs.forecast_time.dt.month == forecast_month)
                ).mean(dim="forecast_time")

                std = train_obs.where(
                    (train_obs.forecast_time.dt.hour == forecast_hour)
                    & (train_obs.forecast_time.dt.month == forecast_month)
                ).std(dim="forecast_time")

                hour_month_climato = xr.combine_nested(
                    [mean, std], concat_dim="parameter"
                ).assign_coords(
                    month=forecast_month, hour=forecast_hour, parameter=["loc", "scale"]
                )
                months.append(hour_month_climato)
            hours.append(months)

        monthly_climato = xr.combine_nested(hours, concat_dim=["hour", "month"])

        climato_preds = (
            monthly_climato.sel(
                month=obs.forecast_time.dt.month,
                hour=obs.forecast_time.dt.hour,
                step=obs.step.isel(step=range(0, 82, 8)),
                station=obs.station,
            )
            .assign_coords(model="monthly_climato", distribution="normal")
            .drop(["t2m", "si10", "month", "hour"])
            .rename({"obs_t2m": "t2m", "obs_si10": "si10"})
        )

        return climato_preds

    def artifact(self):
        return aq.LocalStoreArtifact("pp2023/predictions/smc01_monthly_climato.nc")


SQRT_PI = math.sqrt(math.pi)


def crps_empirical_np(Q: np.array, y: np.array, sorted=False):
    """Compute the CRPS of an empirical distribution. Q is the sorted samples of the empirical distribution,
    where the last dimension is the sample dimension.
    Q and y should have the same shape except for the last dimension.

    Args:
        Q: The samples that form the empirical distribution.
        y: The observations.

    Return:
        A tensor containing the CRPS of each distribution given their respective observations.
    """

    if not sorted:
        Q = np.sort(Q, axis=-1)

    if len(y.shape) == len(Q.shape) - 1:
        y = np.expand_dims(y, axis=-1)

    N = Q.shape[-1]

    right_width = np.concatenate(
        [
            Q[..., 0:1] - y,
            Q[..., 1:] - np.maximum(y, Q[..., :-1]),
            np.zeros(
                (*Q.shape[:-1], 1)
            ),  # Right integral is never used if the obs is to the right of the distribution, so set width to zero.
        ],
        axis=-1,
    )

    left_width = np.concatenate(
        [
            np.zeros(
                (*Q.shape[:-1], 1)
            ),  # Left integral is never used if the obs is to the left of the distribution.
            np.minimum(y, Q[..., 1:]) - Q[..., :-1],
            y - Q[..., [-1]],
        ],
        axis=-1,
    )

    weights = np.arange(0, N + 1) / N
    right_weights = (1 - weights) ** 2
    left_weights = weights**2

    left = np.clip(left_width, a_min=0, a_max=None) * left_weights
    right = np.clip(right_width, a_min=0, a_max=None) * right_weights

    return (left + right).sum(axis=-1)


def crps_energy_np(Q, y, sorted=False):
    """Energy CRPS as in equation (eNRG) of Zamo2018."""
    lhs = np.abs(Q - y).mean(axis=-1)

    inter_q_distances = (
        np.abs(np.expand_dims(Q, axis=-1) - np.expand_dims(Q, axis=-2))
        .sum(axis=-1)
        .sum(axis=-1)
    )

    rhs = (1.0 / (2 * Q.shape[-1] * Q.shape[-1])) * inter_q_distances

    return lhs - rhs


def crps_fair_np(Q, y, sorted=False):
    """Fair CRPS as in equation (eFAIR) of Zamo2018."""
    lhs = np.abs(Q - y).mean(axis=-1)

    inter_q_distances = (
        np.abs(np.expand_dims(Q, axis=-1) - np.expand_dims(Q, axis=-2))
        .sum(axis=-1)
        .sum(axis=-1)
    )

    rhs = (1.0 / (2 * Q.shape[-1] * (Q.shape[-1] - 1))) * inter_q_distances

    return lhs - rhs


def crps_gaussian_np(Q, y):
    loc = Q.sel(parameter="loc")
    scale = Q.sel(parameter="scale")

    centered_dist = scipy.stats.norm(loc=np.zeros_like(loc), scale=np.ones_like(scale))

    centered_sample = (y - loc) / scale

    cdf = centered_dist.cdf(centered_sample)
    pdf = centered_dist.pdf(centered_sample)

    centered_crps = centered_sample * (2 * cdf - 1) + 2 * pdf - (1 / SQRT_PI)
    crps = scale * centered_crps

    return crps


def task_of_run_id(run_id, test_set=True):
    mlflow_run = get_run_object(run_id)

    if mlflow_run.data.params["dataset_name"].startswith("gdps"):
        model_prediction_task = aq.as_artifact(
            SMC01_ModelPredictions(run_id=run_id, test_set=test_set)
        )
    elif mlflow_run.data.params["dataset_name"].startswith("ens10"):
        model_prediction_task = aq.as_artifact(
            ModelPredictions(
                run_id=run_id,
                station_set="ens10_stations_smc.csv",
                test_set=test_set,
            )
        )
    else:
        raise RuntimeError()

    return model_prediction_task


def requirements_for_run_metrics(run_id, test_set=True):
    mlflow_run = get_run_object(run_id)

    model_prediction_task = task_of_run_id(run_id, test_set=test_set)

    if mlflow_run.data.params["dataset_name"].startswith("gdps"):
        obs_task = SMC01_ValTestObs()
        outlier_mask = SMC01_QualityControlMask()
        baseline_metrics = NWPModelMetrics(
            dataset="gdps", test_set=test_set, variant="naive"
        )

        requirements = [model_prediction_task, obs_task, outlier_mask, baseline_metrics]

    elif mlflow_run.data.params["dataset_name"].startswith("ens10"):
        obs_task = ValTestObs(station_set="ens10_stations_smc.csv")
        baseline_metrics = NWPModelMetrics(
            dataset="ens10", test_set=test_set, variant="naive"
        )
        requirements = [model_prediction_task, obs_task, baseline_metrics]

    else:
        raise KeyError("Did not recognize dataset name in MLFlow run.")

    return requirements


def do_one_chunk(inputs, fn=crps_empirical_np):
    p_dataset, obs_dataset = inputs
    crps = fn(p_dataset, obs_dataset)
    crps = xr.DataArray(crps, coords=obs_dataset.coords)

    return crps


def crps_of_prediction(prediction: xr.DataArray, observation: xr.DataArray):
    chunk_size = 10
    n_chunks = math.ceil(len(prediction.forecast_time) / chunk_size)

    crps_chunks = []
    mse_chunks = []
    mae_chunks = []
    bias_chunks = []
    for i in range(n_chunks):
        chunk_begin = i * chunk_size
        chunk_end = (i + 1) * chunk_size

        p_dataset = prediction.isel(forecast_time=slice(chunk_begin, chunk_end))
        obs_dataset = observation.isel(forecast_time=slice(chunk_begin, chunk_end))

        crps_chunks.append(do_one_chunk((p_dataset, obs_dataset)))
        # crps_fair_chunks.append(do_one_chunk((p_dataset, obs_dataset), fn=crps_fair_np))
        # crps_energy_chunks.append(
        #     do_one_chunk((p_dataset, obs_dataset), fn=crps_energy_np)
        # )
        mse_chunks.append(np.square(p_dataset.mean(dim="parameter") - obs_dataset))
        mae_chunks.append(np.abs(p_dataset.median(dim="parameter") - obs_dataset))
        bias_chunks.append(p_dataset.mean(dim="parameter") - obs_dataset)

    crps_xr = xr.combine_nested(crps_chunks, concat_dim="forecast_time")
    mse_xr = xr.combine_nested(mse_chunks, concat_dim="forecast_time")
    mae_xr = xr.combine_nested(mae_chunks, concat_dim="forecast_time")
    bias_xr = xr.combine_nested(bias_chunks, concat_dim="forecast_time")
    # crps_fair_xr = xr.combine_nested(crps_fair_chunks, concat_dim="forecast_time")
    return crps_xr, mse_xr, mae_xr, bias_xr


def quantile_values_normal(
    params: xr.DataArray, quantile_levels: list[float]
) -> xr.DataArray:
    loc = params.sel(parameter="loc")
    scale = params.sel(parameter="scale")

    dist = torch.distributions.Normal(
        torch.from_numpy(loc.fillna(0.0).values).unsqueeze(-1),
        torch.from_numpy(scale.fillna(1.0).values).unsqueeze(-1) + 1e-6,
        validate_args=None,
    )

    all_thresholds = dist.icdf(torch.tensor(quantile_levels))
    all_thresholds_numpy = all_thresholds.numpy()
    all_thresholds_numpy[np.isnan(loc)] = np.nan

    thresholds_xr = xr.DataArray(
        all_thresholds_numpy,
        dims=[*loc.dims, "quantile_level"],
        coords={**loc.coords, "quantile_level": quantile_levels},
    )

    return thresholds_xr.drop("parameter")


def interpolate_quantile_values(q, quantile_levels):
    if "quantile_level" not in q.coords:
        n_quantiles = q.sizes["parameter"]
        q = q.assign_coords(
            parameter=np.array(range(1, n_quantiles + 1)) / (n_quantiles + 1)
        ).rename({"parameter": "quantile_level"})

    return q.interp(quantile_level=quantile_levels, method="linear")


def compute_quantile_score(quantile_values, obs):
    quantile_score = xr.where(
        quantile_values <= obs,
        quantile_values.quantile_level * (obs - quantile_values),
        (1.0 - quantile_values.quantile_level) * (quantile_values - obs),
    )

    return quantile_score


def compute_model_metrics(
    preds: xr.Dataset,
    obs: xr.Dataset,
    baseline=None,
    qc_mask: xr.Dataset = None,
) -> xr.Dataset:
    distribution = preds.coords["distribution"]
    dataset = preds.coords["dataset"]

    obs = obs.reindex({"forecast_time": preds.forecast_time, "step": preds.step})

    # Only do t2m for now.
    preds = preds.t2m
    obs = obs.obs_t2m

    if distribution in ["emos", "normal"]:
        crps = crps_gaussian_np(preds, obs).drop_vars("parameter")
        mse = np.square(preds.sel(parameter="loc") - obs).drop_vars("parameter")
        mae = np.abs(preds.sel(parameter="loc") - obs).drop_vars("parameter")
        bias = (preds.sel(parameter="loc") - obs).drop_vars("parameter")
        spread = (preds.sel(parameter="scale")).drop_vars("parameter")

        quantile_values = quantile_values_normal(preds, [0.05, 0.1, 0.9, 0.95])
        quantile_score = compute_quantile_score(quantile_values, obs)
        quantile_score_05 = quantile_score.sel(quantile_level=0.05)
        quantile_score_95 = quantile_score.sel(quantile_level=0.95)

        quantile_spread = quantile_values.sel(quantile_level=0.9) - quantile_values.sel(
            quantile_level=0.1
        )

        dataset_dict = {
            "crps": crps,
            "mse": mse,
            "bias": bias,
            "mae": mae,
            "spread": spread,
            "quantile_score_05": quantile_score_05.drop("quantile_level"),
            "quantile_score_95": quantile_score_95.drop("quantile_level"),
            "quantile_spread": quantile_spread,
            "composite_spread": quantile_spread,
        }
    elif distribution in ["bernstein", "quantile"]:
        crps, mse, mae, bias = crps_of_prediction(preds, obs)
        n = preds.sizes["parameter"]
        spread = np.sqrt(n / (n - 1)) * preds.std(dim="parameter")

        quantile_values = interpolate_quantile_values(preds, [0.05, 0.1, 0.9, 0.95])
        quantile_score = compute_quantile_score(quantile_values, obs)
        quantile_score_05 = quantile_score.sel(quantile_level=0.05)
        quantile_score_95 = quantile_score.sel(quantile_level=0.95)

        quantile_spread = quantile_values.sel(
            quantile_level=0.95
        ) - quantile_values.sel(quantile_level=0.05)

        dataset_dict = {
            "crps": crps,
            "mse": mse,
            "bias": bias,
            "mae": mae,
            "spread": spread,
            "composite_spread": compute_composite_spread(preds, width=0.8),
            "quantile_spread": quantile_spread,
            "quantile_score_05": quantile_score_05.drop("quantile_level"),
            "quantile_score_95": quantile_score_95.drop("quantile_level"),
        }
    elif distribution == "deterministic":
        crps = np.abs(preds - obs).squeeze()
        mse = np.square(preds - obs).squeeze()
        bias = (preds - obs).squeeze()
        mae = crps
        spread = None

        dataset_dict = {"crps": crps, "mse": mse, "bias": bias, "mae": mae}
    else:
        raise KeyError("Could not compute crps for distribution type.")

    metrics_dataset = xr.Dataset(
        {
            k: v.drop_vars(["elevation", "latitude", "longitude"], errors="ignore")
            for k, v in dataset_dict.items()
        }
    )

    # metrics_dataset.assign_coords(
    #     {
    #         "elevation": obs.elevation,
    #         "latitude": obs.latitude,
    #         "longitude": obs.longitude,
    #     }
    # )

    if baseline is not None:
        metrics_dataset["crpss"] = 1.0 - (metrics_dataset["crps"] / baseline["crps"])

    valid_time = metrics_dataset.forecast_time + metrics_dataset.step

    if qc_mask is not None:
        mask_for_metrics = qc_mask.sel(valid_time=valid_time)
    else:
        mask_for_metrics = xr.zeros_like(valid_time, dtype=bool)

    for v in metrics_dataset.data_vars:
        masked_dataarray = (
            metrics_dataset[v]
            .where(~mask_for_metrics)
            .to_array()
            .drop("variable")
            .squeeze()
        )

        metrics_dataset[v] = masked_dataarray
    return metrics_dataset


def compute_composite_spread(Q, width=0.9):
    sorted_q = Q.isel(parameter=Q.argsort(axis=-1))
    bin_width = sorted_q.isel(parameter=slice(1, None)) - sorted_q.isel(
        parameter=slice(0, -1)
    )
    sorted_width = bin_width.isel(parameter=bin_width.argsort(axis=-1))

    n_bins = bin_width.shape[-1]
    n_for_metric = math.floor(n_bins * width)

    base_metric = sorted_width[..., :n_for_metric].sum(axis=-1)

    leftover_cdf = width - (n_for_metric / n_bins)
    last_column_ratio = leftover_cdf / (1 / n_bins)
    last_column_value = sorted_width[..., n_for_metric] * last_column_ratio

    full_metric = last_column_value + base_metric

    return full_metric


class ModelPredictionsDispatch(aq.Task):
    def __init__(
        self,
        type,
        dataset,
        run_id=None,
        nwp_variant=None,
        experiment_name=None,
        distribution=None,
        test_set=True,
        **kwargs,
    ):
        self.type = type
        self.run_id = run_id
        self.nwp_variant = nwp_variant
        self.dataset = dataset
        self.test_set = test_set
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.kwargs = kwargs

    def requirements(self):
        if self.type == "mlflow":
            return task_of_run_id(self.run_id, test_set=self.test_set)
        elif self.type == "nwp":
            return aq.as_artifact(
                NWPModelPredictionsDispatch(
                    dataset=self.dataset,
                    test_set=self.test_set,
                    variant=self.nwp_variant,
                )
            )
        elif self.type == "ensemble":
            return aq.as_artifact(
                ModelPredictionsEnsemble(
                    experiment_name=self.experiment_name,
                    distribution=self.distribution,
                    n_runs=5,
                    test_set=self.test_set,
                    **self.kwargs,
                )
            )
        else:
            raise KeyError()

    def run(self, requirements):
        return xr.open_dataset(requirements.path)


class EnsembleModelMetrics(aq.Task):
    def __init__(
        self,
        experiment_name,
        distribution,
        test_set=True,
        step_embedding=True,
        step_feature=True,
        step_idx=None,
        n_members=None,
        dataset="gdps",
    ):
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.test_set = test_set
        self.step_embedding = step_embedding
        self.step_feature = step_feature
        self.step_idx = step_idx
        self.dataset = dataset
        self.n_members = n_members

    def requirements(self):
        preds = ModelPredictionsEnsemble(
            self.experiment_name,
            self.distribution,
            n_runs=5,
            test_set=self.test_set,
            step_embedding=self.step_embedding,
            step_feature=self.step_feature,
            step_idx=self.step_idx,
            n_members=self.n_members,
        )

        if self.dataset == "gdps":
            return [preds, SMC01_ValTestObs(), SMC01_QualityControlMask()]
        elif self.dataset == "ens10":
            return [preds, ValTestObs(station_set="ens10_stations_smc.csv")]
        else:
            raise KeyError("Unhandled dataset")

    def run(self, reqs):
        if self.dataset == "gdps":
            preds_artifact, obs_artifact, qc_mask = reqs
        else:
            preds_artifact, obs_artifact = reqs
            qc_mask = None

        preds = xr.open_dataset(preds_artifact.path)
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        return compute_model_metrics(preds, obs, baseline=None, qc_mask=qc_mask)

    def artifact(self):
        step_idx_label = f"_s{self.step_idx}" if self.step_idx is not None else ""
        n_member_label = f"_m{self.n_members}" if self.n_members is not None else ""
        label = f"{self.experiment_name}_{self.distribution}_{self.test_set}_{self.step_embedding}_{self.step_feature}{step_idx_label}{n_member_label}.nc"
        return aq.LocalStoreArtifact(f"pp2023/metrics/ensemble/{label}.nc")


class ModelMetricsDispatch(aq.Task):
    def __init__(
        self,
        type,
        dataset="gdps",
        run_id=None,
        nwp_variant="naive",
        experiment_name="pp2023_mlp_table_08",
        distribution=None,
        test_set=True,
        **kwargs,
    ):
        self.type = type
        self.run_id = run_id
        self.nwp_variant = nwp_variant
        self.dataset = dataset
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.test_set = test_set
        self.kwargs = kwargs

    def requirements(self):
        preds = ModelPredictionsDispatch(
            self.type,
            self.dataset,
            self.run_id,
            self.nwp_variant,
            self.experiment_name,
            self.distribution,
            self.test_set,
            **self.kwargs,
        )

        if self.dataset == "gdps":
            return [preds, SMC01_ValTestObs(), SMC01_QualityControlMask()]
        elif self.dataset == "ens10":
            return [preds, ValTestObs(station_set="ens10_stations_smc.csv")]
        else:
            raise KeyError()

    def run(self, requirements) -> xr.Dataset:
        if len(requirements) == 3:
            preds, obs_artifact, qc_mask = requirements
        else:
            preds, obs_artifact = requirements
            qc_mask = None

        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        metrics = compute_model_metrics(preds, obs, qc_mask=qc_mask)

        return metrics

    def artifact(self):
        set_label = "test" if self.test_set else "val"

        if self.type == "mlflow":
            label = f"{self.run_id}_{set_label}.nc"
        elif self.type == "nwp":
            label = f"{self.dataset}_{self.nwp_variant}_{set_label}.nc"
        elif self.type == "ensemble":
            label = f"{self.experiment_name}_{self.distribution}_{set_label}.nc"
        else:
            raise KeyError()

        return aq.LocalStoreArtifact(f"pp2023/metrics/{self.type}/{label}.nc")


class ModelMetrics(aq.Task):
    def __init__(self, run_id: str, test_set: bool = True):
        self.run_id = run_id
        self.test_set = test_set

    def requirements(self):
        return requirements_for_run_metrics(self.run_id, self.test_set)

    def run(self, requirements) -> xr.Dataset:
        if len(requirements) == 4:
            preds, obs_artifact, qc_mask, baseline_metrics = requirements
        else:
            preds, obs_artifact, baseline_metrics = requirements
            qc_mask = None

        preds = xr.open_dataset(preds.path)

        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        metrics = compute_model_metrics(preds, obs, baseline_metrics, qc_mask=qc_mask)

        return metrics

    def artifact(self):
        set_label = "test" if self.test_set else "val"
        return aq.LocalStoreArtifact(f"pp2023/metrics/{self.run_id}_{set_label}.nc")


class NWPModelMetrics(aq.Task):
    def __init__(self, dataset: str = "gdps", test_set=True, variant="debias"):
        self.dataset = dataset
        self.test_set = test_set
        self.variant = variant

    def requirements(self):
        if self.dataset == "gdps":
            obs_task = SMC01_ValTestObs()
        elif self.dataset == "ens10":
            obs_task = ValTestObs(station_set="ens10_stations_smc.csv")
        else:
            KeyError()

        return [
            NWPModelPredictionsDispatch(
                dataset=self.dataset, test_set=self.test_set, variant=self.variant
            ),
            obs_task,
            SMC01_QualityControlMask(),
        ]

    def run(self, reqs):
        preds, obs_artifact, qc_mask = reqs
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        return compute_model_metrics(
            preds, obs, baseline=None, qc_mask=qc_mask
        ).assign_coords(model=self.variant)

    def artifact(self):
        model_label = self.variant
        set_label = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(
            f"pp2023/metrics/{self.dataset}_{model_label}_{set_label}.nc"
        )


class TableCRPS(aq.Task):
    """Data for the headline table of the paper that has the mean CRPS over test
    set for all model/distribution combinations."""

    def __init__(self):
        pass

    def requirements(self):
        model_tasks = []
        for experiment_name in ["pp2023_mlp_table_08"]:
            for distribution in ["deterministic", "quantile", "bernstein", "emos"]:
                model_tasks.append(
                    ModelMetricsDispatch(
                        type="ensemble",
                        experiment_name=experiment_name,
                        distribution=distribution,
                        test_set=True,
                    )
                )

        model_tasks.extend(
            [
                ModelMetrics(
                    run_id="e9ef58e43d584ab9bdc1f00067533123", test_set=True
                ),  # EMOS.
                ModelMetrics(
                    run_id="b00adfeb13bf4944a86af9d20d76a789", test_set=True
                ),  # LBQ.
                ModelMetrics(
                    run_id="899a9e1b93bd4978b10203022fcf277e", test_set=True
                ),  # LQR.
                ModelMetrics(
                    run_id="383d53fda0e0427882958c033e3dc3bc", test_set=True
                ),  # Deterministic.
            ]
        )

        for dataset in ["gdps"]:
            model_tasks.append(
                NWPModelMetrics(dataset=dataset, test_set=True, variant="raw")
            )
            model_tasks.append(
                NWPModelMetrics(dataset=dataset, test_set=True, variant="debiased")
            )
            model_tasks.append(
                NWPModelMetrics(dataset=dataset, test_set=True, variant="naive")
            )

        return model_tasks

    def run(self, requirements: list[xr.Dataset]):
        rows = []
        for full_metrics in requirements:
            for lead in ["all"]:
                metrics = full_metrics.sel(step=full_metrics.step > pd.to_timedelta(0))

                run_id = metrics["run_id"].item() if "run_id" in metrics else None

                crps_mean = metrics.crps.mean()
                rmse = np.sqrt(metrics.mse.mean())

                row_dict = {
                    "rmse": rmse.item(),
                    "crps": crps_mean.item(),
                    "distribution": metrics.coords["distribution"].item(),
                    "dataset": metrics.coords["dataset"].item(),
                    "model": metrics.coords["model"].item(),
                    "run_id": run_id,
                    "lead": lead,
                }

                if "quantile_score_05" in metrics:
                    row_dict["quantile_score_05"] = (
                        metrics.quantile_score_05.mean().item()
                    )
                    row_dict["quantile_score_95"] = (
                        metrics.quantile_score_95.mean().item()
                    )

                rows.append(row_dict)

        df = pd.DataFrame(rows)
        df["dataset"] = df["dataset"].replace(
            {"gdps_prebatch_24h": "gdps", "ens10_prebatch": "ens10"}
        )

        return df


class MetricsByStep(TableCRPS):
    def run(self, requirements):
        dfs = []

        for model_metrics in requirements:
            q_df = model_metrics.quantile(
                [0.05, 0.95], dim=["forecast_time", "station"]
            ).to_dataframe()

            df = model_metrics.mean(dim=["forecast_time", "station"]).to_dataframe()

            df["rmse"] = np.sqrt(df["mse"])

            dfs.append(df)

        df = pd.concat(dfs)
        df["dataset"] = df["dataset"].replace(
            {
                "gdps_prebatch_24h": "gdps",
                "ens10_prebatch": "ens10",
                "gdps_hdf_24h_no6": "gdps",
            }
        )

        # Filter case where dataset is gdps and step is 0
        df = df[~((df["dataset"] == "gdps") & (df.index == pd.to_timedelta(0)))]

        return df


class CRPSByStep(TableCRPS):
    def run(self, requirements: list[xr.Dataset]) -> pd.DataFrame:
        dfs = []
        for metrics in requirements:
            crps_by_step = metrics.mean(dim=["forecast_time", "station"])
            dfs.append(crps_by_step.to_dataframe())

        df = pd.concat(dfs)
        df["dataset"] = df["dataset"].replace(
            {"gdps_prebatch_24h": "gdps", "ens10_prebatch": "ens10"}
        )

        return df


class DispersionByStep(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        linear_experiment = client.get_experiment_by_name("pp2023_linear_table")
        linear_runs = client.search_runs(
            experiment_ids=[linear_experiment.experiment_id],
            max_results=5000,
        )

        linear_runs = [
            r.info.run_id
            for r in linear_runs
            if r.info.run_name == "no_share_stations"
            and r.data.params["distribution_name"] in ["quantile", "emos"]
            and r.data.params["dataset_name"].startswith("gdps")
        ]

        mlp_experiment = client.get_experiment_by_name("pp2023_mlp_table")
        mlp_runs_response = client.search_runs(
            experiment_ids=[mlp_experiment.experiment_id],
            max_results=5000,
        )

        mlp_runs = []
        for r in mlp_runs_response:
            if (
                r.data.params["scheduler.instance._target_"]
                == "pp2023.scheduler.reduce_lr_plateau"
                and r.data.params["distribution_name"]
                in [
                    "quantile",
                    "emos",
                ]
                and r.data.params["dataset_name"].startswith("gdps")
            ):
                mlp_runs.append(r.info.run_id)

        all_runs = [*linear_runs, *mlp_runs]

        requirements = []
        for run_id in all_runs:
            preds_task, _ = requirements_for_run_metrics(run_id, test_set=True)
            requirements.append(preds_task)

        return requirements

    def run(self, reqs: list[xr.Dataset]) -> pd.DataFrame:
        dfs = []
        for preds in reqs:
            if preds.coords["distribution"] == "emos":
                df = (
                    preds.sel(parameter="scale")
                    .mean(dim=["forecast_time", "station"])
                    .to_dataframe()
                    .rename(columns={"t2m": "value"})
                )
                df["metric"] = "std"
                dfs.append(df)
            elif preds.coords["distribution"] == "quantile":
                bin_sizes = preds.sel(
                    parameter=slice(1, preds.sizes["parameter"])
                ) - preds.sel(parameter=slice(0, preds.sizes["parameter"] - 1))

                sorted_bin_sizes = (
                    bin_sizes.t2m.isel(parameter=bin_sizes.t2m.argsort())
                    .cumsum(dim="parameter")
                    .mean(dim=["forecast_time", "station"])
                )

                for idx in [7, 13, 14]:
                    n_bins = sorted_bin_sizes.sizes["parameter"]
                    df = sorted_bin_sizes.isel(parameter=idx).to_dataframe(name="value")
                    df["metric"] = f"{(idx+1):02}/{n_bins}"
                    dfs.append(df)
            else:
                raise KeyError()

        return pd.concat(dfs)


class ModelSpreadErrorRatio(aq.Task):
    def __init__(
        self, run_id: str = None, nwp_variant: str = None, test_set: bool = True
    ):
        self.run_id = run_id
        self.nwp_variant = nwp_variant
        self.test_set = test_set

        if nwp_variant is not None and run_id is not None:
            raise RuntimeError("run_id and nwp_variant are mutually exclusive")

    def requirements(self):
        if self.nwp_variant:
            return [
                aq.as_artifact(
                    NWPModelPredictionsDispatch(
                        variant=self.nwp_variant, test_set=self.test_set
                    )
                ),
                SMC01_ValTestObs(),
                SMC01_QualityControlMask(),
            ]
        else:
            return requirements_for_run_metrics(self.run_id, self.test_set)

    def artifact(self) -> ArtifactSpec | None:
        label = self.run_id if self.run_id else self.nwp_variant
        set_label = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(
            f"pp2023/metrics/spread_error_ratio/{label}_{set_label}.nc"
        )

    def run(self, requirements: tuple[xr.Dataset, Any]):
        preds, obs_artifact, qc_mask = requirements
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path).sel(step=preds.step)

        qc_mask = qc_mask.to_array().squeeze().drop("variable")

        valid_time = preds.forecast_time + preds.step
        mask_for_preds = qc_mask.sel(valid_time=valid_time)
        masked_preds = preds.t2m.where(~mask_for_preds)

        valid_time = obs.forecast_time + obs.step
        mask_for_obs = qc_mask.sel(valid_time=valid_time)
        masked_obs = obs.obs_t2m.where(~mask_for_obs)

        distribution = preds.coords["distribution"].item()
        dataset = preds.coords["dataset"]

        # Only do t2m for now.
        preds = masked_preds
        obs = masked_obs

        if distribution in ["bernstein", "quantile"]:
            ensemble_mean = preds.mean(dim="parameter")
            rmse_of_ensemble_mean = np.sqrt(
                np.square(ensemble_mean - obs).mean(dim=["station", "forecast_time"])
            )

            ensemble_variance = np.square(ensemble_mean - preds).sum(
                dim="parameter"
            ) / (preds.sizes["parameter"] - 1)

            root_mean_ensemble_variance = np.sqrt(
                ensemble_variance.mean(dim=["station", "forecast_time"])
            )

        elif distribution in ["emos", "normal"]:
            rmse_of_ensemble_mean = np.sqrt(
                np.square(preds.sel(parameter="loc") - obs).mean(
                    dim=["station", "forecast_time"]
                )
            ).drop("parameter")

            ensemble_variance = np.square(preds.sel(parameter="scale"))
            root_mean_ensemble_variance = np.sqrt(
                ensemble_variance.mean(dim=["station", "forecast_time"])
            ).drop("parameter")
        else:
            raise KeyError("Could not identify distribution")

        n_quantiles = preds.sizes["parameter"]
        spread_error_ratio = np.sqrt((n_quantiles + 1) / n_quantiles) * (
            root_mean_ensemble_variance / rmse_of_ensemble_mean
        )

        dataset = xr.Dataset(
            {
                "spread": root_mean_ensemble_variance,
                "error": rmse_of_ensemble_mean,
                "spread_error_ratio": spread_error_ratio,
            }
        )

        return dataset


class ModelSpreadErrorRatioDispatch(ModelSpreadErrorRatio):
    def __init__(
        self,
        type: str,
        dataset="gdps",
        run_id=None,
        experiment_name: str = None,
        distribution: str = None,
        test_set: bool = True,
        nwp_variant: str = None,
    ):
        self.type = type
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.test_set = test_set
        self.nwp_variant = nwp_variant
        self.run_id = run_id
        self.dataset = dataset

    def artifact(self):
        set_label = "test" if self.test_set else "val"

        if self.type == "mlflow":
            label = f"{self.run_id}_{set_label}.nc"
        elif self.type == "nwp":
            label = f"{self.dataset}_{self.nwp_variant}_{set_label}.nc"
        elif self.type == "ensemble":
            label = f"{self.experiment_name}_{self.distribution}_{set_label}.nc"
        else:
            raise KeyError()

        return aq.LocalStoreArtifact(
            f"pp2023/metrics/spread_error_ratio/{self.type}/{label}.nc"
        )

    def requirements(self):
        if self.type == "ensemble":
            predictions_task = ModelPredictionsDispatch(
                dataset="gdps",
                type=self.type,
                experiment_name=self.experiment_name,
                distribution=self.distribution,
                test_set=self.test_set,
            )
        elif self.type == "nwp":
            predictions_task = NWPModelPredictionsDispatch(
                dataset="gdps", variant=self.nwp_variant, test_set=self.test_set
            )
        elif self.type == "mlflow":
            predictions_task = ModelPredictionsDispatch(
                type="mlflow",
                dataset="gdps",
                run_id=self.run_id,
                test_set=self.test_set,
            )
        else:
            raise NotImplementedError()

        return [
            predictions_task,
            SMC01_ValTestObs(),
            SMC01_QualityControlMask(),
        ]


class SpreadErrorRatio(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        linear_experiment = client.get_experiment_by_name("pp2023_linear_table_06")
        mlp_experiment = client.get_experiment_by_name("pp2023_mlp_table_08")

        runs = client.search_runs(
            experiment_ids=[
                linear_experiment.experiment_id,
                mlp_experiment.experiment_id,
            ],
            max_results=5000,
        )

        filtered_runs = [
            r.info.run_id
            for r in runs
            if r.data.params["distribution_name"] != "deterministic"
        ]

        nwp_variants = ["naive"]

        requirements = [
            *[
                ModelSpreadErrorRatioDispatch(
                    type="ensemble",
                    experiment_name=experiment_name,
                    distribution=distribution,
                    test_set=True,
                )
                for experiment_name in ["pp2023_linear_table_06", "pp2023_mlp_table_08"]
                for distribution in ["quantile", "emos", "bernstein"]
            ],
            *[
                ModelSpreadErrorRatioDispatch(type="nwp", nwp_variant=x, test_set=True)
                for x in nwp_variants
            ],
        ]

        return requirements

    def run(self, requirements) -> pd.DataFrame:
        dfs = []
        for model_stats in requirements:
            ratio = model_stats.spread / model_stats.error
            df = ratio.to_dataframe(name="spread_error_ratio")
            dfs.append(df)

            spread = model_stats.spread.to_dataframe(name="spread")
            df["spread"] = spread["spread"]

        df = pd.concat(dfs).reset_index()
        df = df[(df.step > pd.to_timedelta(0))]

        return df


class SpreadWithTime(SpreadErrorRatio):
    def run(self, requirements):
        dfs = []
        for model_stats in requirements:
            df = model_stats.spread.to_dataframe("spread")
            dfs.append(df)

        df = pd.concat(dfs).reset_index()

        df = df[(df["dataset"] != "gdps_prebatch_24h") | (df.step > pd.to_timedelta(0))]

        return df


class JointTrainingGain(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        partition_experiment = client.get_experiment_by_name(
            "pp2023_paper_step_partition"
        )
        partition_runs = client.search_runs(
            experiment_ids=[partition_experiment.experiment_id],
            max_results=5000,
        )

        partition_run_ids = [r.info.run_id for r in partition_runs]

        partition_run_tasks = [
            ModelMetrics(run_id=i, test_set=True) for i in partition_run_ids
        ]

        joint_run_id = "31213ca0ee91402b9798c82718946a7c"  # GDPS, Reduce Lr plateau, quantile. Taken from pp2023_mlp_table.

        # See experiment pp2023_table_linear.
        shared_linear_id = "bdc296547d4743f792630712eaf93af6"
        partitioned_linear_id = "c118adf84d1f44eb8ed37f835a9268fd"

        return [
            ModelMetrics(run_id=joint_run_id, test_set=True),
            ModelMetrics(run_id=shared_linear_id, test_set=True),
            ModelMetrics(run_id=partitioned_linear_id, test_set=True),
            *partition_run_tasks,
        ]

    def run(self, requirements):
        joint_run, shared_linear, partitioned_linear, *partitioned_runs = requirements

        joint_df = joint_run.crps.mean(dim=["forecast_time", "station"]).to_dataframe()
        joint_df["condition"] = "joint"
        dfs = [joint_df]

        for run in partitioned_runs:
            df = run.crps.mean(dim=["forecast_time", "station"]).to_dataframe()
            df["condition"] = "partitioned"
            dfs.append(df)

        shared_linear_df = shared_linear.mean(
            dim=["forecast_time", "station"]
        ).to_dataframe()
        shared_linear_df["condition"] = "joint"
        dfs.append(shared_linear_df)

        partitioned_linear_df = partitioned_linear.mean(
            dim=["forecast_time", "station"]
        ).to_dataframe()
        partitioned_linear_df["condition"] = "partitioned"
        dfs.append(partitioned_linear_df)

        return pd.concat(dfs)


def strategy_of_run(run_xr):
    conditioning_config = (
        run_xr.step_embedding.item(),
        run_xr.step_feature.item(),
    )

    if conditioning_config == ("True", "True"):
        return "both"
    elif conditioning_config == ("True", "False"):
        return "embedding"
    elif conditioning_config == ("False", "True"):
        return "feature"
    elif conditioning_config == ("False", "False"):
        return "none"
    else:
        raise RuntimeError("Unknown conditioning config")


class StepCondition(aq.Task):
    def requirements(self):
        full_runs = []
        partition_runs = []
        for distribution in ["quantile", "emos", "bernstein"]:
            for use_step_embedding in [True, False]:
                for use_step_feature in [True, False]:
                    full_runs.append(
                        EnsembleModelMetrics(
                            experiment_name="pp2023_mlp_step_other_condition_08",
                            distribution=distribution,
                            test_set=True,
                            step_embedding=use_step_embedding,
                            step_feature=use_step_feature,
                        )
                    )

            partitioned_runs_of_distribution = []
            for step in range(1, 11):
                partitioned_runs_of_distribution.append(
                    EnsembleModelMetrics(
                        experiment_name="pp2023_mlp_step_partition_09",
                        distribution=distribution,
                        test_set=True,
                        step_idx=step,
                        step_embedding=False,
                        step_feature=False,
                    )
                )
            partition_runs.append(partitioned_runs_of_distribution)

        return (full_runs, partition_runs)

    def run(self, requirements):
        (
            mlp_runs,
            partitioned_runs,
        ) = requirements

        partition_xr = xr.combine_nested(
            partitioned_runs, concat_dim=["distribution", "step"]
        ).assign_coords(strategy="partition")

        mlp_runs_nested = []
        for _ in ["quantile", "emos", "bernstein"]:
            runs_of_distribution = []
            for _ in [True, False]:
                for _ in [True, False]:
                    current_run = mlp_runs.pop(0)
                    runs_of_distribution.append(
                        current_run.assign(strategy=strategy_of_run(current_run))
                    )
            mlp_runs_nested.append(xr.concat(runs_of_distribution, dim="strategy"))

        mlp_xr = xr.concat(mlp_runs_nested, dim="distribution").sel(
            step=slice("1D", None)
        )

        full_xr = xr.concat([mlp_xr, partition_xr], dim="strategy")
        full_xr = full_xr[["crps"]]

        return full_xr

    def artifact(self):
        return aq.LocalStoreArtifact("pp2023/step_condition_crps.nc")


class SkillWithMembers(aq.Task):
    def requirements(self):
        reqs = []
        for distribution in ["emos", "bernstein", "quantile"]:
            metrics_of_distribution = []
            for n in range(1, 11):
                metrics_of_distribution.append(
                    EnsembleModelMetrics(
                        experiment_name="pp2023_mlp_n_members",
                        distribution=distribution,
                        test_set=True,
                        n_members=n,
                        dataset="ens10",
                    )
                )
            reqs.append(metrics_of_distribution)

        return reqs

    # def run(self, requirements):
    #     dfs = []

    #     for run in requirements:
    #         df = run.crps.mean(dim=["forecast_time", "station"]).to_dataframe()
    #         dfs.append(df)

    #     return pd.concat(dfs)

    def run(self, requirements):
        combined = xr.combine_nested(
            requirements, concat_dim=["distribution", "n_members"]
        )[["crps"]]

        return combined.assign_coords(n_members=combined.n_members.astype(int))


class CalibrationPlot(aq.Task):
    def __init__(self, run_id=None, dataset="gdps", variant=None):
        self.run_id = run_id
        self.test_set = True
        self.dataset = dataset
        self.variant = variant

    def requirements(self):
        if self.variant is not None:
            model_prediction_task = NWPModelPredictionsDispatch(
                test_set=self.test_set, variant=self.variant
            )
            obs_task = SMC01_ValTestObs()
        elif self.dataset == "gdps":
            model_prediction_task = SMC01_ModelPredictions(
                run_id=self.run_id, test_set=self.test_set
            )
            obs_task = SMC01_ValTestObs()
        elif self.dataset == "ens10":
            station_set = "ens10_stations_smc.csv"

            model_prediction_task = ModelPredictions(
                run_id=self.run_id,
                test_set=self.test_set,
                station_set=station_set,
            )
            obs_task = ValTestObs(station_set=station_set)
        else:
            raise KeyError()

        return [model_prediction_task, obs_task]

    def run(self, requirements):
        preds, obs_artifact = requirements
        return self.compute_counts(preds, obs_artifact)

    def compute_counts(self, preds, obs_artifact):
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        preds = preds.t2m.sel(step=preds.step > pd.to_timedelta(0, unit="d"))
        obs = obs.obs_t2m.sel(step=obs.step > pd.to_timedelta(0, unit="d"))
        obs = obs.reindex({"forecast_time": preds.forecast_time, "step": preds.step})

        bin_ids = (obs < preds).sum(dim="parameter")
        bin_ids_df = bin_ids.to_dataframe("bin_id")

        obs_df = obs.to_dataframe("obs")

        bin_ids_filtered = bin_ids_df[~np.isnan(obs_df["obs"])]
        bin_ids_total = len(bin_ids_filtered.index)
        bin_ids_counts = (
            bin_ids_filtered.groupby("bin_id")
            .count()[["model"]]
            .rename(columns={"model": "bin_count"})
        )

        bin_ids_counts["total"] = bin_ids_total
        bin_ids_counts["model"] = preds.coords["model"].item()
        bin_ids_counts["distribution"] = preds.coords["distribution"].item()
        bin_ids_counts["dataset"] = preds.coords["dataset"].item()

        return bin_ids_counts

    def artifact(self):
        return aq.LocalStoreArtifact(
            f"pp2023/calibration_plots/{self.run_id}_{self.dataset}.parquet"
        )


class CalibrationPlotEnsemble(CalibrationPlot):
    def __init__(self, experiment_name, distribution, test_set=True):
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.test_set = test_set

    def requirements(self):
        preds_task = ModelPredictionsEnsemble(
            self.experiment_name, self.distribution, test_set=self.test_set
        )
        obs_task = SMC01_ValTestObs()

        return preds_task, obs_task

    def artifact(self):
        set_label = "test" if self.test_set else "val"

        return aq.LocalStoreArtifact(
            f"pp2023/calibration_plots/ensemble_{self.experiment_name}_{self.distribution}_{set_label}.parquet"
        )

    def run(self, requirements):
        preds_artifact, obs = requirements
        preds = xr.open_dataset(preds_artifact.path)
        return super().run(preds, obs)


def pit_transform_normal(preds: np.array, n_bins: int = 33) -> np.array:
    loc = np.expand_dims(preds.sel(parameter="loc"), axis=-1)
    scale = np.expand_dims(preds.sel(parameter="scale"), axis=-1)

    bin_size = 1.0 / n_bins
    sample_points = np.tile(
        np.linspace(bin_size, 1.0 - bin_size, n_bins - 1), loc.shape
    )

    ppf = scipy.stats.norm.ppf(sample_points, loc=loc, scale=scale)

    return ppf


class CalibrationPlotNormal(CalibrationPlot):
    def __init__(
        self,
        dataset,
        run_id=None,
        variant=None,
        experiment_name=None,
        distribution=None,
        test_set=True,
    ):
        self.dataset = dataset
        self.run_id = run_id
        self.variant = variant
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.test_set = test_set

    def requirements(self):
        if self.run_id:
            return super().requirements()
        elif self.variant:
            return [
                NaiveNWPModelPredictions(self.test_set),
                SMC01_ValTestObs(),
            ]
        else:
            return [
                ModelPredictionsDispatch(
                    type="ensemble",
                    experiment_name=self.experiment_name,
                    distribution=self.distribution,
                    test_set=self.test_set,
                    dataset="gdps",
                ),
                SMC01_ValTestObs(),
            ]

    def run(self, requirements) -> pd.DataFrame:
        preds, obs_artifact = requirements

        pit_preds = pit_transform_normal(preds.t2m)

        preds_coords = {k: preds.coords[k] for k in preds.coords if k != "parameter"}

        pit_preds_xr = xr.DataArray(
            pit_preds,
            dims=["forecast_time", "step", "station", "parameter"],
            coords=preds_coords,
        )

        pit_preds_dataset = xr.Dataset({"t2m": pit_preds_xr})

        return self.compute_counts(pit_preds_dataset, obs_artifact)

    def artifact(self):
        set_label = "test" if self.test_set else "val"

        if self.run_id:
            return aq.LocalStoreArtifact(
                f"pp2023/calibration_plots/{self.run_id}_{set_label}.parquet"
            )
        elif self.variant:
            return aq.LocalStoreArtifact(
                f"pp2023/calibration_plots/nwp_{self.variant}_{set_label}.parquet"
            )
        else:
            return aq.LocalStoreArtifact(
                f"pp2023/calibration_plots/{self.experiment_name}_{self.distribution}_{set_label}.parquet"
            )


class CalibrationPlotDispatch(aq.Task):
    def __init__(
        self,
        type,
        run_id=None,
        experiment_name=None,
        distribution=None,
        nwp_variant=None,
        test_set=True,
    ):
        self.type = type
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.nwp_variant = nwp_variant
        self.test_set = test_set

    def requirements(self):
        if self.type == "ensemble":
            return CalibrationPlotEnsemble(
                self.experiment_name,
                distribution=self.distribution,
                experiment_name=self.experiment_name,
                test_set=self.test_set,
            )
        elif self.type == "mlflow":
            return CalibrationPlotNormal(self.run_id, test_set=self.test_set)
        elif self.type == "nwp":
            return CalibrationPlot(variant=self.nwp_variant, test_set=self.test_set)
        else:
            raise KeyError()

    def run(self, requirements):
        return requirements


class CalibrationPlots(aq.Task):
    def requirements(self):
        return [
            CalibrationPlotNormal("gdps", variant="naive"),  # Naive baseline.
            CalibrationPlotEnsemble(
                "pp2023_mlp_table_08", "bernstein", True
            ),  # MLP Bernstein
            CalibrationPlotEnsemble(
                "pp2023_mlp_table_08", "quantile", True
            ),  # MLP Quantile
            CalibrationPlotNormal(
                "gdps",
                experiment_name="pp2023_mlp_table_08",
                distribution="emos",
                test_set=True,
            ),  # MLP Normal
            CalibrationPlot("b00adfeb13bf4944a86af9d20d76a789"),  # Linear Bernstein
            CalibrationPlot("899a9e1b93bd4978b10203022fcf277e"),  # Linear Quantile
            CalibrationPlotNormal(
                "gdps", run_id="e9ef58e43d584ab9bdc1f00067533123"
            ),  # Linear Normal
        ]

    def run(self, requirements) -> pd.DataFrame:
        return pd.concat(requirements)


class ConditioningMechanismsTable(aq.Task):
    def requirements(self):
        experiments = [
            "pp2023_linear_share_step",
            "pp2023_linear_nosharestep",
            "pp2023_paper_step_partition",
            "pp2023_paper_step_condition",
            "pp2023_space_condition",
        ]

        client = mlflow.client.MlflowClient()

        experiment_ids = [
            client.get_experiment_by_name(experiment_name).experiment_id
            for experiment_name in experiments
        ]
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            max_results=5000,
        )

        requirements = [ModelMetrics(run_id=r.info.run_id, test_set=True) for r in runs]

        return requirements

    def run(self, requirements):
        rows = []
        for r in requirements:
            metadata = {
                k: r.coords[k].item()
                for k in [
                    "distribution",
                    "dataset",
                    "model",
                    "step_feature",
                    "step_embedding",
                    "space_features",
                    "station_embedding",
                    "step_partition",
                    "share_step",
                ]
            }

            dims = ["forecast_time", "station", "step"]
            mean = r.mean(dim=dims)
            count = np.count_nonzero(~np.isnan(r.crps))

            rows.append(
                {
                    **metadata,
                    "crps": mean.crps.item(),
                    "mse": mean.mse.item(),
                    "count": count,
                }
            )

        df = pd.DataFrame(rows)
        df["step_partition"] = df["step_partition"].astype("str")
        return df

    def artifact(self):
        return aq.LocalStoreArtifact("pp2023/condition_strategies_table.parquet")


class ConditioningTableLinearStep(aq.Task):
    EXPERIMENT_NAME = ["pp2023_condition_linear_step"]
    METADATA = [
        "model",
        "dataset",
        "step_feature",
        "share_step",
        "distribution",
        "run_id",
    ]

    def requirements(self):
        client = mlflow.client.MlflowClient()

        requirements = []
        for experiment_name in self.EXPERIMENT_NAME:
            experiment = client.get_experiment_by_name(experiment_name)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], max_results=5000
            )

            for r in runs:
                requirements.append(ModelMetrics(run_id=r.info.run_id, test_set=True))

        return requirements

    def run(self, requirements: list[xr.Dataset]):
        rows = []
        for full_run in requirements:
            if "step" in full_run.sizes:
                full_run = full_run.sel(
                    step=full_run.step > pd.to_timedelta(0, unit="d")
                )
            elif int(full_run.step_partition.item()) == 0:
                continue

            for lead in ["first_two", "all"]:
                if lead == "first_two":
                    run = full_run.sel(
                        step=full_run.step <= pd.to_timedelta(2, unit="d")
                    )
                else:
                    run = full_run

                means = run.mean(dim=["station", "forecast_time", "step"])
                for metric in ["crps"]:
                    row = {"lead": lead}
                    row[metric] = means[metric].item()

                    for metadata in self.METADATA:
                        row[metadata] = means.coords[metadata].item()

                    rows.append(row)

        return pd.DataFrame(rows)

    def artifact(self):
        return aq.LocalStoreArtifact(
            f"pp2023/table_conditioning_{self.EXPERIMENT_NAME}.parquet"
        )


class ConditioningTableLinearStation(ConditioningTableLinearStep):
    EXPERIMENT_NAME = ["pp2023_condition_linear_station"]
    METADATA = [
        "model",
        "dataset",
        "share_station",
        "space_features",
        "distribution",
    ]


class ConditioningTableMLPStation(ConditioningTableLinearStep):
    EXPERIMENT_NAME = [
        "pp2023_condition_mlp_station",
        "pp2023_condition_mlp_station_02",
    ]
    METADATA = [
        "model",
        "dataset",
        "space_features",
        "station_embedding",
        "distribution",
    ]


class ConditioningTableMLPStep(aq.Task):
    METADATA = [
        "model",
        "dataset",
        "step_feature",
        "step_embedding",
        "step_partition",
        "distribution",
        "run_id",
    ]

    def requirements(self):
        client = mlflow.client.MlflowClient()

        # experiment_names = [
        #     "pp2023_mlp_condition_step_partition_03",
        #     "pp2023_condition_mlp_step_partition_02",
        #     "pp2023_mlp_condition_step_02",
        #     "pp2023_mlp_condition_step_04",
        #     "pp2023_condition_mlp_step",
        # ]

        experiment_names = [
            "pp2023_mlp_step",
            "pp2023_mlp_other_condition",
        ]

        experiment_ids = []
        for name in experiment_names:
            experiment = client.get_experiment_by_name(name)
            experiment_ids.append(experiment.experiment_id)

        runs = client.search_runs(experiment_ids=experiment_ids, max_results=5000)

        return [ModelMetrics(run_id=r.info.run_id, test_set=True) for r in runs]

    def run(self, requirements: list[xr.Dataset]):
        dfs = []
        for full_run in requirements:
            if "step" not in full_run.sizes:
                full_run = full_run.expand_dims(dim=["step"])

            full_run = full_run.sel(step=full_run.step > pd.to_timedelta(0, unit="d"))

            for lead in ["first_two", "all"]:
                if lead == "first_two":
                    run = full_run.sel(
                        step=full_run.step <= pd.to_timedelta(2, unit="d")
                    )
                else:
                    run = full_run

                if "step" in run.sizes and run.sizes["step"] > 1:
                    df = run.mean(dim=["station", "forecast_time"]).to_dataframe()

                    count = run.crps.count(
                        dim=["station", "forecast_time"]
                    ).to_dataframe(name="count")

                    df["count"] = count["count"]
                    df["lead"] = lead

                    dfs.append(df)
                else:
                    row = {}
                    for metric in ["crps"]:
                        row[metric] = run[metric].mean().item()

                    for metadata in self.METADATA:
                        row[metadata] = run.coords[metadata].item()

                    row["count"] = run.crps.count().item()
                    index = [
                        pd.to_timedelta(
                            int(run.coords["step_partition"].item()), unit="h"
                        )
                    ]

                    df = pd.DataFrame([row], index=index)
                    df["lead"] = lead

                    dfs.append(df)

        dataframe = pd.concat(dfs)

        dataframe["dataset"] = dataframe["dataset"].replace(
            {
                "gdps_prebatch_partition": "gdps_prebatch_24h",
                "ens10_prebatch_partition": "ens10_prebatch",
            }
        )

        return dataframe


class SpatialCRPS(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        experiment_mlp = client.get_experiment_by_name("pp2023_mlp_table_02")
        all_runs = [
            r.info.run_id
            for r in client.search_runs(
                experiment_ids=[experiment_mlp.experiment_id], max_results=5000
            )
            if r.data.params["distribution_name"] == "bernstein"
        ]

        tasks = [ModelMetrics(i, test_set=True) for i in all_runs]

        for dataset in ["gdps", "ens10"]:
            tasks.append(NWPModelMetrics(dataset=dataset, test_set=True, debias=True))

        return tasks

    def run(self, requirements):
        dfs = []

        for metrics in requirements:
            metrics = metrics.sel(step=((metrics.step > pd.to_timedelta(0))))

            mean = metrics.crps.mean(dim=["forecast_time", "step"])

            dfs.append(mean.to_dataframe())

        df = pd.concat(dfs)
        df["dataset"] = df["dataset"].replace(
            {"gdps_prebatch_24h": "gdps", "ens10_prebatch": "ens10"}
        )

        return df


def sort_mean(q):
    return np.sort(q, axis=-1)


class ModelPredictionsEnsemble(aq.IOTask):
    def __init__(
        self,
        experiment_name,
        distribution,
        n_runs=5,
        test_set=True,
        step_embedding=True,
        step_feature=True,
        step_idx=None,
        n_members=None,
    ):
        self.experiment_name = experiment_name
        self.distribution = distribution
        self.n_runs = n_runs
        self.test_set = test_set
        self.step_embedding = step_embedding
        self.step_feature = step_feature
        self.step_idx = step_idx
        self.n_members = n_members

    def requirements(self):
        client = mlflow.client.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], max_results=5000
        )

        run_ids = []
        for r in runs:
            if (
                r.data.params["distribution_name"] == self.distribution
                and r.data.params["model.use_step_embedding"]
                == str(self.step_embedding)
                and r.data.params["model.use_step_feature"] == str(self.step_feature)
            ):
                if self.step_idx is not None:
                    if "dataset.maker.step_idx" in r.data.params and r.data.params[
                        "dataset.maker.step_idx"
                    ] == str(self.step_idx):
                        run_ids.append(r.info.run_id)
                elif self.n_members is not None:
                    if "dataset.n_members" in r.data.params and r.data.params[
                        "dataset.n_members"
                    ] == str(self.n_members):
                        run_ids.append(r.info.run_id)
                else:
                    run_ids.append(r.info.run_id)

        run_ids = run_ids[: self.n_runs]

        return [task_of_run_id(run_id=i, test_set=self.test_set) for i in run_ids]

    def run(self, reqs):
        all_preds = xr.open_mfdataset(
            [r.path for r in reqs], combine="nested", concat_dim="run_id"
        )

        if all_preds.distribution.item() in ["bernstein", "quantile"]:
            all_preds = xr.apply_ufunc(
                sort_mean,
                all_preds,
                dask="allowed",
            )

        all_preds.mean(dim="run_id").to_netcdf(self.artifact().path)

    def artifact(self):
        set_label = "test" if self.test_set else "val"
        step_idx_label = f"_{self.step_idx}" if self.step_idx else ""
        n_member_label = f"_n{self.n_members}" if self.n_members else ""

        return aq.LocalStoreArtifact(
            f"pp2023/predictions/ensemble/{self.experiment_name}_{self.distribution}_{set_label}_{self.step_embedding}_{self.step_feature}{step_idx_label}{n_member_label}.nc"
        )


class AllFigures(aq.Task):
    def requirements(self):
        return [
            MetricsByStep(),
            SkillWithMembers(),
            JointTrainingGain(),
            SpreadErrorRatio(),
            DispersionByStep(),
            CRPSByStep(),
            TableCRPS(),
        ]
