from typing import Optional, Any

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
from eddie.pp2023.smc01 import SMC01_RescaleStatistics, SMC01_ValTestObs

from ..lightning import PP2023Module
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

    merged_overrides = [*run_overrides, "ex.dataset.maker.cache=False", *overrides]

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

        mlflow_run = get_run_object(run_id)

        rescaled_predictions = rescaled_predictions.assign_coords(
            distribution=mlflow_run.data.params["distribution_name"],
            dataset=mlflow_run.data.params["dataset_name"],
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
    def __init__(self, dataset="gdps", test_set=False, debias=True):
        self.dataset = dataset
        self.test_set = test_set
        self.debias = debias

    def requirements(self):
        if self.dataset == "gdps":
            return SMC01_ValTestObs()
        elif self.dataset == "ens10":
            return ValTestObs(station_set="ens10_stations_smc.csv")
        else:
            raise KeyError()

    def run(self, preds_artifacts):
        val_artifact, test_artifact = preds_artifacts.artifacts

        preds_artifact = test_artifact if self.test_set else val_artifact
        preds = xr.open_dataset(preds_artifact.path)

        if self.dataset == "gdps":
            preds = preds.isel(step=slice(0, 82, 8)).expand_dims(dim="number")

        if self.debias:
            obs = preds[["obs_t2m"]].rename({"obs_t2m": "t2m"})
            bias = (obs - preds[["t2m"]]).mean(dim=["forecast_time", "number"])

            preds = preds + bias

        model_name = "debiased" if self.debias else "raw_nwp"

        return (
            preds[["t2m"]]
            .assign_coords(
                dataset=self.dataset, distribution="quantile", model=model_name
            )
            .rename({"number": "parameter"})
            .transpose("forecast_time", "step", "station", "parameter")
        )


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


def requirements_for_run_metrics(run_id, test_set=False):
    mlflow_run = get_run_object(run_id)

    if mlflow_run.data.params["dataset_name"].startswith("gdps"):
        model_prediction_task = SMC01_ModelPredictions(run_id=run_id, test_set=test_set)
        obs_task = SMC01_ValTestObs()
    elif mlflow_run.data.params["dataset_name"].startswith("ens10"):
        model_prediction_task = ModelPredictions(
            run_id=run_id,
            station_set="ens10_stations_smc.csv",
            test_set=test_set,
        )
        obs_task = ValTestObs(station_set="ens10_stations_smc.csv")

    return [
        model_prediction_task,
        obs_task,
    ]


def do_one_chunk(inputs):
    p_dataset, obs_dataset = inputs
    crps = crps_empirical_np(p_dataset, obs_dataset)
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
        mse_chunks.append(np.square(p_dataset.mean(dim="parameter") - obs_dataset))
        mae_chunks.append(np.abs(p_dataset.median(dim="parameter") - obs_dataset))
        bias_chunks.append(p_dataset.mean(dim="parameter") - obs_dataset)

    crps_xr = xr.combine_nested(crps_chunks, concat_dim="forecast_time")
    mse_xr = xr.combine_nested(mse_chunks, concat_dim="forecast_time")
    mae_xr = xr.combine_nested(mae_chunks, concat_dim="forecast_time")
    bias_xr = xr.combine_nested(bias_chunks, concat_dim="forecast_time")
    return crps_xr, mse_xr, mae_xr, bias_xr


def compute_model_metrics(preds: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
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
    elif distribution in ["bernstein", "quantile"]:
        crps, mse, mae, bias = crps_of_prediction(preds, obs)
    elif distribution == "deterministic":
        crps = np.abs(preds - obs).squeeze()
        mse = np.square(preds - obs).squeeze()
        bias = (preds - obs).squeeze()
        mae = crps
    else:
        raise KeyError("Could not compute crps for distribution type.")

    return xr.Dataset({"crps": crps, "mse": mse, "bias": bias, "mae": mae})


class ModelMetrics(aq.Task):
    def __init__(self, run_id: str, test_set: bool = False):
        self.run_id = run_id
        self.test_set = test_set

    def requirements(self):
        return requirements_for_run_metrics(self.run_id, self.test_set)

    def run(self, requirements) -> xr.Dataset:
        preds, obs_artifact = requirements
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        return compute_model_metrics(preds, obs)

    def artifact(self):
        set_label = "test" if self.test_set else "val"
        return aq.LocalStoreArtifact(f"pp2023/metrics/{self.run_id}_{set_label}.nc")


class NWPModelMetrics(aq.Task):
    def __init__(self, dataset: str = "gdps", test_set=False, debias=True):
        self.dataset = dataset
        self.test_set = test_set
        self.debias = debias

    def requirements(self):
        if self.dataset == "gdps":
            obs_task = SMC01_ValTestObs()
        elif self.dataset == "ens10":
            obs_task = ValTestObs(station_set="ens10_stations_smc.csv")
        else:
            KeyError()

        return [
            NWPModelPredictions(
                dataset=self.dataset, test_set=self.test_set, debias=self.debias
            ),
            obs_task,
        ]

    def run(self, reqs):
        preds, obs_artifact = reqs
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        return compute_model_metrics(preds, obs)

    def artifact(self):
        model_label = "debias" if self.debias else "raw_nwp"
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
        client = mlflow.client.MlflowClient()
        linear_experiment = client.get_experiment_by_name("pp2023_linear_table")
        linear_runs = client.search_runs(
            experiment_ids=[linear_experiment.experiment_id]
        )

        linear_runs = [
            r.info.run_id for r in linear_runs if r.info.run_name == "no_share_stations"
        ]

        mlp_experiment = client.get_experiment_by_name("pp2023_mlp_table")
        mlp_runs = client.search_runs(experiment_ids=[mlp_experiment.experiment_id])
        mlp_runs = [
            r.info.run_id
            for r in mlp_runs
            if r.data.params["scheduler.instance._target_"]
            == "pp2023.scheduler.reduce_lr_plateau"
        ]

        all_runs = [*linear_runs, *mlp_runs]

        runs_tasks = [ModelMetrics(run_id=run_id, test_set=True) for run_id in all_runs]

        for dataset in ["gdps", "ens10"]:
            runs_tasks.append(
                NWPModelMetrics(dataset=dataset, test_set=True, debias=False)
            )
            runs_tasks.append(
                NWPModelMetrics(dataset=dataset, test_set=True, debias=True)
            )

        return runs_tasks

    def run(self, requirements: list[xr.Dataset]):
        rows = []
        for metrics in requirements:
            crps_mean = metrics.crps.mean()
            rmse = np.sqrt(metrics.mse.mean())
            rows.append(
                {
                    "rmse": rmse.item(),
                    "crps": crps_mean.item(),
                    "distribution": metrics.coords["distribution"].item(),
                    "dataset": metrics.coords["dataset"].item(),
                    "model": metrics.coords["model"].item(),
                }
            )

        df = pd.DataFrame(rows)
        df["dataset"] = df["dataset"].replace(
            {"gdps_prebatch_24h": "gdps", "ens10_prebatch": "ens10"}
        )

        return df


class MetricsByStep(TableCRPS):
    def run(self, requirements):
        dfs = []

        for model_metrics in requirements:
            df = model_metrics.mean(dim=["forecast_time", "station"]).to_dataframe()
            df["rmse"] = np.sqrt(df["mse"])
            dfs.append(df)

        df = pd.concat(dfs)
        df["dataset"] = df["dataset"].replace(
            {"gdps_prebatch_24h": "gdps", "ens10_prebatch": "ens10"}
        )

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
            experiment_ids=[linear_experiment.experiment_id]
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
            experiment_ids=[mlp_experiment.experiment_id]
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
    def __init__(self, run_id: str, test_set: bool = False):
        self.run_id = run_id
        self.test_set = test_set

    def requirements(self):
        return requirements_for_run_metrics(self.run_id, self.test_set)

    def run(self, requirements: tuple[xr.Dataset, Any]):
        preds, obs_artifact = requirements
        obs_artifact = (
            obs_artifact.artifacts[1] if self.test_set else obs_artifact.artifacts[0]
        )
        obs = xr.open_dataset(obs_artifact.path)

        distribution = preds.coords["distribution"]
        dataset = preds.coords["dataset"]

        # Only do t2m for now.
        preds = preds.t2m
        obs = obs.obs_t2m

        ensemble_mean = preds.mean(dim="parameter")
        rmse_of_ensemble_mean = np.sqrt(
            np.square(ensemble_mean - obs).mean(dim=["station", "forecast_time"])
        )

        ensemble_variance = np.square(ensemble_mean - preds).sum(dim="parameter") / (
            preds.sizes["parameter"] - 1
        )

        root_mean_ensemble_variance = np.sqrt(
            ensemble_variance.mean(dim=["station", "forecast_time"])
        )

        return root_mean_ensemble_variance / rmse_of_ensemble_mean


class SpreadErrorRatio(aq.Task):
    def requirements(self):
        run_ids = [
            "006631e1f6a049aba484f7ec99f9ddaf",  # GDPS EMOS Reduce Lr Plateau MLP
            "31213ca0ee91402b9798c82718946a7c",  # GDPS Quantile Reduce Lr Plateau MLP
            "206af3ad4d75499bb104b127bdfdceba",  # GDPS Bernstein MLP
            "bbd31bd582e04a29807a4d0715e00c4f",  # Linear EMOS,
            "a65ad3d0748448d296987bf06a629b30",  # Linear Quantile
            "b3c22606d4cd495eb562c7c0a4d091a6",  # Linear Bernstein
        ]

        return [ModelSpreadErrorRatio(run_id=i, test_set=True) for i in run_ids]

    def run(self, requirements) -> pd.DataFrame:
        dfs = []
        for model_stats in requirements:
            df = model_stats.to_dataframe(name="spread_error_ratio")
            dfs.append(df)

        return pd.concat(dfs)


class JointTrainingGain(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        partition_experiment = client.get_experiment_by_name(
            "pp2023_paper_step_partition"
        )
        partition_runs = client.search_runs(
            experiment_ids=[partition_experiment.experiment_id]
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


class StepCondition(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        partition_experiment = client.get_experiment_by_name(
            "pp2023_paper_step_partition"
        )
        partition_runs = client.search_runs(
            experiment_ids=[partition_experiment.experiment_id]
        )

        partition_run_ids = [r.info.run_id for r in partition_runs]

        partition_run_tasks = [
            ModelMetrics(run_id=i, test_set=True) for i in partition_run_ids
        ]

        no_condition_run = ModelMetrics(
            run_id="14a8b2308ce146c3af44705a43bad932", test_set=True
        )
        only_feature = ModelMetrics(
            run_id="dc32dbffc8814d14a05e18ba9dae36d5", test_set=True
        )
        only_embedding = ModelMetrics(
            run_id="f88f2177b5d14a8c885b9d3027e47ff1", test_set=True
        )
        both = ModelMetrics(run_id="d687fba27d584681b0e68ecc969b0cac", test_set=True)

        return [
            no_condition_run,
            only_feature,
            only_embedding,
            both,
            *partition_run_tasks,
        ]

    def run(self, requirements):
        (
            no_condition,
            only_feature,
            only_embedding,
            both,
            *partitioned_runs,
        ) = requirements

        no_condition_df = no_condition.mean(
            dim=["forecast_time", "station"]
        ).to_dataframe()
        no_condition_df["condition"] = "none"

        only_feature_df = only_feature.mean(
            dim=["forecast_time", "station"]
        ).to_dataframe()
        only_feature_df["condition"] = "feature"

        only_embedding_df = only_embedding.mean(
            dim=["forecast_time", "station"]
        ).to_dataframe()
        only_embedding_df["condition"] = "embedding"

        both_df = both.mean(dim=["forecast_time", "station"]).to_dataframe()
        both_df["condition"] = "both"

        dfs = [no_condition_df, only_feature_df, only_embedding_df, both_df]

        for run in partitioned_runs:
            df = run.crps.mean(dim=["forecast_time", "station"]).to_dataframe()
            df["condition"] = "partitioned"
            dfs.append(df)

        return pd.concat(dfs)


class SkillWithMembers(aq.Task):
    def requirements(self):
        client = mlflow.client.MlflowClient()
        partition_experiment = client.get_experiment_by_name("pp2023_paper_n_members")
        partition_runs = client.search_runs(
            experiment_ids=[partition_experiment.experiment_id]
        )

        partition_run_ids = [r.info.run_id for r in partition_runs]

        partition_run_tasks = [
            ModelMetrics(run_id=i, test_set=True) for i in partition_run_ids
        ]

        return partition_run_tasks

    def run(self, requirements):
        dfs = []

        for run in requirements:
            df = run.crps.mean(dim=["forecast_time", "station"]).to_dataframe()
            dfs.append(df)

        return pd.concat(dfs)


class CalibrationPlot(aq.Task):
    def __init__(self, run_id: str, dataset="gdps", lead_time=2):
        self.run_id = run_id
        self.test_set = True
        self.dataset = dataset
        self.lead_time = lead_time

    def requirements(self):
        if self.dataset == "gdps":
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

        obs = obs.reindex({"forecast_time": preds.forecast_time, "step": preds.step})

        preds = preds.t2m.sel(step=pd.to_timedelta(self.lead_time, unit="days"))
        obs = obs.obs_t2m.sel(step=pd.to_timedelta(self.lead_time, unit="days"))

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


def pit_transform_normal(preds: np.array, n_bins: int = 17) -> np.array:
    loc = np.expand_dims(preds.sel(parameter="loc"), axis=-1)
    scale = np.expand_dims(preds.sel(parameter="scale"), axis=-1)

    bin_size = 1.0 / n_bins
    sample_points = np.tile(
        np.linspace(bin_size, 1.0 - bin_size, n_bins - 1), loc.shape
    )

    ppf = scipy.stats.norm.ppf(sample_points, loc=loc, scale=scale)

    return ppf


class CalibrationPlotNormal(CalibrationPlot):
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


class CalibrationPlots(aq.Task):
    def requirements(self):
        return [
            CalibrationPlot("206af3ad4d75499bb104b127bdfdceba"),  # MLP GDPS Bernstein
            CalibrationPlot("31213ca0ee91402b9798c82718946a7c"),  # MLP GDPS Quantile
            CalibrationPlotNormal(
                "006631e1f6a049aba484f7ec99f9ddaf"
            ),  # MLP GDPS Normal
            CalibrationPlot(
                "3f0ec04ff96944d9a531452de24e7bdb", dataset="ens10"
            ),  # MLP ENS10 Bernstein
            CalibrationPlot(
                "ed36cb3ef34d413889a103008cd01713", dataset="ens10"
            ),  # MLP ENS10 Quantile
            CalibrationPlotNormal(
                "b2d718d49f9748529da630d8ae211f24", dataset="ens10"
            ),  # MLP ENS10 Normal
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
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])

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

        experiment_names = [
            "pp2023_mlp_condition_step_partition_03",
            "pp2023_condition_mlp_step_partition_02",
            "pp2023_mlp_condition_step_02",
            "pp2023_mlp_condition_step_04",
            "pp2023_condition_mlp_step",
        ]

        experiment_ids = []
        for name in experiment_names:
            experiment = client.get_experiment_by_name(name)
            experiment_ids.append(experiment.experiment_id)

        runs = client.search_runs(experiment_ids=experiment_ids)

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
