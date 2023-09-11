from typing import Callable
import math
import torch
import torch.distributions as td


LOG_SI10_CLAMP = -0.2


class DistributionalForecast:
    def crps(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DistributionalForecastStrategy:
    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        raise NotImplementedError

    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class NormalParametricStrategy(DistributionalForecastStrategy):
    def __init__(self, variable_idx=None, loss_fn: str = "crps"):
        self.variable_idx = variable_idx
        self.loss_fn = loss_fn

    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        forecast_mu = features[..., 0]
        log_forecast_sigma = features[..., 1]

        forecast_sigma = torch.exp(log_forecast_sigma)
        processed_features = torch.stack([forecast_mu, forecast_sigma], dim=-1)

        return NormalParametric(processed_features, loss_fn=self.loss_fn)

    def nwp_base(self, batch):
        forecast_mu = batch["forecast"].mean(dim=1)

        if batch["forecast"].shape[1] == 1:
            forecast_sigma = torch.full_like(forecast_mu, 1.0)
        else:
            forecast_sigma = batch["forecast"].std(dim=1)

        log_forecast_sigma = torch.log(forecast_sigma + 1e-6)

        processed_forecast_params = torch.stack(
            [forecast_mu, log_forecast_sigma], dim=-1
        )

        return processed_forecast_params


class NormalParametric(DistributionalForecast):
    def __init__(self, params, loss_fn: str = "crps"):
        LOSSES_FOR_NORMAL = {
            "crps": self.crps,
            "log_likelihood": self.log_likelihood,
            "dincae": self.dincae_loss,
        }

        self.loss_fn = LOSSES_FOR_NORMAL[loss_fn]

        loc = params[..., 0]

        loc_t2m, loc_log_si10 = loc[..., 0], loc[..., 1]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        loc_log_si10 = torch.clamp(loc_log_si10, min=LOG_SI10_CLAMP)

        loc = torch.stack([loc_t2m, loc_log_si10], dim=-1)

        scale = params[..., 1]

        self.distribution = td.Normal(loc=loc, scale=scale)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x)

    def crps(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)

    def log_likelihood(self, x):
        return -self.distribution.log_prob(x)

    def dincae_loss(self, x):
        """See DOI: 10.5194/gmd-15-2183-2022"""
        squared_scaled_error = torch.square(
            (x - self.distribution.loc) / self.distribution.scale
        )

        sigma_squared = torch.square(self.distribution.scale)

        return squared_scaled_error + sigma_squared


class DeterministicStrategy(DistributionalForecastStrategy):
    def __init__(self, variable_idx=None, use_base=True):
        """Args:
        variable_idx: The index of the variable to make distributions for. If None,
            will train on all the variables jointly.
        use_base: Wether to use the NWP forecast as a basis and only predict the
            residual, or to make our own forecast from scratch."""
        self.variable_idx = variable_idx
        self.use_base = use_base

    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        forecast = batch["forecast"]
        forecast = forecast.mean(dim=1).unsqueeze(-1)

        if self.variable_idx is not None:
            forecast = forecast[..., [self.variable_idx], :]

        # Return the mean over the ensemble member dimension.
        # Unsqueeze the last dimension which is the "parameter" dimension. In the
        # case of a deterministic forecast, the parameter is 1.
        return (
            forecast
            if self.use_base
            else torch.zeros_like(forecast, device=forecast.device)
        )

    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        return DeterministicForecast(features, variable_idx=self.variable_idx)


class DeterministicForecast(DistributionalForecast):
    def __init__(self, params, variable_idx=None):
        self.variable_idx = variable_idx

        t2m, log_si10 = params[..., 0, :], params[..., 1, :]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        log_si10 = torch.clamp(log_si10, min=LOG_SI10_CLAMP)

        params = torch.stack([t2m, log_si10], dim=-2)

        if variable_idx is not None:
            params = params[..., [variable_idx], :]

        self.preds = params.squeeze(-1)

    def loss(self, x: torch.Tensor):
        if self.variable_idx is not None:
            x = x[..., [self.variable_idx]]

        return torch.square(x - self.preds)

    def crps(self, x: torch.Tensor):
        """Args:
        x: [B, v] tensor where B is the batch dimension and v the variable dimension."""
        if self.variable_idx is not None:
            x = x[..., [self.variable_idx]]

        return torch.abs(x - self.preds)


class QuantileRegression(DistributionalForecast):
    def __init__(self, params, regularization=1e-5):
        t2m, log_si10 = params[..., 0, :], params[..., 1, :]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        log_si10 = torch.clamp(log_si10, min=LOG_SI10_CLAMP)

        params = torch.stack([t2m, log_si10], dim=-2)
        params, _ = torch.sort(params)

        self.parameters = params
        self.r = regularization

    def loss(self, x: torch.Tensor):
        misalignments = self.r * -torch.square(
            torch.clamp(self.parameters[..., 1:] - self.parameters[..., :-1], max=0)
        ).sum(dim=-1)

        return self.crps(x) + misalignments

    def auc(self, x: torch.Tensor):
        p = self.parameters
        n_quantiles = p.shape[-1]
        x = x.unsqueeze(-1)

        crps = torch.square(torch.abs(x - p).sum(dim=-1) / n_quantiles)

        return crps

    def crps(self, x: torch.Tensor):
        return crps_empirical(self.parameters, x.unsqueeze(-1), sorted=True)


def quantile_loss(quantile_values, obs, quantiles=None):
    """Args
    quantile_values: Tensor with the last dimension reserved for quantile values.
    obs: The observation to score against.
    quantiles: Tensor containing the quantile that are target by the values. If
        None, will consider the quantiles are uniformly spred from 0 to 1. Must
        have the same length as quantile_values.shape[-1]"""

    if quantiles is None:
        n_quantiles = quantile_values.shape[-1]
        lower_bound = 1.0 / (n_quantiles + 1)
        upper_bound = 1.0 - lower_bound
        quantiles = torch.linspace(
            lower_bound, upper_bound, n_quantiles, device=quantile_values.device
        )

    obs = obs.unsqueeze(-1)
    left_quantile_loss = -1.0 * (quantiles.unsqueeze(0)) * (quantile_values - obs)
    right_quantile_loss = (1 - quantiles.unsqueeze(0)) * (quantile_values - obs)

    quantile_loss = torch.where(
        quantile_values < obs, left_quantile_loss, right_quantile_loss
    )

    return quantile_loss.mean(dim=-1)


class BernsteinQuantileFunctionForecast(DistributionalForecast):
    def __init__(self, params: torch.Tensor, degree: int, device=None):
        self.coefficients = params
        self.degree = degree
        self.poly = bernstein_polynomial(self.degree, device=params.device)
        self.n_samples = 100

    def make_quantile_values(self, n_samples, device):
        quantiles = torch.linspace(0.0, 1.0, n_samples, device=device)

        poly_values = self.poly(quantiles)

        quantile_values = (poly_values * self.coefficients.unsqueeze(-1)).sum(dim=-2)
        return quantile_values

    def loss(self, x):
        quantile_values = self.make_quantile_values(self.n_samples, x.device)

        return quantile_loss(quantile_values, x)

    def crps(self, x):
        quantile_values = self.make_quantile_values(self.n_samples, x.device)
        return crps_empirical(quantile_values, x.unsqueeze(-1))


class BernsteinQuantileFunctionStrategy(DistributionalForecastStrategy):
    def __init__(self, n_parameters: int, use_base=True, variable_idx=None):
        self.degree = n_parameters - 1
        self.variable_idx = None
        self.use_base = use_base

    def nwp_base(self, batch):
        forecast = batch["forecast"]
        forecast = forecast.mean(dim=1).unsqueeze(-1)

        if self.variable_idx is not None:
            forecast = forecast[..., [self.variable_idx], :]

        # Return the mean over the ensemble member dimension.
        # Unsqueeze the last dimension which is the "parameter" dimension. In the
        # case of a deterministic forecast, the parameter is 1.
        return (
            forecast
            if self.use_base
            else torch.zeros_like(forecast, device=forecast.device)
        )

    def from_tensor(self, x: torch.Tensor):
        return BernsteinQuantileFunctionForecast(x, degree=self.degree)


def quantiles_to_deltas(quantiles):
    REG = 1e-6

    midpoint = quantiles.shape[-1] // 2

    mid_value = quantiles[..., [midpoint]]

    left_values = mid_value - quantiles[..., 0 : (midpoint + 1)]
    left_deltas = left_values[..., :-1] - left_values[..., 1:] + REG
    log_left_deltas = torch.log(left_deltas)

    right_values = quantiles[..., midpoint:] - mid_value
    right_deltas = right_values[..., 1:] - right_values[..., :-1] + REG
    log_right_deltas = torch.log(right_deltas)

    deltas = torch.cat([log_left_deltas, mid_value, log_right_deltas], dim=-1)
    return deltas


def deltas_to_quantiles(deltas):
    midpoint = deltas.shape[-1] // 2
    mid_value = deltas[..., [midpoint]]

    log_left_deltas = deltas[..., :midpoint]
    left_deltas = torch.exp(log_left_deltas)
    left_quantiles = mid_value - (
        torch.flip(
            torch.cumsum(torch.flip(left_deltas, dims=(-1,)), dim=-1), dims=(-1,)
        )
    )

    log_right_deltas = deltas[..., midpoint + 1 :]
    right_deltas = torch.exp(log_right_deltas)
    right_quantiles = mid_value + torch.cumsum(right_deltas, dim=-1)

    quantiles = torch.cat([left_quantiles, mid_value, right_quantiles], dim=-1)
    return quantiles
