import math
import torch


from .distribution import (
    NormalDistribution,
    QuantileDistribution,
    DeterministicDistribution,
)

from ..bernstein import bernstein_polynomial_torch


class PP2023_DistributionMapping:
    """Map a bunch of weights from a neural net to a distribution on which we
    can compute a loss."""

    def __init__(self, predict_wind=True):
        raise NotImplementedError()

    def make_distribution(self, forecast, parameters):
        raise NotImplementedError()


class SimpleNormalMapping(PP2023_DistributionMapping):
    def make_distribution(self, forecast, parameters):
        forecast_mu = forecast.mean(dim=1)

        if forecast.shape[1] == 1:
            forecast_sigma = torch.full_like(forecast_mu, 1.0)
        else:
            forecast_sigma = forecast.std(dim=1)

        log_forecast_sigma = torch.log(forecast_sigma + 1e-6)

        mu_base = parameters[..., 0]
        sigma_base = parameters[..., 1]

        mu = forecast_mu + mu_base
        log_sigma = log_forecast_sigma + sigma_base

        sigma = torch.exp(log_sigma)

        return NormalDistribution(mu, sigma)


class EMOSMapping(PP2023_DistributionMapping):
    def __init__(self, predict_wind=True):
        # For this mapping, predict wind has no effect.
        self.predict_wind = predict_wind

    def make_distribution(self, forecast, parameters):
        forecast_mu = forecast.mean(dim=1)

        if forecast.shape[1] == 1:
            forecast_sigma = torch.full_like(forecast_mu, 1.0)
        else:
            forecast_sigma = forecast.std(dim=1)

        log_forecast_sigma = torch.log(forecast_sigma + 1e-6)

        mu_coef = parameters[..., 0] + 1.0
        mu_base = parameters[..., 1]
        sigma_coef = parameters[..., 2] + 1.0
        sigma_base = parameters[..., 3]

        mu = mu_coef * forecast_mu + mu_base
        log_sigma = sigma_coef * log_forecast_sigma + sigma_base

        sigma = torch.exp(log_sigma)

        return NormalDistribution(mu, sigma)


class ConstructiveQuantileMapping(PP2023_DistributionMapping):
    def __init__(self, n_quantiles):
        self.initial = math.log(n_quantiles)

    def make_distribution(
        self, forecast: torch.Tensor, parameters: torch.Tensor
    ) -> QuantileDistribution:
        forecast_mean = forecast.mean(dim=1).unsqueeze(-1)

        midpoint = parameters.shape[-1] // 2

        mid_value = parameters[..., [midpoint]] + forecast_mean

        log_left_deltas = parameters[..., :midpoint]
        left_deltas = torch.exp(log_left_deltas - self.initial)
        left_quantiles = mid_value - (
            torch.flip(
                torch.cumsum(torch.flip(left_deltas, dims=(-1,)), dim=-1), dims=(-1,)
            )
        )

        log_right_deltas = parameters[..., midpoint + 1 :]
        right_deltas = torch.exp(log_right_deltas)

        right_quantiles = mid_value + torch.cumsum(right_deltas, dim=-1)
        quantiles = torch.cat([left_quantiles, mid_value, right_quantiles], dim=-1)

        return QuantileDistribution(quantiles)


class NaiveQuantileMapping(PP2023_DistributionMapping):
    def __init__(self, n_quantiles, predict_wind=True, use_base=True):
        self.n_quantiles = n_quantiles
        self.predict_wind = predict_wind
        self.use_base_quantiles = use_base

    def make_distribution(self, forecast, parameters):
        if self.use_base_quantiles:
            n_members = forecast.shape[1]

            member_of_quantile = torch.round(
                torch.linspace(0, n_members - 1, self.n_quantiles)
            ).int()

            # Move the member dimension (which on forecast is dim 1 to the
            # parameter dimension (which is the last one).
            base_quantiles = forecast[:, member_of_quantile].transpose(1, 2)
            quantiles = base_quantiles + parameters
        else:
            quantiles = parameters

        t2m = quantiles[..., [0], :]
        vars_to_return = [t2m]

        if self.predict_wind:
            log_si10 = quantiles[..., [1], :]
            si10 = torch.exp(log_si10)
            vars_to_return.append(si10)

        quantiles = torch.cat(vars_to_return, dim=-2)

        return QuantileDistribution(quantiles)


class DeterministicMapping(PP2023_DistributionMapping):
    def __init__(self, predict_wind=True):
        self.predict_wind = predict_wind

    def make_distribution(self, forecast, parameters):
        t2m = parameters[..., 0, :]
        vars_params = [t2m]

        if self.predict_wind:
            log_si10 = parameters[..., 1, :]
            si10 = torch.exp(log_si10)
            vars_params.append(si10)

        prediction = torch.stack(vars_params, dim=-2)
        return DeterministicDistribution(prediction)


class BernsteinQuantileFunctionMapping(PP2023_DistributionMapping):
    N_SAMPLES = 100

    def __init__(self, n_parameters: int, use_base=True, predict_wind=True):
        self.degree = n_parameters - 1
        self.use_base = use_base
        self.predict_wind = predict_wind

        polynomial_maker = bernstein_polynomial_torch(self.degree)
        sample_points = torch.linspace(0.0, 1.0, self.N_SAMPLES)
        self.poly_table = polynomial_maker(sample_points)

    def make_distribution(
        self, forecast: torch.Tensor, parameters: torch.Tensor
    ) -> QuantileDistribution:
        coefficients = parameters

        if self.use_base:
            coefficients += forecast.mean(dim=1).unsqueeze(-1)

        quantile_values = (
            self.poly_table.to(parameters.device) * coefficients.unsqueeze(-1)
        ).sum(dim=-2)

        return QuantileDistribution(quantile_values)
