import math
import torch


from .distribution import (
    NormalDistribution,
    QuantileDistribution,
    DeterministicDistribution,
)


class PP2023_DistributionMapping:
    """Map a bunch of weights from a neural net to a distribution on which we
    can compute a loss."""

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

        print(quantiles.std())

        return QuantileDistribution(quantiles)


class NaiveQuantileMapping(PP2023_DistributionMapping):
    def __init__(self, n_quantiles):
        self.n_quantiles = n_quantiles

    def make_distribution(self, forecast, parameters):
        n_members = forecast.shape[1]

        member_of_quantile = torch.round(
            torch.linspace(0, n_members - 1, self.n_quantiles)
        ).int()

        base_quantiles = forecast[:, member_of_quantile].transpose(1, 2).transpose(2, 3)
        quantiles = base_quantiles + parameters

        t2m, log_si10 = quantiles[..., 0, :], quantiles[..., 1, :]
        si10 = torch.exp(log_si10)

        quantiles = torch.cat([t2m, si10], dim=-2)

        return QuantileDistribution(quantiles)


class DeterministicMapping(PP2023_DistributionMapping):
    def make_distribution(self, forecast, parameters):
        t2m, log_si10 = parameters[..., 0, :], parameters[..., 1, :]

        si10 = torch.exp(log_si10)

        prediction = torch.stack([t2m, si10], dim=-2)

        return DeterministicDistribution(prediction)
