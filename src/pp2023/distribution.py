import math
import torch
import torch.distributions as td

SQRT_PI = math.sqrt(math.pi)


def crps_normal(dist: torch.distributions.Normal, sample: torch.Tensor):
    """See http://cran.nexr.com/web/packages/scoringRules/vignettes/crpsformulas.html#Normal."""
    mean = dist.loc
    std = dist.scale
    centered_dist = torch.distributions.Normal(
        torch.zeros_like(mean), scale=torch.ones_like(std)
    )

    centered_sample = (sample - mean) / std

    cdf = centered_dist.cdf(centered_sample)
    pdf = torch.exp(centered_dist.log_prob(centered_sample))

    centered_crps = centered_sample * (2 * cdf - 1) + 2 * pdf - (1 / SQRT_PI)
    crps = std * centered_crps

    return crps


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
    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        forecast_mu = features[..., 0]
        log_forecast_sigma = features[..., 1]

        forecast_sigma = torch.exp(log_forecast_sigma)
        processed_features = torch.stack([forecast_mu, forecast_sigma], dim=-1)

        return NormalParametric(processed_features)

    def nwp_base(self, batch):
        forecast_params = batch["forecast_parameters"]

        forecast_mu = forecast_params[..., 0]
        forecast_sigma = forecast_params[..., 1]
        log_forecast_sigma = torch.log(forecast_sigma + 1e-6)

        processed_forecast_params = torch.stack(
            [forecast_mu, log_forecast_sigma], dim=-1
        )

        return processed_forecast_params


class NormalParametric(DistributionalForecast):
    def __init__(self, params):
        loc = params[..., 0]
        scale = params[..., 1]

        self.distribution = td.Normal(loc=loc, scale=scale)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.distribution.log_prob(x)

    def crps(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)


class DeterministicStrategy(DistributionalForecastStrategy):
    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["forecast_parameters"][..., [0]]

    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        return DeterministicForecast(features)


class DeterministicForecast(DistributionalForecast):
    def __init__(self, params):
        self.preds = params.squeeze()

    def loss(self, x: torch.Tensor):
        return torch.square(x - self.preds)

    def crps(self, x: torch.Tensor):
        return torch.abs(x - self.preds)
