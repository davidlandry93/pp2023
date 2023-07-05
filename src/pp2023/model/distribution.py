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

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DistributionalForecastStrategy:
    def from_features(self, features: torch.Tensor) -> DistributionalForecast:
        raise NotImplementedError

    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class NormalParametricStrategy(DistributionalForecastStrategy):
    def __init__(self, n_variables):
        self.n_variables = n_variables

    def from_features(self, features: torch.Tensor) -> DistributionalForecast:
        forecast_mu = features[..., 0]
        log_forecast_sigma = features[..., 1]

        forecast_sigma = torch.expm1(log_forecast_sigma)

        processed_features = torch.stack([forecast_mu, forecast_sigma], dim=-1)

        return NormalParametric(processed_features)

    def nwp_base(self, batch):
        forecast_params = batch["forecast_parameters"]

        forecast_mu = forecast_params[..., 0]
        forecast_sigma = forecast_params[..., 1]
        log_forecast_sigma = torch.log1p(forecast_sigma)

        processed_forecast_params = torch.stack(
            [forecast_mu, log_forecast_sigma], dim=-1
        )

        target = batch["target"]
        return processed_forecast_params.reshape(*target.shape, -1)


class NormalParametric(DistributionalForecast):
    def __init__(self, params):
        loc = params[..., 0]
        scale = params[..., 1]

        self.distribution = td.Normal(loc=loc, scale=scale)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def crps(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)
