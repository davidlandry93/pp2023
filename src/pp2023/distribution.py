import math
import torch
import torch.distributions as td

SQRT_PI = math.sqrt(math.pi)

LOG_SI10_CLAMP = -0.2


def crps_empirical(Q: torch.tensor, y: torch.tensor, sorted=False):
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
        values, _ = torch.sort(Q, dim=-1)
        Q = values

    N = Q.shape[-1]

    right_width = torch.cat(
        [
            Q[..., 0:1] - y,
            Q[..., 1:] - torch.maximum(y, Q[..., :-1]),
            torch.zeros(
                *Q.shape[:-1], 1, device=Q.device
            ),  # Right integral is never used if the obs is to the right of the distribution, so set width to zero.
        ],
        dim=-1,
    )

    left_width = torch.cat(
        [
            torch.zeros(
                *Q.shape[:-1], 1, device=Q.device
            ),  # Left integral is never used if the obs is to the left of the distribution.
            torch.minimum(y, Q[..., 1:]) - Q[..., :-1],
            y - Q[..., [-1]],
        ],
        dim=-1,
    )

    weights = torch.arange(0, N + 1, device=Q.device) / N
    right_weights = (1 - weights) ** 2
    left_weights = weights**2

    left = torch.clamp(left_width, min=0) * left_weights
    right = torch.clamp(right_width, min=0) * right_weights

    return (left + right).sum(dim=-1)


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

        loc_t2m, loc_log_si10 = loc[..., 0], loc[..., 1]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        loc_log_si10 = torch.clamp(loc_log_si10, min=LOG_SI10_CLAMP)

        loc = torch.stack([loc_t2m, loc_log_si10], dim=-1)

        scale = params[..., 1]

        scale = torch.clamp(scale, min=1e-6)

        self.distribution = td.Normal(loc=loc, scale=scale)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.crps(x)

    def crps(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)


class DeterministicStrategy(DistributionalForecastStrategy):
    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["forecast_parameters"][..., [0]]

    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        return DeterministicForecast(features)


class DeterministicForecast(DistributionalForecast):
    def __init__(self, params):
        t2m, log_si10 = params[..., 0, :], params[..., 1, :]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        log_si10 = torch.clamp(log_si10, min=LOG_SI10_CLAMP)

        params = torch.stack([t2m, log_si10], dim=-2)
        self.preds = params.squeeze()

    def loss(self, x: torch.Tensor):
        return torch.square(x - self.preds)

    def crps(self, x: torch.Tensor):
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

    def quantile_loss(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        n_quantiles = self.parameters.shape[-1]

        weights = torch.linspace(0.0, 1.0, n_quantiles)

        left_loss = weights * (x - self.parameters)
        right_loss = (1.0 - weights) * (self.parameters - x)

        losses = torch.abs(torch.stack([left_loss, right_loss], dim=-1))
        loss, indices = losses.max(dim=-1)

        return loss.mean(dim=-1)

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


class QuantileRegressionStrategy(DistributionalForecastStrategy):
    def __init__(self, n_quantiles: int, regularization: float = 1e-6):
        self.n_quantiles = n_quantiles
        self.regularization = regularization

    def nwp_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        members_values = torch.gather(batch["forecast"], 1, batch["forecast_sort_idx"])

        n_members = batch["forecast"].shape[1]
        n_quantile = self.n_quantiles
        member_idx = torch.round(torch.linspace(0, n_members - 1, n_quantile)).int()

        # Use the nearest member quantile as a base for the forcetast.
        # This way we decouple the number of members and the number of quantiles.
        return members_values[:, member_idx].transpose(1, 2).transpose(2, 3)

    def from_tensor(self, x: torch.Tensor) -> "QuantileRegression":
        return QuantileRegression(x, regularization=self.regularization)
