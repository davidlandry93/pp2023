from typing import Any

import math
import torch.distributions

from .base import quantile_loss

SQRT_PI = math.sqrt(math.pi)


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

    if len(y.shape) == len(Q.shape) - 1:
        y = y.unsqueeze(-1)

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


class PP2023_Distribution:
    def loss(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def crps(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError()


class NormalDistribution(PP2023_Distribution):
    def __init__(self, loc, scale):
        self.distribution = torch.distributions.Normal(loc=loc, scale=scale)

    def loss(self, observation):
        return self.crps(observation)

    def crps(self, observation):
        return crps_normal(self.distribution, observation)

    def to_dict(self):
        return {
            "distribution_type": "normal",
            "parameters": torch.stack(
                [self.distribution.loc, self.distribution.scale], dim=-1
            ),
        }


class QuantileDistribution(PP2023_Distribution):
    def __init__(self, quantiles, loss_fn="crps", sorted=False):
        quantiles, _ = torch.sort(quantiles)

        self.quantiles = quantiles
        self.loss_fn = loss_fn
        self.sorted = sorted

    def loss(self, observation):
        match self.loss_fn:
            case "crps":
                return crps_empirical(self.quantiles, observation)
            case "quantile_loss":
                return quantile_loss(self.quantiles, observation)
            case _:
                raise KeyError("Unknown loss function.")

    def crps(self, observation):
        return crps_empirical(self.quantiles, observation, sorted=self.sorted)

    def to_dict(self):
        return {"distribution_type": "quantile", "parameters": self.quantiles}


class DeterministicDistribution(PP2023_Distribution):
    def __init__(self, predictions: torch.Tensor):
        self.preds = predictions.squeeze(
            -1
        )  # Remove params dimension since we have only one.

    def loss(self, observation):
        return torch.square(observation - self.preds)

    def crps(self, observation):
        return torch.abs(observation - self.preds)

    def to_dict(self):
        return {
            "distribution_type": "deterministic",
            "parameters": self.preds.unsqueeze(dim=-1),
        }
