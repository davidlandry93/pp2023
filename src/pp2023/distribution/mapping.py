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
    def make_distribution(self, forecast, parameters, std_prior=None):
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


class NaiveMapping(PP2023_DistributionMapping):
    def __init__(self, predict_wind=True):
        # For this mapping, predict wind has no effect.
        self.predict_wind = predict_wind

    def make_distribution(self, forecast, parameters, std_prior=None):
        forecast_mu = forecast.mean(dim=1)

        if std_prior is None:
            raise ValueError("NaiveMapping must have std prior.")

        # If no prior is available for that station-month, revert to yearly std prior.
        # This is a rare case in GDPS.
        std_prior = torch.nan_to_num(std_prior, nan=1.0)

        mu_coef = parameters[..., 0] + 1.0
        mu_base = parameters[..., 1]

        mu = mu_coef * forecast_mu + mu_base

        return NormalDistribution(mu, std_prior)


class EMOSMapping(PP2023_DistributionMapping):
    def __init__(self, predict_wind=True, use_std_prior=False):
        # For this mapping, predict wind has no effect.
        self.predict_wind = predict_wind
        self.use_std_prior = use_std_prior

    def make_distribution(self, forecast, parameters, std_prior=None):
        forecast_mu = forecast.mean(dim=1)

        if forecast.shape[1] == 1 and not self.use_std_prior:
            forecast_sigma = torch.full_like(forecast_mu, 0.0)
            sigma_coef = torch.tensor([0.0], device=forecast.device)
        elif forecast.shape[1] == 1 and self.use_std_prior:
            forecast_sigma = torch.nan_to_num(std_prior, nan=0.5)
            sigma_coef = parameters[..., 2] + 1.0
        else:
            forecast_sigma = forecast.std(dim=1)
            sigma_coef = parameters[..., 2] + 1.0

        log_forecast_sigma = torch.log(forecast_sigma + 1e-6)

        mu_coef = parameters[..., 0] + 1.0
        mu_base = parameters[..., 1]
        sigma_base = parameters[..., 3]

        mu = mu_coef * forecast_mu + mu_base
        log_sigma = sigma_coef * log_forecast_sigma + sigma_base

        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-6)

        return NormalDistribution(mu, sigma)


class NaiveQuantileMapping(PP2023_DistributionMapping):
    def __init__(self, n_quantiles, predict_wind=True, use_base=True, loss="crps"):
        self.n_quantiles = n_quantiles
        self.predict_wind = predict_wind
        self.use_base_quantiles = use_base
        self.loss = loss

    def make_distribution(self, forecast, parameters, std_prior=None):
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

        return QuantileDistribution(quantiles, loss_fn=self.loss)


class DeterministicMapping(PP2023_DistributionMapping):
    def __init__(self, predict_wind=True):
        self.predict_wind = predict_wind

    def make_distribution(self, forecast, parameters, std_prior=None):
        t2m = parameters[..., 0, :]
        vars_params = [t2m]

        if self.predict_wind:
            log_si10 = parameters[..., 1, :]
            si10 = torch.exp(log_si10)
            vars_params.append(si10)

        prediction = torch.stack(vars_params, dim=-2)
        return DeterministicDistribution(prediction)


class BernsteinQuantileFunctionMapping(PP2023_DistributionMapping):
    N_SAMPLES = 98  # 98 quantiles = 99 bins --> dividable by 3.

    def __init__(
        self,
        n_parameters: int,
        use_base=True,
        predict_wind=True,
        loss="crps",
        use_std_prior=False,
    ):
        self.degree = n_parameters - 1
        self.use_base = use_base
        self.predict_wind = predict_wind
        self.loss = loss

        polynomial_maker = bernstein_polynomial_torch(self.degree)
        sample_points = torch.linspace(
            1.0 / self.N_SAMPLES, (self.N_SAMPLES - 1) / self.N_SAMPLES, self.N_SAMPLES
        )
        self.poly_table = polynomial_maker(sample_points)

    def make_distribution(
        self, forecast: torch.Tensor, parameters: torch.Tensor, std_prior=None
    ) -> QuantileDistribution:
        coefficients = parameters

        coefficients, _ = torch.sort(coefficients, dim=-1)

        if self.use_base:
            coefficients += forecast.mean(dim=1).unsqueeze(-1)

        quantile_values = (
            self.poly_table.to(parameters.device) * coefficients.unsqueeze(-1)
        ).sum(dim=-2)

        return QuantileDistribution(quantile_values, loss_fn=self.loss)
