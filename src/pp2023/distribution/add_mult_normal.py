"""Additive Multiplicative normal distribution. This may be better than what I was
doing before which was simply additive."""


class AddMultNormalParametricStrategy(DistributionalForecastStrategy):
    def __init__(self, variable_idx=None, loss_fn: str = "crps"):
        self.variable_idx = variable_idx
        self.loss_fn = loss_fn

    def from_tensor(self, features: torch.Tensor) -> DistributionalForecast:
        forecast_mu = features[..., 0]
        log_forecast_sigma = features[..., 1]

        forecast_sigma = torch.exp(log_forecast_sigma)
        processed_features = torch.stack([forecast_mu, forecast_sigma], dim=-1)

        return AddMultNormalParametric(processed_features, loss_fn=self.loss_fn)

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


class AddMultNormalParametric(DistributionalForecast):
    def __init__(self, params):
        loc = params[..., 0]

        loc_t2m, loc_log_si10 = loc[..., 0], loc[..., 1]

        # Restrain log_si10 to a minimum of zero so that expm1(log_si10) will also be
        # restrainted to zero when rescaling to the usual units.
        loc_log_si10 = torch.clamp(loc_log_si10, min=LOG_SI10_CLAMP)

        loc = torch.stack([loc_t2m, loc_log_si10], dim=-1)

        scale = params[..., 1]

        self.distribution = td.Normal(loc=loc, scale=scale)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)

    def crps(self, x: torch.Tensor) -> torch.Tensor:
        return crps_normal(self.distribution, x)
