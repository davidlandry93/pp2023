import math
import torch
import torch.nn as nn


N_DAYS_YEAR = 365
# I remap day 366 to day 365 for leap years. I thought this was better than
# to have a unique day one year out of four. So no need to use 366 day per
# year here.


class EMOS(nn.Module):
    def __init__(
        self,
        in_features=4,
        out_features=2,
        out_params=2,
        n_stations=None,
        n_steps=3,
        n_time_models=1,
        n_step_models=1,
    ):
        super().__init__()
        self.time_model_span = math.ceil(N_DAYS_YEAR / n_time_models)
        self.step_model_span = math.ceil(n_steps / n_step_models)

        self.coefs = nn.Parameter(
            torch.empty(
                n_time_models,
                n_step_models,
                n_stations,
                in_features,
                out_features,
                out_params,
            )
        )

        torch.nn.init.normal_(self.coefs, 0, (1 / math.sqrt(in_features)))

        self.biases = nn.Parameter(
            torch.zeros(
                n_time_models, n_step_models, n_stations, out_features, out_params
            )
        )

    def forward(self, batch):
        time_group_id = (batch["day_of_year"] // self.time_model_span).long()
        step_group_id = (batch["step_idx"] // self.step_model_span).long()
        epsilon = 1e-6

        forecast_params = batch["forecast_parameters"]
        forecast_mu = forecast_params[..., 0]
        forecast_sigma = forecast_params[..., 1]
        log_forecast_sigma = torch.log(forecast_sigma + epsilon)

        processed_forecast_params = torch.stack(
            [forecast_mu, log_forecast_sigma], dim=-1
        )

        features = batch["features"][:, 0]  # remove members for now
        features = features.unsqueeze(-1).unsqueeze(-1)
        coefs = self.coefs[time_group_id, step_group_id]
        biases = self.biases[time_group_id, step_group_id]

        prediction = processed_forecast_params + biases + (coefs * features).sum(dim=-3)

        mu_parameter = prediction[..., 0]
        log_sigma_parameter = prediction[..., 1]

        prediction = torch.stack(
            [mu_parameter, torch.exp(log_sigma_parameter) - epsilon], dim=-1
        )

        return prediction
