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
        n_variables=2,
        n_parameters=2,
        n_stations=0,
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
                n_variables,
                n_parameters,
            )
        )

        torch.nn.init.normal_(self.coefs, 0, (1 / math.sqrt(in_features)))

        self.biases = nn.Parameter(
            torch.zeros(
                n_time_models,
                n_step_models,
                n_stations,
                n_variables,
                n_parameters,
            )
        )

    def forward(self, batch):
        time_group_id = (batch["day_of_year"] // self.time_model_span).long()
        step_group_id = (batch["step_idx"] // self.step_model_span).long()

        features = batch["features"][:, 0]  # remove members for now
        features = features.unsqueeze(-1).unsqueeze(-1)
        coefs = self.coefs[time_group_id, step_group_id]
        biases = self.biases[time_group_id, step_group_id]

        prediction = biases + (coefs * features).sum(dim=-3)

        return prediction
