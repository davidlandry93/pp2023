import math
import torch
import torch.nn as nn

N_DAYS_YEAR = 365


class Linear(nn.Module):
    def __init__(
        self,
        in_features=4,
        n_variables=2,
        n_parameters=2,
        n_stations=None,
        n_forecasts=None,
        n_steps=None,
        n_members=None,
        n_time_models=1,
        share_member=True,
        share_station=True,
        share_step=True,
        share_time=True,
        embedding_member=False,
        embedding_station=False,
        embedding_step=False,
        embedding_time=False,
        use_metadata_features=True,
        use_step_feature=True,
        use_spatial_features=True,
    ):
        super().__init__()
        self.time_model_span = math.ceil(N_DAYS_YEAR / n_time_models)

        self.share_member = share_member
        self.share_station = share_station
        self.share_step = share_step
        self.share_time = share_time

        self.use_metadata_features = use_metadata_features
        self.use_spatial_features = use_spatial_features
        self.use_step_feature = use_step_feature

        model_size_station = 1 if share_station else n_stations
        model_size_member = 1 if share_member else n_members
        model_size_step = 1 if share_step else n_steps
        model_size_time = 1 if share_time else n_time_models

        if use_metadata_features:
            in_features += 7

        if not use_step_feature:
            in_features -= 1

        if not use_spatial_features:
            in_features -= 4

        self.coefs = nn.Parameter(
            torch.empty(
                model_size_time,
                model_size_step,
                model_size_member,
                model_size_station,
                in_features,
                n_variables,
                n_parameters,
            )
        )

        torch.nn.init.normal_(self.coefs, 0, (1 / math.sqrt(in_features)))

        self.biases = nn.Parameter(
            torch.zeros(
                model_size_time,
                model_size_step,
                model_size_station,
                n_variables,
                n_parameters,
            )
        )

    def forward(self, batch):
        if self.share_time:
            time_idx = 0
        else:
            time_idx = (batch["day_of_year"] // self.time_model_span).long()

        if self.share_step:
            step_idx = 0
        else:
            step_idx = batch["step_idx"]

        if self.use_metadata_features:
            features_to_keep = list(range(7))

            if not self.use_spatial_features:
                features_to_keep.remove(3)
                features_to_keep.remove(4)
                features_to_keep.remove(5)
                features_to_keep.remove(6)

            if not self.use_step_feature:
                features_to_keep.remove(2)

            metadata_features = batch["metadata_features"][..., features_to_keep]

            features = torch.cat([batch["features"], metadata_features], dim=-1)
        else:
            features = batch["features"]

        features = features.unsqueeze(-1).unsqueeze(
            -1
        )  # Add placeholder dimensions for output variable (t2m, si10) and output parameter (mu, sigma for instance)

        coefs = self.coefs[time_idx, step_idx]
        biases = self.biases[time_idx, step_idx]

        prediction = biases + (coefs * features).mean(dim=1).sum(
            dim=-3
        )  # Mean over ensemble members, sum over features.

        return prediction
