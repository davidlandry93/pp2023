import logging
import math
import torch
import torch.nn as nn
import pandas as pd


_logger = logging.getLogger(__name__)


class TransposeBatchNorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class MLP(nn.Module):
    def __init__(
        self,
        in_features=4,
        n_variables=2,
        n_parameters=2,
        n_stations=0,
        n_steps=3,
        n_forecasts=1,
        n_members=10,
        embedding_size=32,
        n_blocks=4,
        use_forecast_embedding=False,
        use_step_embedding=True,
        use_station_embedding=True,
        embedding_activation=True,
        use_metadata_features=True,
        use_step_feature=True,
        use_spatial_features=True,
        use_batch_norm=False,
        use_forecast_time_feature=False,
        use_model_version_feature=False,
        use_std_prior=False,
    ):
        super().__init__()

        self.n_embeddings = 0
        if use_station_embedding:
            self.n_embeddings += 1
        if use_step_embedding:
            self.n_embeddings += 1
        if use_forecast_embedding:
            self.n_embeddings += 1

        self.n_variables = n_variables
        self.n_parameters = n_parameters

        self.use_metadata_features = use_metadata_features
        self.use_step_feature = use_step_feature
        self.use_spatial_features = use_spatial_features
        self.use_forecast_time_feature = use_forecast_time_feature
        self.use_model_version_feature = use_model_version_feature
        self.embedding_activation = embedding_activation
        self.use_std_prior = use_std_prior

        # Add to in_features because we concatenate with time features.

        if self.use_metadata_features:
            in_features += 9

            if not self.use_step_feature:
                in_features -= 1

            if not self.use_spatial_features:
                in_features -= 4

            if not self.use_forecast_time_feature:
                in_features -= 1

            if not self.use_std_prior:
                in_features -= 1

        if self.use_model_version_feature:
            in_features += 1

        feature_embedding_size = self.feature_embedding_size(
            embedding_size, self.n_embeddings
        )

        embedding_blocks = [nn.Linear(in_features, feature_embedding_size)]

        embedding = nn.Sequential(*embedding_blocks)

        self.initialize_embeddings(
            embedding_size,
            use_station_embedding,
            use_step_embedding,
            use_forecast_embedding,
            n_stations,
            n_forecasts,
            n_steps,
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Linear(embedding_size, embedding_size))

            if use_batch_norm:
                blocks.append(TransposeBatchNorm(embedding_size))

            blocks.append(nn.SiLU())
        hidden = nn.Sequential(*blocks)
        head = nn.Linear(embedding_size, n_variables * n_parameters)

        self.projection = embedding
        self.mlp = nn.Sequential(hidden, head)

        if self.embedding_activation:
            self.embedding_activation_layer = nn.SiLU()
            self.embedding_batch_norm = TransposeBatchNorm(embedding_size)

    def feature_embedding_size(self, embedding_size, n_embeddings):
        return embedding_size

    def initialize_embeddings(
        self,
        embedding_size,
        use_station_embedding,
        use_step_embedding,
        use_forecast_embedding,
        n_stations,
        n_forecasts,
        n_steps,
    ):
        if use_station_embedding:
            self.station_embedding = nn.Parameter(
                torch.empty(n_stations, embedding_size)
            )
            torch.nn.init.normal_(self.station_embedding, 0, (1 / n_stations))

        if use_step_embedding:
            self.step_embedding = nn.Parameter(torch.empty(n_steps, 1, embedding_size))
            torch.nn.init.normal_(self.step_embedding, 0, (1 / n_steps))

        if use_forecast_embedding:
            self.forecast_embedding = nn.Parameter(
                torch.empty(n_forecasts, 1, embedding_size)
            )
            torch.nn.init.normal_(self.forecast_embedding, 0, (1 / n_forecasts))

    def inject_embeddings(self, batch, pooled_features):
        if hasattr(self, "step_embedding"):
            pooled_features += self.step_embedding[batch["step_idx"]]

        if hasattr(self, "forecast_embedding"):
            pooled_features += self.forecast_embedding[batch["forecast_idx"].squeeze()]

        if hasattr(self, "station_embedding"):
            pooled_features += self.station_embedding

        return pooled_features

    def forward(self, batch):
        batch_size, n_members, n_stations, _ = batch["features"].shape

        if self.use_metadata_features:
            features_to_keep = list(range(9))

            if not self.use_step_feature:
                features_to_keep.remove(2)

            if not self.use_spatial_features:
                features_to_keep.remove(3)
                features_to_keep.remove(4)
                features_to_keep.remove(5)
                features_to_keep.remove(6)

            if not self.use_forecast_time_feature:
                features_to_keep.remove(7)

            if not self.use_std_prior:
                features_to_keep.remove(8)

            metadata_features = batch["metadata_features"][..., features_to_keep]

            features = torch.cat([batch["features"], metadata_features], dim=-1)
        else:
            features = batch["features"]

        if self.use_model_version_feature:
            model_version_feature = (
                batch["forecast_time"] >= pd.to_datetime("2019-07-03T12").value
            )  # GDPS 7 start date.
            model_version_feature = torch.broadcast_to(
                model_version_feature.reshape(batch_size, 1, 1, 1),
                (*features.shape[:-1], 1),
            )
            features = torch.cat([features, model_version_feature], dim=-1)

        projected_features = self.projection(features)
        pooled_features = projected_features.mean(dim=1)

        features_with_embeddings = self.inject_embeddings(batch, pooled_features)
        if self.embedding_activation:
            features_with_embeddings = self.embedding_activation_layer(
                self.embedding_batch_norm(features_with_embeddings)
            )
        correction = self.mlp(features_with_embeddings)

        return correction.reshape(
            *correction.shape[:-1], self.n_variables, self.n_parameters
        )


class ConcatMLP(MLP):
    """Same as the MLP but concatenates the embedding with the features instead of
    adding them."""

    def feature_embedding_size(self, embedding_size, n_embeddings):
        return embedding_size // (n_embeddings + 1) + embedding_size % (
            n_embeddings + 1
        )

    def initialize_embeddings(
        self,
        embedding_size,
        use_station_embedding,
        use_step_embedding,
        use_forecast_embedding,
        n_stations,
        n_forecasts,
        n_steps,
    ):
        self.n_embeddings = 0
        if use_station_embedding:
            self.n_embeddings += 1
        if use_step_embedding:
            self.n_embeddings += 1
        if use_forecast_embedding:
            self.n_embeddings += 1

        if use_station_embedding:
            station_embedding_size = embedding_size // (self.n_embeddings + 1)
            self.station_embedding = nn.Parameter(
                torch.empty(n_stations, station_embedding_size)
            )
            torch.nn.init.normal_(self.station_embedding, 0, (1 / n_stations))

        if use_step_embedding:
            step_embedding_size = embedding_size // (self.n_embeddings + 1)

            self.step_embedding = nn.Parameter(
                torch.empty(n_steps, 1, step_embedding_size)
            )
            torch.nn.init.normal_(self.step_embedding, 0, (1 / n_steps))

        if use_forecast_embedding:
            forecast_embedding_size = embedding_size // (self.n_embeddings + 1)
            self.forecast_embedding = nn.Parameter(
                torch.empty(n_forecasts, 1, forecast_embedding_size)
            )
            torch.nn.init.normal_(self.forecast_embedding, 0, (1 / n_forecasts))

    def inject_embeddings(self, batch, pooled_features):
        to_concat = [pooled_features]

        if hasattr(self, "step_embedding"):
            steps = self.step_embedding[batch["step_idx"]]
            steps = torch.broadcast_to(steps, (*pooled_features.shape[:-1], -1))
            to_concat.append(steps)

        if hasattr(self, "forecast_embedding"):
            forecasts = self.forecast_embedding[batch["forecast_idx"].squeeze()]
            forecasts = torch.broadcast_to(forecasts, (*pooled_features.shape[:-1], -1))
            to_concat.append(forecasts)

        if hasattr(self, "station_embedding"):
            stations = torch.broadcast_to(
                self.station_embedding, (*pooled_features.shape[:-1], -1)
            )
            to_concat.append(stations)

        return torch.cat(to_concat, dim=-1)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        in_features=4,
        n_variables=2,
        n_parameters=2,
        n_stations=0,
        n_steps=3,
        n_forecasts=1,
        n_members=1,
        embedding_size=32,
        n_hidden_layers=4,
        use_forecast_time_embedding=False,
        use_step_embedding=True,
        use_station_embedding=True,
        use_metadata_features=True,
        use_step_feature=True,
        use_spatial_features=True,
        use_forecast_time_feature=False,
        use_model_version_feature=False,
    ):
        super().__init__()

        self.use_station_embedding = use_station_embedding
        self.use_forecast_time_embedding = use_forecast_time_embedding
        self.use_step_embedding = use_step_embedding
        self.use_step_feature = use_step_feature
        self.use_metadata_features = use_metadata_features
        self.use_spatial_features = use_spatial_features
        self.use_forecast_time_feature = use_forecast_time_feature
        self.use_model_version_feature = use_model_version_feature

        self.n_forecasts = n_forecasts
        self.n_steps = n_steps
        self.n_stations = n_stations

        input_len = in_features * n_members

        if self.use_metadata_features:
            input_len += 9

            if not self.use_step_feature:
                input_len -= 1

            if not self.use_spatial_features:
                input_len -= 4

            if not self.use_forecast_time_feature:
                input_len -= 1

            if not self.use_model_version_feature:
                input_len -= 1

        if use_station_embedding:
            input_len += n_stations

        if use_forecast_time_embedding:
            input_len += n_forecasts

        if use_step_embedding:
            input_len += n_steps

        _logger.info(f"Input vector length: {input_len}")
        layers = [
            nn.Linear(input_len, embedding_size),
            TransposeBatchNorm(embedding_size),
            nn.SiLU(),
        ]

        for _ in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(embedding_size, embedding_size),
                    TransposeBatchNorm(embedding_size),
                    nn.SiLU(),
                ]
            )

        layers.append(nn.Linear(embedding_size, n_variables * n_parameters))

        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        batch_size, n_members, _, _ = batch["features"].shape

        if self.use_metadata_features:
            features_to_keep = list(range(8))

            if not self.use_step_feature:
                features_to_keep.remove(2)

            if not self.use_spatial_features:
                features_to_keep.remove(3)
                features_to_keep.remove(4)
                features_to_keep.remove(5)
                features_to_keep.remove(6)

            if not self.use_forecast_time_feature:
                features_to_keep.remove(7)

            metadata_features = batch["metadata_features"][..., features_to_keep]

            features = torch.cat([batch["features"], metadata_features], dim=-1)
        else:
            features = batch["features"]

        if self.use_metadata_features and self.use_model_version_feature:
            model_version_feature = (
                batch["forecast_time"] >= pd.to_datetime("2019-07-03T12").value
            )  # GDPS 7 start date.
            model_version_feature = torch.broadcast_to(
                model_version_feature.reshape(batch_size, 1, 1, 1),
                (*features.shape[:-1], 1),
            )
            features = torch.cat([features, model_version_feature], dim=-1)

        if self.use_station_embedding:
            one_hot_stations = torch.broadcast_to(
                torch.nn.functional.one_hot(
                    torch.arange(self.n_stations, device=features.device),
                    num_classes=self.n_stations,
                ),
                (batch_size, n_members, self.n_stations, self.n_stations),
            )
            features = torch.cat([features, one_hot_stations], dim=-1)

        if self.use_forecast_time_embedding:
            one_hot_forecast_time = torch.broadcast_to(
                torch.nn.functional.one_hot(
                    batch["forecast_idx"], num_classes=self.n_forecasts
                ).unsqueeze(-2),
                (batch_size, n_members, self.n_stations, self.n_forecasts),
            )
            features = torch.cat([features, one_hot_forecast_time], dim=-1)

        if self.use_step_embedding:
            one_hot_steps = torch.broadcast_to(
                torch.nn.functional.one_hot(
                    batch["step_idx"], num_classes=self.n_steps
                ).reshape(batch_size, 1, 1, self.n_steps),
                (batch_size, n_members, self.n_stations, self.n_steps),
            )
            features = torch.cat([features, one_hot_steps], dim=-1)

        # Remove the members dimension.
        features = features.reshape(features.shape[0], features.shape[2], -1)

        return self.model(features)
