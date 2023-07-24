import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features=4,
        n_variables=2,
        n_parameters=2,
        n_stations=0,
        n_steps=3,
        n_members=10,
        embedding_size=32,
        n_blocks=4,
    ):
        super().__init__()

        self.n_variables = n_variables
        self.n_parameters = n_parameters

        # Add to in_features because we concatenate with time features.
        embedding = nn.Sequential(nn.Linear(in_features + 3, embedding_size), nn.SiLU())

        self.station_embedding = nn.Parameter(torch.empty(n_stations, embedding_size))
        torch.nn.init.normal_(self.station_embedding, 0, (1 / n_stations))

        self.member_embedding = nn.Parameter(torch.empty(n_members, 1, embedding_size))
        torch.nn.init.normal_(self.member_embedding, 0, (1 / n_members))

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Linear(embedding_size, embedding_size))
            blocks.append(nn.SiLU())
        hidden = nn.Sequential(*blocks)
        head = nn.Linear(embedding_size, n_variables * n_parameters)

        self.projection = embedding
        self.mlp = nn.Sequential(hidden, head)

    def forward(self, batch):
        features = torch.cat([batch["features"], batch["time_features"]], dim=-1)

        projected_features = self.projection(features)

        pooled_features = torch.cat(
            [(self.member_embedding + projected_features).mean(dim=1)]
        )  # Pool over member dimension.

        station_embedding = self.station_embedding

        correction = self.mlp(pooled_features + station_embedding)

        return correction.reshape(
            *correction.shape[:-1], self.n_variables, self.n_parameters
        )
