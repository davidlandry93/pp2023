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
        embedding_size=32,
        n_blocks=4,
    ):
        super().__init__()

        # Add to in_features because we concatenate with time features.
        embedding = nn.Sequential(nn.Linear(in_features + 3, embedding_size), nn.SiLU())

        self.station_embedding = nn.Parameter(torch.empty(n_stations, embedding_size))
        torch.nn.init.normal_(self.station_embedding, 0, (1 / n_stations))

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Linear(embedding_size, embedding_size))
            blocks.append(nn.SiLU())
        hidden = nn.Sequential(*blocks)
        head = nn.Linear(embedding_size, n_parameters)

        self.projection = embedding
        self.mlp = nn.Sequential(hidden, head)

    def forward(self, batch):
        features = torch.cat(batch["features"], batch["time_features"])
        projection = self.projection(features)
        station_embedding = self.station_embedding[batch["station_idx"]]

        return self.mlp(projection + station_embedding)
