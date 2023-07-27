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
        use_member_embedding=True,
        use_step_embedding=True,
        use_station_embedding=True,
        embedding_activation=True,
    ):
        super().__init__()

        self.n_variables = n_variables
        self.n_parameters = n_parameters

        # Add to in_features because we concatenate with time features.
        embedding_blocks = [nn.Linear(in_features + 3, embedding_size)]
        if embedding_activation:
            embedding_blocks.append(nn.SiLU())
        embedding = nn.Sequential(*embedding_blocks)

        if use_station_embedding:
            self.station_embedding = nn.Parameter(
                torch.empty(n_stations, embedding_size)
            )
            torch.nn.init.normal_(self.station_embedding, 0, (1 / n_stations))

        if use_member_embedding:
            self.member_embedding = nn.Parameter(
                torch.empty(n_members, 1, embedding_size)
            )
            torch.nn.init.normal_(self.member_embedding, 0, (1 / n_members))

        if use_step_embedding:
            self.step_embedding = nn.Parameter(torch.empty(n_steps, 1, embedding_size))
            torch.nn.init.normal_(self.step_embedding, 0, (1 / n_steps))

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

        if hasattr(self, "member_embedding"):
            projected_features += self.member_embedding

        pooled_features = projected_features.mean(dim=1)

        if hasattr(self, "step_embedding"):
            pooled_features += self.step_embedding[batch["step_idx"]]

        if hasattr(self, "station_embedding"):
            pooled_features += self.station_embedding

        correction = self.mlp(pooled_features)

        return correction.reshape(
            *correction.shape[:-1], self.n_variables, self.n_parameters
        )
