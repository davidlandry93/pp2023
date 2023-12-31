import math
import torch
import torch.nn as nn
import torch.nn.functional as F


N_DAYS_YEAR = 365  # We remap day 366 to day 365 on leap years.


class AttentionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight_key = nn.Linear(in_features, out_features, bias=False)
        self.weight_query = nn.Linear(in_features, out_features, bias=False)
        self.weight_value = nn.Linear(in_features, out_features, bias=False)

        self.n_out_features = torch.tensor(float(out_features))

    def forward(self, x):
        keys = self.weight_key(x)
        query = self.weight_query(x)
        value = self.weight_value(x)

        attention = torch.bmm(query, keys.transpose(-2, -1)) / torch.sqrt(
            self.n_out_features
        )
        attention = F.softmax(attention, dim=0)

        return torch.bmm(attention, value)


class AttentionLayer(nn.Module):
    def __init__(self, n_heads, in_features, out_features, activation_function=nn.SiLU):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(in_features, out_features // n_heads)
                for _ in range(n_heads)
            ]
        )

        self.linear = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm([out_features])
        self.activation = activation_function()

    def forward(self, x):
        xs = []
        for head in self.heads:
            out = head(x)
            xs.append(out)

        heads_cat = torch.cat(xs, dim=-1)
        out = self.linear(heads_cat)

        return self.activation(self.layer_norm(out))


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_heads,
        in_features,
        out_features,
        dropout=0.0,
        activation_function=nn.SiLU,
    ):
        super().__init__()

        self.attention_layer = AttentionLayer(
            n_heads, in_features, out_features, activation_function=activation_function
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(out_features, out_features),
            activation_function(),
            nn.Linear(out_features, out_features),
        )

        self.layer_norm = nn.LayerNorm([out_features])
        self.layer_norm_2 = nn.LayerNorm([out_features])

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(self.attention_layer(x))
        x = self.dropout_1(x)

        x = self.layer_norm_2(self.feed_forward(x) + x)
        x = self.dropout_2(x)
        return x


ACTIVATION_FNS = {"silu": nn.SiLU, "relu": nn.ReLU, "lrelu": nn.LeakyReLU}


class TransformerModel(nn.Module):
    """Transformer-like model that performs post-processing on the SMC dataset."""

    def __init__(
        self,
        in_features: int,
        n_variables: int,
        n_parameters: int,
        n_heads=8,
        embedding_size=256,
        n_blocks=4,
        dropout=0.0,
        add_meta_tokens=False,
        n_stations=None,
        n_steps=None,
        n_forecasts=1,
        n_members=10,
        n_time_models=12,
        select_member="mean",
        activation_function="silu",
        use_forecast_time_embedding=False,
    ):
        super().__init__()

        activation_function_class = ACTIVATION_FNS[activation_function]

        self.n_variables = n_variables
        self.n_parameters = n_parameters
        self.select_member = select_member

        out_features = n_variables * n_parameters

        self.add_meta_tokens = add_meta_tokens

        # Here we create one more embedding for the "padding" station.
        self.station_embedding = nn.Parameter(
            torch.rand(n_stations + 1, embedding_size)
        )

        self.forecast_time_embedding = nn.Parameter(
            torch.rand(n_forecasts, embedding_size)
        )

        self.use_forecast_time_embedding = use_forecast_time_embedding

        self.time_model_span = math.ceil(N_DAYS_YEAR / n_time_models)

        embedding_in_features = (
            in_features * n_members if select_member == "feature" else in_features
        )
        self.embedding = nn.Linear(embedding_in_features, embedding_size, bias=True)

        self.day_of_year_embedding = nn.Parameter(
            torch.rand(n_time_models + 1, embedding_size)
        )
        self.step_embedding = nn.Parameter(torch.rand(n_steps, embedding_size))

        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(
                    n_heads,
                    embedding_size,
                    embedding_size,
                    dropout=dropout,
                    activation_function=activation_function_class,
                )
                for _ in range(n_blocks)
            ]
        )

        self.regression = nn.Linear(embedding_size, out_features, bias=True)

    def forward(self, batch):
        batch_size, n_members, n_stations, n_features = batch["features"].shape

        if self.select_member == "feature":
            pre_embedding_features = batch["features"].reshape(
                batch_size, n_stations, n_features * n_members
            )
        else:
            pre_embedding_features = batch["features"]

        embedded_features = self.embedding(pre_embedding_features)

        if self.select_member == "first":
            selected_features = embedded_features[:, 0]
        elif self.select_member == "random":
            if self.train:
                selected_features = embedded_features[
                    torch.arange(0, batch_size),
                    torch.randint(0, n_members, (batch_size,)),
                ]
            else:
                selected_features = embedded_features[:, 0]

        elif self.select_member == "mean":
            selected_features = embedded_features.mean(dim=1)
        elif self.select_member == "all":
            selected_features = embedded_features.reshape(batch_size, -1, n_features)
        elif self.select_member == "feature":
            selected_features = embedded_features
        else:
            raise ValueError(f"Invalid selection strategy {self.select_member}")

        step_id = batch["step_idx"]
        day_of_year = batch["day_of_year"]

        repeat_member = n_members if self.select_member == "all" else 1
        station_id = (
            torch.arange(n_stations, device=batch["features"].device)
            .repeat((batch_size, repeat_member, 1))
            .reshape(batch_size, -1)
        )

        # If obs are missing, use the empty station embedding.
        has_no_obs = torch.isnan(batch["target"]).any(dim=-1)
        torch.where(
            has_no_obs,
            torch.tensor(-1, device=has_no_obs.device),
            station_id,
            out=station_id,
        )

        station_embeddings = self.station_embedding[station_id]

        attention_in_features = selected_features + station_embeddings

        # Add tokens at the end of the sequence that describe the context (forecast id,
        # step id, etc).
        if self.add_meta_tokens:
            n_meta_tokens = 2
            day_of_year_token = self.day_of_year_embedding[
                day_of_year // self.time_model_span
            ]

            step_token = self.step_embedding[step_id]

            attention_in_features = torch.cat(
                [
                    attention_in_features,
                    day_of_year_token.unsqueeze(1),
                    step_token.unsqueeze(1),
                ],
                dim=1,
            )

            if self.use_forecast_time_embedding:
                n_meta_tokens += 1
                forecast_time_token = self.forecast_time_embedding[
                    batch["forecast_idx"]
                ]

                attention_in_features = torch.cat(
                    [attention_in_features, forecast_time_token], dim=1
                )

        annotated_features = self.attention_layers(attention_in_features)

        # Remove the metadata tokens if necessary.
        if self.add_meta_tokens:
            annotated_features = annotated_features[:, :-n_meta_tokens]

        if self.select_member == "all":
            annotated_features_by_station = annotated_features.reshape(
                batch_size, n_members, n_stations, -1
            )
            aggregated_features = annotated_features_by_station.mean(dim=1)
        else:
            aggregated_features = annotated_features

        correction = self.regression(aggregated_features)

        return correction.reshape(
            *correction.shape[:-1], self.n_variables, self.n_parameters
        )
