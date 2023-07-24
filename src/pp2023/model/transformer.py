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
    def __init__(self, n_heads, in_features, out_features):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(in_features, out_features // n_heads)
                for _ in range(n_heads)
            ]
        )

        self.linear = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm([out_features])
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        xs = []
        for head in self.heads:
            out = head(x)
            xs.append(out)

        heads_cat = torch.cat(xs, dim=-1)
        out = self.linear(heads_cat)

        return self.relu(self.layer_norm(out))


class AttentionBlock(nn.Module):
    def __init__(self, n_heads, in_features, out_features, dropout=0.0):
        super().__init__()

        self.attention_layer = AttentionLayer(n_heads, in_features, out_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(0.1),
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


class TransformerModel(nn.Module):
    """Transformer-like model that performs post-processing on the SMC dataset."""

    def __init__(
        self,
        in_features: int,
        n_parameters: int,
        n_heads=4,
        embedding_size=128,
        n_blocks=4,
        dropout=0.0,
        add_meta_tokens=False,
        n_stations=None,
        n_steps=None,
        n_time_models=12,
    ):
        super().__init__()

        self.add_meta_tokens = add_meta_tokens

        # Here we create one more embedding for the "padding" station.
        self.station_embedding = nn.Parameter(
            torch.rand(n_stations + 1, embedding_size)
        )

        self.time_model_span = math.ceil(N_DAYS_YEAR / n_time_models)

        self.embedding = nn.Linear(in_features, embedding_size, bias=True)

        self.day_of_year_embedding = nn.Parameter(
            torch.rand(n_time_models + 1, embedding_size)
        )
        self.step_embedding = nn.Parameter(torch.rand(n_steps, embedding_size))

        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(n_heads, embedding_size, embedding_size, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )

        self.regression = nn.Linear(embedding_size, n_parameters, bias=True)

    def forward(self, batch):
        station_id = batch["station_id"]
        features = batch["features"]
        step_id = batch["step_id"]
        forecast_time_id = batch["forecast_time_id"]
        day_of_year = batch["day_of_year"]

        station_embeddings = self.station_embedding[station_id]

        embedded_features = self.embedding(features)
        attention_in_features = embedded_features + station_embeddings

        # Add tokens at the end of the sequence that describe the context (forecast id,
        # step id, etc).
        if self.add_meta_tokens:
            day_of_year_token = self.day_of_year_embedding[
                day_of_year // self.time_model_span
            ]
            step_token = self.step_embedding[step_id[:, 0]]

            attention_in_features = torch.cat(
                [
                    attention_in_features,
                    day_of_year_token.unsqueeze(1),
                    step_token.unsqueeze(1),
                    forecast_time_id.unsqueeze(1),
                ],
                dim=1,
            )

        annotated_features = self.attention_layers(attention_in_features)

        # Remove the metadata tokens if necessary.
        if self.add_meta_tokens:
            n_meta_tokens = 4 if self.use_model_embedding else 3
            annotated_features = annotated_features[:, :-n_meta_tokens]

        correction = self.regression(annotated_features)

        return correction.squeeze()
