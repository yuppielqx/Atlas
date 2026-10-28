"""
STID-style MLP encoder for single-node federated time series.

Adapted from STID (Shao et al., CIKM 2022), whose backbone first embeds the
whole input window, concatenates identity embeddings, then applies residual MLP
blocks. In this federated setting each client owns one sensor/node, so the node
embedding is a local learnable identity vector.
"""

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """STID residual MLP block implemented with 1x1 convolutions."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return hidden + x


class Encoder(nn.Module):
    """STID encoder adapted to one-node/client forecasting.

    Pipeline:
        X(B, T, D) -> window Conv2d embedding -> concat node/time/day identity
        embeddings -> ResMLP blocks -> Z(B, 1, hid_dim)

    Channels are assumed to be [target_value, time_of_day, day_of_week] (or the
    time/day indices are supplied separately via ``temporal_id``). The target
    history drives the time-series embedding; the last-step time/day indices
    select STID-style temporal identity embeddings, which are always used (they
    are STID's defining component and improve accuracy on traffic data).
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=3,
        dropout=0.1,
        multivariate=False,
        no_temporal_emb=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        self.no_temporal_emb = no_temporal_emb
        # When multivariate=True, all input channels are treated as series features
        # (weather variables, exogenous drivers, etc.). Time features come only
        # from temporal_id. The default (multivariate=False) preserves backward
        # compatibility with METR-LA-style [value, tod, dow] format.
        if multivariate:
            self.series_input_dim = input_dim
        else:
            self.series_input_dim = 1 if input_dim >= 3 else input_dim
        self.time_of_day_size = 288
        self.day_of_week_size = 7

        dims = self._split_embedding_dims(hid_dim, no_temporal_emb=no_temporal_emb)
        self.time_series_emb_dim = dims["series"]
        self.node_emb_dim = dims["node"]
        self.time_in_day_emb_dim = dims.get("time_in_day", 0)
        self.day_in_week_emb_dim = dims.get("day_in_week", 0)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=seq_len * self.series_input_dim,
            out_channels=self.time_series_emb_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.node_emb = nn.Parameter(torch.empty(1, self.node_emb_dim, 1, 1))
        if not no_temporal_emb:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.time_in_day_emb_dim)
            )
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.day_in_week_emb_dim)
            )
        else:
            self.time_in_day_emb = None
            self.day_in_week_emb = None

        self.encoder = nn.Sequential(*[
            MultiLayerPerceptron(hid_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.reset_parameters()

    @staticmethod
    def _split_embedding_dims(hid_dim, no_temporal_emb=False):
        """Split hid_dim across STID embedding parts (sum=hid_dim).

        When ``no_temporal_emb`` is True, time-of-day and day-of-week
        embeddings are skipped and their share is given to the series
        embedding (75% series, 25% node instead of 50/25/12.5/12.5).
        """
        if no_temporal_emb:
            names = ["series", "node"]
            ratios = [0.75, 0.25]
        else:
            names = ["series", "node", "time_in_day", "day_in_week"]
            ratios = [0.5, 0.25, 0.125, 0.125]
        if hid_dim < len(names):
            raise ValueError(f"hid_dim={hid_dim} is too small for {len(names)} STID embedding parts")

        dims = [max(1, int(hid_dim * r)) for r in ratios]
        while sum(dims) < hid_dim:
            dims[0] += 1
        while sum(dims) > hid_dim:
            idx = max(range(len(dims)), key=dims.__getitem__)
            dims[idx] -= 1
        return dict(zip(names, dims))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb)
        if self.time_in_day_emb is not None:
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.day_in_week_emb is not None:
            nn.init.xavier_uniform_(self.day_in_week_emb)

    def _temporal_identity_embeddings(self, x, temporal_id=None):
        if self.no_temporal_emb:
            return ()
        if temporal_id is None:
            if x.shape[-1] < 3:
                raise ValueError(
                    "STID requires temporal_id (B, 2) or x with time_of_day/"
                    "day_of_week channels [1:3]"
                )
            temporal_id = x[:, -1, 1:3]

        tod = torch.clamp(
            (temporal_id[:, 0] * self.time_of_day_size).long(),
            0,
            self.time_of_day_size - 1,
        )
        dow = torch.clamp(
            (temporal_id[:, 1] * self.day_of_week_size).long(),
            0,
            self.day_of_week_size - 1,
        )

        tod_emb = self.time_in_day_emb[tod].unsqueeze(-1).unsqueeze(-1)
        dow_emb = self.day_in_week_emb[dow].unsqueeze(-1).unsqueeze(-1)
        return tod_emb, dow_emb

    def forward(self, x, temporal_id=None):
        """Encode an input window.

        Args:
            x: (B, T, input_dim) input time series
            temporal_id: optional (B, 2) [time_of_day, day_of_week]

        Returns:
            z: (B, 1, hid_dim) STID window representation
        """
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        series = x[:, :, :self.series_input_dim]
        series = series.transpose(1, 2).reshape(B, T * self.series_input_dim, 1, 1)
        hidden_parts = [self.time_series_emb_layer(series)]
        hidden_parts.append(self.node_emb.expand(B, -1, -1, -1))
        hidden_parts.extend(self._temporal_identity_embeddings(x, temporal_id))

        hidden = torch.cat(hidden_parts, dim=1)
        hidden = self.encoder(hidden)  # (B, hid_dim, 1, 1)
        return hidden.squeeze(-1).transpose(1, 2)  # (B, 1, hid_dim)
