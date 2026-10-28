"""NLinear-style encoder adapter for single-node federated forecasting.

Adapted from NLinear (Zeng et al., AAAI 2023): subtract the last value of the
input window (a simple non-stationarity normalisation), apply a single linear
map, then add the offset back. Here the linear map projects the window into a
``hid_dim`` representation for the existing PrototypeMatching + Predictor head
instead of regressing directly to the horizon.

Pure linear + a constant subtraction, so it is fully ``torch.func.vmap`` safe.
"""

import torch.nn as nn


class NLinearEncoder(nn.Module):
    """NLinear backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: unused (kept for the common encoder signature).
        dropout: dropout on the projected representation.
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        self.linear = nn.Linear(seq_len, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temporal_id=None):
        """Encode a window.

        Args:
            x: (B, T, input_dim) input window.
            temporal_id: optional (B, 2) [time_of_day, day_of_week].

        Returns:
            z: (B, 1, hid_dim) window representation.
        """
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        series = x[:, :, 0]                 # (B, T)
        last = series[:, -1:]              # (B, 1) — NLinear last-value normalisation
        repr_ = self.linear(series - last)  # (B, hid_dim)
        repr_ = self.dropout(repr_)
        return repr_.unsqueeze(1)           # (B, 1, hid_dim)
