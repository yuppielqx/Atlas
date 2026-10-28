"""DLinear-style encoder adapter for single-node federated forecasting.

Adapted from DLinear (Zeng et al., AAAI 2023): decompose the input window into a
moving-average trend and the seasonal residual, then apply a separate linear map
to each. Here, instead of regressing directly to the horizon, the two linear maps
project the window into a ``hid_dim`` window representation, so the existing
PrototypeMatching + Predictor head can run unchanged.

The moving average is implemented as a fixed matmul against a precomputed
averaging matrix (replicate-padded), which is fully ``torch.func.vmap`` safe.
"""

import torch
import torch.nn as nn


def _moving_average_matrix(seq_len, kernel_size):
    """(seq_len, seq_len) row-stochastic matrix for replicate-padded moving avg."""
    k = kernel_size
    pad = (k - 1) // 2
    A = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for d in range(-pad, pad + 1):
            j = min(max(i + d, 0), seq_len - 1)
            A[i, j] += 1.0 / k
    return A


class DLinearEncoder(nn.Module):
    """DLinear backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: unused (kept for the common encoder signature).
        dropout: dropout on the projected representation.
        kernel_size: moving-average window for trend extraction (forced odd and
            clamped to <= seq_len).
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=3,
        dropout=0.1,
        kernel_size=5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        k = min(int(kernel_size), seq_len)
        if k % 2 == 0:
            k = max(1, k - 1)
        self.kernel_size = k
        # Non-persistent: a deterministic constant rebuilt in __init__, so it
        # stays out of state_dict() (the vectorized sync_to_clients loads params
        # with strict=True) while still being stacked/moved for vmap.
        self.register_buffer(
            "avg_matrix", _moving_average_matrix(seq_len, k), persistent=False
        )

        self.trend_linear = nn.Linear(seq_len, hid_dim)
        self.seasonal_linear = nn.Linear(seq_len, hid_dim)
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

        series = x[:, :, 0]                       # (B, T)
        trend = series @ self.avg_matrix.t()      # (B, T)
        seasonal = series - trend

        repr_ = self.trend_linear(trend) + self.seasonal_linear(seasonal)  # (B, hid_dim)
        repr_ = self.dropout(repr_)
        return repr_.unsqueeze(1)                 # (B, 1, hid_dim)
