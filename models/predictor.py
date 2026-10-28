"""STID-style regression head for time series forecasting."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Map hidden representations to future values with STID's 1x1 Conv head.

    STID uses a point-wise convolution over the hidden channel dimension as its
    regression layer. Here each client has one node, so the head maps
    ``(B, C, 1, 1)`` directly to ``(B, horizon, 1)``.

    Args:
        hid_dim: hidden dimension
        horizon: number of future steps to predict
        use_residual: if True, concatenate ``z_inv`` and ``z_res`` on channels
        dropout: kept for constructor compatibility; STID's regression head does
            not use dropout
    """

    def __init__(self, hid_dim, horizon, use_residual=True, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.use_residual = use_residual
        in_dim = hid_dim * 2 if use_residual else hid_dim
        self.regression_layer = nn.Conv2d(
            in_channels=in_dim,
            out_channels=horizon,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, z_inv, z_res=None):
        """Predict future values.

        Args:
            z_inv: (B, T, hid_dim) invariant representation
            z_res: (B, T, hid_dim) residual representation, used when enabled

        Returns:
            y_hat: (B, horizon, 1) predictions
        """
        if self.use_residual:
            if z_res is None:
                raise ValueError("z_res must be provided when use_residual=True")
            feat = torch.cat([z_inv[:, -1:, :], z_res[:, -1:, :]], dim=-1)
        else:
            feat = z_inv[:, -1:, :]

        feat = feat.transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        out = self.regression_layer(feat)          # (B, horizon, 1, 1)
        return out.squeeze(-1)                     # (B, horizon, 1)
