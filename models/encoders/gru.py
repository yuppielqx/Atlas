"""GRU encoder adapter for single-node federated forecasting (FedGRU baseline).

Reproduces the FedGRU recipe (Liu et al., IEEE IoT-J 2020: FedAvg + GRU) inside
this framework: run with ``--fl_mode fedavg --encoder_type gru``.

The recurrence is hand-rolled from Linear layers (matmul + sigmoid/tanh over a
fixed-length window) rather than ``nn.GRU``: cuDNN RNN kernels are not
``torch.func.vmap`` compatible, while this unrolled form is, so the encoder
runs on the default vectorized training path.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim.
        hid_dim: hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of stacked GRU layers.
        dropout: dropout between layers and on the output representation.
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
        self.num_layers = num_layers

        self.ih = nn.ModuleList()
        self.hh = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hid_dim
            self.ih.append(nn.Linear(in_dim, 3 * hid_dim))
            self.hh.append(nn.Linear(hid_dim, 3 * hid_dim))
        self.dropout = nn.Dropout(dropout)

    def _cell(self, layer, x_t, h):
        """One GRU step: gates from input/hidden projections (vmap-safe)."""
        gi = self.ih[layer](x_t)                    # (B, 3H)
        gh = self.hh[layer](h)                      # (B, 3H)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

    def forward(self, x, temporal_id=None):
        """Encode a window.

        Args:
            x: (B, T, input_dim) input window.
            temporal_id: optional (B, 2) [time_of_day, day_of_week] (unused).

        Returns:
            z: (B, 1, hid_dim) window representation (last hidden state).
        """
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        out = x
        for layer in range(self.num_layers):
            h = out.new_zeros(B, self.hid_dim)
            hs = []
            for t in range(T):
                h = self._cell(layer, out[:, t], h)
                hs.append(h)
            out = torch.stack(hs, dim=1)            # (B, T, H)
            if layer < self.num_layers - 1:
                out = self.dropout(out)

        return self.dropout(out[:, -1]).unsqueeze(1)  # (B, 1, hid_dim)
