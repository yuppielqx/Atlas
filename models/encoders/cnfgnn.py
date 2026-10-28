"""CNFGNN-style graph-context encoder for federated traffic forecasting.

This is a lightweight baseline adapter for datasets with ``adj_mx.pkl``. The
data loader supplies two channels per client: the node series and an adjacency-
weighted neighbor mean. The encoder fuses temporal GRU states from both streams
with a small graph gate, producing the standard (B, 1, hid_dim) representation.
"""

import torch
import torch.nn as nn

from .gru import GRUEncoder


class CNFGNNEncoder(nn.Module):
    """Graph-context recurrent encoder.

    Args:
        input_dim: expected to be 2 when ``--graph_context true`` is used.
        hid_dim: output hidden dimension.
        seq_len: input window length.
        num_layers: GRU layers in each temporal branch.
        dropout: dropout on fused representation.
        graph_gate: initial neighbor mixing gate in [0, 1].
    """

    def __init__(
        self,
        input_dim=2,
        hid_dim=64,
        seq_len=12,
        num_layers=2,
        dropout=0.1,
        graph_gate=0.5,
    ):
        super().__init__()
        if input_dim < 2:
            raise ValueError(
                "CNFGNNEncoder needs graph_context inputs: run with "
                "--encoder_type cnfgnn --graph_context true"
            )
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        self.self_encoder = GRUEncoder(
            input_dim=1, hid_dim=hid_dim, seq_len=seq_len,
            num_layers=num_layers, dropout=dropout,
        )
        self.neigh_encoder = GRUEncoder(
            input_dim=1, hid_dim=hid_dim, seq_len=seq_len,
            num_layers=num_layers, dropout=dropout,
        )
        gate = min(max(float(graph_gate), 1e-4), 1.0 - 1e-4)
        self.graph_gate = nn.Parameter(torch.logit(torch.tensor(gate)))
        self.fuse = nn.Sequential(
            nn.LayerNorm(hid_dim * 2),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, temporal_id=None):
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        z_self = self.self_encoder(x[:, :, :1], temporal_id=temporal_id)
        z_neigh = self.neigh_encoder(x[:, :, 1:2], temporal_id=temporal_id)
        gate = torch.sigmoid(self.graph_gate)
        mixed = torch.cat([z_self, gate * z_neigh], dim=-1)
        return self.fuse(mixed)
