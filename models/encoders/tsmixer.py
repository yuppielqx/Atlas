"""TSMixer-style encoder adapter for single-node federated forecasting.

Adapted from TSMixer (Chen et al., 2023): an all-MLP architecture that alternates
"time-mixing" (an MLP across the time axis) and "feature-mixing" (an MLP across
the feature axis) residual blocks. The single target channel is first lifted to a
small feature dimension, ``num_layers`` mixer blocks are applied, then the result
is pooled and projected to ``hid_dim`` for the existing PrototypeMatching +
Predictor head.

Only Linear / LayerNorm / GELU, so it is fully ``torch.func.vmap`` safe.
"""

import torch.nn as nn

class _MixerBlock(nn.Module):
    """One TSMixer block: time-mixing MLP then feature-mixing MLP (pre-norm)."""

    def __init__(self, seq_len, d_feat, dropout):
        super().__init__()
        self.time_norm = nn.LayerNorm(d_feat)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feat_norm = nn.LayerNorm(d_feat)
        self.feat_mlp = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_feat),
            nn.Dropout(dropout),
        )

    def forward(self, h):  # h: (B, T, d_feat)
        # Time-mixing: MLP over the time axis (transpose so Linear acts on T).
        u = self.time_norm(h)
        u = self.time_mlp(u.transpose(1, 2)).transpose(1, 2)
        h = h + u
        # Feature-mixing: MLP over the feature axis.
        h = h + self.feat_mlp(self.feat_norm(h))
        return h


class TSMixerEncoder(nn.Module):
    """TSMixer backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of mixer blocks.
        dropout: dropout throughout.
        d_feat: width the single channel is lifted to for mixing.
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=3,
        dropout=0.1,
        d_feat=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        self.embed = nn.Linear(1, d_feat)
        self.blocks = nn.ModuleList(
            [_MixerBlock(seq_len, d_feat, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_feat)
        self.head = nn.Linear(d_feat, hid_dim)

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

        h = self.embed(x[:, :, :1])         # (B, T, d_feat)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        repr_ = self.head(h.mean(dim=1))    # (B, hid_dim)
        return repr_.unsqueeze(1)           # (B, 1, hid_dim)
