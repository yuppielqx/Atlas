"""Autoformer-style encoder adapter for single-node federated forecasting.

Adapted from Autoformer (Wu et al., NeurIPS 2021): series decomposition (moving
average) plus an Auto-Correlation mechanism that discovers period-based
dependencies via FFT and aggregates time-delayed values. We keep only the
encoder (the decoder/forecast head is replaced by the project's Predictor) and
keep the original FFT + top-k delay selection, so this encoder is NOT vmap safe
and runs only on the per-client (``--vectorized false``) training path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class _MovingAvg(nn.Module):
    """Moving average with replicate padding (kernel forced odd, <= seq_len*2)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):  # (B, T, C)
        front = x[:, :1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x):
        mean = self.moving_avg(x)
        return x - mean, mean


class _AutoCorrelation(nn.Module):
    def __init__(self, factor, dropout):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _time_delay_agg(self, values, corr):  # values/corr: (B, H, C, L)
        H, C, L = values.shape[1], values.shape[2], values.shape[3]
        top_k = max(1, int(self.factor * math.log(L)))
        mean_value = corr.mean(dim=1).mean(dim=1)            # (B, L)
        index = torch.topk(mean_value.mean(dim=0), top_k, dim=-1)[1]  # (top_k,)
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)            # (B, top_k)
        agg = torch.zeros_like(values)
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), dims=-1)
            w = tmp_corr[:, i].view(-1, 1, 1, 1)
            agg = agg + pattern * w
        return agg

    def forward(self, queries, keys, values):  # (B, L, H, E)
        B, L, H, E = queries.shape
        q = queries.permute(0, 2, 3, 1).contiguous()        # (B, H, E, L)
        k = keys.permute(0, 2, 3, 1).contiguous()
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)
        v = values.permute(0, 2, 3, 1).contiguous()         # (B, H, E, L)
        out = self._time_delay_agg(v, corr)                 # (B, H, E, L)
        return out.permute(0, 3, 1, 2).contiguous()         # (B, L, H, E)


class _AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor, dropout):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner = _AutoCorrelation(factor, dropout)
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, L, H, -1)
        k = self.k_proj(x).view(B, L, H, -1)
        v = self.v_proj(x).view(B, L, H, -1)
        out = self.inner(q, k, v).view(B, L, -1)
        return self.out_proj(out)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, moving_avg, dropout):
        super().__init__()
        self.attention = _AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.decomp1 = _SeriesDecomp(moving_avg)
        self.decomp2 = _SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(x))
        x, _ = self.decomp1(x)
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res


class AutoformerEncoder(nn.Module):
    """Autoformer encoder producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of Auto-Correlation encoder layers.
        dropout: dropout throughout.
        d_model: encoder width.
        n_heads: Auto-Correlation heads.
        factor: controls the number of delays aggregated (factor*log(L)).
        moving_avg: series-decomposition kernel (forced odd).
        d_ff: feed-forward width.
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=2,
        dropout=0.1,
        d_model=32,
        n_heads=4,
        factor=1,
        moving_avg=5,
        d_ff=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        if moving_avg % 2 == 0:
            moving_avg += 1
        self.value_embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.empty(seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, factor, moving_avg, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, hid_dim)

    def forward(self, x, temporal_id=None):
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        h = self.value_embed(x[:, :, :1]) + self.pos_emb
        h = self.embed_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        repr_ = self.head(h.mean(dim=1))            # (B, hid_dim)
        return repr_.unsqueeze(1)                   # (B, 1, hid_dim)
