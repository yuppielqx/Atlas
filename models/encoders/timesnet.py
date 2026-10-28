"""TimesNet-style encoder adapter for single-node federated forecasting.

Adapted from TimesNet (Wu et al., ICLR 2023): detect the dominant periods of the
window via FFT, fold the 1D series into 2D (period x cycles), apply an Inception
2D-conv block, and aggregate the periods weighted by their amplitude. We keep the
original FFT + top-k period selection, so this encoder is NOT vmap safe and runs
only on the per-client (``--vectorized false``) training path. The stacked
TimesBlocks output is pooled and projected to ``hid_dim`` for the existing
PrototypeMatching + Predictor head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _fft_for_period(x, k):
    """Top-k periods of x:(B,T,C) by mean FFT amplitude, plus their amplitudes."""
    xf = torch.fft.rfft(x, dim=1)
    freq_amp = xf.abs().mean(0).mean(-1)        # (freq,)
    freq_amp[0] = 0                              # drop the DC component
    k = min(k, freq_amp.shape[0] - 1)
    _, top_idx = torch.topk(freq_amp, k)
    top_idx = top_idx.detach().cpu().numpy()
    periods = x.shape[1] // top_idx
    periods = [max(1, int(p)) for p in periods]
    return periods, xf.abs().mean(-1)[:, top_idx]  # list[k], (B, k)


class _InceptionBlock(nn.Module):
    """Multi-kernel 2D convolution (parameter-light Inception)."""

    def __init__(self, in_ch, out_ch, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
             for i in range(num_kernels)]
        )

    def forward(self, x):
        return sum(k(x) for k in self.kernels) / len(self.kernels)


class _TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            _InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            _InceptionBlock(d_ff, d_model, num_kernels),
        )

    def forward(self, x):  # (B, T, C)
        B, T, C = x.shape
        periods, period_weight = _fft_for_period(x, self.top_k)
        res = []
        for period in periods:
            if T % period != 0:
                length = (T // period + 1) * period
                pad = torch.zeros(B, length - T, C, device=x.device, dtype=x.dtype)
                out = torch.cat([x, pad], dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, C).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, length, C)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)                       # (B, T, C, k)
        weight = F.softmax(period_weight, dim=1)             # (B, k)
        weight = weight.unsqueeze(1).unsqueeze(1)            # (B, 1, 1, k)
        return torch.sum(res * weight, dim=-1) + x


class TimesNetEncoder(nn.Module):
    """TimesNet backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of TimesBlocks.
        dropout: dropout on the embedding and projected representation.
        d_model: TimesNet width.
        d_ff: Inception hidden width.
        top_k: number of FFT periods to fold over.
        num_kernels: Inception kernels per conv.
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=2,
        dropout=0.1,
        d_model=32,
        d_ff=32,
        top_k=3,
        num_kernels=6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        self.value_embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.empty(seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temporal_id=None):
        B, T, D = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {T}")
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {D}")

        h = self.value_embed(x[:, :, :1]) + self.pos_emb   # (B, T, d_model)
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = self.norm(block(h))
        repr_ = self.dropout(self.head(h.mean(dim=1)))      # (B, hid_dim)
        return repr_.unsqueeze(1)                           # (B, 1, hid_dim)
