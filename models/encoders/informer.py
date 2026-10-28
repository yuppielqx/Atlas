"""Informer-style encoder adapter for single-node federated forecasting.

Adapted from Informer (Zhou et al., AAAI 2021): ProbSparse self-attention that
samples a subset of dominant queries, with optional self-attention distilling.
We keep only the encoder (the decoder/forecast head is replaced by the project's
Predictor) and keep the original sampling-based query selection, so this encoder
is NOT vmap safe and runs only on the per-client (``--vectorized false``) path.
For the short windows used here ProbSparse degrades gracefully to (near) full
attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ProbAttention(nn.Module):
    def __init__(self, factor, dropout):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # Q,K: (B,H,L,E)
        B, H, L_K, E = K.shape
        L_Q = Q.shape[2]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        idx_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), idx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - Q_K_sample.sum(-1) / L_K
        M_top = M.topk(n_top, sorted=False)[1]               # (B, H, n_top)
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))     # (B, H, n_top, L_K)
        return Q_K, M_top

    def _initial_context(self, V, L_Q):
        ctx = V.mean(dim=-2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        return ctx

    def _update_context(self, ctx, V, scores, index):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)
        ctx[torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(ctx)
        return ctx

    def forward(self, queries, keys, values):  # (B, L, H, E)
        B, L_Q, H, E = queries.shape
        L_K = keys.shape[1]
        q = queries.transpose(2, 1)
        k = keys.transpose(2, 1)
        v = values.transpose(2, 1)
        U_part = min(self.factor * int(math.ceil(math.log(L_K))), L_K)
        u = min(self.factor * int(math.ceil(math.log(L_Q))), L_Q)
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
        scores_top = scores_top * (1.0 / math.sqrt(E))
        ctx = self._initial_context(v, L_Q)
        ctx = self._update_context(ctx, v, scores_top, index)
        return ctx.transpose(2, 1).contiguous()              # (B, L_Q, H, E)


class _AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor, dropout):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner = _ProbAttention(factor, dropout)
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
    def __init__(self, d_model, n_heads, d_ff, factor, dropout):
        super().__init__()
        self.attention = _AttentionLayer(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _DistilLayer(nn.Module):
    """Self-attention distilling: Conv1d + pool halves the time length."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2))
        y = F.elu(self.norm(y))
        y = self.pool(y).transpose(1, 2)
        return y


class InformerEncoder(nn.Module):
    """Informer encoder producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of ProbSparse encoder layers.
        dropout: dropout throughout.
        d_model: encoder width.
        n_heads: attention heads.
        factor: ProbSparse sampling factor.
        distil: insert distilling layers between encoder layers (off by default;
            the windows here are short).
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
        factor=5,
        distil=False,
        d_ff=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        self.value_embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.empty(seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, factor, dropout) for _ in range(num_layers)]
        )
        self.distil_layers = nn.ModuleList(
            [_DistilLayer(d_model) for _ in range(num_layers - 1)] if distil else []
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
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.distil_layers):
                h = self.distil_layers[i](h)
        h = self.norm(h)
        repr_ = self.head(h.mean(dim=1))            # (B, hid_dim)
        return repr_.unsqueeze(1)                   # (B, 1, hid_dim)
