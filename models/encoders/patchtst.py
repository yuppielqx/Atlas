"""PatchTST-style encoder adapter for single-node federated forecasting.

Adapted from PatchTST (Nie et al., ICLR 2023): split the window into fixed-length
patches, linearly embed each patch, add a learnable positional embedding, run a
small Transformer encoder, then pool patches into a ``hid_dim`` window
representation for the existing PrototypeMatching + Predictor head.

Patching uses a precomputed index buffer (plain advanced indexing) and the
Transformer uses an explicit matmul/softmax attention rather than
``nn.MultiheadAttention``'s fused fast path, so the whole module is
``torch.func.vmap`` safe on the vectorized training path.
"""

import torch
import torch.nn as nn

class _MultiHeadAttention(nn.Module):
    """Explicit (vmap-safe) multi-head self-attention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, P, d_model)
        B, P, D = x.shape
        qkv = self.qkv(x).reshape(B, P, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, h, P, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, P, P)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                             # (B, h, P, head_dim)
        out = out.transpose(1, 2).reshape(B, P, D)
        return self.proj(out)


class _EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model, n_heads, dropout, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PatchTSTEncoder(nn.Module):
    """PatchTST backbone producing a (B, 1, hid_dim) window representation.

    Args:
        input_dim: input feature dim; channel 0 is the target series.
        hid_dim: model hidden dimension (output channels).
        seq_len: input window length.
        num_layers: number of Transformer encoder blocks.
        dropout: dropout throughout.
        patch_len: length of each patch.
        stride: stride between patches.
        d_model: Transformer width.
        n_heads: attention heads.
    """

    def __init__(
        self,
        input_dim=1,
        hid_dim=64,
        seq_len=12,
        num_layers=3,
        dropout=0.1,
        patch_len=4,
        stride=2,
        d_model=64,
        n_heads=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.seq_len = seq_len

        patch_len = min(int(patch_len), seq_len)
        stride = max(1, int(stride))
        self.patch_len = patch_len
        self.stride = stride
        num_patches = (seq_len - patch_len) // stride + 1
        self.num_patches = num_patches

        # Precomputed patch index matrix (num_patches, patch_len) — vmap-safe gather.
        # Non-persistent: a deterministic constant rebuilt in __init__, kept out of
        # state_dict() so the vectorized sync_to_clients (strict=True) still matches.
        starts = torch.arange(num_patches) * stride
        idx = starts.unsqueeze(1) + torch.arange(patch_len).unsqueeze(0)
        self.register_buffer("patch_index", idx.long(), persistent=False)

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.empty(num_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, n_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, hid_dim)

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

        series = x[:, :, 0]                         # (B, T)
        patches = series[:, self.patch_index]       # (B, num_patches, patch_len)

        h = self.patch_embed(patches) + self.pos_emb  # (B, num_patches, d_model)
        h = self.input_dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        repr_ = self.head(h.mean(dim=1))            # (B, hid_dim)
        return repr_.unsqueeze(1)                   # (B, 1, hid_dim)
