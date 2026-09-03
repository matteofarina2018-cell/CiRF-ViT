"""Shared-weight transformer blocks with a 2D rotary position embedding.

Every neuron's cropped token sequence is processed by the same blocks; the
only thing that distinguishes neurons is which tokens they see and where
those tokens sit relative to their own centre, which RoPE encodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import RoPE2D


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE 2D on positions relative to the
    neuron's centre.

    Input:  [B, T, emb_dim]
    Output: [B, T, emb_dim]

    rel_row, rel_col : [T] float offsets from the centre of the crop, handed
    to RoPE as the positional information.
    """

    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert emb_dim % num_heads == 0, \
            f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        assert (emb_dim // num_heads) % 4 == 0, \
            f"head_dim ({emb_dim // num_heads}) must be divisible by 4 (RoPE 2D)"

        self.num_heads = num_heads
        self.head_dim  = emb_dim // num_heads
        self.dropout_p = dropout

        self.W_q   = nn.Linear(emb_dim, emb_dim)
        self.W_k   = nn.Linear(emb_dim, emb_dim)
        self.W_v   = nn.Linear(emb_dim, emb_dim)
        self.W_out = nn.Linear(emb_dim, emb_dim)

        self.rope = RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : [B, T, D]
        rel_row         : [T] or [B, T]  row offset from the centre
        rel_col         : [T] or [B, T]  column offset from the centre
        key_padding_mask: [B, T] bool, True = token to ignore
        """
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K = self.rope.apply(Q, K, rel_row, rel_col)

        # Float additive mask (0 = keep, -inf = drop) rather than a bool one:
        # bool masks can hit a slow path in scaled_dot_product_attention.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                B, 1, 1, T, dtype=x.dtype, device=x.device
            ).masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))

        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=attn_mask,
                                             dropout_p=drop_p)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_out(out)


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, emb_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One block: LN -> attention -> residual -> LN -> MLP -> residual.
    """

    def __init__(self, emb_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp   = MLP(emb_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel_row, rel_col, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    A stack of TransformerBlocks, shared by every neuron.

    Sequences of all neurons are folded into the batch dimension by
    RetinalModel, so one set of weights serves the whole population.

      Input:  [B, T, in_dim]
      Output: [B, T, emb_dim]

    A linear projection is inserted when emb_dim differs from in_dim.
    """

    def __init__(self, in_dim: int, num_heads: int, mlp_dim: int, num_blocks: int,
                 emb_dim: int = None, dropout: float = 0.0):
        super().__init__()

        emb_dim = emb_dim if emb_dim is not None else in_dim

        self.proj = nn.Linear(in_dim, emb_dim) if emb_dim != in_dim else nn.Identity()

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor,
                rel_row: torch.Tensor,
                rel_col: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : [B, T, in_dim]
        rel_row         : [T] or [B, T]
        rel_col         : [T] or [B, T]
        key_padding_mask: [B, T] bool, True = token to ignore
        output          : [B, T, emb_dim]
        """
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rel_row, rel_col, key_padding_mask)
        return x
