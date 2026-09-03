"""2D rotary position embedding on offsets relative to a neuron's centre.

Encoding the offset (d_row, d_col) from the neuron's own centre rather than the
absolute position in the feature map makes the token representation
translation-invariant: two neurons with identical receptive fields at different
locations produce identical attention patterns. Offsets are floats, since the
centres are trainable and land between pixels, so cos/sin are evaluated
directly instead of being looked up per integer position.

Constraint: head_dim must be divisible by 4 -- the first half encodes d_row,
the second half d_col, and each half is rotated in pairs.
"""

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """Rotary embedding of the (d_row, d_col) offsets, applied to Q and K."""

    def __init__(self, head_dim: int):
        super().__init__()

        assert head_dim % 4 == 0, \
            f"head_dim ({head_dim}) must be divisible by 4"

        half   = head_dim // 2
        n_freq = half // 2

        theta = 1.0 / (10000 ** (2 * torch.arange(n_freq, dtype=torch.float32) / half))
        self.register_buffer('theta', theta)   # [n_freq]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        a = x[..., 0::2]
        b = x[..., 1::2]
        return torch.stack([-b, a], dim=-1).flatten(-2)

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              rel_row: torch.Tensor, rel_col: torch.Tensor):
        """
        q, k    : [B, num_heads, T, head_dim]
        rel_row : [T] shared by the batch, or [B, T] when neurons are
                  folded into the batch dimension (one offset set per neuron)
        rel_col : same shape as rel_row
        """
        ang_r = rel_row.unsqueeze(-1) * self.theta   # [..., T, n_freq]
        ang_c = rel_col.unsqueeze(-1) * self.theta

        cos_r = torch.cos(ang_r).repeat_interleave(2, dim=-1)  # [..., T, half]
        sin_r = torch.sin(ang_r).repeat_interleave(2, dim=-1)
        cos_c = torch.cos(ang_c).repeat_interleave(2, dim=-1)
        sin_c = torch.sin(ang_c).repeat_interleave(2, dim=-1)

        cos = torch.cat([cos_r, cos_c], dim=-1)   # [..., T, head_dim]
        sin = torch.cat([sin_r, sin_c], dim=-1)

        if rel_row.dim() == 1:
            cos = cos.unsqueeze(0).unsqueeze(0)    # [1, 1, T, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)                 # [B, 1, T, D]
            sin = sin.unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
