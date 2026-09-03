"""Per-neuron factorised readout: a spatial softmax over the crop, then
per-channel feature weights.

For neuron n, with x its cropped token sequence,

    spatial_n = softmax(u_n) . x_n           weights over the neuron's tokens
    output_n  = ELU(spatial_n . v_n + b_n) + 1

The ELU + 1 keeps the predicted rate strictly positive, as the Poisson
likelihood requires. `u` starts at zero, i.e. a flat weighting over the
neuron's tokens: the crop already says where the neuron looks, so the readout
is not given a head start from the data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedReadout(nn.Module):
    """
    Per-neuron factorized readout.

    For each neuron n:
        spatial_n = softmax(u_n) · x_n      [D]
        output_n  = ELU(spatial_n · v_n + bias_n) + 1  → always > 0

    u [N, T] starts at zero: softmax(0) is a flat weighting over the tokens.
    v [N, D] is initialized to small random values.

    Input:  [B, N, T, D]
    Output: [B, N]
    """

    def __init__(self, n_neurons: int, n_tokens: int, emb_dim: int):
        super().__init__()

        T = n_tokens

        self.u    = nn.Parameter(torch.zeros(n_neurons, T))
        self.v    = nn.Parameter(torch.randn(n_neurons, emb_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:        [B, N, T, D]
        pad_mask: [N, T] bool, True = token outside this neuron's circle (ignore).
        output:   [B, N]
        """
        if pad_mask is not None:
            u_norm = F.softmax(self.u.masked_fill(pad_mask, float('-inf')), dim=-1)
        else:
            u_norm = F.softmax(self.u, dim=-1)                      # [N, T]
        spatial = torch.einsum('nt,bntd->bnd', u_norm, x)          # [B, N, D]
        out     = (spatial * self.v).sum(dim=-1) + self.bias        # [B, N]
        return F.elu(out) + 1.0
