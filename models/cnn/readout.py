"""Per-neuron factorised readout with a receptive-field locality prior.

For neuron n, over the feature map A,

    r_n(X) = softplus( sum_c v_nc sum_ij u_nij A_ijc + b_n )

with u (spatial mask) and v (feature weights) constrained non-negative. Instead
of a uniform L1 on the mask, the penalty is weighted by the distance of each
feature pixel from that neuron's measured RF centre,

    penalty = sum_n sum_ij d_n(i,j) |u_nij|

so weights far from the centre cost more and the mask stays localised on the
receptive field. Near the centre d ~ 0 and the penalty vanishes.

With `circle` ([N, H, W] bool) the mask is additionally confined to a hard
circle per neuron -- CiRF-ViT's crop, transplanted onto the CNN's readout
(`cfg['READOUT_RADII']`, the `cnn_crop` control of the ablation). Weights
outside the circle are held at zero, so they are not free parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_distance_maps(centers_feat, feat_h, feat_w):
    """Normalised distance-from-RF-centre maps, one per neuron.

    centers_feat : [N, 2] as (row, col) in feature-map coordinates
    returns      : [N, feat_h, feat_w] euclidean distance / feat_h
    """
    yy, xx = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing='ij')
    maps = np.zeros((len(centers_feat), feat_h, feat_w), dtype=np.float32)
    for n, (cy, cx) in enumerate(centers_feat):
        maps[n] = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    maps /= float(feat_h)   # normalised, so lambda_spatial stays O(1)
    return maps


class FactorizedReadout(nn.Module):
    """Factorised readout whose mask penalty is weighted by RF distance."""

    def __init__(self, n_neurons, n_features, feat_h, feat_w,
                 mask_init=None, bias_init=None, dist_maps=None, circle=None):
        super().__init__()

        # circle: None, or [N, H, W] bool, True inside the neuron's circle
        if circle is not None:
            circle = torch.as_tensor(np.asarray(circle), dtype=torch.float32)
            assert circle.shape == (n_neurons, feat_h, feat_w), circle.shape
        self.register_buffer('circle', circle)

        # u: per-neuron spatial mask, kept non-negative during training.
        if mask_init is None:
            u = torch.rand(n_neurons, feat_h, feat_w) * 0.01
        else:
            u = torch.as_tensor(mask_init, dtype=torch.float32).clone()
        if circle is not None:
            u = u * circle
        self.u = nn.Parameter(u)

        # v: per-neuron feature weights, also non-negative.
        self.v = nn.Parameter(torch.full((n_neurons, n_features), 0.1))

        # per-neuron bias, inside the softplus
        b = torch.zeros(n_neurons) if bias_init is None \
            else torch.as_tensor(bias_init, dtype=torch.float32).clone()
        self.bias = nn.Parameter(b)

        # distance maps: a buffer, so they follow the model across devices
        if dist_maps is None:
            dist_maps = np.zeros((n_neurons, feat_h, feat_w), dtype=np.float32)
        self.register_buffer("dist", torch.as_tensor(dist_maps, dtype=torch.float32))

    def spatial_mask(self):
        """The mask actually applied: `u`, zero outside the circle if there is one."""
        return self.u if self.circle is None else self.u * self.circle

    def forward(self, f):
        """f: [B, C, H, W] -> [B, N]."""
        masked = torch.einsum('bchw,nhw->bnc', f, self.spatial_mask())   # [B, N, C]
        h = (masked * self.v).sum(dim=-1) + self.bias        # [B, N]
        return F.softplus(h)

    def distance_penalty(self):
        """sum_n sum_ij d_n(i,j) |u_nij| -- the locality prior on the mask."""
        return (self.dist * self.spatial_mask().abs()).sum()

    @torch.no_grad()
    def clamp_nonneg(self):
        self.u.clamp_(min=0.0)
        if self.circle is not None:
            self.u.mul_(self.circle)
        self.v.clamp_(min=0.0)
