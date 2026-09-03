"""The differentiable per-neuron circular crop -- the core of the model.

For each neuron the crop keeps the tokens of the feature map that fall inside a
circle centred on its receptive field. Two details make it trainable and cheap:

  * the sampling offsets are shared by all neurons (the largest radius defines
    the token set) so the tensor shape is static, and a per-neuron boolean mask
    excludes the tokens outside that neuron's own radius, both in the attention
    (as a key-padding mask) and in the readout's spatial softmax;
  * the centres are continuous parameters and the feature values are read by
    bilinear sampling (grid_sample), so gradients flow back to the centre
    coordinates and the crop can drift away from the measured centre it started
    at. All neurons of an image are sampled in one call, with a [N, T] grid
    over that image's feature map, so the map is never replicated per neuron.

Input:  [B, C, H, W]
Output: tokens   [B, N, T, C]
        rel_row  [T]      row offset from the centre, for RoPE
        rel_col  [T]      column offset from the centre
        pad_mask [N, T]   True = token outside this neuron's circle

Trainable: cx, cy (one pair per neuron, in feature pixels).
Buffers:   neuron_mask / pad_mask, the circle offsets and their indices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronCircle(nn.Module):
    """Differentiable circular crop, one circle per neuron."""

    def __init__(
        self,
        n_neurons:  int,
        img_h:      int,
        img_w:      int,
        centers_x:  "array-like",
        centers_y:  "array-like",
        radii:      "array-like",   # [N] radii in feature-space pixels
        crop_half:  int,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.img_h     = img_h
        self.img_w     = img_w
        self.crop_half = crop_half

        self.cx = nn.Parameter(torch.tensor(centers_x, dtype=torch.float32))
        self.cy = nn.Parameter(torch.tensor(centers_y, dtype=torch.float32))

        # ── precompute circle token positions ─────────────────────────────────
        h       = crop_half
        offsets = torch.arange(-h, h + 1, dtype=torch.float32)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing='ij')
        dist    = torch.sqrt(off_y ** 2 + off_x ** 2).reshape(-1)  # [T_sq]

        r_arr = torch.tensor(np.asarray(radii, dtype=np.float32))   # [N] feature px
        r_max = float(r_arr.max().item())

        # tokens in the union of all circles (set by the largest radius)
        idx          = (dist <= r_max).nonzero(as_tuple=True)[0]   # [T_circle]
        self.T_circle = int(idx.shape[0])

        self.register_buffer('circle_off_x',   off_x.reshape(-1)[idx])   # [T_circle]
        self.register_buffer('circle_off_y',   off_y.reshape(-1)[idx])   # [T_circle]
        self.register_buffer('dist_circle',    dist[idx])                 # [T_circle]
        self.register_buffer('circle_indices', idx)                       # [T_circle]

        # static per-neuron mask: True = token INSIDE this neuron's circle
        neuron_mask = r_arr.unsqueeze(1) >= dist[idx].unsqueeze(0)   # [N, T_circle] bool
        self.register_buffer('neuron_mask', neuron_mask)
        # pad_mask: True = token OUTSIDE (used as key_padding_mask in transformer)
        self.register_buffer('pad_mask', ~neuron_mask)

    # ── helper ────────────────────────────────────────────────────────────────
    def _to_normalized(self, px: torch.Tensor, size: int) -> torch.Tensor:
        return 2.0 * px / (size - 1) - 1.0

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        Returns: tokens [B, N, T_circle, C], rel_row [T], rel_col [T], pad_mask [N, T]
        """
        B, C, H, W = x.shape
        N = self.n_neurons
        T = self.T_circle

        cx_norm = self._to_normalized(self.cx, W)
        cy_norm = self._to_normalized(self.cy, H)

        off_x_norm = self.circle_off_x / (W / 2.0)
        off_y_norm = self.circle_off_y / (H / 2.0)

        grid_x = cx_norm.view(N, 1) + off_x_norm.view(1, T)          # [N, T]
        grid_y = cy_norm.view(N, 1) + off_y_norm.view(1, T)          # [N, T]
        grid   = torch.stack([grid_x, grid_y], dim=-1)               # [N, T, 2]
        grid   = grid.unsqueeze(0).expand(B, -1, -1, -1)             # [B, N, T, 2]

        feat = F.grid_sample(x, grid,
                             mode='bilinear', padding_mode='border',
                             align_corners=True)                     # [B, C, N, T]
        feat = feat.permute(0, 2, 3, 1)                              # [B, N, T, C]

        return feat, self.circle_off_y, self.circle_off_x, self.pad_mask
