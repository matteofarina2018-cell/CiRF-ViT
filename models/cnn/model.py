"""CNN baseline: a shared convolutional core with a per-neuron readout.

    [B, 1, P, P]
      -> DeepCNNCore          [B, C, F, F]     F = P - sum(k - 1), valid padding
      -> FactorizedReadout    [B, N]           softplus(sum_c v_nc sum_hw u_nhw f)

The locality prior here is the convolution itself plus a distance-weighted
penalty that keeps each neuron's spatial mask near its measured RF centre --
population-level locality, not per-neuron as in CiRF-ViT. With
`cfg['READOUT_RADII']` (one radius per neuron, stimulus pixels) the mask is
also confined to a hard circle around that centre: CiRF-ViT's crop on the
CNN's readout, the `cnn_crop` control of the ablation.

Neuron count comes from the RF centres (or `cfg['N_CELLS']`), image size from
`cfg['IMG_SIZE']`, so the same code fits any recording.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import IN_CHANNELS, IMG_SIZE_DEFAULT
from .cnn import DeepCNNCore
from .readout import FactorizedReadout, build_distance_maps


class RetinalModel(nn.Module):
    """Shared conv core + per-neuron factorised readout with an RF prior.

    cfg        : hyper-parameters (see config.DEFAULT_CFG)
    centers_x/y: (N,) RF centres in stimulus pixels; None centres every mask
    """

    def __init__(self, cfg: dict, centers_x=None, centers_y=None):
        super().__init__()

        kernels = tuple(cfg['CNN_KERNELS'])
        channels = cfg['CNN_CHANNELS']
        img_size = int(cfg.get('IMG_SIZE') or IMG_SIZE_DEFAULT)
        n_cells = int(cfg.get('N_CELLS') or cfg.get('N_CELLS_OVERRIDE')
                      or (len(centers_x) if centers_x is not None else 0))
        if not n_cells:
            raise ValueError('n_cells unknown: pass centers or set cfg["N_CELLS"]')

        feat = img_size - sum(k - 1 for k in kernels)   # 'valid' feature map
        off = sum((k - 1) // 2 for k in kernels)        # image -> feature offset
        if feat <= 0:
            raise ValueError(f'kernels {kernels} are larger than the {img_size} px '
                             f'stimulus; shrink CNN_KERNELS')

        self.n_cells = n_cells
        self.feat = feat
        self.offset = off

        self.core = DeepCNNCore(
            in_channels=IN_CHANNELS, kernels=kernels, channels=channels,
            bn_momentum=cfg.get('BN_MOMENTUM', 0.02),
        )
        n_features = self.core.out_channels

        if centers_x is not None and centers_y is not None:
            cx = np.clip(np.asarray(centers_x, dtype=np.float32) - off, 0, feat - 1)
            cy = np.clip(np.asarray(centers_y, dtype=np.float32) - off, 0, feat - 1)
            centers_feat = np.stack([cy, cx], axis=1)          # (N, 2) = (row, col)
        else:
            centers_feat = np.tile([feat / 2.0, feat / 2.0], (n_cells, 1)).astype(np.float32)
        self.register_buffer('rf_centers_feat',
                             torch.as_tensor(centers_feat, dtype=torch.float32))

        dist_maps = build_distance_maps(centers_feat, feat, feat)
        circle = None
        radii = cfg.get('READOUT_RADII')
        if radii is not None:
            if centers_x is None or len(radii) != n_cells:
                raise ValueError('READOUT_RADII needs the RF centres and one '
                                 'radius per neuron')
            # the conv stack has stride 1, so a radius in stimulus pixels is
            # the same number of feature pixels; dist_maps is normalised by feat
            r = np.asarray(radii, dtype=np.float32).reshape(-1, 1, 1) / feat
            circle = dist_maps <= r

        self.readout = FactorizedReadout(
            n_neurons=n_cells, n_features=n_features, feat_h=feat, feat_w=feat,
            dist_maps=dist_maps, circle=circle,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.core(x))

    def conv_kernels(self):
        """Kernel tensors of every conv layer (for the smoothness penalty)."""
        return self.core.kernel_weights()

    def clamp_nonneg(self):
        """Project the readout back onto its non-negative constraint."""
        self.readout.clamp_nonneg()
