"""CiRF-ViT: a vision transformer with a per-neuron circular receptive-field crop.

    [B, 1, P, P]
      -> CNNTokenizer        [B, C, H, W]        H = ceil(P / stride)
      -> NeuronCircle        [B, N, T, C]        differentiable per-neuron crop
      -> TransformerEncoder  [B*N, T, D]         shared weights, RoPE 2D
      -> FactorizedReadout   [B, N]              per-neuron u . v + bias, ELU+1

The crop is what makes the model usable at small scale: each neuron only ever
sees the tokens inside a circle centred on its own receptive field, so the cost
of attention scales with the RF radius instead of the image, and the locality
prior comes from a measurement rather than from data. Centres are initialised
from the recorded RF centres and refined by backpropagation; radii are fixed
hyper-parameters.

Nothing here is tied to a particular recording: the neuron count comes from the
centres passed in, and the image size from `cfg['IMG_SIZE']` (written by
`build_model`, defaulting to 108).
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .config import IN_CHANNELS, IMG_SIZE_DEFAULT, resolve_radii
from .tokenizer import CNNTokenizer, feat_size
from .neuron_circle import NeuronCircle
from .transformer import TransformerEncoder
from .readout import FactorizedReadout


class RetinalModel(nn.Module):
    """CNN tokeniser -> per-neuron circular crop -> transformer -> readout.

    cfg        : hyper-parameters (see config.DEFAULT_CFG); 'IMG_SIZE' is read
                 from it when present
    centers_x/y: (N,) receptive-field centres in stimulus pixels
    radii      : (N,) crop radii in stimulus pixels; None -> cfg
    """

    def __init__(self, cfg: dict, centers_x, centers_y, radii=None):
        super().__init__()

        n_cells = len(np.asarray(centers_x))
        if radii is None:
            radii = resolve_radii(cfg, n_cells)

        img_size = int(cfg.get('IMG_SIZE') or IMG_SIZE_DEFAULT)
        cnn_dim = cfg['CNN_DIM']
        emb_dim = cfg['EMB_DIM']
        out_dim = emb_dim if emb_dim is not None else cnn_dim
        stride = cfg.get('CNN_STRIDE', 1)

        fh = feat_size(img_size, stride)

        cx = np.asarray(centers_x, dtype=np.float32) / stride
        cy = np.asarray(centers_y, dtype=np.float32) / stride
        r_feat = np.asarray(radii, dtype=np.float32) / stride

        crop_half = max(1, math.ceil(float(np.asarray(radii).max()) / stride))

        self.n_cells = n_cells
        self.img_size = img_size
        self.crop_half = crop_half

        self.cnn = CNNTokenizer(
            in_channels=IN_CHANNELS,
            cnn_dim=cnn_dim,
            cnn_layers=cfg['CNN_LAYERS'],
            cnn_kernel=cfg['CNN_KERNEL'],
            dropout=cfg['DROPOUT'],
            stride=stride,
        )

        self.circle = NeuronCircle(
            n_neurons=n_cells, img_h=fh, img_w=fh,
            centers_x=cx, centers_y=cy, radii=r_feat, crop_half=crop_half,
        )

        T_circle = self.circle.T_circle

        self.transformer = TransformerEncoder(
            in_dim=cnn_dim, num_heads=cfg['NUM_HEADS'], mlp_dim=cfg['MLP_DIM'],
            num_blocks=cfg['NUM_BLOCKS'], emb_dim=emb_dim, dropout=cfg['DROPOUT'],
        )

        self.readout = FactorizedReadout(
            n_neurons=n_cells, n_tokens=T_circle, emb_dim=out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        N = self.n_cells

        feat = self.cnn(x)
        C = feat.shape[1]

        tokens, rel_row, rel_col, pad_mask = self.circle(feat)
        T = tokens.shape[2]

        # Neurons are folded into the batch. The token offsets are the same
        # for every neuron (the circle is centred on each one), so they are
        # passed once, as [T], and the rotary embedding is evaluated once per
        # block rather than per sequence; only the padding mask is per neuron.
        tokens = tokens.reshape(B * N, T, C)
        pad_exp = pad_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, T)

        out = self.transformer(tokens, rel_row, rel_col, pad_exp)
        out = out.reshape(B, N, T, out.shape[-1])
        return self.readout(out, pad_mask)
