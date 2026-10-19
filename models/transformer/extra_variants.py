"""Further ablations of CiRF-ViT: the core, the crop centres, the radius, the sampling.

Every variant here is a CiRF-ViT built from the unmodified blocks of this
package with one thing changed, trained by the same `fit` (Poisson NLL, Adam,
early stopping), on the same K-fold partition, so a difference from `full` is
the effect of that one change.

Core: attention or convolution
------------------------------
Together with `full`, `readout_crop` and `no_crop` they make a controlled
{attention, convolution} x {crop in the core, crop in the readout, no crop}
table in which only the token-mixing operator changes: same tokeniser, same
crop, same readout, same loss, same seeds.

    conv_core_crop     the transformer blocks replaced by convolutional blocks
                       that run on each neuron's crop: tokeniser -> crop ->
                       conv blocks -> readout. The crop sits after the first
                       convolutional layer and before the next ones, i.e. a
                       CNN with the measured locality prior in its core.
    conv_readout_crop  the same convolutional blocks on the whole feature map,
                       the circle only in the readout softmax (the CNN
                       counterpart of `readout_crop`)
    conv_no_crop       the same convolutional blocks on the whole feature map,
                       readout over the whole map (counterpart of `no_crop`)

A convolutional block is  x + Dropout(GELU(Conv3x3(LN(x)))),  with a 3x3
convolution at the token width: 9,312 parameters against the 8,544 of a
transformer block (attention 4,224 + MLP 4,192 + two LayerNorms), so the two
cores are matched to within 1.5k parameters on a ~40-60k model. On the crop the
grid outside the neuron's circle is held at zero before every convolution,
which is what the key-padding mask does for the attention.

Centres
-------
    centers_random   centres drawn uniformly over the image, then trained
    centers_shared   every neuron starts at the population mean centre
    centers_cluster  every neuron starts at the centroid of its k-means
                     cluster of measured centres (k = 4)
    centers_data     centres estimated from the training folds alone, without
                     the STA: argmax of |corr(response, pixel)| over the
                     natural images, smoothed (sigma 3 px)

Radii are the tuned per-neuron ones in every case; only the starting centre
changes. The trained centres are in the state dict, so reloading a checkpoint
recovers them whatever the initialisation was.

Radius
------
    radius_x0.5, radius_x0.75, radius_x1.5, radius_x2   every tuned radius
    multiplied by that factor (rounded, at least 2 px)

Sampling and offsets
--------------------
    sampling_nearest  nearest-neighbour instead of bilinear sampling: no
                      gradient reaches the centres, so they stay where measured
    sampling_bicubic  bicubic sampling
    offsets_fixed     offsets normalised by (W-1)/2, consistent with
                      align_corners=True (the default divides by W/2,
                      shrinking every radius by (W-1)/W = 0.981 on 54x54)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cirf import data as dat

from .config import resolve_radii
from .model import RetinalModel


# -----------------------------------------------------------------------------
# convolutional core
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """x + Dropout(GELU(Conv3x3(LN(x)))) on a [B, C, H, W] map."""

    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep=None):
        y = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if keep is not None:
            y = y * keep
        x = x + self.drop(F.gelu(self.conv(y)))
        return x if keep is None else x * keep


class ConvEncoder(nn.Module):
    """A stack of ConvBlocks, the drop-in for TransformerEncoder."""

    def __init__(self, dim: int, num_blocks: int, kernel: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(dim, kernel, dropout)
                                     for _ in range(num_blocks)])

    def forward(self, x, keep=None):
        for blk in self.blocks:
            x = blk(x, keep)
        return x


class ConvCoreCropModel(RetinalModel):
    """CiRF-ViT with the attention replaced by convolution on each crop.

    The crop's tokens sit on a (2h+1) x (2h+1) grid of offsets; they are
    scattered back onto that grid, convolved with the neuron's circle as the
    support, and gathered again for the unchanged readout.
    """

    def __init__(self, cfg, centers_x, centers_y, radii=None):
        super().__init__(cfg, centers_x, centers_y, radii)
        if cfg.get('EMB_DIM') not in (None, cfg['CNN_DIM']):
            raise ValueError('the convolutional core keeps the tokeniser width')
        self.transformer = ConvEncoder(cfg['CNN_DIM'], cfg['NUM_BLOCKS'],
                                       dropout=cfg['DROPOUT'])
        self.side = 2 * self.crop_half + 1

    def forward(self, x):
        B, N, S = x.shape[0], self.n_cells, self.side
        feat = self.cnn(x)
        C = feat.shape[1]
        tokens, _, _, pad_mask = self.circle(feat)              # [B, N, T, C]
        T = tokens.shape[2]
        idx = self.circle.circle_indices                         # [T] into S*S

        grid = tokens.new_zeros(B * N, S * S, C)
        grid[:, idx] = tokens.reshape(B * N, T, C)
        grid = grid.view(B * N, S, S, C).permute(0, 3, 1, 2)    # [BN, C, S, S]

        keep = tokens.new_zeros(N, S * S)
        keep[:, idx] = (~pad_mask).to(tokens.dtype)
        keep = keep.view(N, 1, S, S).unsqueeze(0).expand(B, -1, -1, -1, -1)
        keep = keep.reshape(B * N, 1, S, S)

        out = self.transformer(grid * keep, keep)               # [BN, C, S, S]
        out = out.permute(0, 2, 3, 1).reshape(B * N, S * S, C)[:, idx]
        return self.readout(out.reshape(B, N, T, C), pad_mask)


def _conv_full_map_model(cfg, ds, device, readout_mask):
    """NoCropModel with the convolutional core on the whole feature map."""
    from .variants import NoCropModel

    class ConvFullMapModel(NoCropModel):
        def __init__(self, cfg, n_cells, readout_mask=None):
            super().__init__(cfg, n_cells, readout_mask)
            self.transformer = ConvEncoder(cfg['CNN_DIM'], cfg['NUM_BLOCKS'],
                                           dropout=cfg['DROPOUT'])

        def forward(self, x):
            feat = self.transformer(self.cnn(x))                # [B, C, H, W]
            out = feat.flatten(2).transpose(1, 2)               # [B, T, C]
            u = self.readout.u
            if self.readout_mask is not None:
                u = u.masked_fill(self.readout_mask, float('-inf'))
            spatial = torch.einsum('nt,btd->bnd', torch.softmax(u, dim=-1), out)
            resp = (spatial * self.readout.v).sum(dim=-1) + self.readout.bias
            return F.elu(resp) + 1.0

    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))
    n = dat.n_cells(ds)
    return ConvFullMapModel(cfg, n, readout_mask).to(device)


# -----------------------------------------------------------------------------
# centres
# -----------------------------------------------------------------------------

def _kmeans(points, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    cent = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(iters):
        lab = ((points[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        new = np.array([points[lab == j].mean(0) if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return cent[lab]


def data_centers(ds, fold, n_folds=None, sigma=3.0):
    """RF centres from the natural-image responses of the training folds only.

    For every neuron, the correlation between its response and each pixel over
    the training stimuli, smoothed with a Gaussian, and its |argmax|. No white
    noise, no STA, no held-out stimulus.
    """
    from scipy.ndimage import gaussian_filter

    i_tr, _ = dat.fold_indices(ds, fold, n_folds)
    d = dat.load(ds)
    X = d['images_pool'][i_tr].squeeze(-1).astype(np.float64)       # [M, P, P]
    Y = d['responses_pool'][i_tr].astype(np.float64)                 # [M, N]
    P = X.shape[1]
    Xc = (X - X.mean(0)).reshape(len(X), -1)
    Xc /= Xc.std(0) + 1e-8
    Yc = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    corr = (Yc.T @ Xc / len(X)).reshape(-1, P, P)                    # [N, P, P]
    cx, cy = [], []
    for m in corr:
        m = gaussian_filter(np.abs(m), sigma)
        r, c = np.unravel_index(np.argmax(m), m.shape)
        cy.append(float(r))
        cx.append(float(c))
    return np.array(cx, np.float32), np.array(cy, np.float32)


def initial_centers(mode, cfg, ds):
    """(cx, cy) in stimulus pixels for a centre-initialisation variant."""
    cx, cy = (np.asarray(a, dtype=np.float32) for a in dat.centers(ds))
    P = dat.img_size(ds)
    fold, seed = int(cfg.get('RUN_FOLD', 0)), int(cfg.get('RUN_SEED', 0))
    if mode == 'random':
        rng = np.random.default_rng(10_000 + 100 * fold + seed)
        return (rng.uniform(0, P - 1, len(cx)).astype(np.float32),
                rng.uniform(0, P - 1, len(cy)).astype(np.float32))
    if mode == 'shared':
        return (np.full_like(cx, cx.mean()), np.full_like(cy, cy.mean()))
    if mode == 'cluster':
        pts = _kmeans(np.stack([cx, cy], 1).astype(np.float64), k=4, seed=0)
        return pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    if mode == 'data':
        return data_centers(ds, fold, cfg.get('RUN_N_FOLDS'))
    raise ValueError(mode)


# -----------------------------------------------------------------------------
# registry
# -----------------------------------------------------------------------------

RADIUS_SCALES = {'radius_x0.5': 0.5, 'radius_x0.75': 0.75,
                 'radius_x1.5': 1.5, 'radius_x2': 2.0}

EXTRA_VARIANTS = ('conv_core_crop', 'conv_readout_crop', 'conv_no_crop',
                   'centers_random', 'centers_shared', 'centers_cluster',
                   'centers_data', *RADIUS_SCALES,
                   'sampling_nearest', 'sampling_bicubic', 'offsets_fixed')

EXTRA_LABELS = {
    'conv_core_crop': 'conv core\ncrop in core',
    'conv_readout_crop': 'conv core\ncrop in readout',
    'conv_no_crop': 'conv core\nno crop',
    'centers_random': 'random\ncentres', 'centers_shared': 'shared\ncentre',
    'centers_cluster': 'cluster\ncentres', 'centers_data': 'centres\nfrom data',
    **{k: f'radius x{v:g}' for k, v in RADIUS_SCALES.items()},
    'sampling_nearest': 'nearest\nsampling', 'sampling_bicubic': 'bicubic\nsampling',
    'offsets_fixed': 'offsets\n(W-1)/2',
}


def _radii(cfg, ds, scale=1.0):
    r = np.asarray(resolve_radii(cfg, dat.n_cells(ds), verbose=False), np.float32)
    return np.maximum(2, np.round(r * scale)).astype(np.float32)


def build_extra_variant(name, cfg, ds=None, device='cpu'):
    from .variants import readout_circle_mask

    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))
    cx, cy = dat.centers(ds)

    if name == 'conv_core_crop':
        return ConvCoreCropModel(cfg, cx, cy, _radii(cfg, ds)).to(device)
    if name == 'conv_readout_crop':
        return _conv_full_map_model(cfg, ds, device, readout_circle_mask(cfg, ds))
    if name == 'conv_no_crop':
        return _conv_full_map_model(cfg, ds, device, None)

    if name.startswith('centers_'):
        cx0, cy0 = initial_centers(name.split('_', 1)[1], cfg, ds)
        return RetinalModel(cfg, cx0, cy0, _radii(cfg, ds)).to(device)

    if name in RADIUS_SCALES:
        return RetinalModel(cfg, cx, cy, _radii(cfg, ds, RADIUS_SCALES[name])).to(device)

    model = RetinalModel(cfg, cx, cy, _radii(cfg, ds)).to(device)
    if name == 'sampling_nearest':
        model.circle.mode = 'nearest'
    elif name == 'sampling_bicubic':
        model.circle.mode = 'bicubic'
    elif name == 'offsets_fixed':
        model.circle.offset_norm = 'corners'
    else:
        raise ValueError(f'unknown variant {name!r}')
    return model
