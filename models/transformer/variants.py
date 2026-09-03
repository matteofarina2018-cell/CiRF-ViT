"""Variants of CiRF-ViT used by the ablation and comparison experiments.

Every variant is built by composing or wrapping the unmodified blocks in this
package -- no copy of the architecture is kept in sync by hand.

    no_crop     every neuron attends to the whole feature map (NoCropModel)
    no_block    NUM_BLOCKS = 0: tokeniser + crop + readout, no transformer
    no_rope     RoPE replaced by the identity, so attention has no positions
    no_attn     self-attention zeroed, so the blocks stop mixing tokens
    focal       the crop replaced by ViT-P's learnable focal attention bias
                (Chen et al., 2022), i.e. soft learned locality instead of a
                hard measured one

Two controls separate what the crop does to the *attention* from what it does
to the *readout*. Removing the crop removes both at once -- the neuron's tokens
are no longer restricted, and the readout no longer knows where the neuron is
-- so a drop cannot be attributed to either alone:

    readout_crop  global attention over the whole feature map, the measured
                  circle applied only in the readout's spatial softmax: the
                  per-neuron spatial prior is kept, the constraint on the
                  attention is removed
    cnn_crop      the CNN baseline with its readout confined to the same
                  circle: the constraint is kept, the backbone is changed

`build_variant(name, cfg, ds, device)` returns any of them, and 'full' returns
the model itself, so an experiment can loop over names. `cnn_crop` is the only
one that is not a CiRF-ViT: it is built by `models.cnn` from the CNN config
(`cnn_crop_cfg`), and trained by the CNN trainer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cirf import data as dat

from .config import IN_CHANNELS, IMG_SIZE_DEFAULT
from .readout import FactorizedReadout
from .tokenizer import CNNTokenizer, feat_size
from .train import build_model
from .transformer import TransformerEncoder


# -----------------------------------------------------------------------------
# no_crop: the same blocks without the per-neuron crop
# -----------------------------------------------------------------------------

class NoCropModel(nn.Module):
    """CiRF-ViT with the crop removed: every neuron sees all H*W tokens.

    With RoPE the attention score depends only on the *relative* position of two
    tokens, so a per-neuron centre cancels in every pairwise difference: without
    a crop (and without a per-neuron mask) the transformer output is identical
    for all neurons. The transformer therefore runs once per image rather than
    once per (image, neuron), and only the cheap per-neuron readout is applied
    afterwards. Absolute token positions are used for RoPE, since no centre is
    involved any more.

    `readout_mask` ([N, H*W] bool, True = token outside the neuron's circle)
    confines each neuron's readout softmax to its own circle while the
    attention stays global -- the `readout_crop` control. Without it every
    neuron reads out from the whole map.
    """

    def __init__(self, cfg, n_cells, readout_mask=None):
        super().__init__()
        cnn_dim = cfg['CNN_DIM']
        emb_dim = cfg['EMB_DIM']
        out_dim = emb_dim if emb_dim is not None else cnn_dim
        stride = cfg.get('CNN_STRIDE', 1)
        img_size = int(cfg.get('IMG_SIZE') or IMG_SIZE_DEFAULT)

        self.fh = feat_size(img_size, stride)
        T = self.fh * self.fh
        self.n_cells = n_cells

        self.cnn = CNNTokenizer(
            in_channels=IN_CHANNELS, cnn_dim=cnn_dim, cnn_layers=cfg['CNN_LAYERS'],
            cnn_kernel=cfg['CNN_KERNEL'], dropout=cfg['DROPOUT'], stride=stride,
        )

        rows = torch.arange(self.fh).view(self.fh, 1).expand(self.fh, self.fh).reshape(-1)
        cols = torch.arange(self.fh).view(1, self.fh).expand(self.fh, self.fh).reshape(-1)
        self.register_buffer('rows_abs', rows.float())
        self.register_buffer('cols_abs', cols.float())

        self.transformer = TransformerEncoder(
            in_dim=cnn_dim, num_heads=cfg['NUM_HEADS'], mlp_dim=cfg['MLP_DIM'],
            num_blocks=cfg['NUM_BLOCKS'], emb_dim=emb_dim, dropout=cfg['DROPOUT'],
        )
        self.readout = FactorizedReadout(n_neurons=n_cells, n_tokens=T,
                                         emb_dim=out_dim)
        if readout_mask is not None:
            readout_mask = torch.as_tensor(readout_mask, dtype=torch.bool)
            assert readout_mask.shape == (n_cells, T), readout_mask.shape
        self.register_buffer('readout_mask', readout_mask)

    def forward(self, x):
        feat = self.cnn(x)
        tokens = feat.flatten(2).transpose(1, 2)               # [B, T, C]
        out = self.transformer(tokens, self.rows_abs, self.cols_abs,
                               key_padding_mask=None)          # [B, T, D]
        u = self.readout.u
        if self.readout_mask is not None:
            u = u.masked_fill(self.readout_mask, float('-inf'))
        u_norm = torch.softmax(u, dim=-1)
        spatial = torch.einsum('nt,btd->bnd', u_norm, out)
        resp = (spatial * self.readout.v).sum(dim=-1) + self.readout.bias
        return F.elu(resp) + 1.0


def build_nocrop_model(cfg, ds=None, device='cpu'):
    """NoCropModel for a dataset: every neuron attends to the whole map."""
    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))
    cx, _cy = dat.centers(ds)
    return NoCropModel(cfg, len(cx)).to(device)


# -----------------------------------------------------------------------------
# readout_crop / cnn_crop: the crop's two roles, separated
# -----------------------------------------------------------------------------

def readout_circle_mask(cfg, ds=None):
    """[N, H*W] bool over the full feature map: True outside neuron n's circle.

    The same circle the crop uses -- the measured centre and the per-neuron
    radius of `cfg`, both in feature pixels -- but drawn on the whole map
    instead of defining a token subset. Only the tokens inside it get a
    readout weight, so the readout has the crop's spatial prior and (for a
    given neuron) the crop's number of free weights, while the attention is
    left unconstrained. The circle is fixed at the measured centre: a hard
    mask carries no gradient to move it, unlike the crop's bilinear sampling.
    """
    from .config import resolve_radii

    stride = cfg.get('CNN_STRIDE', 1)
    fh = feat_size(int(cfg.get('IMG_SIZE') or dat.img_size(ds)), stride)
    cx, cy = dat.centers(ds)
    radii = np.asarray(resolve_radii(cfg, len(cx), verbose=False), dtype=np.float32)
    rows = torch.arange(fh, dtype=torch.float32).view(fh, 1).expand(fh, fh).reshape(-1)
    cols = torch.arange(fh, dtype=torch.float32).view(1, fh).expand(fh, fh).reshape(-1)
    cx_f = torch.as_tensor(np.asarray(cx, dtype=np.float32) / stride).view(-1, 1)
    cy_f = torch.as_tensor(np.asarray(cy, dtype=np.float32) / stride).view(-1, 1)
    r_f = torch.as_tensor(radii / stride).view(-1, 1)
    dist = torch.sqrt((rows[None] - cy_f) ** 2 + (cols[None] - cx_f) ** 2)   # [N, T]
    return dist > r_f


def build_readout_crop_model(cfg, ds=None, device='cpu'):
    """Global attention, the measured circle applied in the readout only.

    The `no_crop` backbone with the readout softmax masked to each neuron's
    circle -- the control that keeps the spatial prior and removes the
    constraint on the attention.
    """
    cfg = dict(cfg)
    cfg.setdefault('IMG_SIZE', dat.img_size(ds))
    mask = readout_circle_mask(cfg, ds)
    return NoCropModel(cfg, mask.shape[0], readout_mask=mask).to(device)


def cnn_crop_cfg(cfg, ds=None, cnn_cfg=None) -> dict:
    """The CNN baseline's config with its readout confined to the crop's circle.

    `cfg` is the CiRF-ViT config the radii are taken from; the CNN's own
    hyper-parameters are `cnn_cfg` or the tuned defaults for the dataset
    (`models.cnn.get_cfg`). The result is a plain CNN config with
    `READOUT_RADII` set, so `models.cnn.build_model` (and the checkpoint
    loader) rebuild the model from it without knowing about this module.
    """
    from models.cnn import get_cfg as cnn_get_cfg
    from .config import resolve_radii

    out = dict(cnn_cfg if cnn_cfg is not None else cnn_get_cfg(ds))
    radii = resolve_radii(cfg, dat.n_cells(ds), verbose=False)
    out['READOUT_RADII'] = [int(r) for r in radii]
    return out


def build_cnn_crop_model(cfg, ds=None, device='cpu'):
    """The CNN baseline with the crop's circle on its readout.

    `cfg` may be either a CiRF-ViT config (the radii are taken from it and the
    CNN's tuned config is used for the rest) or a CNN config that already
    carries `READOUT_RADII`, which is what a saved checkpoint records.
    """
    from models.cnn import build_model as cnn_build_model

    cnn_cfg = cfg if 'READOUT_RADII' in cfg else cnn_crop_cfg(cfg, ds)
    return cnn_build_model(cnn_cfg, ds, device)


# -----------------------------------------------------------------------------
# no_rope / no_attn: runtime surgery on the built model
# -----------------------------------------------------------------------------

class IdentityRoPE(nn.Module):
    """RoPE replaced by the identity: attention keeps no positional information."""

    def apply(self, q, k, rel_row, rel_col):
        return q, k


class ZeroAttention(nn.Module):
    """Self-attention replaced by zero: each block becomes MLP-only."""

    def forward(self, x, rel_row, rel_col, key_padding_mask=None):
        return torch.zeros_like(x)


# -----------------------------------------------------------------------------
# focal: ViT-P's learnable attention bias in place of the crop
# -----------------------------------------------------------------------------

SUPPRESS = -100.0       # initial logit outside the window (ViT-P default)
CIRCULAR = True         # circular window, to match the shape of the crop


class FocalAttentionBias(nn.Module):
    """Learnable relative attention bias over a full fh x fh token grid.

    Following Swin / ViT-P, the bias is a table indexed by the relative offset
    between query and key, so every query shares the same window shape. It is
    initialised to 0 inside a window of the given radius and to `suppress`
    outside, then learned freely: locality starts strong and the model may relax
    it wherever the data support that.
    """

    def __init__(self, fh: int, window: float, suppress=SUPPRESS, circular=CIRCULAR):
        super().__init__()
        self.fh = fh
        rows = torch.arange(fh).view(fh, 1).expand(fh, fh).reshape(-1)
        cols = torch.arange(fh).view(1, fh).expand(fh, fh).reshape(-1)
        self.register_buffer('ri', ((rows[:, None] - rows[None, :]) + (fh - 1)).long())
        self.register_buffer('ci', ((cols[:, None] - cols[None, :]) + (fh - 1)).long())

        span = torch.arange(-(fh - 1), fh, dtype=torch.float32)
        oy, ox = torch.meshgrid(span, span, indexing='ij')
        inside = ((oy ** 2 + ox ** 2) <= float(window) ** 2 if circular
                  else (oy.abs() <= window) & (ox.abs() <= window))
        table = torch.zeros(2 * fh - 1, 2 * fh - 1)
        table[~inside] = suppress
        self.B_hat = nn.Parameter(table)

    def forward(self) -> torch.Tensor:
        return self.B_hat[self.ri, self.ci]        # [T, T]


class FocalMHSA(nn.Module):
    """Wraps a MultiHeadSelfAttention, adding the focal bias to its logits."""

    def __init__(self, base_attn: nn.Module, focal: FocalAttentionBias):
        super().__init__()
        self.base = base_attn
        self.focal = focal

    def forward(self, x, rel_row, rel_col, key_padding_mask=None):
        B, T, D = x.shape
        a = self.base
        Q = a.W_q(x).view(B, T, a.num_heads, a.head_dim).transpose(1, 2)
        K = a.W_k(x).view(B, T, a.num_heads, a.head_dim).transpose(1, 2)
        V = a.W_v(x).view(B, T, a.num_heads, a.head_dim).transpose(1, 2)
        Q, K = a.rope.apply(Q, K, rel_row, rel_col)
        attn_mask = self.focal().to(x.dtype)[None, None]        # [1, 1, T, T]
        drop_p = a.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask,
                                             dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return a.W_out(out)


def focal_window(cfg, n_cells=None) -> float:
    """Window radius in feature pixels: the mean crop radius over stride.

    That puts the soft window on the same spatial scale as the average hard
    crop, which is what makes the two locality mechanisms comparable.
    """
    from .config import resolve_radii

    radii = np.asarray(resolve_radii(cfg, n_cells or len(cfg['NEURON_RADII']),
                                     verbose=False), dtype=np.float32)
    return max(1.0, float(round(radii.mean() / cfg.get('CNN_STRIDE', 1))))


def build_focal_model(cfg, ds=None, device='cpu', window=None):
    """NoCrop backbone with a learnable focal bias in every block.

    Each block gets its own table, so depth can specialise, all starting from
    the same window.
    """
    model = build_nocrop_model(cfg, ds, device)
    window = window or focal_window(cfg, dat.n_cells(ds))
    for blk in model.transformer.blocks:
        blk.attn = FocalMHSA(blk.attn, FocalAttentionBias(model.fh, window).to(device)).to(device)
    return model


# -----------------------------------------------------------------------------

VARIANTS = ('full', 'no_crop', 'no_block', 'no_rope', 'no_attn', 'focal',
            'readout_crop', 'cnn_crop')

#: the variants that are CiRF-ViTs, trained by `models.transformer.fit`
TRANSFORMER_VARIANTS = tuple(v for v in VARIANTS if v != 'cnn_crop')

VARIANT_LABELS = {
    'full': 'CiRF-ViT', 'no_crop': 'no crop', 'no_block': 'no transformer\nblock',
    'no_rope': 'no RoPE', 'no_attn': 'no attention', 'focal': 'ViT-P',
    'readout_crop': 'global attention\ncrop in readout', 'cnn_crop': 'CNN\ncrop in readout',
}


def build_variant(name, cfg, ds=None, device='cpu'):
    """Any of `VARIANTS`, built from the same config.

    `cnn_crop` is built from the CNN config instead (see `cnn_crop_cfg`); `cfg`
    may be either that or the CiRF-ViT config the radii come from.
    """
    if name == 'full':
        return build_model(cfg, ds, device)
    if name == 'no_crop':
        return build_nocrop_model(cfg, ds, device)
    if name == 'readout_crop':
        return build_readout_crop_model(cfg, ds, device)
    if name == 'cnn_crop':
        return build_cnn_crop_model(cfg, ds, device)
    if name == 'focal':
        return build_focal_model(cfg, ds, device)
    if name == 'no_block':
        return build_model({**cfg, 'NUM_BLOCKS': 0}, ds, device)

    model = build_model(cfg, ds, device)
    if name == 'no_rope':
        for blk in model.transformer.blocks:
            blk.attn.rope = IdentityRoPE()
    elif name == 'no_attn':
        for blk in model.transformer.blocks:
            blk.attn = ZeroAttention()
    else:
        raise ValueError(f'unknown variant {name!r}; choose from {VARIANTS}')
    return model
