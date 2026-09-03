"""CiRF-ViT hyper-parameters.

`DEFAULT_CFG` is the architecture reported in the paper (Table 1). It is
dataset-independent except for two things that a Bayesian search tuned per
recording -- the tokeniser kernel size and the per-neuron crop radii -- which
live in `TUNED`, keyed by dataset. A dataset that has never been tuned simply
gets the defaults and a uniform radius, so the model runs on any recording out
of the box; `--radius` (scripts/train.py) or a `NEURON_RADII` override sets it
explicitly.

Resolution order, later winning:

    DEFAULT_CFG  ->  TUNED[dataset]  ->  Dataset.config['transformer']  ->  kwargs
"""

from __future__ import annotations

IN_CHANNELS = 1
IMG_SIZE_DEFAULT = 108      # only a fallback: the image size comes from the data


DEFAULT_CFG = {
    # -- CNN tokeniser ------------------------------------------------------
    'CNN_DIM': 32,
    'CNN_LAYERS': 1,
    'CNN_KERNEL': 3,
    'CNN_STRIDE': 2,
    'DROPOUT': 0.1,

    # -- transformer encoder ------------------------------------------------
    # EMB_DIM = None keeps the tokeniser width (no input projection).
    'EMB_DIM': None,
    'NUM_HEADS': 1,
    'NUM_BLOCKS': 2,
    'MLP_DIM': 64,

    # -- training -----------------------------------------------------------
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 0.0,
    'MAX_EPOCHS': 200,
    'EARLY_STOP': 30,
    # Fuse the model's kernels with torch.compile for training (GPU only; a
    # few seconds of warm-up per run, the model is memory-bound otherwise).
    # Numerically the same model up to float reordering.
    'COMPILE': True,

    # -- per-neuron crop ----------------------------------------------------
    # Radius of each neuron's circular crop, in pixels of the stimulus frame.
    # NEURON_RADII (one per neuron) takes precedence when its length matches the
    # neuron count; otherwise every neuron gets DEFAULT_RADIUS.
    'NEURON_RADII': None,
    'DEFAULT_RADIUS': 12,
}


#: Published per-dataset settings (Bayesian search; paper Table 1).
TUNED = {
    'mice': {
        'NEURON_RADII': [
            23,  8,  8,  4, 16, 14, 13, 19,  6,  4,
             9, 21, 12, 25,  7, 11,  4, 16, 17,  8,
            22, 15, 25,  8,  4, 18,  6,  4,  4, 25,
             9, 25, 16, 13, 21,  4, 13, 13, 21, 20,
             8,
        ],
    },
    'salamander': {
        'CNN_KERNEL': 11,
        'NEURON_RADII': [
            11, 25, 14, 15, 22, 14, 11, 22, 14, 22,
            17,  8,  6, 10,  7, 21,  9, 10, 22, 18,
            22, 21, 19, 20,  6, 15, 10, 12, 10,  4,
            22, 17, 17, 24, 10,  5, 15, 21, 10, 11,
            20, 25, 13, 11, 20, 12, 23, 25, 22,
        ],
    },
}


def get_cfg(ds=None, **overrides) -> dict:
    """Config for a dataset, with the resolution order documented above."""
    from cirf import config as _cfg

    cfg = dict(DEFAULT_CFG)
    if ds is not None:
        dataset = _cfg.resolve(ds)
        cfg.update(TUNED.get(dataset.key, {}))
        cfg.update(dataset.model_config('transformer'))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    check_rope(cfg)
    return cfg


def check_rope(cfg: dict) -> None:
    """The 2D rotary embedding splits the head dimension into four quarters."""
    emb = cfg['EMB_DIM'] if cfg['EMB_DIM'] is not None else cfg['CNN_DIM']
    heads = cfg['NUM_HEADS']
    if emb % heads or (emb // heads) % 4:
        raise ValueError(
            f'RoPE 2D needs emb_dim ({emb}) divisible by NUM_HEADS ({heads}) '
            f'and the head dimension divisible by 4; got head dim '
            f'{emb / heads:g}.')


def resolve_radii(cfg: dict, n_cells: int, verbose=True) -> list:
    """Per-neuron crop radii for a recording with `n_cells` neurons.

    Uses `NEURON_RADII` when it has one entry per neuron, otherwise a uniform
    `DEFAULT_RADIUS`. The uniform fallback is what an untuned dataset gets: the
    crop still restricts attention to each neuron's neighbourhood, it is just
    not sized per cell.
    """
    radii = cfg.get('NEURON_RADII')
    if radii is not None and len(radii) == n_cells:
        return list(radii)
    if isinstance(radii, (int, float)):
        return [int(radii)] * n_cells
    r = int(cfg.get('DEFAULT_RADIUS', 12))
    if verbose:
        have = 0 if radii is None else len(radii)
        print(f'[config] NEURON_RADII ({have}) does not match n_cells '
              f'({n_cells}); using a uniform radius of {r} px.')
    return [r] * n_cells
