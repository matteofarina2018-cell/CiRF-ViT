"""LN baseline hyper-parameters.

One linear-nonlinear cascade per neuron, fitted on a square window centred on
that neuron's receptive field -- the same per-neuron RF information CiRF-ViT
uses for its crop, which makes the LN the "locality only, no shared features"
end of the comparison.

`CROP_SIZE` is clipped to the stimulus size at build time, so smaller images
work unchanged.
"""

from __future__ import annotations

DEFAULT_CFG = {
    # -- receptive-field window --
    'CROP_SIZE': 64,             # side of the window around each RF centre

    # -- filter regularisation --
    'SPARSITY_FACTOR': 1.0e-3,   # L1 on the filter
    'SMOOTHNESS_FACTOR': 1.0e-2,  # squared spatial gradient of the filter
    'MAX_NORM': 1.0,             # ||w|| <= MAX_NORM after every step

    # -- training --
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 2.0e-3,
    'MAX_EPOCHS': 1000,
    'EARLY_STOP': 30,
    'LR_DECAY': 0.1,
    'LR_PATIENCE': 10,
    'N_REPEATS': 5,              # independent fits per neuron, averaged
}

TUNED: dict = {}


def get_cfg(ds=None, **overrides) -> dict:
    """Config for a dataset (the LN uses the same settings everywhere)."""
    from cirf import config as _cfg

    cfg = dict(DEFAULT_CFG)
    if ds is not None:
        dataset = _cfg.resolve(ds)
        cfg.update(TUNED.get(dataset.key, {}))
        cfg.update(dataset.model_config('LN'))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
