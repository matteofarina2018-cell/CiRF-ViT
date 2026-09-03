"""CNN baseline hyper-parameters (paper Table 2).

A shared convolutional core plus a per-neuron factorised readout, i.e. the
standard "what / where" encoding model. Only the readout L1 was retuned per
dataset; everything else is shared.

Resolution order, later winning:

    DEFAULT_CFG  ->  TUNED[dataset]  ->  Dataset.config['cnn']  ->  kwargs
"""

from __future__ import annotations

IN_CHANNELS = 1
IMG_SIZE_DEFAULT = 108      # only a fallback: the image size comes from the data


DEFAULT_CFG = {
    # -- convolutional core (valid padding, so the map shrinks per layer) ----
    'CNN_KERNELS': (15, 3),      # kernel size per layer
    'CNN_CHANNELS': (8, 16),     # channels per layer
    'BN_MOMENTUM': 0.02,

    # -- regularisation -----------------------------------------------------
    'LAMBDA_SMOOTH': 2e-06,      # Laplacian smoothness on the conv kernels
    'LAMBDA_SPATIAL': 5e-03,     # RF-distance-weighted penalty on the mask u
    'LAMBDA_FEATURE': 4e-05,     # L1 on the feature weights v

    # -- training -----------------------------------------------------------
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-3,
    'LR_DECAY': 0.5,
    'LR_PATIENCE': 10,
    'MAX_EPOCHS': 1000,
    'EARLY_STOP': 20,
}


#: Published per-dataset settings.
#:
#: The salamander readout L1 is five times the mouse one. It was picked from a
#: sweep as the value that both maximises the mean adjusted R^2 and leaves no
#: neuron flat-lined: a larger L1 (an earlier choice) drives u and v of the
#: weakest cells to exactly zero, so their output collapses to a constant and
#: their explained variance to zero, trading the tail for the mean.
TUNED = {
    'salamander': {'LAMBDA_FEATURE': 2e-4},
}


def get_cfg(ds=None, **overrides) -> dict:
    """Config for a dataset, with the resolution order documented above."""
    from cirf import config as _cfg

    cfg = dict(DEFAULT_CFG)
    if ds is not None:
        dataset = _cfg.resolve(ds)
        cfg.update(TUNED.get(dataset.key, {}))
        cfg.update(dataset.model_config('cnn'))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
