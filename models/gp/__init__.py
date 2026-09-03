"""Gaussian-process baseline (Goldin et al., PNAS 2023).

One sparse variational GP per neuron, with an arc-cosine kernel whose covariance
is shaped by a learnable Gaussian envelope over pixels centred on that neuron's
receptive field. Locality and smoothness are priors here, not learned features.

The `eigenspace_*`, `kernels`, `likelihoods` and `utils` modules are the
reference implementation of that model; `train.py` is the entry point used by
this repository.
"""

from .train import DEFAULT_CFG, train, train_cell

__all__ = ['DEFAULT_CFG', 'train', 'train_cell']
