"""Poisson negative log-likelihood.

Spike counts are Poisson-distributed given the rate, so
    -log P(k | lambda) = lambda - k log(lambda) + const
and the constant is dropped. The predicted rate must be strictly positive,
which the readout's ELU + 1 guarantees.
"""

import torch


def poisson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """y_pred, y_true: [B, N] predicted rate and recorded spike counts."""
    return (y_pred - y_true * torch.log(y_pred + 1e-8)).mean()
