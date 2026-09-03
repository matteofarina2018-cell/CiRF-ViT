# model/losses.py
#
# Poisson NLL (identical form to the other models — do not modify) plus the
# LN filter regularization (L1 sparsity + 2D smoothness), mirroring the
# reference L1Smooth2DRegularizer.

import torch


def poisson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood: mean(y_pred - y_true*log(y_pred)).
    y_pred must be > 0 (guaranteed by the parametric-softplus output)."""
    return (y_pred - y_true * torch.log(y_pred + 1e-8)).mean()


def filter_regularization(w: torch.Tensor, sparsity_factor: float,
                          smoothness_factor: float) -> torch.Tensor:
    """L1 (sparsity) + squared-gradient (smoothness) penalty on the 2D filter,
    mirroring pyretina's L1Smooth2DRegularizer.
        w : [H, W] linear filter
    """
    l1 = sparsity_factor * w.abs().sum()
    dh = (w[1:, :] - w[:-1, :]).pow(2).sum()
    dw = (w[:, 1:] - w[:, :-1]).pow(2).sum()
    smooth = smoothness_factor * (dh + dw)
    return l1 + smooth
