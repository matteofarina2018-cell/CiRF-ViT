"""Poisson NLL plus the three regularisation terms of the CNN baseline.

    L = Poisson NLL
        + lambda_smooth  * Laplacian smoothness of the conv kernels
        + lambda_spatial * distance-weighted penalty on the readout mask u
        + lambda_feature * L1 on the readout feature weights v
"""

import torch
import torch.nn.functional as F


def poisson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """mean(y_pred - y_true * log(y_pred)); y_pred must be > 0."""
    return (y_pred - y_true * torch.log(y_pred + 1e-8)).mean()


# Laplacian used by the reference implementation
_LAPLACIAN = torch.tensor([
    [0.25, 0.50, 0.25],
    [0.50, -3.0, 0.50],
    [0.25, 0.50, 0.25],
], dtype=torch.float32)


def laplacian_penalty(kernel: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness penalty on a conv kernel.

    Per kernel: ||laplacian * k||^2 / (eps + ||k||^2), summed over kernels, so
    the penalty is scale-free.

    kernel: [out_ch, in_ch, K, K]
    """
    out_ch, in_ch, kh, kw = kernel.shape
    lap = _LAPLACIAN.to(kernel.device).view(1, 1, 3, 3)
    k   = kernel.reshape(out_ch * in_ch, 1, kh, kw)
    k_lap = F.conv2d(k, lap, padding=1)                       # [out*in, 1, K, K]
    num = (k_lap ** 2).sum(dim=(1, 2, 3))
    den = 1e-8 + (k ** 2).sum(dim=(1, 2, 3))
    return (num / den).sum()


def l1_penalty(p: torch.Tensor) -> torch.Tensor:
    """L1 sparsity penalty."""
    return p.abs().sum()


def total_loss(y_pred, y_true, model, cfg):
    """Full objective; returns (total loss, Poisson term)."""
    pois = poisson_loss(y_pred, y_true)
    smooth = sum(laplacian_penalty(w) for w in model.conv_kernels())
    reg = (
        cfg['LAMBDA_SMOOTH']    * smooth
        + cfg['LAMBDA_SPATIAL'] * model.readout.distance_penalty()
        + cfg['LAMBDA_FEATURE'] * l1_penalty(model.readout.v)
    )
    return pois + reg, pois
