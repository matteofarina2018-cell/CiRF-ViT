# PyTorch reimplementation of the Linear-Nonlinear (LN) baseline, which the
# reference work implements in TensorFlow/Keras. One model per cell:
#
#   crop(image around RF centre)  →  linear filter (no bias, ‖w‖≤1)
#     →  parametric softplus  →  predicted firing rate
#
# The linear filter starts from a truncated normal. The nonlinearity is a
# parametric softplus with learnable alpha (output scale), beta (input gain) and
# gamma (threshold/bias), floored at a small value to keep the Poisson loss
# stable.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricSoftplus(nn.Module):
    """f(x) = alpha * softplus(beta * x + gamma) + minimum   (all scalars learned)."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, minimum=1.0e-8):
        super().__init__()
        self.alpha   = nn.Parameter(torch.tensor(float(alpha)))
        self.beta    = nn.Parameter(torch.tensor(float(beta)))
        self.gamma   = nn.Parameter(torch.tensor(float(gamma)))
        self.minimum = float(minimum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * F.softplus(self.beta * x + self.gamma) + self.minimum


class LNModel(nn.Module):
    """Single-cell LN model over a CROP_SIZE×CROP_SIZE window centred on the RF.

    Input : [B, crop, crop]   (image already cropped around the cell's RF centre)
    Output: [B]               (predicted firing rate, > 0)
    """

    def __init__(self, crop_size: int, mean_rate: float = None):
        super().__init__()
        self.crop_size = int(crop_size)

        w = torch.empty(self.crop_size, self.crop_size)
        nn.init.trunc_normal_(w, std=0.05)      # inside the ‖w‖≤1 ball
        self.filter = nn.Parameter(w)

        # init the nonlinearity so that at zero drive the output ≈ mean firing rate
        gamma0 = 0.0
        if mean_rate is not None:
            mr = max(float(mean_rate), 1e-3)
            gamma0 = float(torch.log(torch.expm1(torch.tensor(mr))))   # softplus^{-1}(mr)
        self.act = ParametricSoftplus(alpha=1.0, beta=1.0, gamma=gamma0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = (x * self.filter).sum(dim=(-2, -1))     # [B] linear projection
        return self.act(lin)                           # [B] rate

    @torch.no_grad()
    def clamp_filter_norm(self, max_value: float = 1.0):
        """MaxNorm constraint: rescale the filter so ‖w‖ ≤ max_value."""
        n = self.filter.norm()
        if n > max_value:
            self.filter.mul_(max_value / n)
