"""Multi-layer convolutional core: conv -> softplus -> BatchNorm per layer.

Padding is 'valid', so the feature map shrinks by (k - 1) at every layer and the
readout works on the interior of the frame.

Input:  [B, 1, P, P]
Output: [B, channels[-1], P - sum(k - 1), P - sum(k - 1)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCNNCore(nn.Module):
    """`kernels` and `channels` give one entry per layer."""

    def __init__(self, in_channels=1, kernels=(13, 3), channels=(8, 16), bn_momentum=0.02):
        super().__init__()
        if isinstance(channels, int):
            channels = (channels,) * len(kernels)
        assert len(channels) == len(kernels), (channels, kernels)
        self.out_channels = channels[-1]

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        c_in = in_channels
        for k, c_out in zip(kernels, channels):
            conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=1, padding=0, bias=True)
            nn.init.trunc_normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.zeros_(conv.bias)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm2d(c_out, momentum=bn_momentum))
            c_in = c_out

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = bn(F.softplus(conv(x)))
        return x

    def kernel_weights(self):
        """Kernel tensors of every layer (for the smoothness penalty)."""
        return [conv.weight for conv in self.convs]
