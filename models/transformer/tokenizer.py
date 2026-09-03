"""Convolutional tokeniser: the stem that turns an image into tokens.

A handful of ordinary convolutions instead of a large-stride patch embedding.
The stride is carried by the first layer only, so the feature map is
ceil(P / stride) on a side and every later layer keeps that resolution.

Input:  [B, in_channels, P, P]
Output: [B, cnn_dim, ceil(P/stride), ceil(P/stride)]
"""

import math
import torch
import torch.nn as nn


class CNNTokenizer(nn.Module):
    """Conv2d -> BatchNorm2d -> GELU (-> Dropout2d) per layer."""

    def __init__(
        self,
        in_channels: int,
        cnn_dim:     int,
        cnn_layers:  int,
        cnn_kernel:  int,
        dropout:     float = 0.0,
        stride:      int   = 1,
    ):
        super().__init__()

        assert cnn_kernel % 2 == 1, \
            f"cnn_kernel must be odd for same-padding, got {cnn_kernel}"
        assert stride >= 1, f"stride must be >= 1, got {stride}"

        padding = cnn_kernel // 2

        layers = []
        in_ch  = in_channels
        for i in range(cnn_layers):
            s = stride if i == 0 else 1   # only the first layer strides
            layers.append(nn.Conv2d(in_ch, cnn_dim, kernel_size=cnn_kernel,
                                    padding=padding, stride=s))
            layers.append(nn.BatchNorm2d(cnn_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = cnn_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def feat_size(img_size: int, stride: int) -> int:
    """Side of the feature map for an img_size x img_size input."""
    return math.ceil(img_size / stride)
