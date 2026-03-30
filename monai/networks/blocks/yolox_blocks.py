# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# Adapted from https://github.com/Megvii-BaseDetection/YOLOX
# which has the following license:
# Apache License 2.0
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE
"""
Dimension-agnostic YOLOX building blocks for 2D and 3D object detection.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from monai.networks.layers.factories import Conv

__all__ = [
    "YOLOXBaseConv",
    "YOLOXDWConv",
    "YOLOXBottleneck",
    "YOLOXCSPLayer",
    "YOLOXSPPBottleneck",
    "YOLOXFocus",
]


def _get_batch_norm(spatial_dims: int, num_channels: int) -> nn.Module:
    """Return the appropriate BatchNorm module for the given spatial dimensionality."""
    if spatial_dims == 2:
        return nn.BatchNorm2d(num_channels)
    if spatial_dims == 3:
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")


def _get_activation(name: str = "silu") -> nn.Module:
    """Return an activation module by name.

    Args:
        name: one of ``"silu"``, ``"relu"``, ``"lrelu"``.
    """
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation '{name}'. Supported: silu, relu, lrelu.")


class YOLOXBaseConv(nn.Module):
    """Dimension-agnostic Conv → BatchNorm → Activation block.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        ksize: convolution kernel size.
        stride: convolution stride.
        groups: number of groups for grouped convolution. Defaults to 1.
        bias: whether to include a bias term. Defaults to False.
        act: activation function name. One of ``"silu"``, ``"relu"``, ``"lrelu"``.
            Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        self.conv = conv_type(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = _get_batch_norm(spatial_dims, out_channels)
        self.act = _get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x: Tensor) -> Tensor:
        """Forward pass fusing conv and BN (for export/inference without separate BN)."""
        return self.act(self.conv(x))


class YOLOXDWConv(nn.Module):
    """Dimension-agnostic depthwise separable convolution.

    Consists of a depthwise convolution followed by a pointwise (1×1) convolution,
    each with BatchNorm and activation.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        ksize: depthwise convolution kernel size.
        stride: depthwise convolution stride. Defaults to 1.
        act: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.dconv = YOLOXBaseConv(
            spatial_dims, in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act
        )
        self.pconv = YOLOXBaseConv(spatial_dims, in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.pconv(self.dconv(x))


class YOLOXBottleneck(nn.Module):
    """Standard YOLOX bottleneck block.

    A 1×...×1 pointwise convolution followed by a 3×...×3 convolution,
    with an optional residual shortcut when ``in_channels == out_channels``.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        shortcut: whether to add a residual shortcut. Defaults to True.
        expansion: channel expansion ratio for the hidden (bottleneck) channels.
            Defaults to 0.5.
        depthwise: if True, use :class:`YOLOXDWConv` for the 3×...×3 conv.
            Defaults to False.
        act: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        ConvBlock = YOLOXDWConv if depthwise else YOLOXBaseConv
        self.conv1 = YOLOXBaseConv(spatial_dims, in_channels, hidden, ksize=1, stride=1, act=act)
        self.conv2 = ConvBlock(spatial_dims, hidden, out_channels, ksize=3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class YOLOXCSPLayer(nn.Module):
    """Cross-Stage Partial (CSP) layer with YOLOX Bottleneck blocks.

    Splits input channels into two branches: one passes through ``n`` Bottleneck
    blocks, the other is a direct projection. The outputs are concatenated and
    projected to ``out_channels``. This is the C3 module from YOLOv5.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        n: number of Bottleneck blocks. Defaults to 1.
        shortcut: whether Bottleneck blocks include residual shortcuts.
            Defaults to True.
        expansion: channel expansion ratio. Defaults to 0.5.
        depthwise: if True, use depthwise separable convolutions inside
            Bottleneck blocks. Defaults to False.
        act: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = YOLOXBaseConv(spatial_dims, in_channels, hidden, ksize=1, stride=1, act=act)
        self.conv2 = YOLOXBaseConv(spatial_dims, in_channels, hidden, ksize=1, stride=1, act=act)
        self.conv3 = YOLOXBaseConv(spatial_dims, 2 * hidden, out_channels, ksize=1, stride=1, act=act)
        self.m = nn.Sequential(
            *[YOLOXBottleneck(spatial_dims, hidden, hidden, shortcut, 1.0, depthwise, act) for _ in range(n)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class YOLOXSPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling (SPP) bottleneck.

    Applies multiple max-pooling kernels and concatenates the results,
    enabling multi-scale context aggregation without changing spatial size.
    Based on the SPP layer used in YOLOv3-SPP.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_sizes: pooling kernel sizes. Defaults to (5, 9, 13).
        activation: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = YOLOXBaseConv(spatial_dims, in_channels, hidden, ksize=1, stride=1, act=activation)
        pool_cls: type[nn.Module] = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        self.m = nn.ModuleList(
            [pool_cls(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]  # type: ignore[operator]
        )
        conv2_in = hidden * (len(kernel_sizes) + 1)
        self.conv2 = YOLOXBaseConv(spatial_dims, conv2_in, out_channels, ksize=1, stride=1, act=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


class YOLOXFocus(nn.Module):
    """Focus spatial information into channel space.

    For 2D: slices input into 4 quadrant patches (halving H and W),
    concatenates along the channel dimension (4×C channels), then applies
    a :class:`YOLOXBaseConv` to reduce channels.

    For 3D: slices input into 8 octant patches (halving D, H, and W),
    concatenates along the channel dimension (8×C channels), then applies
    a :class:`YOLOXBaseConv` to reduce channels.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        in_channels: number of input channels.
        out_channels: number of output channels.
        ksize: convolution kernel size for the output projection. Defaults to 1.
        stride: convolution stride for the output projection. Defaults to 1.
        act: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        factor = 2**spatial_dims  # 4 for 2D, 8 for 3D
        self.conv = YOLOXBaseConv(spatial_dims, in_channels * factor, out_channels, ksize=ksize, stride=stride, act=act)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor, shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.

        Returns:
            Tensor of shape (B, out_channels, H/2, W/2) for 2D
            or (B, out_channels, D/2, H/2, W/2) for 3D.
        """
        if self.spatial_dims == 2:
            patches = [
                x[..., ::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, ::2],
                x[..., 1::2, 1::2],
            ]
        else:
            patches = [
                x[..., ::2, ::2, ::2],
                x[..., ::2, ::2, 1::2],
                x[..., ::2, 1::2, ::2],
                x[..., ::2, 1::2, 1::2],
                x[..., 1::2, ::2, ::2],
                x[..., 1::2, ::2, 1::2],
                x[..., 1::2, 1::2, ::2],
                x[..., 1::2, 1::2, 1::2],
            ]
        return self.conv(torch.cat(patches, dim=1))
