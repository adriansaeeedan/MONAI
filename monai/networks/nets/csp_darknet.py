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
Dimension-agnostic CSPDarkNet backbone for 2D and 3D object detection.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from monai.networks.blocks.yolox_blocks import (
    YOLOXBaseConv,
    YOLOXCSPLayer,
    YOLOXDWConv,
    YOLOXFocus,
    YOLOXSPPBottleneck,
)

__all__ = ["CSPDarkNet"]


class CSPDarkNet(nn.Module):
    """Dimension-agnostic CSPDarkNet backbone.

    Implements the CSP (Cross-Stage Partial) DarkNet backbone used in YOLOX,
    supporting both 2D and 3D inputs.

    The network produces multi-scale feature maps at four downsampling levels
    (``dark2`` through ``dark5``). The default output features are from
    ``dark3``, ``dark4``, and ``dark5``, corresponding to strides 8, 16, and 32
    relative to the input.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        depth_mul: depth multiplier that scales the number of CSP blocks.
            Common values: 0.33 (tiny/nano), 0.67 (small), 1.0 (medium), 1.33 (large).
        width_mul: width multiplier that scales the number of channels.
            Common values: 0.25 (tiny), 0.375 (nano), 0.5 (small), 0.75 (medium), 1.0 (large).
        in_channels: number of input image channels. Defaults to 1 (grayscale / single-channel).
        out_features: names of the feature maps to return. Must be a non-empty
            subset of ``("dark2", "dark3", "dark4", "dark5")``.
            Defaults to ``("dark3", "dark4", "dark5")``.
        depthwise: if True, use depthwise separable convolutions in CSP layers.
            Defaults to False.
        act: activation function name. One of ``"silu"``, ``"relu"``, ``"lrelu"``.
            Defaults to ``"silu"``.

    Example:

        .. code-block:: python

            import torch
            from monai.networks.nets.csp_darknet import CSPDarkNet

            # 2D backbone, small variant (depth=0.33, width=0.50)
            backbone_2d = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.50)
            x2d = torch.randn(2, 1, 256, 256)
            feat2d = backbone_2d(x2d)  # {"dark3": ..., "dark4": ..., "dark5": ...}

            # 3D backbone, medium variant
            backbone_3d = CSPDarkNet(spatial_dims=3, depth_mul=1.0, width_mul=1.0)
            x3d = torch.randn(2, 1, 64, 64, 64)
            feat3d = backbone_3d(x3d)
    """

    def __init__(
        self,
        spatial_dims: int,
        depth_mul: float,
        width_mul: float,
        in_channels: int = 1,
        out_features: Sequence[str] = ("dark3", "dark4", "dark5"),
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()

        if not out_features:
            raise ValueError("out_features must not be empty.")
        valid = {"dark2", "dark3", "dark4", "dark5"}
        invalid = set(out_features) - valid
        if invalid:
            raise ValueError(f"Invalid out_features {invalid}. Valid names: {valid}.")

        self.out_features = tuple(out_features)
        ConvBlock = YOLOXDWConv if depthwise else YOLOXBaseConv

        base_ch = int(width_mul * 64)      # base channel count at first block
        base_d = max(round(depth_mul * 3), 1)  # base number of CSP blocks

        # Stem: Focus halves all spatial dims, expands channels
        self.stem = YOLOXFocus(spatial_dims, in_channels, base_ch, ksize=3, act=act)

        # dark2: stride-2 conv → CSP
        self.dark2 = nn.Sequential(
            ConvBlock(spatial_dims, base_ch, base_ch * 2, ksize=3, stride=2, act=act),
            YOLOXCSPLayer(spatial_dims, base_ch * 2, base_ch * 2, n=base_d, depthwise=depthwise, act=act),
        )

        # dark3: stride-2 conv → CSP  (output for large-stride, small objects)
        self.dark3 = nn.Sequential(
            ConvBlock(spatial_dims, base_ch * 2, base_ch * 4, ksize=3, stride=2, act=act),
            YOLOXCSPLayer(spatial_dims, base_ch * 4, base_ch * 4, n=base_d * 3, depthwise=depthwise, act=act),
        )

        # dark4: stride-2 conv → CSP  (output for medium objects)
        self.dark4 = nn.Sequential(
            ConvBlock(spatial_dims, base_ch * 4, base_ch * 8, ksize=3, stride=2, act=act),
            YOLOXCSPLayer(spatial_dims, base_ch * 8, base_ch * 8, n=base_d * 3, depthwise=depthwise, act=act),
        )

        # dark5: stride-2 conv → SPP → CSP  (output for small-stride, large objects)
        self.dark5 = nn.Sequential(
            ConvBlock(spatial_dims, base_ch * 8, base_ch * 16, ksize=3, stride=2, act=act),
            YOLOXSPPBottleneck(spatial_dims, base_ch * 16, base_ch * 16, activation=act),
            YOLOXCSPLayer(
                spatial_dims, base_ch * 16, base_ch * 16, n=base_d, shortcut=False, depthwise=depthwise, act=act
            ),
        )

        # expose for downstream (neck/head can query this)
        self.out_channels: dict[str, int] = {
            "dark2": base_ch * 2,
            "dark3": base_ch * 4,
            "dark4": base_ch * 8,
            "dark5": base_ch * 16,
        }

    def forward(self, x: nn.Module) -> dict[str, nn.Module]:  # type: ignore[override]
        """
        Args:
            x: input tensor, shape (B, in_channels, H, W) or (B, in_channels, D, H, W).

        Returns:
            A dict mapping feature name → tensor for each name in ``self.out_features``.
            Feature map strides relative to input (approximate):
            ``dark2`` → 4, ``dark3`` → 8, ``dark4`` → 16, ``dark5`` → 32.
        """
        outputs: dict[str, nn.Module] = {}
        x = self.stem(x)  # type: ignore[assignment]

        x = self.dark2(x)  # type: ignore[assignment]
        if "dark2" in self.out_features:
            outputs["dark2"] = x

        x = self.dark3(x)  # type: ignore[assignment]
        if "dark3" in self.out_features:
            outputs["dark3"] = x

        x = self.dark4(x)  # type: ignore[assignment]
        if "dark4" in self.out_features:
            outputs["dark4"] = x

        x = self.dark5(x)  # type: ignore[assignment]
        if "dark5" in self.out_features:
            outputs["dark5"] = x

        return outputs
