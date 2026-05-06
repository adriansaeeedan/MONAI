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
Dimension-agnostic YOLOX detection network for 2D and 3D object detection.

This module provides:

- :class:`YOLOXHeadModule` — the per-level decoupled detection head.
- :class:`YOLOPAFPN` — Path Aggregation FPN neck that wraps :class:`CSPDarkNet`.
- :class:`YOLOXNetwork` — the complete network (backbone + neck + head) that
  follows the same interface as :class:`~monai.apps.detection.networks.retinanet_network.RetinaNet`.
- :func:`yolox_darknet_pafpn_network` — convenience factory.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from monai.networks.blocks.yolox_blocks import YOLOXBaseConv, YOLOXCSPLayer, YOLOXDWConv
from monai.networks.layers.factories import Conv
from monai.networks.nets.csp_darknet import CSPDarkNet
from monai.utils import ensure_tuple_rep, look_up_option

__all__ = ["YOLOPAFPN", "YOLOXHeadModule", "YOLOXNetwork", "yolox_darknet_pafpn_network"]


# ---------------------------------------------------------------------------
# YOLO Path Aggregation FPN (neck)
# ---------------------------------------------------------------------------


class YOLOPAFPN(nn.Module):
    """Dimension-agnostic Path Aggregation FPN neck for YOLOX.

    Wraps a :class:`CSPDarkNet` backbone and builds top-down + bottom-up
    feature aggregation paths, producing three feature maps at strides 8, 16,
    and 32 relative to the network input.

    The ``out_channels`` attribute exposes the number of channels per output
    feature map, which is needed by downstream head and neck modules.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        depth: depth multiplier for CSP blocks. Defaults to 1.0.
        width: width multiplier for channel counts. Defaults to 1.0.
        in_channels: number of input image channels. Defaults to 1.
        in_features: backbone feature names to aggregate, from coarse to fine.
            Defaults to ``("dark3", "dark4", "dark5")``.
        backbone_channels: base channel counts for the three backbone features
            (before width scaling). Defaults to ``[256, 512, 1024]``.
        depthwise: use depthwise separable convolutions. Defaults to False.
        act: activation function name. Defaults to ``"silu"``.
    """

    def __init__(
        self,
        spatial_dims: int,
        depth: float = 1.0,
        width: float = 1.0,
        in_channels: int = 1,
        in_features: Sequence[str] = ("dark3", "dark4", "dark5"),
        backbone_channels: Sequence[int] = (256, 512, 1024),
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.in_features = tuple(in_features)
        ch = [int(c * width) for c in backbone_channels]  # scaled channels

        self.backbone = CSPDarkNet(
            spatial_dims=spatial_dims,
            depth_mul=depth,
            width_mul=width,
            in_channels=in_channels,
            out_features=in_features,
            depthwise=depthwise,
            act=act,
        )

        ConvBlock: Callable = YOLOXDWConv if depthwise else YOLOXBaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down path: dark5 (stride 32) → dark4 (stride 16)
        self.lateral_conv0 = YOLOXBaseConv(spatial_dims, ch[2], ch[1], ksize=1, stride=1, act=act)
        self.C3_p4 = YOLOXCSPLayer(
            spatial_dims, 2 * ch[1], ch[1], n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )

        # Top-down path: dark4 (stride 16) → dark3 (stride 8)
        self.reduce_conv1 = YOLOXBaseConv(spatial_dims, ch[1], ch[0], ksize=1, stride=1, act=act)
        self.C3_p3 = YOLOXCSPLayer(
            spatial_dims, 2 * ch[0], ch[0], n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )

        # Bottom-up path: dark3 (stride 8) → dark4 (stride 16)
        self.bu_conv2 = ConvBlock(spatial_dims, ch[0], ch[0], ksize=3, stride=2, act=act)
        self.C3_n3 = YOLOXCSPLayer(
            spatial_dims, 2 * ch[0], ch[1], n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )

        # Bottom-up path: dark4 (stride 16) → dark5 (stride 32)
        self.bu_conv1 = ConvBlock(spatial_dims, ch[1], ch[1], ksize=3, stride=2, act=act)
        self.C3_n4 = YOLOXCSPLayer(
            spatial_dims, 2 * ch[1], ch[2], n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )

        # Each output level has ch[0] channels (the PAFPN normalises to ch[0])
        self.out_channels: int = ch[0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: input image tensor, shape (B, C, H, W) or (B, C, D, H, W).

        Returns:
            A tuple ``(p3, p4, p5)`` of feature maps:
            - ``p3``: stride 8, shape (..., H/8, W/8[, D/8]).
            - ``p4``: stride 16.
            - ``p5``: stride 32.
        """
        out_feats = self.backbone(x)
        x2, x1, x0 = [out_feats[f] for f in self.in_features]  # coarse → fine

        # Top-down
        fpn_out0 = self.lateral_conv0(x0)          # ch[2] → ch[1]
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], dim=1)
        f_out0 = self.C3_p4(f_out0)                # stride 16

        fpn_out1 = self.reduce_conv1(f_out0)        # ch[1] → ch[0]
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], dim=1)
        pan_out2 = self.C3_p3(f_out1)              # stride 8 (p3)

        # Bottom-up
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)              # stride 16 (p4)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)              # stride 32 (p5)

        return pan_out2, pan_out1, pan_out0


# ---------------------------------------------------------------------------
# Decoupled detection head
# ---------------------------------------------------------------------------


class YOLOXHeadModule(nn.Module):
    """Dimension-agnostic YOLOX decoupled detection head.

    For each FPN level, independently processes classification, box regression,
    and objectness predictions through separate branches. This is the
    "decoupled head" design from YOLOX (as opposed to the coupled head in
    earlier YOLO variants).

    The head outputs *raw* (un-decoded) predictions. Decoding (converting
    offsets + log-sizes to absolute centre-size boxes) is the responsibility
    of :class:`YOLOXDetector`.

    Output channels per level:
    - ``cls_key``: (B, num_classes, *spatial)
    - ``box_reg_key``: (B, 2*spatial_dims, *spatial) —
      first ``spatial_dims`` channels are raw centre offsets (before adding
      the grid and multiplying by stride), last ``spatial_dims`` channels
      are raw log-size predictions.
    - ``obj_key``: (B, 1, *spatial)

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        num_classes: number of object categories (excluding background).
        width: channel width multiplier applied to the 256-channel base.
            Defaults to 1.0.
        in_channels: list of input channel counts, one per FPN level.
            Defaults to ``[256, 512, 1024]``.
        act: activation function name. Defaults to ``"silu"``.
        depthwise: use depthwise separable convolutions in classification and
            regression branches. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_classes: int,
        width: float = 1.0,
        in_channels: Sequence[int] = (256, 512, 1024),
        act: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        n_levels = len(in_channels)
        hidden = int(256 * width)

        ConvBlock: Callable = YOLOXDWConv if depthwise else YOLOXBaseConv
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for i in range(n_levels):
            in_ch = int(in_channels[i] * width)
            self.stems.append(YOLOXBaseConv(spatial_dims, in_ch, hidden, ksize=1, stride=1, act=act))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBlock(spatial_dims, hidden, hidden, ksize=3, stride=1, act=act),
                    ConvBlock(spatial_dims, hidden, hidden, ksize=3, stride=1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBlock(spatial_dims, hidden, hidden, ksize=3, stride=1, act=act),
                    ConvBlock(spatial_dims, hidden, hidden, ksize=3, stride=1, act=act),
                )
            )
            self.cls_preds.append(conv_type(hidden, num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(conv_type(hidden, 2 * spatial_dims, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(conv_type(hidden, 1, kernel_size=1, stride=1, padding=0))

    def initialize_biases(self, prior_prob: float = 0.01) -> None:
        """Initialise classification and objectness prediction biases.

        Sets biases to ``-log((1 - prior_prob) / prior_prob)`` so that the
        initial sigmoid output equals ``prior_prob`` (typically 0.01), which
        improves training stability.

        Args:
            prior_prob: initial probability for foreground predictions.
                Defaults to 0.01.
        """
        if not math.isfinite(prior_prob) or prior_prob <= 0.0 or prior_prob >= 1.0:
            raise ValueError(f"prior_prob must be finite and in the open interval (0, 1), got {prior_prob}.")
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_pred, obj_pred in zip(self.cls_preds, self.obj_preds):
            nn.init.constant_(cls_pred.bias, bias_value)
            nn.init.constant_(obj_pred.bias, bias_value)

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """
        Args:
            features: list of FPN feature maps, one per level.
                Each tensor has shape (B, in_channels[i], *spatial).

        Returns:
            Three lists, one per FPN level:
            - ``cls_outputs``: list of (B, num_classes, *spatial) tensors.
            - ``reg_outputs``: list of (B, 2*spatial_dims, *spatial) tensors.
            - ``obj_outputs``: list of (B, 1, *spatial) tensors.
        """
        cls_outputs: list[Tensor] = []
        reg_outputs: list[Tensor] = []
        obj_outputs: list[Tensor] = []

        for k, feat in enumerate(features):
            x = self.stems[k](feat)

            cls_feat = self.cls_convs[k](x)
            cls_out = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            reg_out = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)

            if not torch.compiler.is_compiling():
                for name, t in [("cls", cls_out), ("reg", reg_out), ("obj", obj_out)]:
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        if torch.is_grad_enabled():
                            raise ValueError(f"YOLOX head {name} output is NaN or Inf at level {k}.")
                        warnings.warn(f"YOLOX head {name} output is NaN or Inf at level {k}.")

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            obj_outputs.append(obj_out)

        return cls_outputs, reg_outputs, obj_outputs


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------


class YOLOXNetwork(nn.Module):
    """YOLOX detection network: backbone + PAFPN neck + decoupled head.

    Takes a batched image tensor as input and outputs raw per-level prediction
    maps grouped by type. This is the network component that holds all
    trainable parameters. Post-processing (grid generation, decoding,
    loss computation, and NMS) is handled by :class:`YOLOXDetector`.

    Output format (``use_list_output=False``):
    A dictionary with three keys:
    - ``self.cls_key``: list of (B, num_classes, *spatial) tensors.
    - ``self.box_reg_key``: list of (B, 2*spatial_dims, *spatial) tensors.
    - ``self.obj_key``: list of (B, 1, *spatial) tensors.

    Output format (``use_list_output=True``):
    A list of 3N tensors: first N cls, next N reg, last N obj.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        num_classes: number of object categories (excluding background).
        in_channels: number of input image channels. Defaults to 1.
        depth: depth multiplier for backbone and neck. Defaults to 1.0.
        width: width multiplier for channel counts. Defaults to 1.0.
        strides: output strides of the three FPN levels.
            Defaults to ``(8, 16, 32)``.
        depthwise: use depthwise separable convolutions. Defaults to False.
        act: activation function. Defaults to ``"silu"``.
        size_divisible: input spatial size must be divisible by this.
            Defaults to 32.
        use_list_output: output a list instead of a dict. Defaults to False.

    Example:

        .. code-block:: python

            import torch
            from monai.apps.detection.networks.yolox_network import YOLOXNetwork

            model = YOLOXNetwork(spatial_dims=2, num_classes=10)
            out = model(torch.randn(2, 1, 256, 256))
            print(out["classification"][0].shape)   # (2, 10, 32, 32)
            print(out["box_regression"][0].shape)   # (2, 4, 32, 32)
            print(out["objectness"][0].shape)        # (2, 1, 32, 32)
    """

    def __init__(
        self,
        spatial_dims: int,
        num_classes: int,
        in_channels: int = 1,
        depth: float = 1.0,
        width: float = 1.0,
        strides: Sequence[int] = (8, 16, 32),
        depthwise: bool = False,
        act: str = "silu",
        size_divisible: Sequence[int] | int = 32,
        use_list_output: bool = False,
    ) -> None:
        super().__init__()

        self.spatial_dims: int = look_up_option(spatial_dims, supported=[2, 3])
        self.num_classes = num_classes
        self.strides: tuple[int, ...] = tuple(strides)
        self.size_divisible = ensure_tuple_rep(size_divisible, self.spatial_dims)
        self.use_list_output = use_list_output

        # Output key names (mirrors RetinaNet convention)
        self.cls_key: str = "classification"
        self.box_reg_key: str = "box_regression"
        self.obj_key: str = "objectness"

        # Base backbone channels at dark3/4/5 (before width scaling).
        # The PAFPN outputs levels at [int(256*w), int(512*w), int(1024*w)] channels.
        # The head receives the UNSCALED base channel counts and applies width internally,
        # matching the original YOLOX architecture.
        base_channels = [256, 512, 1024]

        self.feature_extractor = YOLOPAFPN(
            spatial_dims=self.spatial_dims,
            depth=depth,
            width=width,
            in_channels=in_channels,
            backbone_channels=base_channels,
            depthwise=depthwise,
            act=act,
        )
        self.out_channels: int = self.feature_extractor.out_channels  # for detectors

        self.head = YOLOXHeadModule(
            spatial_dims=self.spatial_dims,
            num_classes=num_classes,
            width=width,
            in_channels=base_channels,  # unscaled; head applies width internally
            act=act,
            depthwise=depthwise,
        )
        self.head.initialize_biases(prior_prob=0.01)

    def forward(self, images: Tensor) -> Any:
        """
        Args:
            images: (B, in_channels, H, W) or (B, in_channels, D, H, W).

        Returns:
            If ``self.use_list_output`` is False: a dict with keys
            ``self.cls_key``, ``self.box_reg_key``, ``self.obj_key``.
            If True: a flat list ``[cls_0, ..., cls_N, reg_0, ..., reg_N, obj_0, ..., obj_N]``.
        """
        features = self.feature_extractor(images)  # tuple of N feature maps
        cls_maps, reg_maps, obj_maps = self.head(list(features))

        if not self.use_list_output:
            return {
                self.cls_key: cls_maps,
                self.box_reg_key: reg_maps,
                self.obj_key: obj_maps,
            }
        return cls_maps + reg_maps + obj_maps


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def yolox_darknet_pafpn_network(
    spatial_dims: int,
    num_classes: int,
    model_size: str = "s",
    in_channels: int = 1,
    size_divisible: Sequence[int] | int = 32,
) -> YOLOXNetwork:
    """Convenience factory for standard YOLOX model sizes.

    Constructs a :class:`YOLOXNetwork` with depth and width multipliers
    matching the standard YOLOX-S/M/L/X variants.

    Args:
        spatial_dims: 2 or 3.
        num_classes: number of object categories.
        model_size: one of ``"nano"``, ``"tiny"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.
            Defaults to ``"s"``.
        in_channels: input image channels. Defaults to 1.
        size_divisible: spatial size divisibility requirement. Defaults to 32.

    Returns:
        A :class:`YOLOXNetwork` configured for the requested model size.

    Example:

        .. code-block:: python

            from monai.apps.detection.networks.yolox_network import yolox_darknet_pafpn_network
            model = yolox_darknet_pafpn_network(spatial_dims=2, num_classes=5, model_size="s")
    """
    configs: dict[str, tuple[float, float]] = {
        "nano": (0.33, 0.25),
        "tiny": (0.33, 0.375),
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.33, 1.25),
    }
    if model_size not in configs:
        raise ValueError(f"model_size '{model_size}' not recognised. Choose from {list(configs)}.")
    depth, width = configs[model_size]

    return YOLOXNetwork(
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        in_channels=in_channels,
        depth=depth,
        width=width,
        size_divisible=size_divisible,
    )
