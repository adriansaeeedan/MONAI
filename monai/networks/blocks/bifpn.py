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

"""
BiFPN: Bidirectional Feature Pyramid Network.

Based on "EfficientDet: Scalable and Efficient Object Detection"
by Tan et al. (https://arxiv.org/abs/1911.09070).

This implementation supports both 2D and 3D images and is designed as a
drop-in alternative to :class:`~monai.networks.blocks.FeaturePyramidNetwork`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from monai.networks.layers.factories import Conv, Norm

from .feature_pyramid_network import ExtraFPNBlock

__all__ = ["FastNormalizedFusion", "BiFPNLayer", "BiFPN"]


class FastNormalizedFusion(nn.Module):
    """
    Fast normalized feature fusion from BiFPN.

    Each input feature map is assigned a learned scalar weight that is kept
    non-negative via ReLU. The weights are normalized by their sum plus a small
    ``epsilon`` for numerical stability, then applied to the inputs before
    summation.

    This is the "fast normalized fusion" variant from the EfficientDet paper,
    preferred over softmax-based fusion for computational efficiency.

    Reference: https://arxiv.org/abs/1911.09070 (Equation 4)

    Args:
        num_inputs: number of input feature maps to fuse (must be >= 2).
        epsilon: small constant added to the weight sum for numerical stability.

    Examples::

        >>> fusion = FastNormalizedFusion(num_inputs=2)
        >>> a = torch.rand(1, 8, 16, 16)
        >>> b = torch.rand(1, 8, 16, 16)
        >>> out = fusion([a, b])
        >>> out.shape
        torch.Size([1, 8, 16, 16])
    """

    def __init__(self, num_inputs: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        if num_inputs < 2:
            raise ValueError(f"num_inputs must be >= 2, got {num_inputs}")
        self.num_inputs = num_inputs
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """
        Args:
            inputs: list of feature tensors to fuse, all with the same shape.

        Returns:
            weighted sum of inputs with learned normalized weights.
        """
        w = F.relu(self.weights)
        w = w / (w.sum() + self.epsilon)
        out = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i] * w[i]
        return out


def _make_bifpn_node_conv(spatial_dims: int, out_channels: int, depthwise_separable: bool) -> nn.Module:
    """Create a conv + BN + SiLU block for a BiFPN fusion node."""
    conv_type: Callable = Conv[Conv.CONV, spatial_dims]
    norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
    if depthwise_separable:
        return nn.Sequential(
            conv_type(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            conv_type(out_channels, out_channels, 1, bias=False),
            norm_type(out_channels),
            nn.SiLU(),
        )
    return nn.Sequential(
        conv_type(out_channels, out_channels, 3, padding=1, bias=False),
        norm_type(out_channels),
        nn.SiLU(),
    )


class BiFPNLayer(nn.Module):
    """
    A single BiFPN layer implementing one round of bidirectional feature fusion.

    Performs a **top-down** path followed by a **bottom-up** path across all
    ``num_levels`` feature pyramid levels. Each fusion node uses
    :class:`FastNormalizedFusion` (weighted sum with learned weights).

    Node connectivity (using level index 0 = finest, L-1 = coarsest):

    *Top-down path*: produces intermediate feature maps ``td[L-2], ..., td[0]``.

    * ``td[L-1]`` is the coarsest input (passed through unchanged).
    * ``td[l] = conv(fuse2(in[l], upsample(td[l+1])))`` for ``l = L-2 .. 0``.

    *Bottom-up path*: produces output feature maps ``out[0], ..., out[L-1]``.

    * ``out[0] = td[0]`` (finest output, already computed above).
    * ``out[l] = conv(fuse3(in[l], td[l], downsample(out[l-1])))`` for ``l = 1 .. L-2``.
    * ``out[L-1] = conv(fuse2(in[L-1], downsample(out[L-2])))`` (coarsest output).

    Intermediate levels use 3-input fusion (original + top-down + bottom-up),
    while the finest and coarsest levels use 2-input fusion, exactly as in the
    EfficientDet paper.

    Args:
        spatial_dims: 2 or 3 for 2D or 3D images.
        num_levels: number of feature pyramid levels (must be >= 2).
        out_channels: number of channels at every level (must be uniform).
        epsilon: small constant for fast normalized fusion stability.
        depthwise_separable: if ``True``, use depthwise separable convolutions
            in each fusion node (matching the EfficientDet architecture).
    """

    def __init__(
        self,
        spatial_dims: int,
        num_levels: int,
        out_channels: int,
        epsilon: float = 1e-4,
        depthwise_separable: bool = False,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}")

        self.num_levels = num_levels

        # Top-down path: L-1 fusion nodes (2-input) and L-1 conv blocks.
        # Index l corresponds to level l (l = 0 .. L-2).
        self.td_fusions: nn.ModuleList = nn.ModuleList(
            [FastNormalizedFusion(2, epsilon) for _ in range(num_levels - 1)]
        )
        self.td_convs: nn.ModuleList = nn.ModuleList(
            [_make_bifpn_node_conv(spatial_dims, out_channels, depthwise_separable) for _ in range(num_levels - 1)]
        )

        # Bottom-up intermediate path: L-2 fusion nodes (3-input) and L-2 conv blocks.
        # Index i corresponds to level i+1 (i = 0 .. L-3).
        self.bu_int_fusions: nn.ModuleList = nn.ModuleList(
            [FastNormalizedFusion(3, epsilon) for _ in range(num_levels - 2)]
        )
        self.bu_int_convs: nn.ModuleList = nn.ModuleList(
            [_make_bifpn_node_conv(spatial_dims, out_channels, depthwise_separable) for _ in range(num_levels - 2)]
        )

        # Bottom-up coarsest node: 1 fusion (2-input) and 1 conv block for level L-1.
        self.bu_top_fusion = FastNormalizedFusion(2, epsilon)
        self.bu_top_conv = _make_bifpn_node_conv(spatial_dims, out_channels, depthwise_separable)

    def get_result_from_td_fusions(self, inputs: list[Tensor], idx: int) -> Tensor:
        """TorchScript-compatible indexed access for td_fusions ModuleList."""
        out = inputs[0]
        for i, module in enumerate(self.td_fusions):
            if i == idx:
                out = module(inputs)
        return out

    def get_result_from_td_convs(self, x: Tensor, idx: int) -> Tensor:
        """TorchScript-compatible indexed access for td_convs ModuleList."""
        out = x
        for i, module in enumerate(self.td_convs):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_bu_int_fusions(self, inputs: list[Tensor], idx: int) -> Tensor:
        """TorchScript-compatible indexed access for bu_int_fusions ModuleList."""
        out = inputs[0]
        for i, module in enumerate(self.bu_int_fusions):
            if i == idx:
                out = module(inputs)
        return out

    def get_result_from_bu_int_convs(self, x: Tensor, idx: int) -> Tensor:
        """TorchScript-compatible indexed access for bu_int_convs ModuleList."""
        out = x
        for i, module in enumerate(self.bu_int_convs):
            if i == idx:
                out = module(x)
        return out

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        Args:
            features: list of feature tensors ordered from finest (index 0) to
                coarsest (index L-1). All tensors must have ``out_channels``
                channels (i.e., after lateral projection by the parent
                :class:`BiFPN` module).

        Returns:
            list of fused feature tensors with the same ordering and spatial
            shapes as the input.
        """
        L = self.num_levels

        # ---- Top-down pass ------------------------------------------------
        # Initialise with placeholder tensors (overwritten below).
        td_pass: list[Tensor] = []
        for i in range(L):
            td_pass.append(features[i])

        # Coarsest level passes through unchanged.
        td_pass[L - 1] = features[L - 1]

        # Propagate from coarser-1 down to finest.
        for l_idx in range(L - 2, -1, -1):
            upsampled = F.interpolate(td_pass[l_idx + 1], size=features[l_idx].shape[2:], mode="nearest")
            fused = self.get_result_from_td_fusions([features[l_idx], upsampled], l_idx)
            td_pass[l_idx] = self.get_result_from_td_convs(fused, l_idx)

        # ---- Bottom-up pass -----------------------------------------------
        bu_pass: list[Tensor] = []
        for i in range(L):
            bu_pass.append(features[i])

        # Finest level output is the top-down result (already fused with level above).
        bu_pass[0] = td_pass[0]

        # Intermediate levels: 3-input fusion.
        # Use interpolation to match the target spatial shape exactly, mirroring
        # how the top-down path handles non-power-of-2 or irregular feature maps.
        for l_idx in range(1, L - 1):
            downsampled = F.interpolate(bu_pass[l_idx - 1], size=features[l_idx].shape[2:], mode="nearest")
            fused = self.get_result_from_bu_int_fusions([features[l_idx], td_pass[l_idx], downsampled], l_idx - 1)
            bu_pass[l_idx] = self.get_result_from_bu_int_convs(fused, l_idx - 1)

        # Coarsest level output: 2-input fusion.
        downsampled_top = F.interpolate(bu_pass[L - 2], size=features[L - 1].shape[2:], mode="nearest")
        fused_top = self.bu_top_fusion([features[L - 1], downsampled_top])
        bu_pass[L - 1] = self.bu_top_conv(fused_top)

        return bu_pass


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN).

    Adds a BiFPN neck on top of a set of feature maps, as described in
    `"EfficientDet: Scalable and Efficient Object Detection"
    <https://arxiv.org/abs/1911.09070>`_.

    This module is a drop-in alternative to
    :class:`~monai.networks.blocks.FeaturePyramidNetwork`. It accepts an
    ``OrderedDict[str, Tensor]`` of multi-scale feature maps (ordered from
    finest to coarsest resolution) and returns an ``OrderedDict[str, Tensor]``
    of the same keys with fused features at each level.

    Compared to standard FPN, BiFPN adds:

    * **Bidirectional connections** — each BiFPN layer has a top-down pass
      followed by a bottom-up pass, enabling richer cross-scale feature fusion.
    * **Learned weighted fusion** — each fusion node learns scalar weights for
      its inputs via fast normalized fusion, allowing the network to adaptively
      emphasise more useful features.
    * **Stacked layers** — the BiFPN layer can be repeated ``num_repeats``
      times for progressively deeper fusion.

    Args:
        spatial_dims: 2 or 3 for 2D or 3D images.
        in_channels_list: number of input channels for each feature level,
            ordered from finest to coarsest (same convention as
            :class:`~monai.networks.blocks.FeaturePyramidNetwork`).
        out_channels: unified output channel count for all feature levels.
        num_repeats: number of BiFPN layers to stack. EfficientDet-D0 uses 3,
            D7 uses 8. Default: ``3``.
        epsilon: small constant for fast normalized fusion numerical stability.
            Default: ``1e-4``.
        extra_blocks: optional
            :class:`~monai.networks.blocks.feature_pyramid_network.ExtraFPNBlock`
            that appends additional feature levels (e.g.,
            :class:`~monai.networks.blocks.LastLevelMaxPool` or
            :class:`~monai.networks.blocks.LastLevelP6P7`).
        depthwise_separable: if ``True``, use depthwise separable convolutions
            in BiFPN fusion nodes, matching the EfficientDet architecture.
            Default: ``False``.

    Examples::

        >>> from collections import OrderedDict
        >>> import torch
        >>> from monai.networks.blocks import BiFPN
        >>> # 2D example
        >>> bifpn = BiFPN(spatial_dims=2, in_channels_list=[32, 64, 128], out_channels=64)
        >>> x = OrderedDict()
        >>> x["p3"] = torch.rand(1, 32, 64, 64)
        >>> x["p4"] = torch.rand(1, 64, 32, 32)
        >>> x["p5"] = torch.rand(1, 128, 16, 16)
        >>> out = bifpn(x)
        >>> [(k, v.shape) for k, v in out.items()]
        [('p3', torch.Size([1, 64, 64, 64])),
         ('p4', torch.Size([1, 64, 32, 32])),
         ('p5', torch.Size([1, 64, 16, 16]))]

        >>> # 3D example (medical imaging)
        >>> bifpn3d = BiFPN(spatial_dims=3, in_channels_list=[32, 64], out_channels=32)
        >>> y = OrderedDict()
        >>> y["feat0"] = torch.rand(2, 32, 16, 32, 16)
        >>> y["feat1"] = torch.rand(2, 64, 8, 16, 8)
        >>> out3d = bifpn3d(y)
        >>> [(k, v.shape) for k, v in out3d.items()]
        [('feat0', torch.Size([2, 32, 16, 32, 16])),
         ('feat1', torch.Size([2, 32, 8, 16, 8]))]
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels_list: list[int],
        out_channels: int,
        num_repeats: int = 3,
        epsilon: float = 1e-4,
        extra_blocks: ExtraFPNBlock | None = None,
        depthwise_separable: bool = False,
    ) -> None:
        super().__init__()

        num_levels = len(in_channels_list)
        if num_levels < 2:
            raise ValueError(f"in_channels_list must have at least 2 entries, got {num_levels}")
        if num_repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        # Lateral 1x1 projections: map each level's input channels to out_channels.
        self.lateral_convs: nn.ModuleList = nn.ModuleList()
        for in_ch in in_channels_list:
            if in_ch == 0:
                raise ValueError("in_channels=0 is currently not supported")
            self.lateral_convs.append(conv_type(in_ch, out_channels, 1))

        # Kaiming initialisation for lateral convs (matches FeaturePyramidNetwork).
        conv_type_: type[nn.Module] = Conv[Conv.CONV, spatial_dims]
        for m in self.lateral_convs.modules():
            if isinstance(m, conv_type_):
                nn.init.kaiming_uniform_(cast(torch.Tensor, m.weight), a=1)
                nn.init.constant_(cast(torch.Tensor, m.bias), 0.0)

        # Stacked BiFPN layers.
        self.bifpn_layers: nn.ModuleList = nn.ModuleList(
            [
                BiFPNLayer(spatial_dims, num_levels, out_channels, epsilon, depthwise_separable)
                for _ in range(num_repeats)
            ]
        )

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise AssertionError
        self.extra_blocks = extra_blocks

    def get_result_from_lateral_convs(self, x: Tensor, idx: int) -> Tensor:
        """TorchScript-compatible indexed access for lateral_convs ModuleList."""
        out = x
        for i, module in enumerate(self.lateral_convs):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Computes BiFPN features for a set of multi-scale feature maps.

        Args:
            x: ordered mapping from level name to feature tensor, ordered from
                finest (highest resolution) to coarsest (lowest resolution).

        Returns:
            ordered mapping from level name to fused feature tensor, in the
            same order as the input. If ``extra_blocks`` is set, additional
            levels may be appended.
        """
        names = list(x.keys())
        x_values: list[Tensor] = list(x.values())

        # Project all levels to out_channels via lateral 1x1 convolutions.
        features: list[Tensor] = []
        for idx in range(len(x_values)):
            features.append(self.get_result_from_lateral_convs(x_values[idx], idx))

        # Apply stacked BiFPN layers sequentially.
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)

        if self.extra_blocks is not None:
            features, names = self.extra_blocks(features, x_values, names)

        return OrderedDict(list(zip(names, features)))
