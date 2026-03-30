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
IoU loss for bounding boxes in MONAI StandardMode (corner format).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from monai.data.box_utils import COMPUTE_DTYPE, box_pair_giou, get_spatial_dims
from monai.utils import LossReduction

__all__ = ["BoxIoULoss"]


def _box_pair_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of matched boxes (same length N).

    Args:
        boxes1: (N, 2*D) boxes in StandardMode. D = spatial_dims.
        boxes2: (N, 2*D) boxes in StandardMode. D = spatial_dims.

    Returns:
        iou: shape (N,), values in [0, 1].
    """
    spatial_dims = get_spatial_dims(boxes=boxes1)

    mins1 = boxes1[:, :spatial_dims]   # (N, D)
    maxs1 = boxes1[:, spatial_dims:]   # (N, D)
    mins2 = boxes2[:, :spatial_dims]
    maxs2 = boxes2[:, spatial_dims:]

    area1 = torch.prod(maxs1 - mins1, dim=-1).clamp(min=0)  # (N,)
    area2 = torch.prod(maxs2 - mins2, dim=-1).clamp(min=0)

    inter_min = torch.max(mins1, mins2)
    inter_max = torch.min(maxs1, maxs2)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter = torch.prod(inter_dims, dim=-1)  # (N,)

    union = area1 + area2 - inter
    return inter / (union + torch.finfo(boxes1.dtype).eps)


class BoxIoULoss(_Loss):
    """IoU or GIoU loss for paired bounding boxes in StandardMode.

    Given two equal-length sets of predicted and target boxes (both in
    ``StandardMode``, i.e., ``[xmin, ymin, xmax, ymax]`` or
    ``[xmin, ymin, zmin, xmax, ymax, zmax]``), computes element-wise
    ``1 - IoU`` or ``1 - GIoU`` loss.

    This loss is typically used with YOLOX-style detectors, where the
    regression target is a decoded box rather than an anchor-encoded offset.

    Args:
        loss_type: ``"iou"`` computes ``1 - IoU²`` (squared IoU, as in YOLOX);
            ``"giou"`` computes ``1 - GIoU``. Defaults to ``"iou"``.
        reduction: ``"none"``, ``"mean"``, or ``"sum"``.
            Defaults to ``"mean"``.

    Example:

        .. code-block:: python

            import torch
            from monai.losses.box_iou_loss import BoxIoULoss

            loss_fn = BoxIoULoss(loss_type="iou", reduction="mean")
            pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # StandardMode 2D
            tgt  = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
            print(loss_fn(pred, tgt))  # 0.0 (perfect overlap)
    """

    def __init__(
        self,
        loss_type: str = "iou",
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if loss_type not in ("iou", "giou"):
            raise ValueError(f"loss_type must be 'iou' or 'giou', got '{loss_type}'.")
        self.loss_type = loss_type

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            input: predicted boxes, shape (N, 2*D), in ``StandardMode``.
            target: ground-truth boxes, shape (N, 2*D), in ``StandardMode``.

        Returns:
            Scalar loss (if reduction is ``"mean"`` or ``"sum"``) or
            per-element loss of shape (N,) (if reduction is ``"none"``).

        Raises:
            ValueError: when ``input`` and ``target`` have different shapes.
        """
        if input.shape != target.shape:
            raise ValueError(f"input shape {input.shape} does not match target shape {target.shape}.")

        input_f = input.to(dtype=COMPUTE_DTYPE)
        target_f = target.to(dtype=COMPUTE_DTYPE)

        if self.loss_type == "iou":
            iou = _box_pair_iou(input_f, target_f)
            loss = 1.0 - iou**2
        else:  # giou — delegate to MONAI's box_pair_giou
            giou = box_pair_giou(input_f, target_f)
            loss = 1.0 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.SUM.value:
            loss = loss.sum()

        return loss.to(input.dtype)
