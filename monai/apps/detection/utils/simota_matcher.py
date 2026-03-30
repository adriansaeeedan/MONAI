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
SimOTA dynamic label assignment matcher for anchor-free detectors.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from monai.data.box_utils import box_iou, convert_box_mode, CenterSizeMode, StandardMode

__all__ = ["SimOTAMatcher"]


class SimOTAMatcher:
    """SimOTA label assignment for anchor-free, single-stage object detectors.

    SimOTA (Simplified Optimal Transport Assignment) dynamically selects the
    number of positive samples *per ground-truth box* by using a cost matrix
    that combines IoU and classification losses. This avoids fixed thresholds
    and adapts to image content.

    Reference:
        `"YOLOX: Exceeding Yolo Series Detectors" <https://arxiv.org/abs/2107.08430>`_.

    Args:
        spatial_dims: number of spatial dimensions, 2 or 3.
        center_radius: radius multiplier for the geometric constraint
            (centre-region filtering). A grid point passes the constraint when
            its pixel-space centre lies within ``center_radius × stride`` of each
            GT box centre along every axis. Defaults to 1.5.
        box_overlap_metric: callable computing box IoU. Receives two tensors in
            ``StandardMode`` and returns an (N, M) IoU matrix.
            Defaults to :func:`monai.data.box_utils.box_iou`.
        debug: if True, prints per-image assignment statistics. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        center_radius: float = 1.5,
        box_overlap_metric: Callable = box_iou,
        debug: bool = False,
    ) -> None:
        self.spatial_dims = spatial_dims
        self.center_radius = center_radius
        self.box_overlap_metric = box_overlap_metric
        self.debug = debug

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        gt_boxes: Tensor,
        gt_classes: Tensor,
        pred_boxes: Tensor,
        pred_cls_logits: Tensor,
        pred_obj_logits: Tensor,
        grid_centers: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Assign ground-truth boxes to predicted grid points via SimOTA.

        All box tensors use ``StandardMode`` (corner format):
        ``[xmin, ymin, xmax, ymax]`` or ``[xmin, ymin, zmin, xmax, ymax, zmax]``.

        Args:
            gt_boxes: (num_gt, 2*D) ground-truth boxes in ``StandardMode``.
            gt_classes: (num_gt,) integer class labels.
            pred_boxes: (N, 2*D) decoded predicted boxes in ``StandardMode``.
            pred_cls_logits: (N, num_classes) raw classification logits.
            pred_obj_logits: (N, 1) raw objectness logits.
            grid_centers: (N, D) pixel-space centre coordinates of each grid point.
            strides: (N,) effective stride of each grid point.

        Returns:
            A tuple of:
            - ``gt_matched_classes``: (num_fg,) GT class for each matched foreground.
            - ``fg_mask``: (N,) boolean mask marking foreground grid points.
            - ``pred_ious``: (num_fg,) IoU of each foreground prediction with its GT.
            - ``matched_gt_inds``: (num_fg,) GT index each foreground is matched to.
            - ``num_fg``: total number of matched foreground grid points.
        """
        device = gt_boxes.device
        num_gt = gt_boxes.shape[0]

        # Compute geometry constraint mask
        fg_mask, in_boxes_and_center = self._get_geometry_constraint(gt_boxes, grid_centers, strides)

        num_candidates = int(fg_mask.sum())
        if num_candidates == 0:
            empty = gt_classes[:0]
            return (
                empty,
                fg_mask,
                pred_boxes.new_zeros(0),
                empty.long(),
                0,
            )

        # Candidate predicted boxes and logits
        pred_boxes_fg = pred_boxes[fg_mask]          # (K, 2*D)
        cls_logits_fg = pred_cls_logits[fg_mask]     # (K, C)
        obj_logits_fg = pred_obj_logits[fg_mask]     # (K, 1)

        # Pairwise IoU: (num_gt, K)
        pair_iou = self.box_overlap_metric(
            gt_boxes.to(pred_boxes_fg.device), pred_boxes_fg
        )  # (num_gt, K)
        pair_iou_loss = -torch.log(pair_iou + 1e-8)

        # Pairwise classification cost: (num_gt, K)
        num_classes = cls_logits_fg.shape[-1]
        with torch.cuda.amp.autocast(enabled=False):
            cls_score = (
                cls_logits_fg.float().sigmoid_() * obj_logits_fg.float().sigmoid_()
            ).sqrt()  # (K, C)
            gt_cls_onehot = F.one_hot(gt_classes.to(torch.int64), num_classes).float()  # (num_gt, C)
            pair_cls_loss = F.binary_cross_entropy(
                cls_score.unsqueeze(0).expand(num_gt, -1, -1),   # (num_gt, K, C)
                gt_cls_onehot.unsqueeze(1).expand(-1, num_candidates, -1),  # (num_gt, K, C)
                reduction="none",
            ).sum(-1)  # (num_gt, K)

        # Combined cost matrix
        cost = pair_cls_loss + 3.0 * pair_iou_loss + 1e6 * (~in_boxes_and_center)  # (num_gt, K)

        # Dynamic top-k matching
        (
            num_fg,
            gt_matched_classes,
            pred_ious,
            matched_gt_inds,
        ) = self._simota_matching(cost, pair_iou, gt_classes, num_gt, fg_mask)

        if self.debug:
            print(
                f"[SimOTAMatcher] num_gt={num_gt}, candidates={num_candidates}, "
                f"matched foreground={num_fg}"
            )

        return gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_geometry_constraint(
        self,
        gt_boxes: Tensor,
        grid_centers: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Filter grid points that are near any GT box centre.

        A grid point passes if its pixel-space centre is within
        ``center_radius × stride`` of at least one GT box centre along
        every axis.

        Args:
            gt_boxes: (num_gt, 2*D) in ``StandardMode``.
            grid_centers: (N, D) pixel-space grid centre coordinates.
            strides: (N,) stride per grid point.

        Returns:
            - ``anchor_filter``: (N,) boolean, True when the point is near any GT.
            - ``geometry_relation``: (num_gt, K) boolean, K = anchor_filter.sum().
        """
        num_gt = gt_boxes.shape[0]
        sdims = self.spatial_dims

        # GT box centres: (num_gt, D)
        gt_mins = gt_boxes[:, :sdims]
        gt_maxs = gt_boxes[:, sdims:]
        gt_centers = (gt_mins + gt_maxs) * 0.5  # (num_gt, D)

        # Per-point radius (pixels)
        radius = strides * self.center_radius  # (N,)

        # grid_centers: (N, D) → (1, N, D)
        gc = grid_centers.unsqueeze(0)          # (1, N, D)
        # gt_centers: (num_gt, D) → (num_gt, 1, D)
        gtc = gt_centers.unsqueeze(1)           # (num_gt, 1, D)
        # radius: (N,) → (1, N, 1)
        r = radius.unsqueeze(0).unsqueeze(-1)   # (1, N, 1)

        # Distance from each grid centre to each GT centre along each axis
        dist = (gc - gtc).abs()                 # (num_gt, N, D)
        is_in_radius = (dist < r).all(dim=-1)   # (num_gt, N)

        anchor_filter = is_in_radius.any(dim=0)        # (N,)
        geometry_relation = is_in_radius[:, anchor_filter]  # (num_gt, K)

        return anchor_filter, geometry_relation

    def _simota_matching(
        self,
        cost: Tensor,
        pair_wise_ious: Tensor,
        gt_classes: Tensor,
        num_gt: int,
        fg_mask: Tensor,
    ) -> tuple[int, Tensor, Tensor, Tensor]:
        """Perform dynamic top-k matching given the cost matrix.

        For each GT box, selects the top ``dynamic_k`` lowest-cost candidates,
        where ``dynamic_k`` is the rounded sum of the top-10 IoU values (clamped
        to at least 1). Conflicts (one candidate matched to multiple GTs) are
        resolved by assigning to the GT with the lowest cost.

        Args:
            cost: (num_gt, K) cost matrix.
            pair_wise_ious: (num_gt, K) pairwise IoU.
            gt_classes: (num_gt,) GT class indices.
            num_gt: number of GT boxes.
            fg_mask: (N,) boolean foreground mask (used to write back results).

        Returns:
            - ``num_fg``: total number of matched foreground points.
            - ``gt_matched_classes``: (num_fg,) GT class for each foreground.
            - ``pred_ious``: (num_fg,) IoU with matched GT.
            - ``matched_gt_inds``: (num_fg,) matched GT index.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks

        # Resolve conflicts: anchor matched to multiple GTs → assign to cheapest GT
        anchor_matching_gt = matching_matrix.sum(0)  # (K,)
        if anchor_matching_gt.max() > 1:
            multi_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multi_mask], dim=0)
            matching_matrix[:, multi_mask] = 0
            matching_matrix[cost_argmin, multi_mask] = 1

        fg_mask_inboxes = anchor_matching_gt > 0       # (K,) bool
        num_fg = int(fg_mask_inboxes.sum().item())

        # Write matched anchors back into the global fg_mask
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # (num_fg,)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious, matched_gt_inds
