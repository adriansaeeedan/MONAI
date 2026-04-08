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
Anchor-free YOLOX detector wrapping :class:`YOLOXNetwork`.

The detector mirrors the :class:`~monai.apps.detection.networks.retinanet_detector.RetinaNetDetector`
interface: during training it returns a dict of losses; during inference it
returns a list of per-image detection dictionaries.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from monai.apps.detection.utils.box_selector import BoxSelector
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_, predict_with_inferer
from monai.apps.detection.utils.simota_matcher import SimOTAMatcher
from monai.data.box_utils import box_iou, CenterSizeMode, convert_box_mode, StandardMode
from monai.inferers import SlidingWindowInferer
from monai.losses.box_iou_loss import BoxIoULoss
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple_rep

__all__ = ["YOLOXDetector"]


def _center_size_to_standard(boxes_cs: Tensor, spatial_dims: int) -> Tensor:
    """Convert center-size boxes to StandardMode (corner format).

    Args:
        boxes_cs: (N, 2*D) tensor in center-size order:
            ``(cx, cy[, cz], w, h[, d])``.
        spatial_dims: 2 or 3.

    Returns:
        (N, 2*D) tensor in StandardMode: ``(xmin, ymin[, zmin], xmax, ymax[, zmax])``.
    """
    centers = boxes_cs[:, :spatial_dims]
    sizes = boxes_cs[:, spatial_dims:]
    half = sizes * 0.5
    return torch.cat([centers - half, centers + half], dim=-1)


def _standard_to_center_size(boxes_std: Tensor, spatial_dims: int) -> Tensor:
    """Convert StandardMode boxes to center-size format.

    Args:
        boxes_std: (N, 2*D) in StandardMode.
        spatial_dims: 2 or 3.

    Returns:
        (N, 2*D) in center-size: ``(cx, cy[, cz], w, h[, d])``.
    """
    mins = boxes_std[:, :spatial_dims]
    maxs = boxes_std[:, spatial_dims:]
    centers = (mins + maxs) * 0.5
    sizes = maxs - mins
    return torch.cat([centers, sizes], dim=-1)


class YOLOXDetector(nn.Module):
    """Anchor-free YOLOX detector.

    Wraps a :class:`~monai.apps.detection.networks.yolox_network.YOLOXNetwork` and
    adds grid generation, SimOTA label assignment, loss computation, and
    post-processing into a single module that shares its interface with
    :class:`~monai.apps.detection.networks.retinanet_detector.RetinaNetDetector`.

    The detector does **not** use anchors. Instead, it generates a dense grid
    of centre points at each FPN level and treats each grid point as a single
    prediction location.

    **Training**: returns ``Dict[str, Tensor]`` — loss terms keyed by
    ``self.cls_key``, ``self.box_reg_key``, ``self.obj_key`` (and optionally
    ``"l1"``).

    **Inference**: returns ``List[Dict[str, Tensor]]`` — one dict per image,
    with keys ``self.target_box_key``, ``self.target_label_key``,
    ``self.pred_score_key``.

    Args:
        network: a :class:`YOLOXNetwork` or any module with attributes
            ``spatial_dims``, ``num_classes``, ``cls_key``, ``box_reg_key``,
            ``obj_key``, ``strides``, ``size_divisible``.
        box_overlap_metric: IoU function used in :class:`SimOTAMatcher` and
            :class:`BoxSelector`. Defaults to
            :func:`~monai.data.box_utils.box_iou`.
        spatial_dims: override ``network.spatial_dims``. Rarely needed.
        num_classes: override ``network.num_classes``. Rarely needed.
        size_divisible: override ``network.size_divisible``. Rarely needed.
        debug: print assignment statistics during training. Defaults to False.

    Example:

        .. code-block:: python

            import torch
            from monai.apps.detection.networks.yolox_network import yolox_darknet_pafpn_network
            from monai.apps.detection.networks.yolox_detector import YOLOXDetector

            network = yolox_darknet_pafpn_network(spatial_dims=2, num_classes=3, model_size="s")
            detector = YOLOXDetector(network)
            detector.set_simota_matcher(center_radius=1.5)

            # Training
            detector.train()
            imgs = [torch.rand(1, 256, 256) for _ in range(2)]
            targets = [
                {"boxes": torch.tensor([[10., 10., 50., 50.]]), "labels": torch.tensor([0])},
                {"boxes": torch.tensor([[20., 20., 80., 80.]]), "labels": torch.tensor([1])},
            ]
            losses = detector(imgs, targets)

            # Inference
            detector.eval()
            detections = detector(imgs)
    """

    def __init__(
        self,
        network: nn.Module,
        box_overlap_metric: Callable = box_iou,
        spatial_dims: int | None = None,
        num_classes: int | None = None,
        size_divisible: Sequence[int] | int = 1,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.network = network
        self.debug = debug
        self.box_overlap_metric = box_overlap_metric

        self.spatial_dims: int = self._get_attr("spatial_dims", default_value=spatial_dims)
        self.num_classes: int = self._get_attr("num_classes", default_value=num_classes)
        self.strides: tuple[int, ...] = tuple(self._get_attr("strides", default_value=(8, 16, 32)))

        _sz_div = self._get_attr("size_divisible", default_value=size_divisible)
        self.size_divisible: tuple[int, ...] = ensure_tuple_rep(_sz_div, self.spatial_dims)

        # Network output keys
        self.cls_key: str = self._get_attr("cls_key", default_value="classification")
        self.box_reg_key: str = self._get_attr("box_reg_key", default_value="box_regression")
        self.obj_key: str = self._get_attr("obj_key", default_value="objectness")

        # Ground truth / output keys (can be customised with set_target_keys)
        self.target_box_key = "boxes"
        self.target_label_key = "labels"
        self.pred_score_key = "labels_scores"

        # Default losses
        self.cls_loss_func: nn.Module = nn.BCEWithLogitsLoss(reduction="none")
        self.box_loss_func: nn.Module = BoxIoULoss(loss_type="iou", reduction="none")
        self.obj_loss_func: nn.Module = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss_func: nn.Module | None = None  # enabled via set_l1_loss()
        self.boxes_xform_clip = math.log(1000.0 / 16)

        # Default SimOTA matcher — initialised without requiring set_simota_matcher()
        self.matcher = SimOTAMatcher(
            spatial_dims=self.spatial_dims,
            center_radius=1.5,
            box_overlap_metric=box_overlap_metric,
            debug=debug,
        )

        # Inferer (sliding window, optional)
        self.inferer: SlidingWindowInferer | None = None

        # Box selector (NMS + score filtering)
        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            score_thresh=0.05,
            topk_candidates_per_level=1000,
            nms_thresh=0.5,
            detections_per_img=300,
            apply_sigmoid=False,  # YOLOX: scores are already computed outside
            select_single_label_per_box=True,
        )

        # Cached grid state (rebuilt when image shape changes)
        self._grid_cache: dict[tuple, tuple[Tensor, Tensor]] | None = None
        self._prev_img_shape: Any | None = None

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def _get_attr(self, name: str, default_value: Any = None) -> Any:
        if hasattr(self.network, name):
            return getattr(self.network, name)
        if default_value is not None:
            return default_value
        raise ValueError(f"network does not have attribute '{name}'. Please pass it to YOLOXDetector.")

    def set_cls_loss(self, cls_loss: nn.Module) -> None:
        """Set the per-element classification loss (with no built-in sigmoid).

        Args:
            cls_loss: loss module that receives logits and targets.
                Defaults to ``BCEWithLogitsLoss(reduction="none")``.
        """
        self.cls_loss_func = cls_loss

    def set_box_regression_loss(self, box_loss: nn.Module) -> None:
        """Set the box regression loss.

        The loss receives predicted boxes and target boxes in ``StandardMode``.
        Defaults to :class:`~monai.losses.box_iou_loss.BoxIoULoss` with
        ``loss_type="iou"``.

        Args:
            box_loss: loss module accepting (N, 2*D) paired predicted/target boxes.
        """
        self.box_loss_func = box_loss

    def set_objectness_loss(self, obj_loss: nn.Module) -> None:
        """Set the per-element objectness loss (with no built-in sigmoid).

        Args:
            obj_loss: loss module receiving logits and binary targets.
        """
        self.obj_loss_func = obj_loss

    def set_l1_loss(self, l1_loss: nn.Module) -> None:
        """Enable an optional L1 regression loss on raw (un-decoded) predictions.

        When set, the detector also supervises the raw regression outputs with
        a normalised L1 target, consistent with the original YOLOX training.

        Args:
            l1_loss: typically ``torch.nn.L1Loss(reduction="none")``.
        """
        self.l1_loss_func = l1_loss

    def set_simota_matcher(
        self,
        center_radius: float = 1.5,
    ) -> None:
        """Configure the SimOTA label assignment matcher.

        Args:
            center_radius: geometry constraint radius multiplier.
                A grid point is a candidate if its centre lies within
                ``center_radius × stride`` of a GT box centre.
                Defaults to 1.5.
        """
        self.matcher = SimOTAMatcher(
            spatial_dims=self.spatial_dims,
            center_radius=center_radius,
            box_overlap_metric=self.box_overlap_metric,
            debug=self.debug,
        )

    def set_box_selector_parameters(
        self,
        score_thresh: float = 0.05,
        topk_candidates_per_level: int = 1000,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
    ) -> None:
        """Configure post-processing box selection parameters.

        Args:
            score_thresh: minimum score (obj × cls) to keep a detection.
            topk_candidates_per_level: maximum candidates per FPN level.
            nms_thresh: NMS IoU threshold.
            detections_per_img: maximum detections per image after NMS.
        """
        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            apply_sigmoid=False,
            score_thresh=score_thresh,
            topk_candidates_per_level=topk_candidates_per_level,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            select_single_label_per_box=True,
        )

    def set_target_keys(self, box_key: str, label_key: str) -> None:
        """Set keys for GT targets (training) and output dicts (inference).

        Args:
            box_key: key for bounding boxes.
            label_key: key for class labels.
        """
        self.target_box_key = box_key
        self.target_label_key = label_key
        self.pred_score_key = label_key + "_scores"

    def set_sliding_window_inferer(
        self,
        roi_size: Sequence[int] | int,
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: BlendMode | str = BlendMode.CONSTANT,
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
    ) -> None:
        """Define and store a sliding-window inferer for large-image inference."""
        self.inferer = SlidingWindowInferer(
            roi_size,
            sw_batch_size,
            overlap,
            mode,
            sigma_scale,
            padding_mode,
            cval,
            sw_device,
            device,
            progress,
            cache_roi_weight_map,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_images: list[Tensor] | Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        use_inferer: bool = False,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Run the detector in training or inference mode.

        Args:
            input_images: list of (C, H, W[, D]) tensors **or** a single
                (B, C, H, W[, D]) tensor. Values should be in ``[0, 1]``.
            targets: during training, a list of dicts, each containing:
                - ``self.target_box_key``: (N, 2*D) GT boxes in ``StandardMode``.
                - ``self.target_label_key``: (N,) integer class labels.
            use_inferer: if True, use ``self.inferer`` (sliding window) for
                inference instead of a single forward pass.

        Returns:
            **Training** — ``Dict[str, Tensor]``: loss terms.

            **Inference** — ``List[Dict[str, Tensor]]``: each dict has keys
            ``self.target_box_key``, ``self.target_label_key``,
            ``self.pred_score_key``.
        """
        if self.training:
            targets = check_training_targets(
                input_images, targets, self.spatial_dims, self.target_label_key, self.target_box_key
            )

        # 1. Pad images to a uniform size divisible by self.size_divisible
        images, image_sizes = preprocess_images(input_images, self.spatial_dims, self.size_divisible)

        # 2. Network forward pass
        if self.training or not use_inferer:
            head_outputs = self.network(images)
            if isinstance(head_outputs, (list, tuple)):
                n = len(head_outputs) // 3
                tmp: dict[str, list[Tensor]] = {
                    self.cls_key: head_outputs[:n],
                    self.box_reg_key: head_outputs[n : 2 * n],
                    self.obj_key: head_outputs[2 * n :],
                }
                head_outputs = tmp
            else:
                ensure_dict_value_to_list_(head_outputs)
        else:
            if self.inferer is None:
                raise ValueError(
                    "`self.inferer` is None. Call set_sliding_window_inferer() first."
                )
            head_outputs = predict_with_inferer(
                images,
                self.network,
                keys=[self.cls_key, self.box_reg_key, self.obj_key],
                inferer=self.inferer,
            )

        # 3. Generate grid points and decode predictions
        cls_maps: list[Tensor] = head_outputs[self.cls_key]
        reg_maps: list[Tensor] = head_outputs[self.box_reg_key]
        obj_maps: list[Tensor] = head_outputs[self.obj_key]

        grid_points, strides_all = self._generate_grids(cls_maps, images.device)
        # Decode all levels: (B, N_total, 2*sdims) in StandardMode
        pred_boxes_std, raw_reg_all = self._decode_predictions(reg_maps, grid_points, strides_all)
        # Reshape cls and obj: (B, N_total, num_classes) and (B, N_total, 1)
        pred_cls = self._concat_level_outputs(cls_maps, self.num_classes)
        pred_obj = self._concat_level_outputs(obj_maps, 1)

        num_anchor_locs_per_level = [int(m.shape[2:].numel()) for m in cls_maps]

        # 4. Training path: compute losses
        if self.training:
            return self._compute_losses(
                pred_boxes_std,
                pred_cls,
                pred_obj,
                raw_reg_all,
                grid_points,
                strides_all,
                targets,  # type: ignore[arg-type]
                num_anchor_locs_per_level,
            )

        # 5. Inference path: post-process detections
        return self._postprocess(
            pred_boxes_std,
            pred_cls,
            pred_obj,
            image_sizes,
            num_anchor_locs_per_level,
        )

    # ------------------------------------------------------------------
    # Grid generation and decoding
    # ------------------------------------------------------------------

    def _generate_grids(
        self,
        cls_maps: list[Tensor],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Generate flat grid-centre coordinates and stride tensors.

        Args:
            cls_maps: list of (B, C, *spatial) tensors, one per FPN level.
            device: target device.

        Returns:
            - ``grid_points``: (N_total, D) grid-point coordinates in pixel
              space, aligned with MONAI's native spatial axis order.
            - ``strides_all``: (N_total,) stride value per point.
        """
        img_shape = tuple(cls_maps[0].shape[2:])
        if self._prev_img_shape == img_shape and self._grid_cache is not None:
            # Reuse cached grids (shape unchanged)
            gc, sa = self._grid_cache[img_shape]
            return gc.to(device), sa.to(device)

        all_centers = []
        all_strides = []

        for feat_map, stride in zip(cls_maps, self.strides):
            spatial = feat_map.shape[2:]  # (H, W) or (D, H, W)
            ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in spatial]

            # torch.meshgrid returns grids in (dim0, dim1[, dim2]) indexing
            grids = torch.meshgrid(*ranges, indexing="ij")  # each: spatial shape

            # Stack to (*spatial, ndims) then flatten to (N, ndims).
            # Keep MONAI's native spatial axis order so decoded boxes, GT boxes,
            # clipping, and NMS all use the same coordinate convention.
            centers = torch.stack(grids, dim=-1)
            n_pts = centers.shape[:-1].numel()
            centers = centers.view(n_pts, self.spatial_dims)  # (N, D)

            # Convert integer grid indices to pixel-space grid origins.
            centers = centers * stride

            strides = torch.full((n_pts,), stride, dtype=torch.float32, device=device)
            all_centers.append(centers)
            all_strides.append(strides)

        grid_centers = torch.cat(all_centers, dim=0)     # (N_total, D)
        strides_all = torch.cat(all_strides, dim=0)      # (N_total,)

        self._grid_cache = {img_shape: (grid_centers.cpu(), strides_all.cpu())}
        self._prev_img_shape = img_shape

        return grid_centers, strides_all

    def _decode_predictions(
        self,
        reg_maps: list[Tensor],
        grid_points: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode raw regression maps to centre-size boxes, then convert to StandardMode.

        Raw regression format for each grid point:
        - First ``spatial_dims`` channels: raw centre offsets (δx, δy[, δz]).
          Decoded: ``cx = (raw_cx + grid_cx_in_stride_units) * stride``.
          Here ``grid_points`` stores ``grid_idx * stride`` in pixel space, so
          decoding is ``cx = raw_cx * stride + grid_point_pixel``.
        - Last ``spatial_dims`` channels: raw log-size (log_w, log_h[, log_d]).
          Decoded: ``w = exp(raw_w) * stride``.

        Args:
            reg_maps: list of (B, 2*D, *spatial) tensors per level.
            grid_points: (N_total, D) pixel-space grid origins.
            strides: (N_total,) stride per grid point.

        Returns:
            - ``pred_boxes_std``: (B, N_total, 2*D) decoded boxes in StandardMode.
            - ``raw_reg_all``: (B, N_total, 2*D) raw (un-decoded) regression outputs,
              used for optional L1 loss.
        """
        B = reg_maps[0].shape[0]
        sdims = self.spatial_dims
        level_preds = []
        level_raw = []
        idx = 0

        for level_idx, (feat_map, stride_val) in enumerate(zip(reg_maps, self.strides)):
            n_pts = feat_map.shape[2:].numel()
            # Flatten: (B, 2*D, *spatial) → (B, n_pts, 2*D)
            raw = feat_map.flatten(start_dim=2).permute(0, 2, 1)  # (B, n_pts, 2*D)
            if not torch.isfinite(raw).all():
                warnings.warn(
                    f"Non-finite YOLOX box regression outputs detected at FPN level {level_idx}; "
                    "replacing invalid values before decode.",
                    stacklevel=2,
                )
                raw = raw.clone()
                raw[..., :sdims] = torch.nan_to_num(raw[..., :sdims], nan=0.0, posinf=0.0, neginf=0.0)
                raw[..., sdims:] = torch.nan_to_num(
                    raw[..., sdims:],
                    nan=0.0,
                    posinf=self.boxes_xform_clip,
                    neginf=-self.boxes_xform_clip,
                )
            level_raw.append(raw)

            gc = grid_points[idx : idx + n_pts].to(dtype=torch.float32)   # (n_pts, D)
            raw_centre = raw[..., :sdims].to(dtype=torch.float32)  # (B, n, D)
            raw_size = raw[..., sdims:].to(dtype=torch.float32).clamp(max=self.boxes_xform_clip)  # (B, n, D)

            pred_centre = raw_centre * stride_val + gc.unsqueeze(0)  # (B, n, D)
            pred_size = torch.exp(raw_size) * stride_val              # (B, n, D)

            # centre-size → StandardMode
            pred_cs = torch.cat([pred_centre, pred_size], dim=-1)  # (B, n, 2*D)
            pred_std = torch.cat([pred_centre - pred_size * 0.5, pred_centre + pred_size * 0.5], dim=-1)
            level_preds.append(pred_std)
            idx += n_pts

        return torch.cat(level_preds, dim=1), torch.cat(level_raw, dim=1)

    def _concat_level_outputs(self, maps: list[Tensor], num_ch: int) -> Tensor:
        """Flatten and concatenate FPN level outputs.

        Args:
            maps: list of (B, num_ch, *spatial) tensors.
            num_ch: number of output channels.

        Returns:
            (B, N_total, num_ch) tensor.
        """
        B = maps[0].shape[0]
        parts = [m.flatten(start_dim=2).permute(0, 2, 1) for m in maps]  # each (B, n, C)
        return torch.cat(parts, dim=1)  # (B, N_total, C)

    # ------------------------------------------------------------------
    # Training: loss computation
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        pred_boxes_std: Tensor,
        pred_cls: Tensor,
        pred_obj: Tensor,
        raw_reg_all: Tensor,
        grid_points: Tensor,
        strides_all: Tensor,
        targets: list[dict[str, Tensor]],
        num_anchor_locs_per_level: list[int],
    ) -> dict[str, Tensor]:
        """Compute SimOTA-assigned losses for a batch.

        Args:
            pred_boxes_std: (B, N, 2*D) decoded boxes in StandardMode.
            pred_cls: (B, N, num_classes) classification logits.
            pred_obj: (B, N, 1) objectness logits.
            raw_reg_all: (B, N, 2*D) raw regression outputs for optional L1.
            grid_points: (N, D) pixel-space grid origins.
            strides_all: (N,) stride per grid point.
            targets: list of per-image target dicts.
            num_anchor_locs_per_level: number of grid points per FPN level.

        Returns:
            Dict of scalar loss tensors.
        """
        device = pred_boxes_std.device
        total_iou_loss = torch.zeros(1, device=device)
        total_cls_loss = torch.zeros(1, device=device)
        total_obj_loss = torch.zeros(1, device=device)
        total_l1_loss = torch.zeros(1, device=device)
        num_fg_total = 0
        num_gt_total = 0
        B = pred_boxes_std.shape[0]
        matching_centers = grid_points + 0.5 * strides_all.unsqueeze(-1)

        for b_idx in range(B):
            gt_boxes = targets[b_idx][self.target_box_key]      # (num_gt, 2*D)
            gt_labels = targets[b_idx][self.target_label_key]   # (num_gt,)
            num_gt = gt_boxes.shape[0]
            num_gt_total += num_gt

            pred_boxes_b = pred_boxes_std[b_idx]    # (N, 2*D)
            pred_cls_b = pred_cls[b_idx]            # (N, C)
            pred_obj_b = pred_obj[b_idx]            # (N, 1)

            if num_gt == 0:
                # No GT boxes: objectness target is all-zero
                obj_target = torch.zeros(pred_obj_b.shape[0], 1, device=pred_obj_b.device)
                total_obj_loss += self.obj_loss_func(pred_obj_b, obj_target).sum()
                continue

            (
                gt_matched_classes,
                fg_mask,
                pred_ious_matched,
                matched_gt_inds,
                num_fg,
            ) = self._run_matcher_with_fallback(
                gt_boxes=gt_boxes.to(pred_boxes_b.device),
                gt_classes=gt_labels.to(pred_boxes_b.device),
                pred_boxes=pred_boxes_b,
                pred_cls_logits=pred_cls_b,
                pred_obj_logits=pred_obj_b,
                matching_centers=matching_centers,
                strides_all=strides_all,
                image_index=b_idx,
            )

            num_fg_total += num_fg

            # ------ Objectness target ------
            obj_target = fg_mask.float().unsqueeze(-1)  # (N, 1)
            total_obj_loss += self.obj_loss_func(pred_obj_b, obj_target).sum()

            if num_fg == 0:
                continue

            # ------ Classification target ------
            # Soft label: class one-hot weighted by IoU (YOLOX style)
            cls_target = (
                F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float()
                * pred_ious_matched.unsqueeze(-1)
            )  # (num_fg, C)
            total_cls_loss += self.cls_loss_func(
                pred_cls_b[fg_mask], cls_target
            ).sum()

            # ------ Box regression target ------
            reg_target_std = gt_boxes.to(pred_boxes_b.device)[matched_gt_inds]  # (num_fg, 2*D)
            total_iou_loss += self.box_loss_func(
                pred_boxes_b[fg_mask], reg_target_std
            ).sum()

            # ------ Optional L1 target ------
            if self.l1_loss_func is not None:
                l1_target = self._get_l1_target(
                    reg_target_std,
                    strides_all[fg_mask],
                    grid_points[fg_mask],
                )
                total_l1_loss += self.l1_loss_func(
                    raw_reg_all[b_idx][fg_mask], l1_target
                ).sum()

        num_fg_total = max(num_fg_total, 1)
        reg_weight = 5.0

        losses = {
            self.cls_key: total_cls_loss / num_fg_total,
            self.box_reg_key: reg_weight * total_iou_loss / num_fg_total,
            self.obj_key: total_obj_loss / num_fg_total,
        }
        if self.l1_loss_func is not None:
            losses["l1"] = total_l1_loss / num_fg_total

        return losses

    def _run_matcher_with_fallback(
        self,
        gt_boxes: Tensor,
        gt_classes: Tensor,
        pred_boxes: Tensor,
        pred_cls_logits: Tensor,
        pred_obj_logits: Tensor,
        matching_centers: Tensor,
        strides_all: Tensor,
        image_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Run SimOTA, retrying on CPU if CUDA runs out of memory."""
        try:
            return self.matcher(
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                pred_boxes=pred_boxes,
                pred_cls_logits=pred_cls_logits,
                pred_obj_logits=pred_obj_logits,
                grid_centers=matching_centers,
                strides=strides_all,
            )
        except RuntimeError as exc:
            if "CUDA out of memory" not in str(exc):
                raise

        warnings.warn(f"OOM in SimOTA for image {image_index}, retrying assignment on CPU.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cpu_outputs = self.matcher(
            gt_boxes=gt_boxes.cpu(),
            gt_classes=gt_classes.cpu(),
            pred_boxes=pred_boxes.cpu(),
            pred_cls_logits=pred_cls_logits.cpu(),
            pred_obj_logits=pred_obj_logits.cpu(),
            grid_centers=matching_centers.cpu(),
            strides=strides_all.cpu(),
        )
        gt_matched_classes, fg_mask, pred_ious_matched, matched_gt_inds, num_fg = cpu_outputs
        out_device = pred_boxes.device
        return (
            gt_matched_classes.to(out_device),
            fg_mask.to(out_device),
            pred_ious_matched.to(out_device),
            matched_gt_inds.to(out_device),
            num_fg,
        )

    def _get_l1_target(
        self,
        gt_boxes_std: Tensor,
        strides: Tensor,
        grid_points: Tensor,
    ) -> Tensor:
        """Compute normalised L1 regression targets for matched foreground points.

        The L1 target encodes GT boxes as normalised centre offsets and log sizes
        relative to the grid point and stride — matching the raw prediction space.

        Args:
            gt_boxes_std: (num_fg, 2*D) GT boxes in StandardMode.
            strides: (num_fg,) stride per foreground grid point.
            grid_points: (num_fg, D) pixel-space grid origin of each foreground point.

        Returns:
            (num_fg, 2*D) normalised L1 targets in raw prediction space.
        """
        sdims = self.spatial_dims
        gt_cs = _standard_to_center_size(gt_boxes_std, sdims)  # (num_fg, 2*D)
        gt_ctr = gt_cs[:, :sdims]
        gt_sz = gt_cs[:, sdims:]

        # Normalised centre offset: (gt_ctr - grid_ctr) / stride
        offset = (gt_ctr - grid_points) / strides.unsqueeze(-1)
        # Log-size target: log(gt_sz / stride)
        log_sz = torch.log(gt_sz / strides.unsqueeze(-1) + 1e-8)

        return torch.cat([offset, log_sz], dim=-1)

    # ------------------------------------------------------------------
    # Inference: post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        pred_boxes_std: Tensor,
        pred_cls: Tensor,
        pred_obj: Tensor,
        image_sizes: list[list[int]],
        num_anchor_locs_per_level: list[int],
    ) -> list[dict[str, Tensor]]:
        """Select final detections from decoded predictions.

        Combines objectness and class scores (obj × cls), then passes per-image
        predictions to :class:`BoxSelector` for score thresholding and NMS.

        Args:
            pred_boxes_std: (B, N, 2*D) decoded boxes in StandardMode.
            pred_cls: (B, N, num_classes) classification logits.
            pred_obj: (B, N, 1) objectness logits.
            image_sizes: original spatial sizes before padding.
            num_anchor_locs_per_level: grid points per FPN level.

        Returns:
            List of detection dicts, one per image.
        """
        # Post-processing is non-differentiable and is used only for inference
        # or optional training-time visualization, so keep it off the autograd graph.
        pred_boxes_std = pred_boxes_std.detach()
        pred_cls = pred_cls.detach()
        pred_obj = pred_obj.detach()

        with torch.no_grad():
            B = pred_boxes_std.shape[0]
            detections: list[dict[str, Tensor]] = []

            # Combined scores: sigmoid(obj) * sigmoid(cls) — YOLOX style
            cls_scores = pred_cls.sigmoid()       # (B, N, C)
            obj_scores = pred_obj.sigmoid()       # (B, N, 1)
            scores = cls_scores * obj_scores  # (B, N, C)

            for b_idx in range(B):
                boxes_b = pred_boxes_std[b_idx]     # (N, 2*D)
                scores_b = scores[b_idx]            # (N, C)
                img_size = image_sizes[b_idx]

                # Split per FPN level for BoxSelector
                boxes_per_level = list(boxes_b.split(num_anchor_locs_per_level, dim=0))
                scores_per_level = list(scores_b.split(num_anchor_locs_per_level, dim=0))

                sel_boxes, sel_scores, sel_labels = self.box_selector.select_boxes_per_image(
                    boxes_per_level, scores_per_level, img_size
                )
                detections.append(
                    {
                        self.target_box_key: sel_boxes,
                        self.pred_score_key: sel_scores,
                        self.target_label_key: sel_labels,
                    }
                )

            return detections
