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

from __future__ import annotations

import unittest

import torch

from monai.apps.detection.networks.yolox_network import YOLOXNetwork
from monai.apps.detection.networks.yolox_detector import YOLOXDetector


def _make_detector(spatial_dims: int, num_classes: int = 3) -> YOLOXDetector:
    net = YOLOXNetwork(
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        depth=0.33,
        width=0.25,
    )
    return YOLOXDetector(net)


def _make_targets_2d(n: int = 2) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
        }
        for _ in range(n)
    ]


def _make_targets_3d(n: int = 2) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor([[4.0, 4.0, 4.0, 20.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
        }
        for _ in range(n)
    ]


class _OOMThenCPUMatcher:
    def __init__(self):
        self.calls = 0
        self.second_call_on_cpu = False

    def __call__(
        self,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        pred_obj_logits: torch.Tensor,
        grid_centers: torch.Tensor,
        strides: torch.Tensor,
    ):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")

        self.second_call_on_cpu = all(
            t.device.type == "cpu"
            for t in (gt_boxes, gt_classes, pred_boxes, pred_cls_logits, pred_obj_logits, grid_centers, strides)
        )
        fg_mask = torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device)
        fg_mask[0] = True
        return gt_classes[:1], fg_mask, pred_boxes.new_tensor([0.5]), gt_classes.new_tensor([0]), 1


class TestYOLOXDetector2D(unittest.TestCase):
    def setUp(self):
        self.detector = _make_detector(spatial_dims=2, num_classes=3)

    def test_training_returns_loss_dict(self):
        self.detector.train()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        losses = self.detector(imgs, _make_targets_2d(2))
        self.assertIsInstance(losses, dict)
        for key in (self.detector.cls_key, self.detector.box_reg_key, self.detector.obj_key):
            self.assertIn(key, losses)
            self.assertFalse(torch.isnan(losses[key]).any(), f"{key} loss is NaN")
            self.assertFalse(torch.isinf(losses[key]).any(), f"{key} loss is Inf")

    def test_eval_returns_detection_list(self):
        self.detector.eval()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        with torch.no_grad():
            dets = self.detector(imgs)
        self.assertIsInstance(dets, list)
        self.assertEqual(len(dets), 2)
        for d in dets:
            self.assertIn(self.detector.target_box_key, d)
            self.assertIn(self.detector.target_label_key, d)
            self.assertIn(self.detector.pred_score_key, d)

    def test_eval_boxes_in_standard_mode(self):
        """Output boxes must be in StandardMode (xmin < xmax)."""
        self.detector.eval()
        imgs = [torch.rand(1, 128, 128) for _ in range(1)]
        with torch.no_grad():
            dets = self.detector(imgs)
        boxes = dets[0][self.detector.target_box_key]
        if boxes.numel() > 0:
            self.assertTrue((boxes[:, 2] >= boxes[:, 0]).all())
            self.assertTrue((boxes[:, 3] >= boxes[:, 1]).all())

    def test_training_gradient_flows(self):
        self.detector.train()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        losses = self.detector(imgs, _make_targets_2d(2))
        total = sum(losses.values())
        total.backward()
        # Check at least one parameter has a gradient
        grads = [p.grad for p in self.detector.network.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_set_target_keys(self):
        self.detector.set_target_keys("bbox", "category")
        self.assertEqual(self.detector.target_box_key, "bbox")
        self.assertEqual(self.detector.target_label_key, "category")
        self.assertEqual(self.detector.pred_score_key, "category_scores")

    def test_empty_targets(self):
        """Detector must handle images with no GT boxes without crashing."""
        self.detector.train()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]
        losses = self.detector(imgs, targets)
        for key, v in losses.items():
            self.assertFalse(torch.isnan(v).any(), f"{key} is NaN with empty targets")

    def test_batch_tensor_input(self):
        """Accepts a single (B, C, H, W) Tensor in addition to a list."""
        self.detector.eval()
        with torch.no_grad():
            dets = self.detector(torch.rand(2, 1, 128, 128))
        self.assertEqual(len(dets), 2)

    def test_generate_grids_preserves_rectangular_axis_order(self):
        """Grid coordinates should follow the input tensor spatial axis order."""
        with torch.no_grad():
            outputs = self.detector.network(torch.randn(1, 1, 32, 64))
        cls_maps = outputs[self.detector.cls_key]
        centers, _ = self.detector._generate_grids(cls_maps, torch.device("cpu"))
        torch.testing.assert_close(centers.max(0).values, torch.tensor([24.0, 56.0]))

    def test_decode_predictions_uses_grid_origins(self):
        """Zero offsets should decode from the grid origin as in YOLOX."""
        cls_maps = [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 1)]
        reg_maps = [torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 1, 1)]
        grid_points, strides = self.detector._generate_grids(cls_maps, torch.device("cpu"))
        pred_boxes, _ = self.detector._decode_predictions(reg_maps, grid_points, strides)
        torch.testing.assert_close(pred_boxes[0, 0], torch.tensor([-4.0, -4.0, 4.0, 4.0]))

    def test_l1_loss_enabled(self):
        self.detector.set_l1_loss(torch.nn.L1Loss(reduction="none"))
        self.detector.train()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        losses = self.detector(imgs, _make_targets_2d(2))
        self.assertIn("l1", losses)

    def test_oom_matcher_retries_on_cpu(self):
        self.detector.train()
        matcher = _OOMThenCPUMatcher()
        self.detector.matcher = matcher  # type: ignore[assignment]
        imgs = [torch.rand(1, 128, 128)]
        losses = self.detector(imgs, _make_targets_2d(1))
        self.assertEqual(matcher.calls, 2)
        self.assertTrue(matcher.second_call_on_cpu)
        for loss in losses.values():
            self.assertFalse(torch.isnan(loss).any())

    def test_postprocess_keeps_only_best_class_per_box(self):
        boxes = [torch.tensor([[1.0, 2.0, 3.0, 4.0]])]
        scores = [torch.tensor([[0.8, 0.6, 0.7]])]
        selected_boxes, selected_scores, selected_labels = self.detector.box_selector.select_boxes_per_image(
            boxes, scores, (10, 10)
        )
        torch.testing.assert_close(selected_boxes, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        torch.testing.assert_close(selected_scores, torch.tensor([0.8]))
        torch.testing.assert_close(selected_labels, torch.tensor([0]))


class TestYOLOXDetector3D(unittest.TestCase):
    def setUp(self):
        self.detector = _make_detector(spatial_dims=3, num_classes=2)

    def test_training_returns_loss_dict(self):
        self.detector.train()
        imgs = [torch.rand(1, 64, 64, 64) for _ in range(1)]
        losses = self.detector(imgs, _make_targets_3d(1))
        self.assertIsInstance(losses, dict)
        for key in (self.detector.cls_key, self.detector.box_reg_key, self.detector.obj_key):
            self.assertIn(key, losses)
            self.assertFalse(torch.isnan(losses[key]).any(), f"{key} is NaN")

    def test_eval_returns_detection_list(self):
        self.detector.eval()
        with torch.no_grad():
            dets = self.detector([torch.rand(1, 64, 64, 64)])
        self.assertIsInstance(dets, list)
        self.assertEqual(len(dets), 1)

    def test_3d_boxes_have_6_coords(self):
        self.detector.eval()
        with torch.no_grad():
            dets = self.detector([torch.rand(1, 64, 64, 64)])
        boxes = dets[0][self.detector.target_box_key]
        if boxes.numel() > 0:
            self.assertEqual(boxes.shape[-1], 6)

    def test_generate_grids_preserves_non_cubic_axis_order(self):
        """3D grid coordinates should follow the input tensor spatial axis order."""
        with torch.no_grad():
            outputs = self.detector.network(torch.randn(1, 1, 32, 64, 128))
        cls_maps = outputs[self.detector.cls_key]
        centers, _ = self.detector._generate_grids(cls_maps, torch.device("cpu"))
        torch.testing.assert_close(centers.max(0).values, torch.tensor([24.0, 56.0, 120.0]))


class TestYOLOXDetectorConfiguration(unittest.TestCase):
    def test_simota_matcher_reconfigurable(self):
        det = _make_detector(2)
        det.set_simota_matcher(center_radius=2.5)
        self.assertAlmostEqual(det.matcher.center_radius, 2.5)

    def test_box_selector_reconfigurable(self):
        det = _make_detector(2)
        det.set_box_selector_parameters(score_thresh=0.1, nms_thresh=0.3, detections_per_img=50)
        self.assertEqual(det.box_selector.score_thresh, 0.1)

    def test_network_only_has_trainable_params(self):
        """Only network parameters should be trainable; the detector itself adds none."""
        det = _make_detector(2)
        net_param_count = sum(p.numel() for p in det.network.parameters())
        det_param_count = sum(p.numel() for p in det.parameters())
        self.assertEqual(net_param_count, det_param_count)


if __name__ == "__main__":
    unittest.main()
