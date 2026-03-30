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

    def test_l1_loss_enabled(self):
        self.detector.set_l1_loss(torch.nn.L1Loss(reduction="none"))
        self.detector.train()
        imgs = [torch.rand(1, 128, 128) for _ in range(2)]
        losses = self.detector(imgs, _make_targets_2d(2))
        self.assertIn("l1", losses)


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
