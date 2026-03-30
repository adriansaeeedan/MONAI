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

from monai.losses.box_iou_loss import BoxIoULoss


class TestBoxIoULoss(unittest.TestCase):
    def test_perfect_overlap_is_zero(self):
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        loss = BoxIoULoss(loss_type="iou", reduction="mean")(box, box)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_no_overlap_is_one(self):
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        tgt = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
        loss = BoxIoULoss(loss_type="iou", reduction="mean")(pred, tgt)
        # IoU=0, so 1 - 0^2 = 1
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_half_overlap_2d(self):
        pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        tgt = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
        # intersection=2, union=6, iou=1/3
        loss_fn = BoxIoULoss(loss_type="iou", reduction="none")
        loss = loss_fn(pred, tgt)
        expected = 1.0 - (1.0 / 3.0) ** 2
        self.assertAlmostEqual(loss[0].item(), expected, places=4)

    def test_giou_perfect_overlap(self):
        box = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        loss = BoxIoULoss(loss_type="giou")(box, box)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_giou_no_overlap(self):
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        tgt = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
        loss = BoxIoULoss(loss_type="giou")(pred, tgt)
        # GIoU should be < 0, so loss > 1
        self.assertGreater(loss.item(), 1.0)

    def test_3d_boxes(self):
        box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 4.0, 4.0]])
        loss = BoxIoULoss(loss_type="iou")(box, box)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_reduction_sum(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
        loss_sum = BoxIoULoss(loss_type="iou", reduction="sum")(boxes, boxes)
        self.assertAlmostEqual(loss_sum.item(), 0.0, places=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            BoxIoULoss()(torch.randn(2, 4), torch.randn(3, 4))

    def test_invalid_loss_type(self):
        with self.assertRaises(ValueError):
            BoxIoULoss(loss_type="ciou")

    def test_gradient_flows(self):
        pred = torch.tensor([[0.5, 0.5, 2.5, 2.5]], requires_grad=True)
        tgt = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        BoxIoULoss(loss_type="iou")(pred, tgt).backward()
        self.assertIsNotNone(pred.grad)


if __name__ == "__main__":
    unittest.main()
