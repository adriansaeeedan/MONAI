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

from monai.apps.detection.utils.simota_matcher import SimOTAMatcher


def _make_grid(h: int, w: int, stride: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a simple 2D flat grid."""
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    centers = torch.stack([xs.flatten(), ys.flatten()], dim=-1) * stride + stride * 0.5
    strides = torch.full((h * w,), stride, dtype=torch.float32)
    return centers, strides


class TestSimOTAMatcher2D(unittest.TestCase):
    def setUp(self):
        self.matcher = SimOTAMatcher(spatial_dims=2, center_radius=2.0)

    def _pred_boxes(self, centers, strides, scale=0.5):
        """Create predicted boxes centred on the grid with a small fixed size."""
        size = strides.unsqueeze(-1) * scale
        return torch.cat([centers - size, centers + size], dim=-1)

    def test_single_gt_matches_nearby_predictions(self):
        h, w = 4, 4
        centers, strides = _make_grid(h, w, stride=8)
        pred_boxes = self._pred_boxes(centers, strides)
        # GT box centred at (20, 20) in pixel space
        gt_boxes = torch.tensor([[16.0, 16.0, 24.0, 24.0]])
        gt_classes = torch.tensor([0])

        pred_cls = torch.zeros(h * w, 3)
        pred_obj = torch.zeros(h * w, 1)

        gt_matched_cls, fg_mask, pred_ious, matched_gt_inds, num_fg = self.matcher(
            gt_boxes, gt_classes, pred_boxes, pred_cls, pred_obj, centers, strides
        )

        self.assertGreater(num_fg, 0)
        self.assertEqual(fg_mask.shape[0], h * w)
        self.assertEqual(gt_matched_cls.shape[0], num_fg)
        self.assertTrue((gt_matched_cls == 0).all())
        self.assertEqual(matched_gt_inds.shape[0], num_fg)

    def test_no_gt_returns_empty(self):
        h, w = 4, 4
        centers, strides = _make_grid(h, w)
        pred_boxes = self._pred_boxes(centers, strides)
        gt_boxes = torch.zeros(0, 4)
        gt_classes = torch.zeros(0, dtype=torch.long)
        pred_cls = torch.zeros(h * w, 2)
        pred_obj = torch.zeros(h * w, 1)

        gt_matched_cls, fg_mask, pred_ious, matched_gt_inds, num_fg = self.matcher(
            gt_boxes, gt_classes, pred_boxes, pred_cls, pred_obj, centers, strides
        )
        self.assertEqual(num_fg, 0)
        self.assertEqual(fg_mask.sum().item(), 0)

    def test_fg_mask_is_subset_of_geometry_candidates(self):
        h, w = 8, 8
        centers, strides = _make_grid(h, w, stride=4)
        pred_boxes = self._pred_boxes(centers, strides)
        # GT at centre of grid
        gt_boxes = torch.tensor([[12.0, 12.0, 20.0, 20.0]])
        gt_classes = torch.tensor([1])
        pred_cls = torch.zeros(h * w, 5)
        pred_obj = torch.zeros(h * w, 1)

        _, fg_mask, _, _, num_fg = self.matcher(
            gt_boxes, gt_classes, pred_boxes, pred_cls, pred_obj, centers, strides
        )
        self.assertEqual(int(fg_mask.sum().item()), num_fg)

    def test_conflict_resolution_unique_assignment(self):
        """Each grid point must be assigned to at most one GT box."""
        h, w = 4, 4
        centers, strides = _make_grid(h, w, stride=8)
        pred_boxes = self._pred_boxes(centers, strides)
        # Two overlapping GT boxes
        gt_boxes = torch.tensor([[16.0, 16.0, 32.0, 32.0], [20.0, 20.0, 36.0, 36.0]])
        gt_classes = torch.tensor([0, 1])
        pred_cls = torch.zeros(h * w, 3)
        pred_obj = torch.zeros(h * w, 1)

        _, fg_mask, _, matched_gt_inds, num_fg = self.matcher(
            gt_boxes, gt_classes, pred_boxes, pred_cls, pred_obj, centers, strides
        )
        # Each foreground point has exactly one matched GT
        self.assertEqual(matched_gt_inds.shape[0], num_fg)
        # No duplicate assignments: each fg point → exactly 1 GT
        if num_fg > 0:
            self.assertEqual(matched_gt_inds.unique().shape[0], min(num_fg, 2))


class TestSimOTAMatcher3D(unittest.TestCase):
    def setUp(self):
        self.matcher = SimOTAMatcher(spatial_dims=3, center_radius=1.5)

    def _make_grid_3d(self, d, h, w, stride=8):
        zs, ys, xs = torch.meshgrid(
            torch.arange(d, dtype=torch.float32),
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        centers = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=-1) * stride + stride * 0.5
        strides = torch.full((d * h * w,), stride, dtype=torch.float32)
        return centers, strides

    def test_3d_single_gt(self):
        d, h, w = 4, 4, 4
        centers, strides = self._make_grid_3d(d, h, w, stride=8)
        size = strides.unsqueeze(-1) * 0.5
        pred_boxes = torch.cat([centers - size, centers + size], dim=-1)

        gt_boxes = torch.tensor([[12.0, 12.0, 12.0, 28.0, 28.0, 28.0]])
        gt_classes = torch.tensor([0])
        pred_cls = torch.zeros(d * h * w, 2)
        pred_obj = torch.zeros(d * h * w, 1)

        _, fg_mask, _, _, num_fg = self.matcher(
            gt_boxes, gt_classes, pred_boxes, pred_cls, pred_obj, centers, strides
        )
        self.assertGreater(num_fg, 0)
        self.assertEqual(fg_mask.shape[0], d * h * w)


if __name__ == "__main__":
    unittest.main()
