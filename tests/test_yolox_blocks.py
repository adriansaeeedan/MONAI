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

from monai.networks.blocks.yolox_blocks import (
    YOLOXBaseConv,
    YOLOXBottleneck,
    YOLOXCSPLayer,
    YOLOXDWConv,
    YOLOXFocus,
    YOLOXSPPBottleneck,
)


class TestYOLOXBaseConv(unittest.TestCase):
    def _run(self, spatial_dims, in_h, in_w, in_d=None):
        if spatial_dims == 2:
            x = torch.randn(2, 8, in_h, in_w)
            mod = YOLOXBaseConv(spatial_dims=2, in_channels=8, out_channels=16, ksize=3, stride=1)
            out = mod(x)
            self.assertEqual(out.shape, (2, 16, in_h, in_w))
        else:
            x = torch.randn(2, 8, in_d, in_h, in_w)
            mod = YOLOXBaseConv(spatial_dims=3, in_channels=8, out_channels=16, ksize=3, stride=1)
            out = mod(x)
            self.assertEqual(out.shape, (2, 16, in_d, in_h, in_w))

    def test_2d(self):
        self._run(2, 16, 16)

    def test_3d(self):
        self._run(3, 8, 8, in_d=8)

    def test_stride_2d(self):
        x = torch.randn(1, 4, 16, 16)
        mod = YOLOXBaseConv(2, 4, 8, ksize=3, stride=2)
        out = mod(x)
        self.assertEqual(out.shape, (1, 8, 8, 8))

    def test_activations(self):
        for act in ("silu", "relu", "lrelu"):
            mod = YOLOXBaseConv(2, 4, 4, ksize=1, stride=1, act=act)
            out = mod(torch.randn(1, 4, 8, 8))
            self.assertFalse(torch.isnan(out).any())

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            YOLOXBaseConv(2, 4, 4, ksize=1, stride=1, act="gelu")


class TestYOLOXDWConv(unittest.TestCase):
    def test_2d(self):
        mod = YOLOXDWConv(2, 16, 32, ksize=3)
        out = mod(torch.randn(2, 16, 8, 8))
        self.assertEqual(out.shape, (2, 32, 8, 8))

    def test_3d(self):
        mod = YOLOXDWConv(3, 16, 32, ksize=3)
        out = mod(torch.randn(2, 16, 4, 4, 4))
        self.assertEqual(out.shape, (2, 32, 4, 4, 4))


class TestYOLOXBottleneck(unittest.TestCase):
    def test_2d_shortcut(self):
        mod = YOLOXBottleneck(2, 16, 16, shortcut=True)
        x = torch.randn(2, 16, 8, 8)
        self.assertEqual(mod(x).shape, x.shape)

    def test_2d_no_shortcut(self):
        mod = YOLOXBottleneck(2, 16, 32, shortcut=False)
        out = mod(torch.randn(2, 16, 8, 8))
        self.assertEqual(out.shape, (2, 32, 8, 8))

    def test_3d(self):
        mod = YOLOXBottleneck(3, 8, 8, shortcut=True)
        x = torch.randn(1, 8, 4, 4, 4)
        self.assertEqual(mod(x).shape, x.shape)

    def test_depthwise(self):
        mod = YOLOXBottleneck(2, 16, 16, depthwise=True)
        out = mod(torch.randn(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))


class TestYOLOXCSPLayer(unittest.TestCase):
    def test_2d(self):
        mod = YOLOXCSPLayer(2, 16, 16, n=2)
        out = mod(torch.randn(2, 16, 8, 8))
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_3d(self):
        mod = YOLOXCSPLayer(3, 16, 16, n=1)
        out = mod(torch.randn(1, 16, 4, 4, 4))
        self.assertEqual(out.shape, (1, 16, 4, 4, 4))

    def test_channel_change(self):
        mod = YOLOXCSPLayer(2, 32, 16, n=1, shortcut=False)
        out = mod(torch.randn(1, 32, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))


class TestYOLOXSPPBottleneck(unittest.TestCase):
    def test_2d(self):
        mod = YOLOXSPPBottleneck(2, 16, 16, kernel_sizes=(3, 5, 7))
        out = mod(torch.randn(1, 16, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8))

    def test_3d(self):
        mod = YOLOXSPPBottleneck(3, 16, 16, kernel_sizes=(3, 5))
        out = mod(torch.randn(1, 16, 8, 8, 8))
        self.assertEqual(out.shape, (1, 16, 8, 8, 8))


class TestYOLOXFocus(unittest.TestCase):
    def test_2d_output_shape(self):
        mod = YOLOXFocus(2, 1, 16, ksize=1)
        x = torch.randn(2, 1, 16, 16)
        out = mod(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_3d_output_shape(self):
        mod = YOLOXFocus(3, 1, 16, ksize=1)
        x = torch.randn(2, 1, 8, 8, 8)
        out = mod(x)
        self.assertEqual(out.shape, (2, 16, 4, 4, 4))

    def test_gradient_flows(self):
        mod = YOLOXFocus(2, 1, 4, ksize=1)
        x = torch.randn(1, 1, 8, 8, requires_grad=True)
        mod(x).sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
