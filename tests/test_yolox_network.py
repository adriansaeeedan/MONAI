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

from monai.apps.detection.networks.yolox_network import (
    YOLOPAFPN,
    YOLOXHeadModule,
    YOLOXNetwork,
    yolox_darknet_pafpn_network,
)


class TestYOLOPAFPN(unittest.TestCase):
    def test_2d_output_shape(self):
        neck = YOLOPAFPN(spatial_dims=2, depth=0.33, width=0.25)
        p3, p4, p5 = neck(torch.randn(2, 1, 128, 128))
        H = 128
        self.assertEqual(p3.shape[-2:], (H // 8, H // 8))
        self.assertEqual(p4.shape[-2:], (H // 16, H // 16))
        self.assertEqual(p5.shape[-2:], (H // 32, H // 32))

    def test_3d_output_shape(self):
        neck = YOLOPAFPN(spatial_dims=3, depth=0.33, width=0.25)
        p3, p4, p5 = neck(torch.randn(1, 1, 64, 64, 64))
        self.assertEqual(p3.shape[-3:], (8, 8, 8))
        self.assertEqual(p5.shape[-3:], (2, 2, 2))

    def test_out_channels_attr(self):
        neck = YOLOPAFPN(spatial_dims=2, depth=0.33, width=0.50)
        self.assertIsInstance(neck.out_channels, int)
        self.assertEqual(neck.out_channels, int(256 * 0.50))


class TestYOLOXNetwork(unittest.TestCase):
    def _make_network(self, spatial_dims, width=0.25, depth=0.33):
        return YOLOXNetwork(
            spatial_dims=spatial_dims,
            num_classes=5,
            depth=depth,
            width=width,
        )

    def test_2d_output_keys(self):
        net = self._make_network(spatial_dims=2)
        out = net(torch.randn(2, 1, 128, 128))
        self.assertIn(net.cls_key, out)
        self.assertIn(net.box_reg_key, out)
        self.assertIn(net.obj_key, out)
        self.assertEqual(len(out[net.cls_key]), 3)   # 3 FPN levels

    def test_2d_cls_shape(self):
        net = self._make_network(spatial_dims=2)
        out = net(torch.randn(2, 1, 128, 128))
        cls = out[net.cls_key][0]
        self.assertEqual(cls.shape[0], 2)       # batch
        self.assertEqual(cls.shape[1], 5)       # num_classes
        self.assertEqual(len(cls.shape), 4)     # 2D spatial

    def test_2d_box_reg_shape(self):
        net = self._make_network(spatial_dims=2)
        out = net(torch.randn(2, 1, 128, 128))
        reg = out[net.box_reg_key][0]
        self.assertEqual(reg.shape[1], 4)       # 2 * spatial_dims

    def test_2d_obj_shape(self):
        net = self._make_network(spatial_dims=2)
        out = net(torch.randn(2, 1, 128, 128))
        obj = out[net.obj_key][0]
        self.assertEqual(obj.shape[1], 1)

    def test_3d_output_keys(self):
        net = self._make_network(spatial_dims=3)
        out = net(torch.randn(1, 1, 64, 64, 64))
        self.assertIn(net.cls_key, out)
        self.assertIn(net.box_reg_key, out)
        self.assertIn(net.obj_key, out)

    def test_3d_box_reg_shape(self):
        net = self._make_network(spatial_dims=3)
        out = net(torch.randn(1, 1, 64, 64, 64))
        reg = out[net.box_reg_key][0]
        self.assertEqual(reg.shape[1], 6)       # 2 * spatial_dims = 6

    def test_list_output_mode(self):
        net = YOLOXNetwork(spatial_dims=2, num_classes=3, width=0.25, depth=0.33, use_list_output=True)
        out = net(torch.randn(1, 1, 64, 64))
        # 3 levels × 3 output types = 9 tensors
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 9)

    def test_attributes_exist(self):
        net = self._make_network(spatial_dims=2)
        for attr in ("spatial_dims", "num_classes", "cls_key", "box_reg_key", "obj_key", "strides", "size_divisible"):
            self.assertTrue(hasattr(net, attr), f"Missing attribute: {attr}")

    def test_gradient_flows_2d(self):
        net = self._make_network(spatial_dims=2)
        x = torch.randn(1, 1, 128, 128, requires_grad=True)
        out = net(x)
        loss = sum(t.sum() for maps in out.values() for t in maps)
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_factory_function(self):
        for size in ("nano", "tiny", "s", "m", "l", "x"):
            net = yolox_darknet_pafpn_network(spatial_dims=2, num_classes=3, model_size=size)
            self.assertIsInstance(net, YOLOXNetwork)

    def test_factory_invalid_size(self):
        with self.assertRaises(ValueError):
            yolox_darknet_pafpn_network(2, 3, model_size="xl")


class TestYOLOXHeadModule(unittest.TestCase):
    def test_initialize_biases_rejects_invalid_prior_probability(self):
        head = YOLOXHeadModule(spatial_dims=2, num_classes=3, width=0.25, in_channels=(256,))
        for prior_prob in (0.0, 1.0, -0.1, float("nan"), float("inf")):
            with self.subTest(prior_prob=prior_prob):
                with self.assertRaises(ValueError):
                    head.initialize_biases(prior_prob=prior_prob)


if __name__ == "__main__":
    unittest.main()
