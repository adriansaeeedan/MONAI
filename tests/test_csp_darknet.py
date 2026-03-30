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

from monai.networks.nets.csp_darknet import CSPDarkNet


class TestCSPDarkNet(unittest.TestCase):
    def test_2d_default(self):
        net = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.50)
        x = torch.randn(1, 1, 128, 128)
        out = net(x)
        self.assertIn("dark3", out)
        self.assertIn("dark4", out)
        self.assertIn("dark5", out)
        # dark3 → stride 8, dark4 → stride 16, dark5 → stride 32
        self.assertEqual(out["dark3"].shape[-2:], (16, 16))
        self.assertEqual(out["dark4"].shape[-2:], (8, 8))
        self.assertEqual(out["dark5"].shape[-2:], (4, 4))

    def test_3d_default(self):
        net = CSPDarkNet(spatial_dims=3, depth_mul=0.33, width_mul=0.25)
        x = torch.randn(1, 1, 64, 64, 64)
        out = net(x)
        self.assertIn("dark3", out)
        self.assertIn("dark4", out)
        self.assertIn("dark5", out)
        self.assertEqual(out["dark3"].shape[-3:], (8, 8, 8))
        self.assertEqual(out["dark5"].shape[-3:], (2, 2, 2))

    def test_custom_out_features(self):
        net = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.25, out_features=("dark4", "dark5"))
        out = net(torch.randn(1, 1, 64, 64))
        self.assertNotIn("dark3", out)
        self.assertIn("dark4", out)
        self.assertIn("dark5", out)

    def test_out_channels_attribute(self):
        net = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.50)
        base_ch = int(0.50 * 64)
        self.assertEqual(net.out_channels["dark3"], base_ch * 4)
        self.assertEqual(net.out_channels["dark5"], base_ch * 16)

    def test_multi_channel_input(self):
        net = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.25, in_channels=3)
        out = net(torch.randn(1, 3, 64, 64))
        self.assertIn("dark3", out)

    def test_invalid_out_features(self):
        with self.assertRaises(ValueError):
            CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.50, out_features=("dark6",))

    def test_gradient_flows_2d(self):
        net = CSPDarkNet(spatial_dims=2, depth_mul=0.33, width_mul=0.25)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = net(x)
        sum(v.sum() for v in out.values()).backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
