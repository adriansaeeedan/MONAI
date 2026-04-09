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
from collections import OrderedDict

import torch
from parameterized import parameterized

from monai.networks.blocks.backbone_fpn_utils import _resnet_bifpn_extractor
from monai.networks.blocks.bifpn import BiFPN, BiFPNLayer, FastNormalizedFusion
from monai.networks.nets.resnet import resnet50
from monai.utils import optional_import
from tests.test_utils import test_script_save as run_script_save

_, has_torchvision = optional_import("torchvision")

# ---------------------------------------------------------------------------
# Test cases for BiFPN forward pass
# Format: [constructor_kwargs, (input_shape_0, input_shape_1), (expected_shape_0, expected_shape_1)]
# ---------------------------------------------------------------------------
TEST_CASES_2LEVEL = [
    # 3D, 2 levels
    [
        {"spatial_dims": 3, "in_channels_list": [32, 64], "out_channels": 6},
        ((7, 32, 16, 32, 64), (7, 64, 8, 16, 32)),
        ((7, 6, 16, 32, 64), (7, 6, 8, 16, 32)),
    ],
    # 2D, 2 levels
    [
        {"spatial_dims": 2, "in_channels_list": [32, 64], "out_channels": 6},
        ((7, 32, 16, 32), (7, 64, 8, 16)),
        ((7, 6, 16, 32), (7, 6, 8, 16)),
    ],
    # 3D, depthwise separable
    [
        {"spatial_dims": 3, "in_channels_list": [32, 64], "out_channels": 16, "depthwise_separable": True},
        ((2, 32, 16, 16, 16), (2, 64, 8, 8, 8)),
        ((2, 16, 16, 16, 16), (2, 16, 8, 8, 8)),
    ],
    # 2D, depthwise separable
    [
        {"spatial_dims": 2, "in_channels_list": [32, 64], "out_channels": 16, "depthwise_separable": True},
        ((2, 32, 16, 16), (2, 64, 8, 8)),
        ((2, 16, 16, 16), (2, 16, 8, 8)),
    ],
    # 3D, num_repeats=1
    [
        {"spatial_dims": 3, "in_channels_list": [32, 64], "out_channels": 8, "num_repeats": 1},
        ((2, 32, 8, 8, 8), (2, 64, 4, 4, 4)),
        ((2, 8, 8, 8, 8), (2, 8, 4, 4, 4)),
    ],
    # 2D, num_repeats=5
    [
        {"spatial_dims": 2, "in_channels_list": [32, 64], "out_channels": 8, "num_repeats": 5},
        ((2, 32, 16, 16), (2, 64, 8, 8)),
        ((2, 8, 16, 16), (2, 8, 8, 8)),
    ],
]

# Test cases with 3 levels
TEST_CASES_3LEVEL = [
    # 2D, 3 levels
    [
        {"spatial_dims": 2, "in_channels_list": [16, 32, 64], "out_channels": 8},
        ((2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 8, 8)),
        ((2, 8, 32, 32), (2, 8, 16, 16), (2, 8, 8, 8)),
    ],
    # 3D, 3 levels
    [
        {"spatial_dims": 3, "in_channels_list": [16, 32, 64], "out_channels": 8},
        ((2, 16, 16, 16, 16), (2, 32, 8, 8, 8), (2, 64, 4, 4, 4)),
        ((2, 8, 16, 16, 16), (2, 8, 8, 8, 8), (2, 8, 4, 4, 4)),
    ],
]

# Test cases for backbone+BiFPN extractor (requires torchvision).
# BiFPN requires at least 2 feature levels, so we return layers [1, 2].
TEST_CASES_BACKBONE = [
    [
        {"spatial_dims": 3, "returned_layers": [1, 2]},
        (2, 3, 32, 64, 32),
        # layer1 output: stride-2, layer2 output: stride-4 relative to input
        ((2, 256, 16, 32, 16), (2, 256, 8, 16, 8), (2, 256, 4, 8, 8)),
    ]
]


def _run_bifpn_training_steps(net: BiFPN, data: OrderedDict[str, torch.Tensor], steps: int = 3) -> None:
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = net(data)
        for value in result.values():
            assert torch.isfinite(value).all()

        loss = sum(value.square().mean() for value in result.values())
        assert torch.isfinite(loss)
        loss.backward()

        for parameter in net.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()

    for parameter in net.parameters():
        assert torch.isfinite(parameter).all()


class TestFastNormalizedFusion(unittest.TestCase):
    def test_output_shape_2input(self):
        fusion = FastNormalizedFusion(num_inputs=2)
        a = torch.rand(2, 8, 16, 16)
        b = torch.rand(2, 8, 16, 16)
        out = fusion([a, b])
        self.assertEqual(out.shape, a.shape)

    def test_output_shape_3input(self):
        fusion = FastNormalizedFusion(num_inputs=3)
        a = torch.rand(2, 8, 4, 4, 4)
        b = torch.rand(2, 8, 4, 4, 4)
        c = torch.rand(2, 8, 4, 4, 4)
        out = fusion([a, b, c])
        self.assertEqual(out.shape, a.shape)

    def test_weights_are_learnable(self):
        fusion = FastNormalizedFusion(num_inputs=2)
        a = torch.rand(1, 4, 8, 8, requires_grad=True)
        b = torch.rand(1, 4, 8, 8, requires_grad=True)
        out = fusion([a, b])
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(fusion.weights.grad)

    def test_invalid_num_inputs(self):
        with self.assertRaises(ValueError):
            FastNormalizedFusion(num_inputs=1)

    def test_weights_non_negative_after_forward(self):
        """Normalized fusion weights should always stay finite and non-negative."""
        fusion = FastNormalizedFusion(num_inputs=2)
        # Manually set weights to negative values.
        with torch.no_grad():
            fusion.weights.fill_(-1.0)
        a = torch.rand(1, 4, 4, 4)
        b = torch.rand(1, 4, 4, 4)
        # With all-negative raw weights, softplus keeps the normalized weights positive.
        out = fusion([a, b])
        # Should not produce NaN/Inf.
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_negative_weights_still_receive_gradients(self):
        fusion = FastNormalizedFusion(num_inputs=2)
        with torch.no_grad():
            fusion.weights.copy_(torch.tensor([-5.0, -4.0]))

        a = torch.rand(1, 4, 4, 4, requires_grad=True)
        b = torch.rand(1, 4, 4, 4, requires_grad=True)
        out = fusion([a, b])
        loss = out.sum()
        loss.backward()

        self.assertGreater(float(out.abs().sum().detach()), 0.0)
        self.assertIsNotNone(fusion.weights.grad)
        self.assertGreater(float(fusion.weights.grad.abs().sum()), 0.0)


class TestBiFPNLayer(unittest.TestCase):
    def test_output_shapes_2d_2level(self):
        layer = BiFPNLayer(spatial_dims=2, num_levels=2, out_channels=8)
        feat0 = torch.rand(2, 8, 16, 16)
        feat1 = torch.rand(2, 8, 8, 8)
        outputs = layer([feat0, feat1])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, feat0.shape)
        self.assertEqual(outputs[1].shape, feat1.shape)

    def test_output_shapes_3d_3level(self):
        layer = BiFPNLayer(spatial_dims=3, num_levels=3, out_channels=16)
        feat0 = torch.rand(2, 16, 8, 8, 8)
        feat1 = torch.rand(2, 16, 4, 4, 4)
        feat2 = torch.rand(2, 16, 2, 2, 2)
        outputs = layer([feat0, feat1, feat2])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, feat0.shape)
        self.assertEqual(outputs[1].shape, feat1.shape)
        self.assertEqual(outputs[2].shape, feat2.shape)

    def test_invalid_num_levels(self):
        with self.assertRaises(ValueError):
            BiFPNLayer(spatial_dims=2, num_levels=1, out_channels=8)


class TestBiFPNBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2LEVEL)
    def test_bifpn_2level(self, input_param, input_shapes, expected_shapes):
        net = BiFPN(**input_param)
        data = OrderedDict()
        data["feat0"] = torch.rand(input_shapes[0])
        data["feat1"] = torch.rand(input_shapes[1])
        result = net(data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["feat0"].shape, expected_shapes[0])
        self.assertEqual(result["feat1"].shape, expected_shapes[1])

    @parameterized.expand(TEST_CASES_3LEVEL)
    def test_bifpn_3level(self, input_param, input_shapes, expected_shapes):
        net = BiFPN(**input_param)
        data = OrderedDict()
        data["feat0"] = torch.rand(input_shapes[0])
        data["feat1"] = torch.rand(input_shapes[1])
        data["feat2"] = torch.rand(input_shapes[2])
        result = net(data)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["feat0"].shape, expected_shapes[0])
        self.assertEqual(result["feat1"].shape, expected_shapes[1])
        self.assertEqual(result["feat2"].shape, expected_shapes[2])

    def test_extra_blocks_last_level_max_pool(self):
        from monai.networks.blocks.feature_pyramid_network import LastLevelMaxPool

        extra = LastLevelMaxPool(spatial_dims=2)
        net = BiFPN(spatial_dims=2, in_channels_list=[32, 64], out_channels=8, extra_blocks=extra)
        data = OrderedDict()
        data["feat0"] = torch.rand(2, 32, 16, 16)
        data["feat1"] = torch.rand(2, 64, 8, 8)
        result = net(data)
        # extra_blocks appends a "pool" level
        self.assertIn("pool", result)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["pool"].shape, (2, 8, 4, 4))

    def test_extra_blocks_p6p7(self):
        from monai.networks.blocks.feature_pyramid_network import LastLevelP6P7

        extra = LastLevelP6P7(spatial_dims=2, in_channels=8, out_channels=8)
        net = BiFPN(spatial_dims=2, in_channels_list=[32, 64], out_channels=8, extra_blocks=extra)
        data = OrderedDict()
        data["feat0"] = torch.rand(2, 32, 64, 64)
        data["feat1"] = torch.rand(2, 64, 32, 32)
        result = net(data)
        self.assertIn("p6", result)
        self.assertIn("p7", result)
        self.assertEqual(len(result), 4)

    def test_gradient_flow_through_weights(self):
        """Ensure gradients flow to the fast normalized fusion weights."""
        net = BiFPN(spatial_dims=2, in_channels_list=[16, 32], out_channels=8, num_repeats=1)
        data = OrderedDict()
        data["feat0"] = torch.rand(1, 16, 8, 8)
        data["feat1"] = torch.rand(1, 32, 4, 4)
        result = net(data)
        loss = sum(v.sum() for v in result.values())
        loss.backward()
        # Check that at least one fusion weight has a gradient.
        fusion_layer = net.bifpn_layers[0]
        self.assertIsNotNone(fusion_layer.td_fusions[0].weights.grad)
        self.assertIsNotNone(fusion_layer.bu_top_fusion.weights.grad)

    def test_invalid_in_channels_list(self):
        with self.assertRaises(ValueError):
            BiFPN(spatial_dims=2, in_channels_list=[32], out_channels=8)

    def test_invalid_num_repeats(self):
        with self.assertRaises(ValueError):
            BiFPN(spatial_dims=2, in_channels_list=[32, 64], out_channels=8, num_repeats=0)

    def test_output_keys_preserved(self):
        """Output OrderedDict must retain original key names."""
        net = BiFPN(spatial_dims=2, in_channels_list=[16, 32, 64], out_channels=8)
        data = OrderedDict()
        data["p3"] = torch.rand(1, 16, 32, 32)
        data["p4"] = torch.rand(1, 32, 16, 16)
        data["p5"] = torch.rand(1, 64, 8, 8)
        result = net(data)
        self.assertEqual(list(result.keys()), ["p3", "p4", "p5"])

    def test_odd_spatial_dims(self):
        """Non-power-of-2 spatial dimensions must not crash (P1 regression guard)."""
        net = BiFPN(spatial_dims=2, in_channels_list=[16, 32, 64], out_channels=8)
        data = OrderedDict()
        data["p3"] = torch.rand(1, 16, 13, 17)  # odd sizes
        data["p4"] = torch.rand(1, 32, 7, 9)
        data["p5"] = torch.rand(1, 64, 4, 5)
        result = net(data)
        self.assertEqual(result["p3"].shape, (1, 8, 13, 17))
        self.assertEqual(result["p4"].shape, (1, 8, 7, 9))
        self.assertEqual(result["p5"].shape, (1, 8, 4, 5))

    def test_3d_4level(self):
        """3D input with 4 feature levels."""
        net = BiFPN(spatial_dims=3, in_channels_list=[16, 32, 64, 128], out_channels=32)
        data = OrderedDict()
        data["p2"] = torch.rand(1, 16, 16, 16, 16)
        data["p3"] = torch.rand(1, 32, 8, 8, 8)
        data["p4"] = torch.rand(1, 64, 4, 4, 4)
        data["p5"] = torch.rand(1, 128, 2, 2, 2)
        result = net(data)
        self.assertEqual(result["p2"].shape, (1, 32, 16, 16, 16))
        self.assertEqual(result["p3"].shape, (1, 32, 8, 8, 8))
        self.assertEqual(result["p4"].shape, (1, 32, 4, 4, 4))
        self.assertEqual(result["p5"].shape, (1, 32, 2, 2, 2))

    def test_group_norm_can_be_selected(self):
        net = BiFPN(
            spatial_dims=2,
            in_channels_list=[16, 32, 64],
            out_channels=8,
            norm=("group", {"num_groups": 4}),
        )
        group_norms = [m for m in net.modules() if isinstance(m, torch.nn.GroupNorm)]
        batch_norms = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.assertGreater(len(group_norms), 0)
        self.assertEqual(len(batch_norms), 0)

    def test_batch_norm_parameters_are_configurable(self):
        net = BiFPN(
            spatial_dims=2,
            in_channels_list=[16, 32, 64],
            out_channels=8,
            norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
        )
        batch_norms = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertGreater(len(batch_norms), 0)
        for norm_layer in batch_norms:
            self.assertAlmostEqual(norm_layer.eps, 1e-3)
            self.assertAlmostEqual(norm_layer.momentum, 0.01)

    def test_training_loop_keeps_outputs_finite_with_group_norm(self):
        net = BiFPN(
            spatial_dims=2,
            in_channels_list=[16, 32, 64],
            out_channels=8,
            num_repeats=2,
            norm=("group", {"num_groups": 4}),
        )
        data = OrderedDict(
            {
                "p3": torch.randn(1, 16, 32, 32),
                "p4": torch.randn(1, 32, 16, 16),
                "p5": torch.randn(1, 64, 8, 8),
            }
        )
        _run_bifpn_training_steps(net, data)

    def test_training_loop_keeps_bn_stats_finite_with_custom_batch_norm(self):
        net = BiFPN(
            spatial_dims=2,
            in_channels_list=[16, 32, 64],
            out_channels=8,
            num_repeats=2,
            norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
        )
        data = OrderedDict(
            {
                "p3": torch.randn(1, 16, 32, 32),
                "p4": torch.randn(1, 32, 16, 16),
                "p5": torch.randn(1, 64, 8, 8),
            }
        )
        _run_bifpn_training_steps(net, data)

        for norm_layer in (m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)):
            self.assertTrue(torch.isfinite(norm_layer.running_mean).all())
            self.assertTrue(torch.isfinite(norm_layer.running_var).all())


class TestBiFPNScript(unittest.TestCase):
    @parameterized.expand(TEST_CASES_2LEVEL[:2])  # 2D and 3D base cases
    def test_script_2level(self, input_param, input_shapes, expected_shapes):
        net = BiFPN(**input_param)
        data = OrderedDict()
        data["feat0"] = torch.rand(input_shapes[0])
        data["feat1"] = torch.rand(input_shapes[1])
        run_script_save(net, data)


@unittest.skipUnless(has_torchvision, "Requires torchvision")
class TestBiFPNWithBackbone(unittest.TestCase):
    @parameterized.expand(TEST_CASES_BACKBONE)
    def test_bifpn_backbone(self, input_param, input_shape, expected_shapes):
        net = _resnet_bifpn_extractor(
            backbone=resnet50(spatial_dims=input_param["spatial_dims"]),
            spatial_dims=input_param["spatial_dims"],
            returned_layers=input_param["returned_layers"],
        )
        data = torch.rand(input_shape)
        result = net(data)
        # Returned layer keys are "0", "1", ...; LastLevelMaxPool adds "pool".
        returned = input_param["returned_layers"]
        for i in range(len(returned)):
            self.assertIn(str(i), result)
            # All output levels should have out_channels=256 channels.
            self.assertEqual(result[str(i)].shape[1], 256)
        self.assertIn("pool", result)
        self.assertEqual(result["pool"].shape[1], 256)

    @parameterized.expand(TEST_CASES_BACKBONE)
    def test_script_backbone(self, input_param, input_shape, expected_shapes):
        net = _resnet_bifpn_extractor(
            backbone=resnet50(spatial_dims=input_param["spatial_dims"]),
            spatial_dims=input_param["spatial_dims"],
            returned_layers=input_param["returned_layers"],
        )
        data = torch.rand(input_shape)
        run_script_save(net, data)


if __name__ == "__main__":
    unittest.main()
