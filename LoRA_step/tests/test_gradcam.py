"""Grad-CAM 数值与张量布局回归测试。"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from gradcam import GradCAM, feature_rank_stats


class ToChannelsLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class TinyImageClassifier(nn.Module):
    def __init__(self, channels_last_output: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.layout = ToChannelsLast() if channels_last_output else nn.Identity()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[-0.5]]]]))
            self.fc.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.conv(x))
        formatted = self.layout(features)
        if formatted.shape[-1] == 2:
            pooled = formatted.mean(dim=(1, 2))
        else:
            pooled = formatted.mean(dim=(2, 3))
        return self.fc(pooled)


class GradCAMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.x = torch.randn(1, 1, 8, 8)

    def test_channels_first_cam_shape_and_range(self) -> None:
        model = TinyImageClassifier().eval()
        with GradCAM(model, model.conv) as cam:
            result = cam(self.x, target_class=0)
        self.assertEqual(tuple(result.cam.shape), (8, 8))
        self.assertGreaterEqual(float(result.cam.min()), 0.0)
        self.assertLessEqual(float(result.cam.max()), 1.0)
        self.assertEqual(result.feature_shape, (1, 2, 8, 8))

    def test_channels_last_cam_shape_and_range(self) -> None:
        model = TinyImageClassifier(channels_last_output=True).eval()
        with GradCAM(model, model.layout, channels_last=True) as cam:
            result = cam(self.x, target_class=0)
        self.assertEqual(tuple(result.cam.shape), (8, 8))
        self.assertGreaterEqual(float(result.cam.min()), 0.0)
        self.assertLessEqual(float(result.cam.max()), 1.0)
        self.assertEqual(result.feature_shape, (1, 8, 8, 2))

    def test_no_grad_context_is_rejected(self) -> None:
        model = TinyImageClassifier().eval()
        with GradCAM(model, model.conv) as cam:
            with torch.no_grad():
                with self.assertRaises(RuntimeError):
                    cam(self.x)


class FeatureRankTest(unittest.TestCase):
    def test_rank_one_feature_map(self) -> None:
        feature_map = torch.zeros(2, 2, 2)
        feature_map[0, 0, 0] = 1.0
        feature_map[1, 0, 0] = 2.0
        stats = feature_rank_stats(feature_map)
        self.assertEqual(stats["raw_rank"], 1)
        self.assertAlmostEqual(float(stats["effective_rank"]), 1.0, places=5)
        self.assertAlmostEqual(float(stats["stable_rank"]), 1.0, places=5)
        self.assertEqual(stats["singular_values"][0], 1.0)

    def test_rank_two_feature_map_and_batch_input(self) -> None:
        feature_map = torch.zeros(1, 2, 2, 2)
        feature_map[0, 0, 0, 0] = 1.0
        feature_map[0, 1, 1, 1] = 1.0
        stats = feature_rank_stats(feature_map)
        self.assertEqual(stats["raw_rank"], 2)
        self.assertAlmostEqual(float(stats["effective_rank"]), 2.0, places=5)
        self.assertAlmostEqual(float(stats["stable_rank"]), 2.0, places=5)
        self.assertEqual(stats["spatial_positions"], 4)

    def test_all_zero_feature_map_has_zero_effective_rank(self) -> None:
        stats = feature_rank_stats(torch.zeros(3, 2, 2))
        self.assertEqual(stats["raw_rank"], 0)
        self.assertEqual(float(stats["effective_rank"]), 0.0)
        self.assertEqual(stats["singular_values"], [])


if __name__ == "__main__":
    unittest.main()
