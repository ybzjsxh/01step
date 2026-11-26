"""运行：uv run --no-sync python3.12 -m unittest discover -s tests -v"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lora_core import (
    LoRAConfig,
    LoRAConv2d,
    LoRALinear,
    classifier_state_dict,
    count_parameters,
    inject_lora,
    load_adapter,
    lora_state_dict,
    merge_lora,
    save_adapter,
)
from train import build_model, stratified_indices


class LoRALinearTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.base = nn.Linear(8, 5)
        self.x = torch.randn(3, 8)

    def test_zero_initialization_preserves_output(self) -> None:
        wrapped = LoRALinear(self.base, r=2, alpha=4)
        wrapped.eval()
        with torch.no_grad():
            torch.testing.assert_close(wrapped(self.x), self.base(self.x))

    def test_backward_only_updates_lora(self) -> None:
        wrapped = LoRALinear(self.base, r=2, alpha=4)
        with torch.no_grad():
            wrapped.lora_B.weight.normal_(0, 0.01)
        wrapped(self.x).sum().backward()
        self.assertIsNone(wrapped.base.weight.grad)
        self.assertIsNone(wrapped.base.bias.grad)
        self.assertIsNotNone(wrapped.lora_A.weight.grad)
        self.assertIsNotNone(wrapped.lora_B.weight.grad)
        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_A.weight.requires_grad)

    def test_merge_preserves_output(self) -> None:
        wrapped = LoRALinear(self.base, r=2, alpha=4)
        with torch.no_grad():
            wrapped.lora_A.weight.normal_(0, 0.02)
            wrapped.lora_B.weight.normal_(0, 0.02)
        wrapped.eval()
        with torch.no_grad():
            expected = wrapped(self.x)
            functional = torch.nn.functional.linear(self.x, wrapped.weight, wrapped.bias)
            torch.testing.assert_close(functional, expected)
            merged = wrapped.merge()
            actual = merged(self.x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


class LoRAConv2dTest(unittest.TestCase):
    def test_merge_preserves_output_for_strided_3x3_conv(self) -> None:
        torch.manual_seed(11)
        base = nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1, bias=True)
        wrapped = LoRAConv2d(base, r=3, alpha=6)
        with torch.no_grad():
            wrapped.lora_A.weight.normal_(0, 0.02)
            wrapped.lora_B.weight.normal_(0, 0.02)
        x = torch.randn(2, 8, 16, 16)
        wrapped.eval()
        with torch.no_grad():
            expected = wrapped(x)
            merged = wrapped.merge()
            actual = merged(x)
        self.assertEqual(expected.shape, (2, 12, 8, 8))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_first_resnet_conv_accepts_rank_four(self) -> None:
        base = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        wrapped = LoRAConv2d(base, r=4, alpha=8)
        x = torch.randn(1, 3, 32, 32)
        self.assertEqual(wrapped(x).shape, base(x).shape)


class ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(6, 4)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.activation(self.block(x)))


class AdapterIOTest(unittest.TestCase):
    def test_save_and_load_restore_lora_and_classifier(self) -> None:
        torch.manual_seed(21)
        model = ToyClassifier()
        config = LoRAConfig(r=2, alpha=4, target="all_linear")
        targets = inject_lora(model, config, head_names=("fc",))
        self.assertEqual(targets, ["block"])
        with torch.no_grad():
            model.block.lora_A.weight.normal_(0, 0.02)
            model.block.lora_B.weight.normal_(0, 0.02)
        x = torch.randn(5, 6)
        model.eval()
        expected = model(x)

        restored = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in restored.parameters():
                if parameter.requires_grad:
                    parameter.normal_(0, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(
                model, path, config, targets, head_names=("fc",), metadata={"test": True}
            )
            self.assertLess(path.stat().st_size, 20_000)
            payload = load_adapter(restored, path)
        self.assertEqual(payload["metadata"]["test"], True)
        restored.eval()
        torch.testing.assert_close(restored(x), expected)
        for key, value in lora_state_dict(model).items():
            torch.testing.assert_close(lora_state_dict(restored)[key], value)
        for key, value in classifier_state_dict(model, ("fc",)).items():
            torch.testing.assert_close(classifier_state_dict(restored, ("fc",))[key], value)

    def test_recursive_merge_restores_plain_modules(self) -> None:
        model = ToyClassifier()
        inject_lora(model, LoRAConfig(r=2, target="all_linear"), head_names=("fc",))
        merge_lora(model)
        self.assertIsInstance(model.block, nn.Linear)
        self.assertNotIsInstance(model.block, LoRALinear)
        self.assertEqual(lora_state_dict(model), {})


class ModelInjectionTest(unittest.TestCase):
    def test_resnet18_all_conv_and_classifier_trainable(self) -> None:
        model, targets, _ = build_model(
            "resnet18", pretrained=False, num_classes=10, image_size=32,
            lora_config=LoRAConfig(r=4, alpha=8, target="all_conv"),
        )
        self.assertEqual(len(targets), 20)
        stats = count_parameters(model)
        self.assertGreater(stats["lora"], 0)
        for name, parameter in model.named_parameters():
            expected_trainable = ".lora_A." in name or ".lora_B." in name or name.startswith("fc.")
            self.assertEqual(parameter.requires_grad, expected_trainable, name)

    def test_swin_has_24_attention_injection_points(self) -> None:
        model, targets, _ = build_model(
            "swin_tiny", pretrained=False, num_classes=10, image_size=64,
            lora_config=LoRAConfig(r=4, alpha=8, target="attention"),
        )
        self.assertEqual(len(targets), 24)
        self.assertTrue(all(name.endswith(("qkv", "proj")) for name in targets))
        self.assertTrue(model.head.weight.requires_grad)
        self.assertFalse(model.features[0][0].weight.requires_grad)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(output.shape, (1, 10))


class DataTest(unittest.TestCase):
    def test_stratified_subset_is_balanced_and_deterministic(self) -> None:
        targets = [index % 10 for index in range(1000)]
        first = stratified_indices(targets, limit=137, seed=42)
        second = stratified_indices(targets, limit=137, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 137)
        counts = [sum(targets[index] == label for index in first) for label in range(10)]
        self.assertLessEqual(max(counts) - min(counts), 1)


if __name__ == "__main__":
    unittest.main()
