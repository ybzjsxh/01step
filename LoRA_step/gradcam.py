"""不依赖 CAM 库的 Grad-CAM 教学实现。

Grad-CAM 的核心：
1. 前向时保存目标层特征 A；
2. 对“预测类别分数”反向传播，得到梯度；
3. 对梯度做全局平均，得到每个通道的权重；
4. 对 A 做通道加权求和、ReLU，再上采样到输入分辨率。

支持 ResNet 的 [B, C, H, W] 特征，也支持 torchvision Swin 的
[B, H, W, C] 和 [B, N, C] 特征。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    cam: Tensor
    predicted_class: int
    confidence: float
    target_score: float
    feature_shape: tuple[int, ...]
    feature_map: Tensor


def feature_rank_stats(feature_map: Tensor) -> dict[str, float | int | list[float]]:
    """分析单个目标层特征矩阵 [C, H*W] 的谱结构。

    原始矩阵秩的上限是 min(C, H*W)，在小特征图上很容易饱和。
    因此同时计算：
    - entropy effective rank：exp(奇异值能量分布的熵)；
    - stable rank：||A||_F² / ||A||_2²；
    - 归一化奇异值谱，用来看能量集中程度。
    """
    if feature_map.dim() == 3:
        matrix = feature_map
    elif feature_map.dim() == 4 and feature_map.shape[0] == 1:
        matrix = feature_map[0]
    else:
        raise ValueError(f"feature_map 应为 [C,H,W]，实际为 {tuple(feature_map.shape)}")
    matrix = matrix.detach().float().cpu().reshape(matrix.shape[0], -1).double()
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        raise ValueError("特征图为空，无法计算秩")
    energy = singular_values.square()
    total_energy = energy.sum()
    if float(total_energy) <= 1e-12:
        return {
            "raw_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "spectral_entropy": 0.0,
            "channel_count": int(matrix.shape[0]),
            "spatial_positions": int(matrix.shape[1]),
            "singular_values": [],
        }

    probabilities = energy / total_energy
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * positive.log()).sum().item())
    effective_rank = math.exp(entropy)
    stable_rank = float(total_energy / energy[0].clamp_min(1e-12))
    raw_rank = int(torch.linalg.matrix_rank(matrix).item())
    normalized_spectrum = (singular_values / singular_values[0].clamp_min(1e-12)).tolist()
    return {
        "raw_rank": raw_rank,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "spectral_entropy": entropy,
        "channel_count": int(matrix.shape[0]),
        "spatial_positions": int(matrix.shape[1]),
        "singular_values": normalized_spectrum,
    }


class GradCAM:
    """对单个输入生成一张归一化到 [0, 1] 的类别激活图。"""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        channels_last: bool = False,
    ) -> None:
        if channels_last not in (True, False):
            raise ValueError("channels_last 必须是布尔值")
        self.model = model
        self.target_layer = target_layer
        self.channels_last = channels_last
        self._activation: Tensor | None = None
        self._gradient: Tensor | None = None
        self._gradient_handles: list[Any] = []
        self._forward_handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            raise RuntimeError("目标层输出不是 Tensor，无法计算 Grad-CAM")
        if not tensor.requires_grad:
            raise RuntimeError(
                "目标层输出没有梯度。请选择包含 LoRA/可训练参数的层，并在 grad 模式下调用。"
            )
        self._activation = tensor
        self._gradient_handles.append(tensor.register_hook(self._capture_gradient))

    def _capture_gradient(self, gradient: Tensor) -> None:
        self._gradient = gradient.detach()

    def _spatial_feature(self, tensor: Tensor, name: str) -> Tensor:
        """统一转换为 [B, C, H, W]，便于通道加权。"""
        if self.channels_last:
            if tensor.dim() == 4:
                batch, height, width, channels = tensor.shape
                return tensor.permute(0, 3, 1, 2).reshape(batch, channels, height, width)
            if tensor.dim() == 3:
                batch, tokens, channels = tensor.shape
                side = math.isqrt(tokens)
                if side * side != tokens:
                    raise RuntimeError(f"{name} 的 token 数 {tokens} 不是完全平方数")
                return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 3:
            batch, tokens, channels = tensor.shape
            side = math.isqrt(tokens)
            if side * side != tokens:
                raise RuntimeError(f"{name} 的 token 数 {tokens} 不是完全平方数")
            return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
        raise RuntimeError(f"{name} 应为 3D 或 4D 特征，实际为 {tuple(tensor.shape)}")

    def __call__(self, inputs: Tensor, target_class: int | None = None) -> GradCAMResult:
        if not torch.is_grad_enabled():
            raise RuntimeError("Grad-CAM 需要梯度，调用时不要包在 torch.no_grad() 中")
        if inputs.shape[0] != 1:
            raise ValueError("为了简单和可视化清晰，本实现一次只处理 batch=1")

        self.model.zero_grad(set_to_none=True)
        self._activation = None
        self._gradient = None
        logits = self.model(inputs)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        if not 0 <= target_class < logits.shape[1]:
            raise ValueError(f"target_class 越界：{target_class}")

        score = logits[0, target_class]
        score.backward()
        if self._activation is None or self._gradient is None:
            raise RuntimeError("没有捕获到目标层特征或梯度，请检查目标层是否参与前向计算")

        activation = self._spatial_feature(self._activation.detach(), "activation")
        gradient = self._spatial_feature(self._gradient, "gradient")
        if activation.shape != gradient.shape:
            raise RuntimeError(
                f"特征和梯度形状不一致：{tuple(activation.shape)} vs {tuple(gradient.shape)}"
            )

        # 每个通道一个重要性权重：对该通道所有空间位置的梯度取平均。
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        maximum = cam.amax(dim=(2, 3), keepdim=True)
        cam = cam / maximum.clamp_min(1e-8)
        cam = cam[0, 0].detach().cpu()

        probabilities = logits.detach().softmax(dim=1)[0].cpu()
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles.clear()
        return GradCAMResult(
            cam=cam,
            predicted_class=target_class,
            confidence=float(probabilities[target_class].item()),
            target_score=float(score.detach().cpu().item()),
            feature_shape=tuple(self._activation.shape),
            feature_map=activation[0].detach().cpu(),
        )

    def close(self) -> None:
        self._forward_handle.remove()
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor):
                return item
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, Tensor):
                return item
    return None
