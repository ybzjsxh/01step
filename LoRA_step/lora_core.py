"""一个只依赖 PyTorch 的教学版 LoRA 实现。

支持：
1. nn.Linear：适用于 Swin Transformer 的 qkv / proj。
2. nn.Conv2d：适用于 ResNet 的普通卷积（groups=1）。
3. 冻结原参数、保存/加载适配器、把 LoRA 合并回原模型。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn


@dataclass
class LoRAConfig:
    """LoRA 超参数。alpha / r 决定低秩增量的缩放系数。"""

    r: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target: str = "attention"

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r 必须是正整数")
        if self.alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间")


class LoRALinear(nn.Module):
    """给已有 Linear 增加 ΔW = scaling · B · A。

    原 Linear 的权重为 W[out, in]，A[r, in]，B[out, r]。
    B 初始化为 0，所以训练开始时 ΔW = 0，不改变原模型输出。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r > min(base.in_features, base.out_features):
            raise ValueError(
                f"r={r} 不能大于 min(in_features, out_features)="
                f"{min(base.in_features, base.out_features)}"
            )
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self._move_adapters_to_base_device()
        self.reset_lora_parameters()
        self.freeze_base()

    def _move_adapters_to_base_device(self) -> None:
        self.lora_A.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.lora_B.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> Tensor:
        """兼容 torchvision Swin 直接读取 qkv.weight / proj.weight 的写法。"""
        if self.lora_dropout.p != 0:
            raise RuntimeError(
                "外部通过 F.linear 读取有效权重时无法应用输入 dropout；"
                "请将 Swin 的 LoRA dropout 设为 0"
            )
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        return self.base.weight + delta.to(dtype=self.base.weight.dtype)

    @property
    def bias(self) -> Tensor | None:
        """与 nn.Linear 接口保持一致，供 F.linear 读取。"""
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        original = self.base(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original + self.scaling * update

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """把 ΔW 加进原权重，返回不含旁路的普通 Linear。"""
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base.weight.add_(delta.to(dtype=self.base.weight.dtype))
        return self.base


class LoRAConv2d(nn.Module):
    """给已有 Conv2d 增加低秩卷积旁路。

    把原卷积核展开成 W[out, in·kh·kw]，同样可写 ΔW = B · A。
    A 负责原卷积的空间变换，B 是 1×1 卷积并恢复输出通道数。
    教学版仅处理 groups=1；ResNet-18 的所有卷积都满足该条件。
    """

    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("教学版 LoRAConv2d 暂不支持分组卷积（groups != 1）")
        if base.padding_mode != "zeros":
            raise ValueError("教学版 LoRAConv2d 暂不支持非 zero padding")
        max_rank = min(base.out_channels, base.in_channels * base.kernel_size[0] * base.kernel_size[1])
        if r > max_rank:
            raise ValueError(f"r={r} 不能大于展开卷积核的最大秩 {max_rank}")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=r,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        self._move_adapters_to_base_device()
        self.reset_lora_parameters()
        self.freeze_base()

    def _move_adapters_to_base_device(self) -> None:
        self.lora_A.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.lora_B.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        original = self.base(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original + self.scaling * update

    @torch.no_grad()
    def merge(self) -> nn.Conv2d:
        """将 B[out,r,1,1] 与 A[r,in,kh,kw] 相乘后加回原卷积核。"""
        flat_a = self.lora_A.weight.reshape(self.r, -1)
        flat_b = self.lora_B.weight.reshape(self.base.out_channels, self.r)
        delta = self.scaling * (flat_b @ flat_a)
        self.base.weight.add_(delta.reshape_as(self.base.weight).to(self.base.weight.dtype))
        return self.base


LoRALayer = LoRALinear | LoRAConv2d


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    """根据点分层级名替换子模块，例如 layers.0.blocks.1.attn.qkv。"""
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent, child_name = model, name
    setattr(parent, child_name, replacement)


def _matches_target(name: str, module: nn.Module, target: str) -> bool:
    if target == "all_conv":
        return isinstance(module, nn.Conv2d)
    if target == "attention":
        return isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"qkv", "proj"}
    if target == "all_linear":
        return isinstance(module, nn.Linear)
    patterns = [item.strip() for item in target.split(",") if item.strip()]
    return bool(patterns) and any(pattern in name for pattern in patterns)


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    head_names: Iterable[str] = ("fc", "head"),
) -> list[str]:
    """原地注入 LoRA，冻结全部原参数，仅放开 LoRA 与新分类头。

    target 可选：
    - all_conv：所有普通卷积，ResNet 示例使用。
    - attention：所有名为 qkv / proj 的 Linear，Swin 示例使用。
    - all_linear：除分类头外的所有 Linear。
    - 逗号分隔的模块名片段：例如 "attn.qkv,attn.proj"。
    """
    head_names = tuple(head_names)
    replaced: list[str] = []
    # 先保存列表，避免遍历 named_modules 时同时修改模块树。
    candidates = list(model.named_modules())
    for name, module in candidates:
        if not name or name in head_names or not _matches_target(name, module, config.target):
            continue
        if isinstance(module, nn.Linear):
            replacement: nn.Module = LoRALinear(
                module, config.r, config.alpha, config.dropout
            )
        elif isinstance(module, nn.Conv2d):
            replacement = LoRAConv2d(module, config.r, config.alpha, config.dropout)
        else:
            continue
        _replace_module(model, name, replacement)
        replaced.append(name)

    if not replaced:
        raise ValueError(f"没有模块匹配 target={config.target!r}，请检查模块名")

    # 先冻结整个模型，再精确放开 LoRA 参数和分类头。
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
    for head_name in head_names:
        try:
            head = model.get_submodule(head_name)
        except AttributeError:
            continue
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    return replaced


def configure_head_only(model: nn.Module, head_names: Iterable[str]) -> None:
    """对照实验：冻结特征提取器，只训练分类头。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for head_name in head_names:
        try:
            head = model.get_submodule(head_name)
        except AttributeError:
            continue
        for parameter in head.parameters():
            parameter.requires_grad_(True)


def merge_lora(model: nn.Module) -> nn.Module:
    """递归合并所有 LoRA 层；合并后模型不再包含 A/B 旁路。"""
    for name, child in list(model.named_children()):
        if isinstance(child, (LoRALinear, LoRAConv2d)):
            setattr(model, name, child.merge())
        else:
            merge_lora(child)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """只提取 A/B 权重，不保存庞大的原始模型参数。"""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def classifier_state_dict(model: nn.Module, head_names: Iterable[str]) -> dict[str, Tensor]:
    """新分类头也不属于原 LoRA 数学定义，因此单独保存。"""
    prefixes = tuple(f"{name}." for name in head_names)
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def save_adapter(
    model: nn.Module,
    path: str | Path,
    config: LoRAConfig,
    target_modules: list[str],
    head_names: Iterable[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    """保存 LoRA、分类头和复现所需配置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head_names = list(head_names)
    payload = {
        "format": "lora-from-scratch-v1",
        "lora_config": asdict(config),
        "target_modules": target_modules,
        "head_names": head_names,
        "lora_state_dict": lora_state_dict(model),
        "classifier_state_dict": classifier_state_dict(model, head_names),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_adapter(
    model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """把适配器加载进结构相同的模型，并检查所有保存权重均已匹配。"""
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != "lora-from-scratch-v1":
        raise ValueError("不是本项目保存的 LoRA 适配器文件")
    saved_state = {
        **payload["lora_state_dict"],
        **payload["classifier_state_dict"],
    }
    incompatible = model.load_state_dict(saved_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"出现未匹配权重：{incompatible.unexpected_keys}")
    not_loaded = [key for key in saved_state if key in incompatible.missing_keys]
    if not_loaded:
        raise RuntimeError(f"以下适配器权重未加载：{not_loaded}")
    return payload


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """返回总参数量、可训练参数量以及 LoRA 参数量。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "classifier_and_other_trainable": trainable - lora,
        "trainable_percent": round(100 * trainable / total, 4),
        "lora_percent": round(100 * lora / total, 4),
    }
