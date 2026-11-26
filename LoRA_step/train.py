"""用 CIFAR-10 演示 ResNet-18 / Swin-Tiny 的 LoRA 微调。

示例：
    uv run --no-sync python3.12 train.py --model resnet18 --epochs 2
    uv run --no-sync python3.12 train.py --model swin_tiny --epochs 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t

from lora_core import (
    LoRAConfig,
    configure_head_only,
    count_parameters,
    inject_lora,
    merge_lora,
    save_adapter,
)


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MODEL_DEFAULTS = {
    "resnet18": {"image_size": 32, "lr": 1e-3, "target": "all_conv", "dropout": 0.05},
    "swin_tiny": {"image_size": 64, "lr": 5e-4, "target": "attention", "dropout": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从零实现 LoRA：ResNet-18 / Swin-Tiny + CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_DEFAULTS, required=True)
    parser.add_argument("--method", choices=("lora", "head", "full"), default="lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=2000, help="0 表示完整训练集")
    parser.add_argument("--test-limit", type=int, default=500, help="0 表示完整测试集")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--target", default=None, help="默认 ResNet=all_conv，Swin=attention")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="仅调试用；正常实验不要加")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--save-merged", action="store_true", help="额外保存完整合并权重")
    args = parser.parse_args()
    defaults = MODEL_DEFAULTS[args.model]
    args.image_size = args.image_size or defaults["image_size"]
    args.lr = args.lr or defaults["lr"]
    args.target = args.target or defaults["target"]
    args.dropout = defaults["dropout"] if args.dropout is None else args.dropout
    args.output_dir = args.output_dir or Path("outputs") / f"{args.model}_{args.method}"
    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    if args.train_limit < 0 or args.test_limit < 0:
        parser.error("--train-limit / --test-limit 不能小于 0")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 MPS，请改用 --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请改用 --device auto/cpu")
    return torch.device(requested)


def build_model(
    model_name: str,
    pretrained: bool,
    num_classes: int,
    image_size: int,
    method: str = "lora",
    lora_config: LoRAConfig | None = None,
) -> tuple[nn.Module, list[str], tuple[str, ...]]:
    """创建原模型、替换分类头，再按 method 配置可训练参数。"""
    head_names: tuple[str, ...]
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        # CIFAR-10 是 10 类；保留 ImageNet 主干，只替换最后的分类器。
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_names = ("fc",)
    elif model_name == "swin_tiny":
        # torchvision 官方 Swin-T；移位窗口支持本教程的 64×64 小输入。
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        head_names = ("head",)
    else:
        raise ValueError(f"未知模型：{model_name}")

    target_modules: list[str] = []
    if method == "lora":
        if lora_config is None:
            raise ValueError("method=lora 时必须提供 lora_config")
        target_modules = inject_lora(model, lora_config, head_names=head_names)
    elif method == "head":
        configure_head_only(model, head_names)
    elif method == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"未知训练方式：{method}")
    return model, target_modules, head_names


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # 使用 ImageNet 均值/方差，因为主干加载的是 ImageNet 预训练权重。
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def stratified_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    """构造类别均衡且可复现的小数据集索引。"""
    if limit == 0 or limit >= len(targets):
        return list(range(len(targets)))
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)
    for indices in by_class.values():
        rng.shuffle(indices)
    classes = sorted(by_class)
    selected: list[int] = []
    base, remainder = divmod(limit, len(classes))
    for position, label in enumerate(classes):
        count = base + int(position < remainder)
        selected.extend(by_class[label][:count])
    rng.shuffle(selected)
    return selected


def build_loaders(
    data_dir: Path,
    image_size: int,
    train_limit: int,
    test_limit: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_transform, test_transform = build_transforms(image_size)
    try:
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, transform=test_transform, download=True
        )
    except Exception as error:
        raise RuntimeError(
            "CIFAR-10 下载或读取失败。请检查网络；若已手动下载，请将 "
            "cifar-10-batches-py 与 cifar-10-python.tar.gz 放入 data-dir。"
        ) from error

    train_indices = stratified_indices(train_dataset.targets, train_limit, seed)
    test_indices = stratified_indices(test_dataset.targets, test_limit, seed + 1)
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_interval: int = 20,
) -> dict[str, float]:
    """optimizer=None 时为验证；否则进行一个 epoch 的训练。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
        if training and batch_index % log_interval == 0:
            logging.info(
                "  step %d/%d | 当前 batch loss %.4f",
                batch_index, len(loader), loss.item(),
            )
    return {
        "loss": total_loss / total_examples,
        "accuracy": 100.0 * total_correct / total_examples,
        "examples": total_examples,
    }


def verify_merge(model: nn.Module, sample: Tensor) -> float:
    """比较合并前后 logits，确认 LoRA 可无损折叠进原模型。"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        original = model(sample).cpu()
        merged_model = copy.deepcopy(model)
        merge_lora(merged_model)
        merged_model.eval()
        merged = merged_model(sample).cpu()
    max_difference = (original - merged).abs().max().item()
    if was_training:
        model.train()
    if max_difference > 2e-4:
        raise AssertionError(f"合并前后输出差异过大：{max_difference}")
    return max_difference


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)
    set_seed(args.seed)
    device = select_device(args.device)
    logging.info("设备：%s | 模型：%s | 方法：%s", device, args.model, args.method)
    logging.info(
        "实验：rank=%d alpha=%g dropout=%g target=%s image_size=%d",
        args.rank, args.alpha, args.dropout, args.target, args.image_size,
    )

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logging.info("数据：训练 %d 张，测试 %d 张", len(train_loader.dataset), len(test_loader.dataset))

    config = LoRAConfig(
        r=args.rank, alpha=args.alpha, dropout=args.dropout, target=args.target
    )
    model, target_modules, head_names = build_model(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        num_classes=len(CIFAR10_CLASSES),
        image_size=args.image_size,
        method=args.method,
        lora_config=config,
    )
    model = model.to(device)
    parameter_stats = count_parameters(model)
    logging.info("参数统计：%s", parameter_stats)
    logging.info("注入了 %d 个 LoRA 模块", len(target_modules))
    if target_modules:
        logging.info("前 5 个注入点：%s", target_modules[:5])

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    initial_metrics: dict[str, float] | None = None
    if not args.skip_initial_eval:
        initial_metrics = run_epoch(model, test_loader, criterion, device)
        logging.info(
            "训练前 | test loss %.4f | test accuracy %.2f%%",
            initial_metrics["loss"], initial_metrics["accuracy"],
        )

    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d 开始", epoch, args.epochs)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        test_metrics = run_epoch(model, test_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(record)
        logging.info(
            "Epoch %d | train loss %.4f acc %.2f%% | test loss %.4f acc %.2f%%",
            epoch,
            train_metrics["loss"], train_metrics["accuracy"],
            test_metrics["loss"], test_metrics["accuracy"],
        )
    elapsed_seconds = time.perf_counter() - start_time

    sample_images, _ = next(iter(test_loader))
    merge_max_difference = verify_merge(model, sample_images[:2].to(device))
    logging.info("LoRA 合并前后最大 logits 误差：%.8f", merge_max_difference)

    metadata = {
        "model": args.model,
        "method": args.method,
        "image_size": args.image_size,
        "num_classes": len(CIFAR10_CLASSES),
        "classes": CIFAR10_CLASSES,
        "pretrained": not args.no_pretrained,
        "best_test_accuracy": max(item["test"]["accuracy"] for item in history),
    }
    adapter_path = args.output_dir / "adapter.pt"
    if args.method == "lora":
        save_adapter(
            model, adapter_path, config, target_modules, head_names, metadata=metadata
        )
        logging.info("LoRA 与分类头已保存：%s", adapter_path)
    else:
        torch.save(
            {"model_state_dict": model.state_dict(), "metadata": metadata},
            args.output_dir / "model.pt",
        )

    if args.save_merged:
        merged_model = copy.deepcopy(model)
        merge_lora(merged_model)
        merged_path = args.output_dir / "merged_model.pt"
        torch.save(
            {"model_state_dict": merged_model.state_dict(), "metadata": metadata},
            merged_path,
        )
        logging.info("完整合并权重已保存：%s", merged_path)

    metrics = {
        "config": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_stats": parameter_stats,
        "target_module_count": len(target_modules),
        "target_modules": target_modules,
        "initial": initial_metrics,
        "history": history,
        "best_test_accuracy": metadata["best_test_accuracy"],
        "merge_max_logit_difference": merge_max_difference,
        "training_seconds": round(elapsed_seconds, 3),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logging.info("指标已保存：%s", args.output_dir / "metrics.json")
    logging.info("最佳测试准确率：%.2f%%", metadata["best_test_accuracy"])


if __name__ == "__main__":
    main()
