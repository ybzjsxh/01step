"""加载 LoRA 适配器，对本地图片或 CIFAR-10 测试图片进行推理。

示例：
    uv run --no-sync python3.12 predict.py \
      --checkpoint outputs/resnet18_lora/adapter.pt --image my_cat.jpg
    uv run --no-sync python3.12 predict.py \
      --checkpoint outputs/swin_tiny_lora/adapter.pt --index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets

from lora_core import LoRAConfig, load_adapter
from train import build_model, build_transforms, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加载教学版 LoRA 适配器并预测单张图片")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="不提供时读取 CIFAR-10 测试集")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 测试图片索引")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    metadata = payload["metadata"]
    if not metadata.get("pretrained", False):
        raise ValueError("独立推理需要预训练主干；--no-pretrained 仅适合调试训练流程")
    config = LoRAConfig(**payload["lora_config"])
    model, _, _ = build_model(
        model_name=metadata["model"],
        pretrained=True,
        num_classes=metadata["num_classes"],
        image_size=metadata["image_size"],
        method="lora",
        lora_config=config,
    )
    load_adapter(model, args.checkpoint)
    device = select_device(args.device)
    model = model.to(device).eval()

    true_label: str | None = None
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        source = str(args.image)
    else:
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True)
        image, label_index = dataset[args.index]
        true_label = dataset.classes[label_index]
        source = f"CIFAR-10 test index={args.index}"

    _, test_transform = build_transforms(metadata["image_size"])
    inputs = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(inputs).softmax(dim=1)[0].cpu()
    scores, indices = probabilities.topk(min(5, len(probabilities)))
    predictions = [
        {"class": metadata["classes"][index], "probability": round(score, 6)}
        for score, index in zip(scores.tolist(), indices.tolist())
    ]
    result = {
        "source": source,
        "true_label": true_label,
        "predicted_label": predictions[0]["class"],
        "top5": predictions,
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
