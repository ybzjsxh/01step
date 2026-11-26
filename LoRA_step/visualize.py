"""生成 Grad-CAM 与训练曲线可视化。

默认一次生成：
- outputs/visualizations/resnet18_gradcam.png
- outputs/visualizations/swin_tiny_gradcam.png
- outputs/visualizations/training_curves.png
- outputs/visualizations/gradcam_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch
from torchvision import datasets

from gradcam import GradCAM, feature_rank_stats
from lora_core import LoRAConfig, load_adapter
from train import build_model, build_transforms, select_device


ROOT = Path(__file__).resolve().parent
VISUALS = {
    "resnet18": {
        "display_name": "ResNet-18",
        "checkpoint": ROOT / "outputs" / "resnet18_lora" / "adapter.pt",
        "layer": "layer2.1.conv2",
        "channels_last": False,
    },
    "swin_tiny": {
        "display_name": "Swin-T",
        "checkpoint": ROOT / "outputs" / "swin_tiny_lora" / "adapter.pt",
        "layer": "features.5",
        "channels_last": True,
    },
}
METRIC_FILES = {
    "ResNet-18 · head": ROOT / "outputs" / "resnet18_head" / "metrics.json",
    "ResNet-18 · LoRA": ROOT / "outputs" / "resnet18_lora" / "metrics.json",
    "Swin-T · LoRA": ROOT / "outputs" / "swin_tiny_lora" / "metrics.json",
}
COLORS = {
    "ResNet-18 · head": "#8C8C8C",
    "ResNet-18 · LoRA": "#457B9D",
    "Swin-T · LoRA": "#E76F51",
}
MODEL_VIOLETS = {
    "ResNet-18": "#5B21B6",
    "Swin-T": "#A21CAF",
}
PURPLE_MAPS = {
    "ResNet-18": LinearSegmentedColormap.from_list(
        "resnet_violet", ["#F5F3FF", "#C4B5FD", "#7C3AED", "#3B0764"]
    ),
    "Swin-T": LinearSegmentedColormap.from_list(
        "swin_violet", ["#FAE8FF", "#E879F9", "#A21CAF", "#4A044E"]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 模型的 Grad-CAM 与训练曲线")
    parser.add_argument(
        "--model", choices=("all", "resnet18", "swin_tiny"), default="all"
    )
    parser.add_argument("--num-images", type=int, default=4, help="每个模型展示的样本数")
    parser.add_argument(
        "--indices", default=None,
        help="逗号分隔的 CIFAR-10 测试索引；不填则自动选择不同类别样本",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "visualizations")
    return parser.parse_args()


def load_lora_model(
    model_name: str, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    metadata = payload["metadata"]
    if not metadata.get("pretrained", False):
        raise ValueError("可视化需要预训练主干；--no-pretrained 适配器无法独立复现")
    config = LoRAConfig(**payload["lora_config"])
    model, _, _ = build_model(
        model_name=model_name,
        pretrained=True,
        num_classes=metadata["num_classes"],
        image_size=metadata["image_size"],
        method="lora",
        lora_config=config,
    )
    load_adapter(model, checkpoint)
    return model.to(device).eval(), metadata


@torch.no_grad()
def choose_diagnostic_indices(
    model: torch.nn.Module,
    dataset: datasets.CIFAR10,
    transform: Any,
    device: torch.device,
    count: int,
    candidate_limit: int = 1000,
) -> list[int]:
    """自动选 2 个正确、2 个错误样本，并在类别上尽量不重复。"""
    model.eval()
    limit = min(len(dataset), max(candidate_limit, count * 20))
    records: list[dict[str, Any]] = []
    batch_size = 64
    for start in range(0, limit, batch_size):
        batch_indices = list(range(start, min(start + batch_size, limit)))
        images = torch.stack([transform(dataset[index][0]) for index in batch_indices]).to(device)
        probabilities = model(images).softmax(dim=1)
        predictions = probabilities.argmax(dim=1).cpu()
        confidences = probabilities.max(dim=1).values.cpu()
        for index, prediction, confidence in zip(batch_indices, predictions, confidences):
            true_index = int(dataset.targets[index])
            records.append({
                "index": index,
                "true_index": true_index,
                "predicted_index": int(prediction),
                "confidence": float(confidence),
                "correct": true_index == int(prediction),
            })

    target_correct = count // 2
    target_wrong = count - target_correct
    selected_correct: list[dict[str, Any]] = []
    selected_wrong: list[dict[str, Any]] = []
    seen_classes: set[int] = set()
    for record in records:
        if (
            record["correct"]
            and len(selected_correct) < target_correct
            and record["true_index"] not in seen_classes
        ):
            selected_correct.append(record)
            seen_classes.add(record["true_index"])
    for record in records:
        if (
            not record["correct"]
            and len(selected_wrong) < target_wrong
            and record["true_index"] not in seen_classes
        ):
            selected_wrong.append(record)
            seen_classes.add(record["true_index"])

    chosen_ids = {
        record["index"] for record in selected_correct + selected_wrong
    }
    wanted = count - len(chosen_ids)
    if wanted > 0:
        if len(selected_correct) < target_correct:
            pool = [record for record in records if record["correct"]]
            existing = selected_correct
        else:
            pool = [record for record in records if not record["correct"]]
            existing = selected_wrong
        for record in pool:
            if record["index"] not in chosen_ids:
                existing.append(record)
                chosen_ids.add(record["index"])
                wanted -= 1
                if wanted == 0:
                    break

    ordered: list[dict[str, Any]] = []
    for position in range(max(len(selected_correct), len(selected_wrong))):
        if position < len(selected_correct):
            ordered.append(selected_correct[position])
        if position < len(selected_wrong):
            ordered.append(selected_wrong[position])
    return [record["index"] for record in ordered]


def display_image(image: Image.Image, size: int = 224) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(resized.convert("RGB")) / 255.0


def save_gradcam_grid(
    model_name: str,
    device: torch.device,
    indices: list[int] | None,
    output_dir: Path,
    num_images: int = 4,
) -> dict[str, Any]:
    config = VISUALS[model_name]
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到适配器：{checkpoint}")

    model, metadata = load_lora_model(model_name, checkpoint, device)
    target_layer = model.get_submodule(config["layer"])
    test_dataset = datasets.CIFAR10(root=ROOT / "data", train=False, download=True)
    _, transform = build_transforms(int(metadata["image_size"]))
    classes = metadata["classes"]
    selection_mode = (
        "manual"
        if indices
        else f"{num_images // 2} correct + {num_images - num_images // 2} wrong (auto)"
    )
    if indices is None:
        indices = choose_diagnostic_indices(
            model, test_dataset, transform, device, num_images
        )

    rows: list[dict[str, Any]] = []
    with GradCAM(model, target_layer, channels_last=config["channels_last"]) as cammer:
        for index in indices:
            image, true_index = test_dataset[index]
            inputs = transform(image).unsqueeze(0).to(device)
            result = cammer(inputs)
            cam = result.cam.numpy()
            rows.append({
                "index": index,
                "true": classes[true_index],
                "predicted": classes[result.predicted_class],
                "confidence": round(result.confidence, 6),
                "correct": true_index == result.predicted_class,
                "feature_shape": result.feature_shape,
                "rank": feature_rank_stats(result.feature_map),
                "display_image": display_image(image),
                "cam": cam,
            })

    row_count = len(rows)
    fig, axes = plt.subplots(row_count, 3, figsize=(10.0, 2.85 * row_count + 1.1))
    if row_count == 1:
        axes = np.asarray([axes])

    heat_image = None
    for row_index, row in enumerate(rows):
        input_axis, cam_axis, overlay_axis = axes[row_index]
        input_axis.imshow(row["display_image"])
        input_axis.set_title(
            f"True: {row['true']}\nPred: {row['predicted']} ({row['confidence']:.1%})",
            fontsize=8.5,
            loc="left",
            pad=5,
        )
        heat_image = cam_axis.imshow(row["cam"], cmap="turbo", vmin=0, vmax=1)
        overlay_axis.imshow(row["display_image"])
        overlay_axis.imshow(row["cam"], cmap="turbo", alpha=0.55, vmin=0, vmax=1)
        for axis in (input_axis, cam_axis, overlay_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#CCCCCC")

    correct = sum(int(row["correct"]) for row in rows)
    fig.suptitle(
        f"{config['display_name']} Grad-CAM · target layer: {config['layer']}\n"
        f"red/orange supports the predicted class · correct: {correct}/{row_count}",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.03, right=0.97, bottom=0.10, top=0.84, hspace=0.48, wspace=0.08
    )

    # 使用独立表头，避免第一行的 True/Pred 标签与列标题重叠。
    column_centers = [
        axes[0, column].get_position().x0 + axes[0, column].get_position().width / 2
        for column in range(3)
    ]
    header_y = axes[0, 0].get_position().y1 + 0.045
    header_titles = (
        "Input",
        f"Grad-CAM · feature {rows[0]['feature_shape']}",
        "Overlay · red = important",
    )
    for center, title in zip(column_centers, header_titles):
        fig.text(center, header_y, title, ha="center", va="bottom", fontsize=12, fontweight="bold")
    if heat_image is not None:
        colorbar_axis = fig.add_axes([0.28, 0.025, 0.44, 0.016])
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="turbo"),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_ticks([0, 0.5, 1])
        colorbar.ax.set_xticklabels(["low", "medium", "high"])
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.set_label("importance", fontsize=8, labelpad=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_gradcam.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "model": model_name,
        "display_name": config["display_name"],
        "checkpoint": str(checkpoint),
        "target_layer": config["layer"],
        "channels_last": config["channels_last"],
        "output_image": str(output_path),
        "selection_mode": selection_mode,
        "indices": indices,
        "samples": [
            {
                key: value
                for key, value in row.items()
                if key not in {"display_image", "cam"}
            }
            for row in rows
        ],
    }


def plot_training_curves(output_dir: Path) -> dict[str, str]:
    available = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in METRIC_FILES.items()
        if path.is_file()
    }
    if not available:
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(
        "CIFAR-10 · 2,000 train / 500 test · 2 epochs · rank 4 / alpha 8",
        fontsize=14,
        fontweight="bold",
    )
    accuracy_axis, loss_axis, gap_axis, parameter_axis = axes.flatten()

    for label, metrics in available.items():
        epochs = [0] + [record["epoch"] for record in metrics["history"]]
        test_accuracy = [metrics["initial"]["accuracy"]] + [
            record["test"]["accuracy"] for record in metrics["history"]
        ]
        test_loss = [metrics["initial"]["loss"]] + [
            record["test"]["loss"] for record in metrics["history"]
        ]
        color = COLORS[label]
        accuracy_axis.plot(epochs, test_accuracy, marker="o", linewidth=2, label=label, color=color)
        loss_axis.plot(epochs, test_loss, marker="o", linewidth=2, label=label, color=color)
        for epoch, value in zip(epochs, test_accuracy):
            accuracy_axis.annotate(
                f"{value:.1f}", (epoch, value), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8, color=color,
            )

    accuracy_axis.set_title("Test accuracy", fontweight="bold")
    accuracy_axis.set_xlabel("epoch (0 = before training)")
    accuracy_axis.set_ylabel("accuracy (%)")
    accuracy_axis.set_xticks([0, 1, 2])
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(frameon=False, fontsize=9)

    loss_axis.set_title("Test loss", fontweight="bold")
    loss_axis.set_xlabel("epoch (0 = before training)")
    loss_axis.set_ylabel("cross entropy")
    loss_axis.set_xticks([0, 1, 2])
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(frameon=False, fontsize=9)

    for label in ("ResNet-18 · LoRA", "Swin-T · LoRA"):
        metrics = available.get(label)
        if metrics is None:
            continue
        color = COLORS[label]
        epochs = [record["epoch"] for record in metrics["history"]]
        train_accuracy = [record["train"]["accuracy"] for record in metrics["history"]]
        test_accuracy = [record["test"]["accuracy"] for record in metrics["history"]]
        gap_axis.plot(epochs, train_accuracy, marker="o", linestyle="--", color=color, label=f"{label} train")
        gap_axis.plot(epochs, test_accuracy, marker="s", color=color, label=f"{label} test")
    gap_axis.set_title("LoRA: train vs test accuracy", fontweight="bold")
    gap_axis.set_xlabel("epoch")
    gap_axis.set_ylabel("accuracy (%)")
    gap_axis.set_xticks([1, 2])
    gap_axis.grid(alpha=0.25)
    gap_axis.legend(frameon=False, fontsize=8, ncol=2)

    bar_labels = list(available)
    counts = [available[label]["parameter_stats"]["trainable"] for label in bar_labels]
    percentages = [available[label]["parameter_stats"]["trainable_percent"] for label in bar_labels]
    bar_colors = [COLORS[label] for label in bar_labels]
    bars = parameter_axis.barh(bar_labels, counts, color=bar_colors, height=0.55)
    for bar, count, percentage in zip(bars, counts, percentages):
        parameter_axis.text(
            count + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({percentage:.4f}%)",
            va="center", fontsize=9,
        )
    parameter_axis.set_title("Trainable parameters (count and % of total)", fontweight="bold")
    parameter_axis.set_xlabel("parameters")
    parameter_axis.set_xlim(0, max(counts) * 1.55)
    parameter_axis.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"training_curves": str(output_path)}


def plot_feature_ranks(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    """用紫色系展示目标层特征矩阵的有效秩与奇异值谱。"""
    if not records:
        return {}

    fig, (rank_axis, spectrum_axis) = plt.subplots(
        1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    positions = np.arange(len(records))
    effective_ranks = [float(record["rank"]["effective_rank"]) for record in records]
    bar_colors = [MODEL_VIOLETS[record["display_name"]] for record in records]
    bars = rank_axis.barh(
        positions,
        effective_ranks,
        color=bar_colors,
        edgecolor="#2E1065",
        linewidth=0.8,
        height=0.58,
        alpha=0.88,
    )
    for bar, record in zip(bars, records):
        if not record["correct"]:
            bar.set_hatch("//")
            bar.set_alpha(0.58)
    rank_axis.invert_yaxis()
    labels = [
        f"{record['display_name']} #{record['index']}  "
        f"{'✓' if record['correct'] else '✗'}  {record['true']}→{record['predicted']}"
        for record in records
    ]
    rank_axis.set_yticks(positions)
    rank_axis.set_yticklabels(labels, fontsize=8)
    max_positions = max(int(record["rank"]["spatial_positions"]) for record in records)
    maximum_rank = max(effective_ranks)
    rank_axis.set_xlim(0, max(max_positions * 1.28, maximum_rank * 1.8))
    for position, record, value in zip(positions, records, effective_ranks):
        stats = record["rank"]
        rank_axis.text(
            value + max_positions * 0.02,
            position,
            f"eff {value:.1f} · stable {stats['stable_rank']:.1f} · raw {stats['raw_rank']}",
            va="center",
            fontsize=8,
            color="#3B0764",
        )
    rank_axis.set_title(
        f"Effective rank per feature map · raw rank max = {max_positions}",
        fontsize=12,
        fontweight="bold",
        color="#4C1D95",
    )
    rank_axis.set_xlabel("entropy effective rank  (higher = more diverse spectrum)", fontsize=9)
    rank_axis.grid(axis="x", alpha=0.2, color="#7C3AED")
    rank_axis.set_facecolor("#FCFBFF")
    rank_axis.legend(
        handles=[
            Patch(facecolor=MODEL_VIOLETS["ResNet-18"], label="ResNet-18"),
            Patch(facecolor=MODEL_VIOLETS["Swin-T"], label="Swin-T"),
            Patch(facecolor="#A78BFA", edgecolor="#2E1065", hatch="//", label="wrong prediction"),
        ],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )

    for model_name in ("ResNet-18", "Swin-T"):
        model_records = [record for record in records if record["display_name"] == model_name]
        if not model_records:
            continue
        purple_map = PURPLE_MAPS[model_name]
        shades = np.linspace(0.36, 0.92, len(model_records))
        for shade, record in zip(shades, model_records):
            spectrum = np.asarray(record["rank"]["singular_values"], dtype=float)
            if spectrum.size == 0:
                continue
            spectrum_axis.plot(
                np.arange(1, len(spectrum) + 1),
                spectrum,
                color=purple_map(shade),
                linestyle="-" if record["correct"] else "--",
                marker="o" if record["correct"] else "x",
                markersize=3.5,
                linewidth=1.4,
                alpha=0.9,
                label=(
                    f"{model_name} #{record['index']} "
                    f"{'✓' if record['correct'] else '✗'} "
                    f"{record['true']}→{record['predicted']}"
                ),
            )
    spectrum_axis.set_title(
        "Normalized singular-value spectrum · flatter = higher effective rank",
        fontsize=12,
        fontweight="bold",
        color="#4C1D95",
    )
    spectrum_axis.set_xlabel("singular value index", fontsize=9)
    spectrum_axis.set_ylabel("σᵢ / σ₁", fontsize=9)
    spectrum_axis.set_ylim(0, 1.05)
    spectrum_axis.set_facecolor("#FCFBFF")
    spectrum_axis.grid(alpha=0.2, color="#7C3AED")
    spectrum_axis.legend(
        fontsize=7,
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.29),
    )
    spectrum_axis.text(
        0.98,
        0.96,
        "solid = correct\n dashed + × = wrong",
        transform=spectrum_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#6D28D9",
        bbox={"facecolor": "#F5F3FF", "edgecolor": "#C4B5FD", "boxstyle": "round,pad=0.35"},
    )
    for axis in (rank_axis, spectrum_axis):
        axis.tick_params(colors="#4C1D95")
        for spine in axis.spines.values():
            spine.set_color("#A78BFA")

    fig.suptitle(
        "Feature-map rank diagnostics · purple palette\n"
        "raw rank saturates at min(C, H×W); effective rank shows spectral concentration",
        fontsize=13,
        fontweight="bold",
        color="#3B0764",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_rank.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"feature_rank": str(output_path)}


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images 必须大于 0")
    device = select_device(args.device)
    output_dir = args.output_dir.resolve()
    dataset = datasets.CIFAR10(root=ROOT / "data", train=False, download=True)

    models = list(VISUALS) if args.model == "all" else [args.model]
    if args.indices:
        indices = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
        for index in indices:
            if not 0 <= index < len(dataset):
                raise ValueError(f"测试索引越界：{index}")
    else:
        indices = None

    summary: dict[str, Any] = {
        "device": str(device),
        "requested_indices": indices,
        "gradcam": {},
    }
    rank_records: list[dict[str, Any]] = []
    for model_name in models:
        result = save_gradcam_grid(
            model_name, device, indices, output_dir, num_images=args.num_images
        )
        selected = result["indices"]
        print(
            f"{VISUALS[model_name]['display_name']} Grad-CAM 样本 {selected} → "
            f"{result['output_image']}",
            flush=True,
        )
        summary["gradcam"][model_name] = result
        for sample in result["samples"]:
            rank_records.append({
                "display_name": result["display_name"],
                "index": sample["index"],
                "true": sample["true"],
                "predicted": sample["predicted"],
                "correct": sample["correct"],
                "rank": sample["rank"],
            })

    rank_figure = plot_feature_ranks(rank_records, output_dir)
    summary.update(rank_figure)
    if rank_figure:
        print(f"特征有效秩分析 → {rank_figure['feature_rank']}", flush=True)

    curves = plot_training_curves(output_dir)
    summary.update(curves)
    summary_path = output_dir / "gradcam_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"汇总文件：{summary_path}", flush=True)


if __name__ == "__main__":
    main()
