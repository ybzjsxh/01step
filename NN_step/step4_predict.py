"""第 4 步：加载训好的模型，看它到底学到了什么。

运行：python step4_predict.py
产出（figures/ 目录）：
    predictions.png     12 张猜对 + 4 张猜错的测试图（错的标红）
    confusion.png       混淆矩阵：错在哪两类之间
    filters.png         第一层 256 个神经元的权重（可视化"它在找什么笔画"）
    saliency.png        输入梯度图：哪些像素在推动模型做这个判断
"""

import os

import numpy as np

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")   # 只跳过系统字体扫描，不影响画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist import load_mnist
from nn import MLP

FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def show_predictions(model, Xte, yte):
    """12 张猜对的 + 4 张猜错的，错的那几张贴出来最有教学价值。"""
    pred = model.predict(Xte)
    right = np.where(pred == yte)[0][:12]
    wrong = np.where(pred != yte)[0][:4]
    sel = list(right) + list(wrong)

    fig, axes = plt.subplots(4, 4, figsize=(7, 7.6))
    for ax, i in zip(axes.ravel(), sel):
        ax.imshow(Xte[i].reshape(28, 28), cmap="gray_r")
        ok = pred[i] == yte[i]
        ax.set_title(f"pred {pred[i]} / true {yte[i]}", fontsize=9,
                     color="tab:green" if ok else "tab:red")
        ax.axis("off")
    fig.suptitle("Predictions (green = correct, red = wrong)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "predictions.png"), dpi=150)
    plt.close(fig)


def show_confusion(model, Xte, yte):
    pred = model.predict(Xte)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted"); ax.set_ylabel("true label")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_title(f"Confusion matrix (acc {(pred == yte).mean()*100:.2f}%)")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion.png"), dpi=150)
    plt.close(fig)

    # 最容易混淆的几对
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(cm[i, j], i, j) for i in range(10) for j in range(10) if off[i, j] > 0]
    pairs.sort(reverse=True)
    print("\n最容易混淆的 5 对（真实 -> 预测，张数）：")
    for cnt, i, j in pairs[:5]:
        print(f"  {i} -> {j} : {cnt}")


def show_filters(model, n_show=64):
    """第一层权重：每个隐藏单元对应 784 个权重，把其中 64 个重排成 28x28 看形状。"""
    W1 = model.layers[0].W                      # (784, 256)
    rows = cols = int(np.sqrt(n_show))
    scale = np.percentile(np.abs(W1), 95)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10.4))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= min(n_show, W1.shape[1]):
            continue
        ax.imshow(W1[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-scale, vmax=scale)
    fig.suptitle(f"First {n_show} of {W1.shape[1]} hidden units: weight vectors as 28x28 "
                 f"(red = +, blue = -)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "filters.png"), dpi=130)
    plt.close(fig)


def show_saliency(model, Xte, yte):
    """反向传播的副产物：算 d(预测类别分数)/d(输入像素)，看模型在"看哪里"。

    做法：给目标类别的 logit 一个梯度 1，其余为 0，然后往前跑一遍 backward。
    红色 = 这些像素变亮会让该类别分数升高；蓝色 = 变亮会压低分数。
    """
    pred = model.predict(Xte)
    pick = [int(np.where(yte == d)[0][0]) for d in range(10)]

    def input_grad(idx, cls):
        model.forward(Xte[idx:idx + 1])
        g = np.zeros((1, 10))
        g[0, cls] = 1.0
        for layer in reversed(model.layers):
            g = layer.backward(g)
        return g[0].reshape(28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(16, 3.8))
    for col, idx in enumerate(pick):
        axes[0, col].imshow(Xte[idx].reshape(28, 28), cmap="gray_r")
        axes[0, col].set_title(f"{yte[idx]}->{pred[idx]}", fontsize=8)
        axes[0, col].axis("off")
        grad = input_grad(idx, pred[idx])
        v = np.abs(grad).max() * 0.6
        axes[1, col].imshow(grad, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("d score / d pixel", fontsize=9)
    fig.suptitle("Saliency via backprop: which pixels raise the predicted class score", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "saliency.png"), dpi=150)
    plt.close(fig)


def export_sample_digit(Xte, yte):
    """导出一张测试图到 data/sample_digit.png，给 predict_image.py 当输入样例。"""
    os.makedirs("data", exist_ok=True)
    idx = int(np.where(yte == 2)[0][0])
    fig = plt.figure(figsize=(2.8, 2.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(Xte[idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    path = os.path.join("data", "sample_digit.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"样例图已导出到 {path}（真实标签 {yte[idx]}）")


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist(verbose=False)
    model = MLP.load("models/mlp_mnist.npz")
    print(f"测试集准确率：{model.accuracy(Xte, yte)*100:.2f}%")
    show_predictions(model, Xte, yte)
    show_confusion(model, Xte, yte)
    show_filters(model)
    show_saliency(model, Xte, yte)
    export_sample_digit(Xte, yte)
    print(f"\n图片已保存到 {FIG}/")
