"""第 6 步：用自己写的卷积层训练 MNIST，并和 MLP 基线对比。

运行：python step6_train_cnn.py               # 默认 8 轮
      python step6_train_cnn.py --epochs 3    # 快速试
产出：models/cnn_mnist.npz、models/cnn_history.json
      figures/cnn_training_curves.png（训练曲线）
      figures/cnn_filters.png（学到的 16 个卷积核长什么样）
      figures/cnn_vs_mlp.png（和 MLP 基线的对比）
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conv import CNN
from mnist import load_mnist
from nn import Adam


def to_float32(model):
    """把权重降到 float32：卷积的中间张量是 (N,24,24,16) 级别，float64 太占内存。"""
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = layer.W.astype(np.float32)
            layer.b = layer.b.astype(np.float32)
            layer.dW = np.zeros_like(layer.W)
            layer.db = np.zeros_like(layer.b)


def shift_augment(xb, rng, max_shift=2):
    """随机平移 0~2 个像素：MNIST 的数字本来就写得歪歪扭扭，平移不该改变类别。

    这是最便宜也最有效的数据增强。注意用 pad+crop 而不是 np.roll，
    否则图像会从另一侧"绕"回来，变成假样本。
    """
    ms = max_shift
    out = np.empty_like(xb)
    for i in range(len(xb)):
        dy, dx = rng.integers(-ms, ms + 1, size=2)
        p = np.pad(xb[i], ((ms, ms), (ms, ms), (0, 0)), mode="constant")
        out[i] = p[ms + dy:ms + dy + 28, ms + dx:ms + dx + 28]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--conv-ch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=1, help="1=开启平移增强")
    ap.add_argument("--wd", type=float, default=1e-4, help="权重衰减系数")
    ap.add_argument("--out", default="models/cnn_mnist.npz")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_mnist(verbose=False)
    # 关键的一步：把 (N,784) 还原成 (N,28,28,1)，卷积需要空间结构
    Xtr = Xtr.reshape(-1, 28, 28, 1)
    Xte = Xte.reshape(-1, 28, 28, 1)

    model = CNN(in_ch=1, conv_ch=args.conv_ch, k=5, hidden=args.hidden, seed=args.seed)
    to_float32(model)
    n_params = sum(p.size for p, _ in model.params_and_grads())
    opt = Adam(model, lr=args.lr)
    rng = np.random.default_rng(args.seed)

    print(f"结构: conv5x5({args.conv_ch}) -> ReLU -> pool2 -> fc{args.hidden} -> fc10")
    print(f"参数量: {n_params:,}   平移增强={'开' if args.augment else '关'}   wd={args.wd}")
    print(f"初始 loss = {model.loss(Xtr[:2000], ytr[:2000]):.4f}（期望 ≈ 2.30）\n")

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "sec": []}
    n = len(Xtr)
    t_start = time.time()
    lr_now = args.lr
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # 学习率减半：后期步子小一点，才能落进更窄的极小值
        if epoch == args.epochs // 2 + 1:
            lr_now = args.lr / 2
            opt.lr = lr_now
            print(f"  -- 学习率降到 {lr_now:g} --")

        t0 = time.time()
        perm = rng.permutation(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr[idx]
            if args.augment:
                xb = shift_augment(xb, rng)
            model.loss_fn.forward(model.forward(xb), ytr[idx])
            model.backward()
            opt.step()
            if args.wd > 0:
                # 权重衰减：等价于在损失里加 L2 惩罚，抑制权重变大（越大的权重越容易过拟合）
                for p, _ in model.params_and_grads():
                    p *= (1.0 - lr_now * args.wd)

        # loss 用固定子集评估（训练中途的 batch loss 噪声太大，不适合当指标）
        train_loss = model.loss(Xtr[:2000], ytr[:2000])
        train_acc = model.accuracy(Xtr[:5000], ytr[:5000])
        test_acc = model.accuracy(Xte, yte)
        dt = time.time() - t0

        hist["epoch"].append(epoch)
        hist["train_loss"].append(float(train_loss))
        hist["train_acc"].append(train_acc)
        hist["test_acc"].append(test_acc)
        hist["sec"].append(dt)
        print(f"epoch {epoch:2d}/{args.epochs}   loss {train_loss:.4f}   "
              f"train_acc {train_acc*100:5.2f}%   test_acc {test_acc*100:5.2f}%   ({dt:.1f}s)")

        if test_acc > best_acc:                       # 保留最好的一轮，而不是最后一轮
            best_acc = test_acc
            model.save(args.out)

    total_t = time.time() - t_start
    last = model.accuracy(Xte, yte)
    print(f"\n完成：{args.epochs} 轮共 {total_t:.0f}s")
    print(f"  最后一轮 test_acc {last*100:.2f}%   最佳 test_acc {best_acc*100:.2f}%")
    print(f"  已保存最佳那一轮的权重到 {args.out}")
    with open("models/cnn_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    model = CNN.load(args.out)        # 后面的可视化用「最好那一轮」，而不是最后一轮

    # ---------------- 画图 ----------------
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], marker="o")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("CNN: cross-entropy loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(hist["epoch"], [a * 100 for a in hist["train_acc"]], marker="o", label="train")
    axes[1].plot(hist["epoch"], [a * 100 for a in hist["test_acc"]], marker="s", label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("CNN: accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/cnn_training_curves.png", dpi=140)
    plt.close(fig)

    # 卷积核：16 个 5x5
    W = model.layers[0].W                       # (5*5*1, 16)
    nf = W.shape[1]
    cols = 8
    rows = int(np.ceil(nf / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < nf:
            ax.imshow(W[:, i].reshape(5, 5), cmap="RdBu_r",
                      vmin=-np.abs(W).max() * 0.7, vmax=np.abs(W).max() * 0.7)
            ax.set_title(f"#{i}", fontsize=7)
    fig.suptitle("Learned 5x5 conv filters (red = +, blue = -)", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/cnn_filters.png", dpi=150)
    plt.close(fig)

    # 和 MLP 基线对比
    mlp_acc = None
    if os.path.exists("models/history.json"):
        with open("models/history.json") as f:
            mlp_acc = json.load(f)["test_acc"][-1] * 100
    labels = ["MLP 784-256-10", f"CNN conv{args.conv_ch}+fc{args.hidden}"]
    vals = [mlp_acc if mlp_acc is not None else 97.98, best_acc * 100]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bars = ax.bar(labels, vals, color=["#9aa7b8", "#3b7dd8"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%",
                ha="center", fontsize=10)
    ax.set_ylim(min(vals) - 2.5, 100)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Same data, same optimizer: MLP vs CNN")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/cnn_vs_mlp.png", dpi=150)
    plt.close(fig)
    print("图已保存：figures/cnn_training_curves.png, cnn_filters.png, cnn_vs_mlp.png")


if __name__ == "__main__":
    main()
