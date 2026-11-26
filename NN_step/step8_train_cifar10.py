"""第 8 步：把同一套 NumPy 网络搬到 CIFAR-10（32x32 彩色，10 类）上跑。

这是"从 0 手写"这条线上真正开始变难的地方。相对 EMNIST 有四个新东西：

1. **输入变成 3 通道**：第一层卷积从 (3,3,3) 的核开始，不再是灰度。
2. **必须做通道标准化**：CIFAR 三个通道的分布不同，不标准化 loss 前几个 epoch
   都在抖（见 cifar10.normalize）。
3. **必须做数据增强**：随机裁剪 ±4 像素 + 水平翻转。不做的话，训练准确率能到
   90% 而测试准确率卡在 55%——模型在背"飞机总在正中间朝左"。
4. **必须堆两层卷积**：32x32 的图，单个 5x5 卷积的感受野只有 5 个像素，
   不够描述"猫的耳朵 + 胡须"。所以换成 VGG 式两层 same 卷积（见 conv.DeepCNN）。

对照实验：同一个数据集上，MLP（直接展平 3072 维）能到多少，CNN 能到多少。
这一步的目的不是刷分，而是让你亲眼看到"卷积为什么值这个钱"。

CD   : .venv/bin/python step8_train_cifar10.py
产物 : models/mlp_cifar10.npz, models/cnn_cifar10.npz, models/cifar10_history.json
       figures/cifar10_curves.png, cifar10_confusion.png, cifar10_samples.png,
       cifar10_augment.png, cifar10_features.png
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

from cifar10 import CLASSES, load_cifar10, normalize, random_crop_flip
from conv import DeepCNN
from nn import MLP, Adam


def to_float32(model):
    """把参数和梯度统统降到 float32。

    CIFAR 的中间张量比 MNIST 大一个数量级（64 张 32x32x3 的 im2col 就有
    64*32*32*27 ≈ 177 万个数），float64 会让内存带宽成为瓶颈。实测这一步能快 1.6 倍，
    而 float32 的精度对训练分类器完全够用。
    """
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = layer.W.astype(np.float32)
            layer.b = layer.b.astype(np.float32)
            layer.dW = np.zeros_like(layer.W)
            layer.db = np.zeros_like(layer.b)
    return model


def train_mlp(Xtr, ytr, Xte, yte, epochs, batch, lr, hidden, seed, out):
    """基线：把 32x32x3 直接展平成 3072 维丢给全连接。

    这个基线很重要——它告诉你"不用卷积能到多少"。CIFAR 上大概 45%~50%，
    比 MNIST 的 98% 惨得多，原因就是全连接看不到"相邻像素"这个结构。
    """
    Xf, Xfe = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    model = MLP(sizes=(3072, hidden, len(CLASSES)), seed=seed, dtype=np.float32)
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    print(f"\n[MLP] 3072-{hidden}-{len(CLASSES)}  "
          f"参数量 {sum(p.size for p, _ in model.params_and_grads()):,}  初始 loss = "
          f"{model.loss(Xf[:2000], ytr[:2000]):.4f}（期望 ≈ {np.log(len(CLASSES)):.2f}）")

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "sec": []}
    n, best, t_all = len(Xf), 0.0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model.loss_fn.forward(model.forward(Xf[idx]), ytr[idx])
            model.backward()
            opt.step()
        tr_loss = model.loss(Xf[:4000], ytr[:4000])
        tr_acc = model.accuracy(Xf[:8000], ytr[:8000])
        te_acc = model.accuracy(Xfe, yte)
        dt = time.time() - t0
        for k, v in zip(("train_loss", "train_acc", "test_acc", "sec"),
                        (tr_loss, tr_acc, te_acc, dt)):
            hist[k].append(float(v))
        hist["epoch"].append(ep)
        print(f"  epoch {ep:2d}/{epochs}  loss {tr_loss:.4f}  train {tr_acc*100:5.2f}%  "
              f"test {te_acc*100:5.2f}%  ({dt:.1f}s)")
        if te_acc > best:
            best = te_acc
            model.save(out)
    print(f"  MLP 最佳 test_acc {best*100:.2f}%  总耗时 {time.time()-t_all:.0f}s")
    return MLP.load(out), hist, best


def train_cnn(Xtr, ytr, Xte, yte, epochs, batch, lr, stages, hidden, seed, wd,
              augment, out):
    model = to_float32(DeepCNN(in_ch=3, in_size=32, stages=stages,
                               num_classes=len(CLASSES), hidden=hidden, seed=seed))
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    n_params = sum(p.size for p, _ in model.params_and_grads())
    print(f"\n[CNN] conv3x3({stages[0]})->pool->conv3x3({stages[1]})->pool->fc{hidden}"
          f"->fc{len(CLASSES)}  参数量 {n_params:,}  初始 loss = "
          f"{model.loss(Xtr[:1000], ytr[:1000]):.4f}  增强={'开' if augment else '关'}")

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [],
            "sec": [], "lr": []}
    n, best, t_all, lr_now = len(Xtr), 0.0, time.time(), lr
    for ep in range(1, epochs + 1):
        # 后半程学习率减半：前期大步走，后期小步磨。
        # 不减的话最后几轮 test acc 会来回荡，取不到最好那个点。
        if ep == epochs // 2 + 1:
            lr_now = lr / 2
            opt.lr = lr_now
        t0 = time.time()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx]
            if augment:                              # 只有训练集才增强
                xb = random_crop_flip(xb, rng)
            model.loss_fn.forward(model.forward(xb), ytr[idx])
            model.backward()
            opt.step()
            if wd > 0:                               # 权重衰减：小步把参数往 0 拉
                for p, _ in model.params_and_grads():
                    p *= (1.0 - lr_now * wd)
        tr_loss = model.loss(Xtr[:1000], ytr[:1000])
        tr_acc = model.accuracy(Xtr[:5000], ytr[:5000])
        te_acc = model.accuracy(Xte, yte)
        dt = time.time() - t0
        for k, v in zip(("train_loss", "train_acc", "test_acc", "sec", "lr"),
                        (tr_loss, tr_acc, te_acc, dt, lr_now)):
            hist[k].append(float(v))
        hist["epoch"].append(ep)
        print(f"  epoch {ep:2d}/{epochs}  loss {tr_loss:.4f}  train {tr_acc*100:5.2f}%  "
              f"test {te_acc*100:5.2f}%  lr {lr_now:.0e}  ({dt:.1f}s)")
        if te_acc > best:
            best = te_acc
            model.save(out)
    print(f"  CNN 最佳 test_acc {best*100:.2f}%  总耗时 {time.time()-t_all:.0f}s")
    return DeepCNN.load(out), hist, best


def confusion_matrix(pred, y, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def plot_all(mlp_h, cnn_h, mlp_acc, cnn_acc, pred_cnn, yte, Xte, model, mean, std):
    os.makedirs("figures", exist_ok=True)
    k = len(CLASSES)

    # 1) MLP vs CNN 两条曲线：一眼看出"卷积值多少钱"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for h, name, c in ((mlp_h, "MLP", "#9aa7b8"), (cnn_h, "CNN", "#3b7dd8")):
        axes[0].plot(h["train_loss"], marker="o", ms=4, label=name, color=c)
        axes[1].plot([a * 100 for a in h["train_acc"]], marker="o", ms=4,
                     label=f"{name} train", color=c, alpha=0.6)
        axes[1].plot([a * 100 for a in h["test_acc"]], marker="s", ms=4,
                     label=f"{name} test", color=c)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("CIFAR-10: loss"); axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("accuracy"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/cifar10_curves.png", dpi=150); plt.close(fig)

    # 2) 混淆矩阵：整张图都是"猫和狗""汽车和卡车"这种邻居
    cm = confusion_matrix(pred_cnn, yte, k)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "#222")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CNN confusion matrix, test acc {cnn_acc*100:.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.045, label="count")
    fig.tight_layout(); fig.savefig("figures/cifar10_confusion.png", dpi=160); plt.close(fig)

    # 3) 每类抽 6 张测试图：画框是绿色=对，红色=错，并写上模型猜的类
    rng = np.random.default_rng(2)
    n_show = 6
    fig, axes = plt.subplots(k, n_show, figsize=(n_show * 1.7, k * 2.05))
    for c in range(k):
        idxs = np.where(yte == c)[0]
        picks = idxs[rng.integers(0, len(idxs), n_show)]
        for j, i in enumerate(picks):
            ax = axes[c, j]
            ax.imshow(np.clip(Xte[i] * std[0] + mean[0], 0, 1))   # 还原成看得懂的像素
            ax.axis("off")
            if pred_cnn[i] == c:
                color, label = "#1a9c50", "✓"
            else:
                color, label = "#d02f2f", CLASSES[pred_cnn[i]]
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.6); sp.set_visible(True)
            # 标签画在图内：用 set_title 会和上一行的图挤在一起
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top",
                    fontsize=7, color=color,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        axes[c, 0].text(-0.35, 0.5, CLASSES[c], transform=axes[c, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("One row per true class, 6 test samples (red = misclassified, "
                 "text = predicted class)")
    fig.tight_layout(); fig.savefig("figures/cifar10_samples.png", dpi=150); plt.close(fig)

    # 4) 增强长什么样：上排原图，下排同一个样本裁剪+翻转之后
    rng = np.random.default_rng(3)
    picks = [np.where(yte == c)[0][0] for c in (0, 2, 4, 8)]   # 飞机/鸟/鹿/船
    xb = Xte[picks]
    raw = np.clip(xb * std + mean, 0, 1)                 # 标准化后的图不能直接看
    aug = np.clip(random_crop_flip(xb, rng) * std + mean, 0, 1)
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 5.2))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(raw[i] if i < 4 else aug[i - 4])
        ax.axis("off")
        ax.set_title(f"original: {CLASSES[yte[picks[i % 4]]]}" if i < 4
                     else "cropped + flipped", fontsize=9)
    fig.suptitle("random crop (±4 px) + horizontal flip: same class, new pose and position")
    fig.tight_layout(h_pad=1.4); fig.savefig("figures/cifar10_augment.png", dpi=150); plt.close(fig)

    # 5) 第一层特征图：同一张图，32 个卷积核各自"看到了什么"
    #    3x3 的核本身只是 9 个像素的颜色组合，单独看毫无意义；有意义的是它对图像的响应。
    conv1 = model.layers[0]
    i = int(np.where(yte == 8)[0][0])                    # 挑一艘船：有天空、海面、船体三种结构
    x = Xte[i:i + 1]
    acts = conv1.forward(x)[0]                           # (32, 32, 32)
    nf = min(acts.shape[-1], 35)
    fig, axes = plt.subplots(6, 6, figsize=(9, 9))
    for ax in axes.ravel():
        ax.axis("off")
    ax0 = axes.ravel()[0]
    ax0.imshow(np.clip(x[0] * std[0] + mean[0], 0, 1))
    ax0.set_title("input (ship)", fontsize=9)
    for j, ax in enumerate(axes.ravel()[1:1 + nf]):
        ax.imshow(acts[..., j], cmap="viridis")
    fig.suptitle("First conv layer: 32 feature maps for one ship image "
                 "(each filter highlights its own color/edge pattern)")
    fig.tight_layout(); fig.savefig("figures/cifar10_features.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-mlp", type=int, default=6)
    ap.add_argument("--epochs-cnn", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-mlp", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256, help="CNN 全连接层宽度")
    ap.add_argument("--stages", type=int, nargs=2, default=(32, 64))
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只用 N 张训练图（调试用）")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_cifar10(limit=args.limit or None, verbose=True)
    Xtr, Xte, mean, std = normalize(Xtr, Xte)
    print(f"数据形状: train {Xtr.shape}  test {Xte.shape}  "
          f"（标准化后均值≈0、方差≈1）")

    mlp, mlp_h, mlp_acc = train_mlp(Xtr, ytr, Xte, yte, args.epochs_mlp, args.batch,
                                    args.lr, args.hidden_mlp, args.seed,
                                    "models/mlp_cifar10.npz")
    cnn, cnn_h, cnn_acc = train_cnn(Xtr, ytr, Xte, yte, args.epochs_cnn, args.batch,
                                    args.lr, tuple(args.stages), args.hidden,
                                    args.seed, args.wd, bool(args.augment),
                                    "models/cnn_cifar10.npz")

    pred_cnn = cnn.predict(Xte)
    plot_all(mlp_h, cnn_h, mlp_acc, cnn_acc, pred_cnn, yte, Xte, cnn, mean, std)

    # 最容易混淆的 6 对（只看"真 a 判成 b"的方向）
    cm = confusion_matrix(pred_cnn, yte, len(CLASSES))
    off = cm.copy(); np.fill_diagonal(off, 0)
    flat = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:8]
    confused = [[CLASSES[t], CLASSES[p], int(off[t, p])] for t, p in flat]
    print("\n最易混淆（真 -> 预测, 次数）:",
          ", ".join(f"{a}->{b}({c})" for a, b, c in confused))

    per_cls = np.diag(cm) / np.maximum(cm.sum(1), 1)
    print("每类准确率:", ", ".join(f"{CLASSES[i]} {v*100:.1f}%" for i, v in
                              enumerate(per_cls)))
    worst = np.argsort(per_cls)[:3]
    print("最差三类:", ", ".join(f"{CLASSES[i]} {per_cls[i]*100:.1f}%" for i in worst))
    print(f"\nMLP {mlp_acc*100:.2f}% -> CNN {cnn_acc*100:.2f}%  "
          f"（提升 {(cnn_acc-mlp_acc)*100:.1f} 个点）")

    summary = {
        "num_classes": len(CLASSES), "classes": CLASSES,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "normalize": {"mean": mean.ravel().tolist(), "std": std.ravel().tolist()},
        "mlp": {"sizes": [3072, args.hidden_mlp, len(CLASSES)],
                "epochs": args.epochs_mlp, "best_test_acc": mlp_acc, "history": mlp_h},
        "cnn": {"stages": list(args.stages), "hidden": args.hidden,
                "epochs": args.epochs_cnn, "wd": args.wd,
                "augment": bool(args.augment), "best_test_acc": cnn_acc,
                "history": cnn_h},
        "confusions": confused,
        "per_class_acc": {CLASSES[i]: float(v) for i, v in enumerate(per_cls)},
    }
    with open("models/cifar10_history.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n已保存 models/cifar10_history.json 和 figures/cifar10_*.png")


if __name__ == "__main__":
    main()
