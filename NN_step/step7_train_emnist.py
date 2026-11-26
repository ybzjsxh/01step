"""第 7 步：把同一套 NumPy 网络搬到 EMNIST balanced（47 类）上跑。

和 MNIST 的区别只有三件事：
- 输出层从 10 改成 47（初始 loss 从 ln10=2.30 变成 ln47=3.85，这是新的体检基准）
- 数据发现躺倒 -> 加载器里 transpose 扶正（见 emnist.py 注释）
- 类别不再均衡到"完美"，但 balanced 子集已经足够接近

CD   : .venv/bin/python step7_train_emnist.py
产物 : models/mlp_emnist.npz, models/cnn_emnist.npz, models/emnist_history.json
       figures/emnist_curves.png, emnist_confusion.png, emnist_per_class.png, emnist_samples.png
"""

import argparse
import json
import os
import time

import numpy as np
import os
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conv import CNN
from emnist import load_emnist, load_mapping
from nn import MLP, Adam

N_CLASSES = 47


def shift_augment(xb, rng, max_shift=2):
    """随机平移 ±2 像素。字母和数字一样，平移不改变类别。"""
    ms = max_shift
    out = np.empty_like(xb)
    for i in range(len(xb)):
        dy, dx = rng.integers(-ms, ms + 1, size=2)
        p = np.pad(xb[i], ((ms, ms), (ms, ms), (0, 0)), mode="constant")
        out[i] = p[ms + dy:ms + dy + 28, ms + dx:ms + dx + 28]
    return out


def train_mlp(Xtr, ytr, Xte, yte, epochs, batch, lr, hidden, seed, out):
    model = MLP(sizes=(784, hidden, N_CLASSES), seed=seed)
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    print(f"\n[MLP] 784-{hidden}-{N_CLASSES}  初始 loss = "
          f"{model.loss(Xtr[:2000], ytr[:2000]):.4f}（期望 ≈ {np.log(N_CLASSES):.2f}）")

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "sec": []}
    n, best, t_all = len(Xtr), 0.0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model.loss_fn.forward(model.forward(Xtr[idx]), ytr[idx])
            model.backward()
            opt.step()
        tr_loss = model.loss(Xtr[:4000], ytr[:4000])
        tr_acc = model.accuracy(Xtr[:8000], ytr[:8000])
        te_acc = model.accuracy(Xte, yte)
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


def train_cnn(Xtr, ytr, Xte, yte, epochs, batch, lr, conv_ch, hidden, seed, wd, out):
    Xtr3 = Xtr.reshape(-1, 28, 28, 1)
    Xte3 = Xte.reshape(-1, 28, 28, 1)
    model = CNN(in_ch=1, conv_ch=conv_ch, k=5, hidden=hidden,
                num_classes=N_CLASSES, seed=seed)
    for layer in model.layers:                    # float32：中间张量太占内存
        if hasattr(layer, "W"):
            layer.W = layer.W.astype(np.float32)
            layer.b = layer.b.astype(np.float32)
            layer.dW = np.zeros_like(layer.W)
            layer.db = np.zeros_like(layer.b)
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(seed)
    n_params = sum(p.size for p, _ in model.params_and_grads())
    print(f"\n[CNN] conv5x5({conv_ch})->ReLU->pool2->fc{hidden}->fc{N_CLASSES}  "
          f"参数量 {n_params:,}  初始 loss = {model.loss(Xtr3[:2000], ytr[:2000]):.4f}")

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "sec": []}
    n, best, t_all, lr_now = len(Xtr), 0.0, time.time(), lr
    for ep in range(1, epochs + 1):
        if ep == epochs // 2 + 1:
            lr_now = lr / 2
            opt.lr = lr_now
        t0 = time.time()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = shift_augment(Xtr3[idx], rng)
            model.loss_fn.forward(model.forward(xb), ytr[idx])
            model.backward()
            opt.step()
            if wd > 0:
                for p, _ in model.params_and_grads():
                    p *= (1.0 - lr_now * wd)
        tr_loss = model.loss(Xtr3[:2000], ytr[:2000])
        tr_acc = model.accuracy(Xtr3[:5000], ytr[:5000])
        te_acc = model.accuracy(Xte3, yte)
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
    print(f"  CNN 最佳 test_acc {best*100:.2f}%  总耗时 {time.time()-t_all:.0f}s")
    return CNN.load(out), hist, best


def confusion_matrix(pred, y, k=N_CLASSES):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def plot_all(mlp_h, cnn_h, mlp_acc, cnn_acc, pred_mlp, pred_cnn, yte, chars, Xte):
    os.makedirs("figures", exist_ok=True)

    # 1) 两条训练曲线（MLP vs CNN）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for h, name, c in ((mlp_h, "MLP", "#9aa7b8"), (cnn_h, "CNN", "#3b7dd8")):
        axes[0].plot(h["train_loss"], marker="o", ms=4, label=name, color=c)
        axes[1].plot([a * 100 for a in h["train_acc"]], marker="o", ms=4,
                     label=f"{name} train", color=c, alpha=0.6)
        axes[1].plot([a * 100 for a in h["test_acc"]], marker="s", ms=4, label=f"{name} test", color=c)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title(f"EMNIST balanced ({N_CLASSES} classes): loss")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("accuracy"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/emnist_curves.png", dpi=150); plt.close(fig)

    # 2) 混淆矩阵：47x47，画成 log 色标，看不清的格子就是"难分的字符对"
    cm = confusion_matrix(pred_cnn, yte)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log1p(cm), cmap="magma_r")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(chars, fontsize=6, rotation=90)
    ax.set_yticklabels(chars, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"CNN confusion matrix (log scale), test acc {cnn_acc*100:.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.045, label="log(1+count)")
    fig.tight_layout(); fig.savefig("figures/emnist_confusion.png", dpi=160); plt.close(fig)

    # 3) 每类准确率：找出最弱的类别
    per_cls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    order = np.argsort(per_cls)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(range(N_CLASSES), per_cls[order] * 100, color="#3b7dd8")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels([chars[i] for i in order], fontsize=8)
    ax.set_ylim(min(per_cls.min() * 100 - 3, 80), 100)
    ax.set_ylabel("per-class accuracy (%)")
    ax.set_title("Per-class accuracy, sorted (right = best)")
    for i, idx in enumerate(order[:3]):
        ax.text(i, per_cls[idx] * 100 - 1.5, f"{per_cls[idx]*100:.0f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/emnist_per_class.png", dpi=150); plt.close(fig)

    # 4) 47 个类各抽一张样本图，检查"标签和图像对得上"
    fig, axes = plt.subplots(5, 10, figsize=(10, 5.6))
    rng = np.random.default_rng(1)
    samples = []
    for c in range(N_CLASSES):
        idxs = np.where(yte == c)[0]
        samples.append(idxs[rng.integers(len(idxs))] if len(idxs) else None)
    for j, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if j >= N_CLASSES:
            continue
        i = samples[j]
        if i is None:
            continue
        ax.imshow(Xte[i].reshape(28, 28), cmap="gray")
        ok = pred_cnn[i] == yte[i]
        ax.set_title(chars[j], fontsize=9,
                     color="black" if ok else "#d02f2f")
    fig.suptitle("One test sample per class (red title = CNN misclassified this one)")
    fig.tight_layout(); fig.savefig("figures/emnist_samples.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-mlp", type=int, default=8)
    ap.add_argument("--epochs-cnn", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--conv-ch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="只用 N 张训练图（调试用）")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_emnist(limit=args.limit or None, verbose=True)
    chars = load_mapping()

    mlp, mlp_h, mlp_acc = train_mlp(Xtr, ytr, Xte, yte, args.epochs_mlp, args.batch,
                                    args.lr, args.hidden, args.seed,
                                    "models/mlp_emnist.npz")
    cnn, cnn_h, cnn_acc = train_cnn(Xtr, ytr, Xte, yte, args.epochs_cnn, args.batch,
                                    args.lr, args.conv_ch, 128, args.seed, args.wd,
                                    "models/cnn_emnist.npz")

    yte3 = yte
    pred_mlp = mlp.predict(Xte)
    pred_cnn = cnn.predict(Xte.reshape(-1, 28, 28, 1))
    plot_all(mlp_h, cnn_h, mlp_acc, cnn_acc, pred_mlp, pred_cnn, yte3, chars, Xte)

    # 最容易混淆的 5 对（只看"真 a 判成 b"的方向）
    cm = confusion_matrix(pred_cnn, yte3)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:8]
    confused = [[chars[t], chars[p], int(off[t, p])] for t, p in flat]
    print("\n最易混淆（真 -> 预测, 次数）:",
          ", ".join(f"{a}->{b}({c})" for a, b, c in confused))

    summary = {
        "split": "balanced", "num_classes": N_CLASSES,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "mlp": {"sizes": [784, args.hidden, N_CLASSES], "epochs": args.epochs_mlp,
                "best_test_acc": mlp_acc, "history": mlp_h},
        "cnn": {"epochs": args.epochs_cnn, "conv_ch": args.conv_ch,
                "best_test_acc": cnn_acc, "history": cnn_h},
        "confusions": confused,
        "per_class_acc": {chars[i]: float(v) for i, v in
                          enumerate(np.diag(cm) / np.maximum(cm.sum(1), 1))},
    }
    with open("models/emnist_history.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n已保存 models/emnist_history.json 和 figures/emnist_*.png")


if __name__ == "__main__":
    main()
