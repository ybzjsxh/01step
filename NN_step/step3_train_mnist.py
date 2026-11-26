"""第 3 步：在 MNIST 上把整个训练闭环跑通。

流程：取数据 -> 初始化 -> 循环{前向 -> 算损失 -> 反向传播 -> 更新参数} -> 评估

运行：
    python step3_train_mnist.py                     # 默认 10 轮，约半分钟
    python step3_train_mnist.py --epochs 20 --hidden 512 --lr 1e-3
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")   # 只跳过系统字体扫描，不影响画图

from mnist import load_mnist
from nn import MLP, Adam

FIG_DIR = "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-out", default="models/mlp_mnist.npz")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist()

    model = MLP(sizes=(784, args.hidden, 10), seed=args.seed)
    opt = Adam(model, lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # 开训前的健康检查：loss 应该 ≈ ln(10) ≈ 2.30
    initial_loss = model.loss(Xtr[:2000], ytr[:2000])
    print(f"\n初始 loss = {initial_loss:.4f}（期望 ≈ 2.30，明显偏离说明初始化/数据有问题）\n")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "sec": []}
    n = len(Xtr)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(n)               # 每轮重新打乱，否则模型会记住样本顺序
        total_loss, seen = 0.0, 0

        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xtr[idx], ytr[idx]

            logits = model.forward(xb)          # 1. 前向
            loss = model.loss_fn.forward(logits, yb)   # 2. 损失
            model.backward()                    # 3. 反向（梯度已写进各层的 dW/db）
            opt.step()                          # 4. 更新参数

            total_loss += loss * len(idx)
            seen += len(idx)

        train_loss = total_loss / seen
        train_acc = model.accuracy(Xtr[:5000], ytr[:5000])
        test_acc = model.accuracy(Xte, yte)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sec"].append(dt)

        print(f"epoch {epoch:2d}/{args.epochs}   loss {train_loss:.4f}   "
              f"train_acc {train_acc*100:5.2f}%   test_acc {test_acc*100:5.2f}%   ({dt:.1f}s)")

    total = time.time() - t_start
    final = model.accuracy(Xte, yte)
    n_err = int((model.predict(Xte) != yte).sum())
    print(f"\n完成：{args.epochs} 轮共 {total:.1f}s，测试集准确率 {final*100:.2f}%"
          f"（{len(yte)} 张里错 {n_err} 张）")

    model.save(args.model_out)
    with open(os.path.join("models", "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"模型已保存到 {args.model_out}")

    # 画 loss / 准确率曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(history["epoch"], history["train_loss"], marker="o")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
        axes[0].set_title("Cross-entropy loss"); axes[0].grid(alpha=0.3)
        axes[1].plot(history["epoch"], [a * 100 for a in history["train_acc"]], marker="o", label="train")
        axes[1].plot(history["epoch"], [a * 100 for a in history["test_acc"]], marker="s", label="test")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "training_curves.png")
        fig.savefig(path, dpi=140)
        print(f"曲线已保存到 {path}")
    except ImportError:
        print("（未安装 matplotlib，跳过画图）")


if __name__ == "__main__":
    main()
