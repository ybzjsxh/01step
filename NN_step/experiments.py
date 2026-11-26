"""对照实验：同一套数据和代码，只改结构/优化器，看什么在起作用。

运行：python experiments.py            # 全部约 3 分钟
      python experiments.py --quick    # 只跑前两个
结果写进 models/experiments.json，可以直接对照 README 第 10 节。
"""

import argparse
import json
import time

import numpy as np

from mnist import load_mnist
from nn import MLP, Adam, SGD


def train(sizes, optimizer="adam", lr=1e-3, epochs=10, batch=64, seed=0):
    Xtr, ytr, Xte, yte = DATA
    model = MLP(sizes=sizes, seed=seed)
    opt = Adam(model, lr=lr) if optimizer == "adam" else SGD(model, lr=lr, momentum=0.9)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    t0 = time.time()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model.loss_fn.forward(model.forward(Xtr[idx]), ytr[idx])
            model.backward()
            opt.step()
    dt = time.time() - t0
    acc = model.accuracy(Xte, yte)
    return acc, dt


CONFIGS = [
    # (标签, sizes, optimizer, lr)
    ("线性模型 784-10", (784, 10), "adam", 1e-3),
    ("窄隐藏层 784-32-10", (784, 32, 10), "adam", 1e-3),
    ("基线 784-256-10", (784, 256, 10), "adam", 1e-3),
    ("宽隐藏层 784-512-10", (784, 512, 10), "adam", 1e-3),
    ("双层 784-256-128-10", (784, 256, 128, 10), "adam", 1e-3),
    ("基线 + SGD+momentum(0.1)", (784, 256, 10), "sgd", 0.1),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    DATA = load_mnist(verbose=False)
    rows = []
    for label, sizes, opt, lr in (CONFIGS[:2] if args.quick else CONFIGS):
        acc, dt = train(sizes, opt, lr)
        rows.append((label, acc * 100, dt))
        print(f"{label:<28} test_acc {acc*100:6.2f}%   {dt:5.1f}s", flush=True)

    with open("models/experiments.json", "w") as f:
        json.dump([{"config": r[0], "test_acc": round(r[1], 2), "sec": round(r[2], 1)} for r in rows],
                  f, indent=2, ensure_ascii=False)
    print("\n结果已写入 models/experiments.json")
