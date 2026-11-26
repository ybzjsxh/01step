"""第 2 步：反向传播 + 梯度检验。

这是整套流程里最容易写错、也最值得投入时间的一步。
思路：解析梯度（backward 算出来的）和数值梯度（对每个参数做 ±eps 的
有限差分）必须几乎相等。相等 = 你的链式法则推导和代码都对。

运行：python step2_gradient_check.py
"""

import numpy as np

from nn import MLP, Linear, ReLU

np.set_printoptions(precision=6, suppress=True)


def analytic_grads(model, X, y, bug=None):
    """跑一次 forward + backward，返回参数梯度列表。

    bug 用来演示「写错反向传播会怎样」，正式训练时永远不要用：
      bug="no_norm"  忘了除以 batch size
      bug="no_mask"  忘了 ReLU 的 mask（梯度传过了被截断的神经元）
    """
    logits = model.forward(X)
    model.loss_fn.forward(logits, y)
    g = model.loss_fn.backward()
    if bug == "no_norm":
        g = g * len(y)          # 抵消掉 /N
    for layer in reversed(model.layers):
        if bug == "no_mask" and isinstance(layer, ReLU):
            continue                      # 错：梯度原样往下传，没有乘 mask

        g = layer.backward(g)
    return [gr.copy() for _, gr in model.params_and_grads()]


def numeric_grads(model, X, y, eps=1e-6):
    """中心差分：(L(θ+eps) - L(θ-eps)) / (2eps)，逐个参数算。"""
    grads = []
    for p, _ in model.params_and_grads():
        g = np.zeros_like(p)
        flat_p, flat_g = p.reshape(-1), g.reshape(-1)
        for i in range(flat_p.size):
            orig = flat_p[i]
            flat_p[i] = orig + eps
            loss_plus = model.loss(X, y)
            flat_p[i] = orig - eps
            loss_minus = model.loss(X, y)
            flat_p[i] = orig                  # 改回去
            flat_g[i] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(g)
    return grads


def rel_error(a, b):
    return np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))


def report(title, analytic, numeric, names):
    print(f"\n{title}")
    print(f"  {'参数':<6}{'形状':<12}{'最大相对误差':>14}")
    worst = 0.0
    for n, A, N in zip(names, analytic, numeric):
        err = rel_error(A, N).max()
        worst = max(worst, err)
        print(f"  {n:<8}{str(A.shape):<14}{err:>12.2e}")
    return worst


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # 一个小网络 + 小样本，让数值梯度可以全量校验
    model = MLP(sizes=(8, 6, 10), seed=1)
    X = rng.normal(0, 1, (5, 8))
    y = rng.integers(0, 10, 5)
    names = ["W1", "b1", "W2", "b2"]

    print("=" * 68)
    print("梯度检验：解析梯度 vs 数值梯度（网络 8-6-10，batch=5）")
    print("=" * 68)

    ana = analytic_grads(model, X, y)
    num = numeric_grads(model, X, y)
    worst = report("正确实现", ana, num, names)
    print(f"\n  最大相对误差 {worst:.2e}  ->  {'通过 ✅' if worst < 1e-6 else '不通过 ❌'}")
    print("  经验阈值：< 1e-6 完美；< 1e-4 可以接受（float32 或 eps 选得不好时）；> 1e-2 一定有 bug。")

    print("\n" + "=" * 68)
    print("反例：故意写错的反向传播，梯度检验会立刻报警")
    print("=" * 68)
    for bug, desc in [("no_norm", "忘了除以 batch size"), ("no_mask", "忘了 ReLU mask")]:
        bad = analytic_grads(model, X, y, bug=bug)
        w = report(f"BUG: {desc}", bad, num, names)
        print(f"  -> 最大相对误差 {w:.2e}，{'一眼看出有问题' if w > 1e-2 else '差异不明显'}")

    # 看一下 8-6-10 网络上第一个样本的梯度长什么样
    model2 = MLP(sizes=(8, 6, 10), seed=1)
    ana2 = analytic_grads(model2, X, y)
    print("\n" + "=" * 68)
    print("梯度实例：W2 的梯度（6x10），注意有正有负")
    print("=" * 68)
    print(ana2[2])
    print("\n解释：梯度 = 损失对该参数的方向导数。下降法就是让参数往梯度的反方向挪，")
    print("      挪的步长是学习率 lr。所以 lr 太大 -> 震荡发散（loss 变 nan），")
    print("      lr 太小 -> 训练慢得像没在学。")
