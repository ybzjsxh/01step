"""第 5 步：卷积层的梯度检验。

卷积层比全连接层多了两个容易写错的地方：
  1. im2col 的列顺序（k*k*C 还是 C*k*k）必须和 W 的行顺序一致；
  2. col2im 是**累加**而不是赋值 —— 相邻窗口在重叠像素上会贡献多次梯度。
这两处错了，代码照样跑，只是不收敛。所以先做题，再训练。

运行：python step5_gradcheck_conv.py
"""

import numpy as np

from conv import Conv2D, MaxPool2D, Flatten, CNN
from nn import Linear, ReLU, SoftmaxCrossEntropy

np.set_printoptions(precision=6, suppress=True)


def build(seed=7, bad_pool=False):
    """8x8 -> conv3x3(3) -> 6x6 -> pool2 -> 3x3 -> flatten(27) -> linear(5)。"""
    rng = np.random.default_rng(seed)
    layers = [Conv2D(1, 3, 3, rng=rng), ReLU(),
              BadPool() if bad_pool else MaxPool2D(2), Flatten(),
              Linear(3 * 3 * 3, 5, rng)]
    return layers


class BadPool(MaxPool2D):
    """反例：把梯度平均分给窗口里 4 个位置（正确做法是只给最大值那个）。"""

    def backward(self, g):
        N, H, W, C = self.shape
        s = self.size
        d = np.repeat(np.repeat(g, s, axis=1), s, axis=2) / (s * s)   # 错在这里
        return d


def forward_loss(layers, loss_fn, X, y):
    h = X
    for layer in layers:
        h = layer.forward(h)
    return loss_fn.forward(h, y)


def backward_all(layers, loss_fn, X, y, return_input_grad=False):
    h = X
    for layer in layers:
        h = layer.forward(h)
    loss_fn.forward(h, y)
    g = loss_fn.backward()
    for layer in reversed(layers):
        g = layer.backward(g)
    params = [p for layer in layers for (p, _) in layer.params_and_grads()]
    grads = [gr for layer in layers for (_, gr) in layer.params_and_grads()]
    return (params, grads, g) if return_input_grad else (params, grads)


def numeric_grads(f, values, eps=1e-6):
    out = []
    for v in values:
        g = np.zeros_like(v)
        flat_v, flat_g = v.reshape(-1), g.reshape(-1)
        for i in range(flat_v.size):
            orig = flat_v[i]
            flat_v[i] = orig + eps
            plus = f()
            flat_v[i] = orig - eps
            minus = f()
            flat_v[i] = orig
            flat_g[i] = (plus - minus) / (2 * eps)
        out.append(g)
    return out


def rel_err(a, b):
    return (np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))).max()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2, 8, 8, 1))          # 2 张 8x8 单通道"图"
    y = np.array([1, 3])
    loss_fn = SoftmaxCrossEntropy()

    print("=" * 70)
    print("A. 正确实现的卷积网络（8x8 -> conv3x3 -> pool -> linear）")
    print("=" * 70)
    layers = build()
    params, grads = backward_all(layers, loss_fn, X, y)
    nums = numeric_grads(lambda: forward_loss(layers, loss_fn, X, y), params)

    names = ["conv.W", "conv.b", "fc.W", "fc.b"]
    for n, p, a, num in zip(names, params, grads, nums):
        print(f"  {n:<8} 形状 {str(p.shape):<12} 最大相对误差 {rel_err(a, num):.2e}")

    # 输入梯度也检验一遍 —— 它检验的是 col2im（最容易写错的地方）
    params2, grads2, dx = backward_all(layers, loss_fn, X, y, return_input_grad=True)
    num_dx = numeric_grads(lambda: forward_loss(layers, loss_fn, X, y), [X])
    print(f"  {'dX':<8} 形状 {str(X.shape):<12} 最大相对误差 {rel_err(dx, num_dx[0]):.2e}")
    print("  （dX 通过 = col2im 的累加逻辑正确；卷积核排列顺序也一致）")

    print("\n" + "=" * 70)
    print("B. 反例：池化层把梯度平均分给 4 个位置")
    print("=" * 70)
    bad = build(bad_pool=True)
    p_bad, g_bad = backward_all(bad, loss_fn, X, y)
    for n, a, num in zip(names, g_bad, nums):
        print(f"  {n:<8} 最大相对误差 {rel_err(a, num):.2e}")
    print("  conv 的参数梯度全错 -> 一眼看出池化反向写错了")
    print("  （fc.W 却是对的：池化在 fc 的下游，梯度还没流到它那儿，所以它察觉不到）")

    print("\n" + "=" * 70)
    print("C. 参数量对比：为什么卷积比全连接省")
    print("=" * 70)
    conv_params = 3 * 3 * 1 * 3 + 3
    cnn = CNN(in_ch=1, conv_ch=16, k=5, hidden=128)
    conv_layer = cnn.layers[0].W.size + cnn.layers[0].b.size
    total = sum(p.size for p, _ in cnn.params_and_grads())
    # 如果不用卷积，而是用全连接去产生同样大小的 24x24x16 特征图：
    dense_equiv = 28 * 28 * (24 * 24 * 16)
    print(f"  本例卷积层(3 个 3x3 核): {conv_params} 个参数")
    print(f"  实战 CNN 第一层(16 个 5x5 核): {conv_layer} 个参数")
    print(f"  若用全连接产生同样的 24x24x16 特征图: {dense_equiv:,} 个参数")
    print(f"  -> 卷积用 {conv_layer} 个参数干完了全连接需要 {dense_equiv:,} 个参数才能干的事，")
    print(f"     差了 {dense_equiv // conv_layer} 倍。原因是这 {conv_layer} 个参数在 24x24 个位置上**共享**，")
    print("     同一组「边缘检测器」扫遍全图 —— 这就是平移不变性的来源。")
    print(f"\n  整个 CNN 一共 {total:,} 个参数，其中大部分({147584:,})集中在最后那个全连接层，")
    print("  因为卷积负责「看局部」，全连接负责「汇总做判断」，后者本来就该更贵。")
