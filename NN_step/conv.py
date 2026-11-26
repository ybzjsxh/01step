"""纯 NumPy 手写卷积：Conv2D / MaxPool2D / Flatten / CNN。

核心思想只有一个——**im2col**：
卷积 = 每个滑动窗口和卷积核做内积。把每个窗口里的像素"拉成一列"，
所有窗口的列拼成矩阵，卷积就变成一次矩阵乘法，可以直接用 BLAS。
反向传播也顺理成章：矩阵乘法的梯度我们第 3 节已经推过。

形状约定（和 nn.py 的 MLP 不同，卷积层需要空间结构）：
    输入 x: (N, H, W, C)
    Conv2D 输出: (N, H_out, W_out, F)
"""

import numpy as np


class Conv2D:
    """二维卷积，stride 可调，可选 zero padding。

    W: (k*k*C_in, F)  —— 把 (F, C_in, k, k) 的卷积核摊平，方便矩阵乘法

    pad 的作用：不加 padding 时每过一层卷积，特征图就缩 k-1 个像素（5x5 卷积
    在 28x28 上做两次就只剩 16x16），信息全被"磨"掉了。pad=1 + k=3 让输出尺寸
    和输入完全相同（same 卷积），这样就能堆很多层而不用担心图越卷越小。
    """

    def __init__(self, in_ch, out_ch, k=5, stride=1, rng=None, pad=0):
        rng = np.random.default_rng(0) if rng is None else rng
        fan_in = k * k * in_ch
        self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, out_ch))
        self.b = np.zeros(out_ch)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.k, self.stride, self.pad = k, stride, pad

    def _im2col(self, x):
        N, H, W, C = x.shape
        k, s = self.k, self.stride
        oh, ow = (H - k) // s + 1, (W - k) // s + 1
        # 每个 kernel 位置单独取一次切片，拼成 (N, oh, ow, k, k, C)
        cols = np.empty((N, oh, ow, k, k, C), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                cols[:, :, :, i, j, :] = x[:, i:i + s * oh:s, j:j + s * ow:s, :]
        return cols.reshape(N, oh, ow, k * k * C), oh, ow

    def forward(self, x):
        N = x.shape[0]
        self.in_shape = x.shape                      # 反向要把梯度裁回这个尺寸
        if self.pad:
            x = np.pad(x, ((0, 0), (self.pad, self.pad),
                           (self.pad, self.pad), (0, 0)))
        self.cols, self.oh, self.ow = self._im2col(x)          # (N, oh, ow, K)
        self.F = self.W.shape[1]
        M = self.cols.reshape(-1, self.cols.shape[-1])          # (N*oh*ow, K)
        out = M @ self.W + self.b                               # 一次矩阵乘法 = 整个卷积
        return out.reshape(N, self.oh, self.ow, self.F)

    def backward(self, g):
        N = g.shape[0]
        g_flat = g.reshape(-1, self.F)                          # (N*oh*ow, F)
        M = self.cols.reshape(-1, self.cols.shape[-1])
        self.dW = M.T @ g_flat
        self.db = g_flat.sum(axis=0)
        dcols = (g_flat @ self.W.T).reshape(N, self.oh, self.ow, self.k, self.k, -1)
        # col2im：把每个窗口的梯度按位置累加回原图
        dx = np.zeros((N, self.oh * self.stride + self.k - 1,
                       self.ow * self.stride + self.k - 1, dcols.shape[-1]),
                      dtype=dcols.dtype)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, i:i + self.stride * self.oh:self.stride,
                   j:j + self.stride * self.ow:self.stride, :] += dcols[:, :, :, i, j, :]
        if self.pad:                                # padding 区不参与运算，梯度直接丢掉
            p = self.pad
            return dx[:, p:p + self.in_shape[1], p:p + self.in_shape[2], :]
        return dx

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class MaxPool2D:
    """2x2、stride 2 的最大池化：每个 2x2 窗口取最大值，反向只把梯度给那个最大值。"""

    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        N, H, W, C = x.shape
        s = self.size
        self.shape = x.shape
        # (N, H/2, 2, W/2, 2, C) -> (N, H/2, W/2, C, 4)：把每个窗口的 4 个数放到最后一维
        xr = x.reshape(N, H // s, s, W // s, s, C).transpose(0, 1, 3, 5, 2, 4)
        xr = xr.reshape(N, H // s, W // s, C, s * s)
        self.argmax = xr.argmax(axis=-1)
        return np.take_along_axis(xr, self.argmax[..., None], axis=-1).squeeze(-1)

    def backward(self, g):
        N, H, W, C = self.shape
        s = self.size
        d = np.zeros((N, H // s, W // s, C, s * s), dtype=g.dtype)
        np.put_along_axis(d, self.argmax[..., None], g[..., None], axis=-1)
        d = d.reshape(N, H // s, W // s, C, s, s).transpose(0, 1, 4, 2, 5, 3)
        return d.reshape(N, H, W, C)

    def params_and_grads(self):
        return []


class Flatten:
    """(N, H, W, C) -> (N, H*W*C)，反向就是 reshape 回去。"""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, g):
        return g.reshape(self.shape)

    def params_and_grads(self):
        return []


# ------------------------------------------------------------------ 网络


class CNN:
    """Conv -> ReLU -> Pool -> Flatten -> Linear -> ReLU -> Linear。

    复用 nn.py 里的 ReLU、SoftmaxCrossEntropy、Adam，因为它们跟卷积无关。
    """

    def __init__(self, in_ch=1, conv_ch=16, k=5, hidden=128, num_classes=10, seed=0):
        from nn import Linear, ReLU

        rng = np.random.default_rng(seed)
        self.config = dict(in_ch=in_ch, conv_ch=conv_ch, k=k, hidden=hidden,
                           num_classes=num_classes)
        h1 = (28 - k + 1) // 2                       # 卷积后 24x24，池化后 12x12
        self.layers = [
            Conv2D(in_ch, conv_ch, k, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Linear(conv_ch * h1 * h1, hidden, rng),
            ReLU(),
            Linear(hidden, num_classes, rng),
        ]
        from nn import SoftmaxCrossEntropy
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        g = self.loss_fn.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def loss(self, x, y):
        return self.loss_fn.forward(self.forward(x), y)

    def params_and_grads(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params_and_grads())
        return out

    def predict(self, x, batch_size=500):
        preds = []
        for i in range(0, len(x), batch_size):
            preds.append(self.forward(x[i:i + batch_size]).argmax(axis=1))
        return np.concatenate(preds)

    def accuracy(self, x, y, batch_size=500):
        return float((self.predict(x, batch_size) == y).mean())

    def save(self, path):
        arrays = {}
        idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2D,)) or layer.__class__.__name__ == "Linear":
                arrays[f"W{idx}"] = layer.W
                arrays[f"b{idx}"] = layer.b
                idx += 1
        arrays["config"] = np.array([self.config[k] for k in
                                     ("in_ch", "conv_ch", "k", "hidden", "num_classes")])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        cfg = dict(zip(("in_ch", "conv_ch", "k", "hidden", "num_classes"),
                       [int(v) for v in d["config"]]))
        model = cls(**cfg)
        idx = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D) or layer.__class__.__name__ == "Linear":
                layer.W = d[f"W{idx}"].copy()
                layer.b = d[f"b{idx}"].copy()
                layer.dW = np.zeros_like(layer.W)
                layer.db = np.zeros_like(layer.b)
                idx += 1
        return model


# ------------------------------------------------------- 更深的卷积网络


class DeepCNN:
    """可堆叠的 CNN：每个 stage = conv(k,pad) -> ReLU -> pool2，最后接 FC 头。

    CIFAR-10 是 32x32 彩色图，比 MNIST 难在两点：
    - 3 个通道、物体有颜色和纹理，单个 5x5 卷积不够看；
    - 类间差异更细（猫 vs 狗、汽车 vs 卡车），需要更大的感受野。

    所以这里用最朴素的 VGG 思路：两个 stage，通道 32 -> 64，每层都 k=3/pad=1
    （same 卷积，尺寸不变），每层之后 2x2 池化把边长减半：
        32x32 --conv32--> 32x32 --pool--> 16x16 --conv64--> 16x16 --pool--> 8x8
    然后 Flatten 成 64*8*8=4096 维接全连接。
    """

    def __init__(self, in_ch=3, in_size=32, stages=(32, 64), num_classes=10,
                 hidden=256, k=3, pad=1, seed=0):
        from nn import Linear, ReLU

        rng = np.random.default_rng(seed)
        self.config = dict(in_ch=in_ch, in_size=in_size, stages=list(stages),
                           num_classes=num_classes, hidden=hidden, k=k, pad=pad)
        self.layers = []
        ch, size = in_ch, in_size
        for out_ch in stages:
            self.layers += [Conv2D(ch, out_ch, k, pad=pad, rng=rng), ReLU(),
                            MaxPool2D(2)]
            ch, size = out_ch, size // 2
        self.layers += [Flatten(), Linear(ch * size * size, hidden, rng), ReLU(),
                        Linear(hidden, num_classes, rng)]
        from nn import SoftmaxCrossEntropy
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        g = self.loss_fn.backward()
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def loss(self, x, y):
        return self.loss_fn.forward(self.forward(x), y)

    def params_and_grads(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params_and_grads())
        return out

    def predict(self, x, batch_size=200):
        preds = []
        for i in range(0, len(x), batch_size):
            preds.append(self.forward(x[i:i + batch_size]).argmax(axis=1))
        return np.concatenate(preds)

    def accuracy(self, x, y, batch_size=200):
        return float((self.predict(x, batch_size) == y).mean())

    def save(self, path):
        import json
        arrays = {}
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D) or layer.__class__.__name__ == "Linear":
                arrays[f"W{idx}"] = layer.W
                arrays[f"b{idx}"] = layer.b
                idx += 1
        arrays["config"] = np.array(json.dumps(self.config))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        import json
        d = np.load(path)
        cfg = json.loads(str(d["config"]))
        model = cls(**cfg)
        idx = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D) or layer.__class__.__name__ == "Linear":
                layer.W = d[f"W{idx}"].copy()
                layer.b = d[f"b{idx}"].copy()
                layer.dW = np.zeros_like(layer.W)
                layer.db = np.zeros_like(layer.b)
                idx += 1
        return model
