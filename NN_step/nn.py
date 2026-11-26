"""纯 NumPy 实现的最小神经网络库：Linear / ReLU / Softmax+交叉熵 / MLP / SGD / Adam。

设计原则：一个「层」只做三件事
    forward(x)  -> 算输出，并缓存反向传播要用的中间量
    backward(g) -> 收上游传来的梯度 g，算出本层参数的梯度，再把梯度传给下游
这样整个网络就是一串层的串联，训练 = 前向 -> 损失 -> 反向 -> 更新 循环。
"""

import numpy as np


# ---------------------------------------------------------------- 层


class Linear:
    """全连接层 y = x @ W + b。

    W: (in_dim, out_dim)   b: (out_dim,)
    """

    def __init__(self, in_dim, out_dim, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        # He 初始化：std = sqrt(2/in_dim)，专门配合 ReLU。
        # 全零初始化会让所有隐藏单元学到同样的东西（对称性无法打破）。
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x                       # 反向传播要拿它算 dW = x^T @ g
        return x @ self.W + self.b

    def backward(self, g):
        # 关键：一次矩阵乘法同时得到整个 batch 上所有梯度之和
        self.dW = self.x.T @ g           # (in, N) @ (N, out) -> (in, out)
        self.db = g.sum(axis=0)          # (out,)
        return g @ self.W.T              # 传给上一层的梯度，形状回到 (N, in)

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """max(0, x)。没有参数，只需要把 x<=0 的位置的梯度清零。"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask

    def params_and_grads(self):
        return []


class SoftmaxCrossEntropy:
    """softmax + 交叉熵，合在一起写是为了反向传播只差一个 p - y。"""

    def __init__(self):
        self.p = None
        self.y = None

    def forward(self, logits, y):
        # 减去每行最大值：防止 exp(大数) 溢出成 inf/nan（数学上等价）
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        self.p = exp / exp.sum(axis=1, keepdims=True)
        self.y = y
        n = y.shape[0]
        # 取出正确类别的概率，取负对数再平均
        loss = -np.log(self.p[np.arange(n), y] + 1e-12).mean()
        return loss

    def backward(self):
        # 整个网络里最漂亮的一步：dL/dlogits = (p - onehot(y)) / N
        d = self.p.copy()
        d[np.arange(len(self.y)), self.y] -= 1.0
        return d / len(self.y)   # 因为 loss 是 batch 平均，梯度也要除以 N

    def params_and_grads(self):
        return []


# ---------------------------------------------------------------- 网络


class MLP:
    """多层感知机：若干 Linear+ReLU，最后接一个 Linear 出 10 个 logits。"""

    def __init__(self, sizes=(784, 256, 10), seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:          # 最后一层不接激活函数
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()
        if dtype is not np.float64:
            for layer in self.layers:
                if isinstance(layer, Linear):
                    layer.W = layer.W.astype(dtype)
                    layer.b = layer.b.astype(dtype)
                    layer.dW = layer.dW.astype(dtype)
                    layer.db = layer.db.astype(dtype)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                          # 返回 logits（未过 softmax）

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

    def predict(self, x, batch_size=1000):
        preds = []
        for i in range(0, len(x), batch_size):
            logits = self.forward(x[i:i + batch_size])
            preds.append(logits.argmax(axis=1))
        return np.concatenate(preds)

    def accuracy(self, x, y, batch_size=1000):
        return float((self.predict(x, batch_size) == y).mean())

    # -------------------------------------------------- 存盘 / 读盘
    def save(self, path):
        arrays = {f"W{i}": l.W for i, l in enumerate(self.layers) if isinstance(l, Linear)}
        arrays.update({f"b{i}": l.b for i, l in enumerate(self.layers) if isinstance(l, Linear)})
        arrays["sizes"] = np.array(self.sizes)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        sizes = tuple(int(s) for s in d["sizes"])
        model = cls(sizes)
        for i, l in enumerate(model.layers):
            if isinstance(l, Linear):
                l.W = d[f"W{i}"].copy()
                l.b = d[f"b{i}"].copy()
                l.dW = np.zeros_like(l.W)
                l.db = np.zeros_like(l.b)
        return model


# ---------------------------------------------------------------- 优化器


class SGD:
    """带动量的随机梯度下降：v = m*v - lr*g;  param += v"""

    def __init__(self, model, lr=0.1, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p) for p, _ in model.params_and_grads()]

    def step(self):
        for i, (p, g) in enumerate(self.model.params_and_grads()):
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            p += self.v[i]


class Adam:
    """Adam：给每个参数自适应学习率，超参几乎不用调，是新手首选。"""

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        params = model.params_and_grads()
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.model.params_and_grads()):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g          # 一阶矩
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)    # 二阶矩
            m_hat = self.m[i] / (1 - self.b1 ** self.t)                  # 偏差修正
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
