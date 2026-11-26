"""第 1 步：只做前向传播 —— 看清神经网络到底在算什么。

运行：python step1_forward.py

看点：
1) 一个 3 -> 4 -> 2 的小网络，用手算级别的数字走一遍，每步都打印出来；
2) 再换成真实尺寸（784 -> 256 -> 10）跑一个 batch，观察形状怎么流动；
3) 观察未训练的模型输出概率 ≈ 1/10 —— 它在瞎猜，loss ≈ ln(10)。
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # 数值稳定：先减最大值
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


print("=" * 68)
print("A. 手算版：3 -> 4 -> 2，单样本 x = [0.5, -1.0, 2.0]")
print("=" * 68)

x = np.array([0.5, -1.0, 2.0])
# 为了口算方便，权重故意取小整数（真实训练时是随机小数）
W1 = np.array([[1.0, -2.0, 0.5, 1.0],
               [0.0, 1.0, -1.0, 2.0],
               [2.0, 0.5, 1.0, -1.0]])      # (3, 4)
b1 = np.array([0.0, 1.0, -0.5, 0.0])        # (4,)
W2 = np.array([[1.0, -1.0],
               [0.5, 2.0],
               [-2.0, 1.0],
               [1.0, 1.5]])                 # (4, 2)
b2 = np.array([0.0, 0.5])                   # (2,)

print(f"输入 x         形状 {x.shape} = {x}")

z1 = x @ W1 + b1                            # (3,) @ (3,4) -> (4,)
print(f"\n第1层 Linear   z1 = x@W1 + b1")
print(f"  x@W1 = {x @ W1}")
print(f"  +b1   = {z1}          形状 {z1.shape}")

a1 = np.maximum(z1, 0)                      # ReLU：负数直接砍成 0
print(f"\nReLU 激活      a1 = max(z1, 0)")
print(f"  a1 = {a1}                            形状 {a1.shape}")

z2 = a1 @ W2 + b2                           # (4,) @ (4,2) -> (2,)
print(f"\n第2层 Linear   z2 = a1@W2 + b2 = {z2}  形状 {z2.shape}  （这 2 个数叫 logits）")

p = softmax(z2)
print(f"\nSoftmax        p = {p}  概率和={p.sum():.4f}")

y = 1                                       # 假设真实标签是第 1 类
loss = -np.log(p[y])
print(f"\n交叉熵损失     loss = -log(p[{y}]) = -log({p[y]:.4f}) = {loss:.4f}")
print("""
一句话总结前向传播：数据从输入层出发，每层做一次「线性变换 + 非线性激活」，
最后用 softmax 把 2 个 logits 变成 2 个概率，用交叉熵衡量「猜对的程度」。
""")

print("=" * 68)
print("B. 真实尺寸：784 -> 256 -> 10，一个 batch 8 张图")
print("=" * 68)

rng = np.random.default_rng(0)
batch = rng.random((8, 784))                # 假装是 8 张展平的手写数字图
W1 = rng.normal(0, np.sqrt(2 / 784), (784, 256))   # He 初始化
b1 = np.zeros(256)
W2 = rng.normal(0, np.sqrt(2 / 256), (256, 10))
b2 = np.zeros(10)

h = batch @ W1 + b1
h = np.maximum(h, 0)
logits = h @ W2 + b2
probs = softmax(logits)

print(f"输入 batch     {batch.shape}   (8 张图，每张 28*28=784 个像素，已归一化到 0~1)")
print(f"隐藏层 h       {h.shape}       (每个样本变成 256 个特征)")
print(f"输出 logits    {logits.shape}        (10 个类别各一个分数)")
print(f"输出 probs     {probs.shape}        (每行加起来 = {probs.sum(axis=1).round(4)}）")
print(f"\n未训练时每行概率都接近 1/10={1/10:.3f}，说明模型在瞎猜：")
print(probs[0].round(3))
print(f"\n对应 loss ≈ -ln(1/10) = {-np.log(1/10):.4f}  <- 训练开始时的 loss 就该是 2.30 左右")
print("如果你第一次训练时初始 loss 是 5.5 或 230，那基本是数据处理或网络结构写错了。")
