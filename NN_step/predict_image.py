"""把训好的模型用在「一张自己的图」上 —— 验证整个流程真的闭环。

运行：python predict_image.py 图片路径 [--invert] [--model 权重路径]
    图片会被转成灰度、缩放到 28x28、归一化到 0~1，再送进网络。
    MNIST 是「黑底白字」，如果你的图是白底黑字，加 --invert。
    默认用 MLP 权重；加 --model models/cnn_mnist.npz 就用卷积网络（脚本会自动识别）。

示例：python predict_image.py data/sample_digit.png
      python predict_image.py data/sample_digit.png --model models/cnn_mnist.npz
"""

import argparse
import os

import numpy as np

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")   # 只跳过系统字体扫描
import matplotlib.image as mpimg

from nn import MLP


def load_image(path, invert=False):
    img = mpimg.imread(path)
    if img.ndim == 3:
        # RGB/RGBA -> 灰度
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    if img.max() > 1.0:                     # 0~255 归一化
        img = img / 255.0
    if invert:                              # 白底黑字 -> 黑底白字
        img = 1.0 - img
    # 最近邻缩放到 28x28（依赖很少，就手写一个简单的采样）
    h, w = img.shape
    idx_h = (np.arange(28) * h / 28).astype(int).clip(0, h - 1)
    idx_w = (np.arange(28) * w / 28).astype(int).clip(0, w - 1)
    small = img[idx_h][:, idx_w]
    return small.reshape(1, 784).astype(np.float64), small


def load_model(path):
    """自动识别是 MLP 还是 CNN：CNN 存了 config，MLP 只存了 sizes。"""
    d = np.load(path)
    if "config" in d.files:
        from conv import CNN
        return CNN.load(path), "cnn"
    return MLP.load(path), "mlp"


def predict(model, x, kind):
    if kind == "cnn":
        x = x.reshape(1, 28, 28, 1)              # 卷积需要空间结构
    logits = model.forward(x)[0]
    logits = logits - logits.max()
    p = np.exp(logits) / np.exp(logits).sum()
    return int(p.argmax()), p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--invert", action="store_true", help="白底黑字的图请加上这个参数")
    ap.add_argument("--model", default="models/mlp_mnist.npz")
    args = ap.parse_args()

    model, kind = load_model(args.model)
    x, small = load_image(args.image, args.invert)
    pred, probs = predict(model, x, kind)

    chars = " .:-=+*#%@"
    print("送入网络的 28x28 图像：")
    for row in small:
        print("".join(chars[min(int(v * 9.0), 9)] for v in row))

    print(f"\n预测：{pred}   置信度 {probs[pred]*100:.1f}%")
    top3 = np.argsort(probs)[::-1][:3]
    print("Top-3：" + "  ".join(f"{int(c)}({probs[c]*100:.1f}%)" for c in top3))
