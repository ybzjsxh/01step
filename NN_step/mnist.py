"""MNIST 数据加载：下载 -> 解析 idx 二进制 -> 缓存 npz。

MNIST 的原始格式（idx）很朴素，自己解析一遍比调库更能看清"数据长什么样"：
- 图片文件：magic(2051) + 图片数 + 行 + 列 + 像素(uint8, 行优先)
- 标签文件：magic(2049) + 标签数 + 标签(uint8)
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# 官方源（Google 托管的 MNIST 镜像），备选镜像见 FALLBACK_BASE
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FALLBACK_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        last_err = None
        for base in (BASE_URL, FALLBACK_BASE_URL):
            try:
                print(f"下载 {base}{fname} ...")
                urllib.request.urlretrieve(base + fname, path)
                last_err = None
                break
            except Exception as e:  # 网络问题就换镜像
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"下载失败：{fname}\n{last_err}")


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"不是图片文件：magic={magic}"
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    assert data.size == n * rows * cols, "像素数量对不上"
    # 展平成 (N, 784)，并归一化到 [0,1]（这步对训练收敛非常关键）
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"不是标签文件：magic={magic}"
        buf = f.read()
    labels = np.frombuffer(buf, dtype=np.uint8)
    assert labels.size == n
    return labels.astype(np.int64)


def load_mnist(data_dir="data/mnist", cache=True, verbose=True):
    """返回 (X_train, y_train, X_test, y_test)。

    X: float32, 形状 (N, 784), 取值 [0,1]
    y: int64,  形状 (N,),   取值 0..9
    """
    cache_path = os.path.join(data_dir, "mnist.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    else:
        _download(data_dir)
        Xtr = _read_images(os.path.join(data_dir, FILES["train_images"]))
        ytr = _read_labels(os.path.join(data_dir, FILES["train_labels"]))
        Xte = _read_images(os.path.join(data_dir, FILES["test_images"]))
        yte = _read_labels(os.path.join(data_dir, FILES["test_labels"]))
        if cache:
            np.savez_compressed(
                cache_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte
            )

    if verbose:
        print(f"训练集: X{Xtr.shape} y{ytr.shape}  像素范围[{Xtr.min():.2f},{Xtr.max():.2f}]")
        print(f"测试集: X{Xte.shape} y{yte.shape}")
        print(f"类别分布(训练): {np.bincount(ytr).tolist()}")
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist()
    # 用字符画看一眼第 0 张图，确认数据没读错
    img = Xtr[0].reshape(28, 28)
    chars = " .:-=+*#%@"
    print(f"\n第 0 张图，标签={ytr[0]}")
    for row in img[::2]:
        print("".join(chars[min(int(v * 9), 9)] for v in row[::1]))
