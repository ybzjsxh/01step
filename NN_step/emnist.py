"""EMNIST 数据加载：从官方 gzip.zip 里取出 balanced 子集 -> 解析 idx -> 缓存 npz。

EMNIST 和 MNIST 用的是完全一样的 idx 格式（见 mnist.py），所以 _read_images /
_read_labels 可以直接复用。但有两处坑必须处理，否则模型会学出"鬼画符"：

1. **图像是转置的**：EMNIST 的 28x28 像素按 (列, 行) 顺序存，直接 reshape 出来
   整个字符是躺倒的（像被旋转过），必须 transpose(0, 2, 1) 把它扶正。
2. **标签从 1 开始**：idx 里是 1..47，而我们的网络输出 0..46，要减 1。

balanced 子集：47 类（0-9 + A-Z 去掉 C/F/I/L/O/P/S/U/X 这些易混合并后的字母），
训练 112,800 / 测试 18,800，每类样本数基本相等。
"""

import gzip
import os
import struct
import urllib.request
import zipfile

import numpy as np

URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
ZIP_MD5 = "58c8d27c78d21e728a6bc7b3cc06412e"

SPLIT = "balanced"
FILES = {
    "train_images": f"emnist-{SPLIT}-train-images-idx3-ubyte.gz",
    "train_labels": f"emnist-{SPLIT}-train-labels-idx1-ubyte.gz",
    "test_images": f"emnist-{SPLIT}-test-images-idx3-ubyte.gz",
    "test_labels": f"emnist-{SPLIT}-test-labels-idx1-ubyte.gz",
    "mapping": f"emnist-{SPLIT}-mapping.txt",
}


def _ensure_files(data_dir, verbose=True):
    """四个 idx.gz + mapping.txt 都到位；缺了就从 gzip.zip 里解出来。

    官方只提供整包（561 MB），但我们只需要 balanced 那 40 多 MB，
    所以下载完只解压需要的成员，其余留在压缩包里。
    """
    os.makedirs(data_dir, exist_ok=True)
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(data_dir, f))]
    if not missing:
        return

    zip_path = os.path.join(data_dir, "gzip.zip")
    if not os.path.exists(zip_path):
        print(f"下载 {URL} （561 MB，只需一次）...")
        urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for fname in missing:
            # zip 内路径形如 gzip/emnist-balanced-train-images-idx3-ubyte.gz
            member = next(n for n in names if n.endswith("/" + fname) or n == fname)
            if verbose:
                print(f"解压 {fname} ...")
            zf.extract(member, data_dir)
            # 拉平到 data/emnist/ 下，避免每次都要拼 gzip/ 前缀
            src = os.path.join(data_dir, member)
            if os.path.dirname(src) != data_dir:
                os.replace(src, os.path.join(data_dir, fname))
        # 清掉空目录
        sub = os.path.join(data_dir, "gzip")
        if os.path.isdir(sub) and not os.listdir(sub):
            os.rmdir(sub)


def _read_images(path):
    """和 mnist.py 同款解析，只多做一步转置。"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"不是图片文件：magic={magic}"
        buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    assert data.size == n * rows * cols, "像素数量对不上"
    # (N, 784) -> (N, 28, 28) -> 转置扶正 -> 再拉平
    imgs = data.reshape(n, rows, cols).transpose(0, 2, 1)
    return imgs.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"不是标签文件：magic={magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    assert labels.size == n
    return labels.astype(np.int64)


def load_mapping(path="data/emnist"):
    """返回 47 个字符的映射表：class_id -> 字符。"""
    map_path = os.path.join(path, FILES["mapping"])
    chars = []
    with open(map_path) as f:
        for line in f:
            line = line.strip()
            if line:
                _, code = line.split()
                chars.append(chr(int(code)))
    return chars


def load_emnist(data_dir="data/emnist", cache=True, verbose=True, seed=None, limit=None):
    """返回 (X_train, y_train, X_test, y_test)。

    X: float32 (N, 784) ∈ [0,1]，已转置扶正
    y: int64   (N,)     ∈ [0, 46]
    """
    cache_path = os.path.join(data_dir, f"emnist_{SPLIT}.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    else:
        _ensure_files(data_dir, verbose)
        Xtr = _read_images(os.path.join(data_dir, FILES["train_images"]))
        Xte = _read_images(os.path.join(data_dir, FILES["test_images"]))
        ytr = _read_labels(os.path.join(data_dir, FILES["train_labels"]))
        yte = _read_labels(os.path.join(data_dir, FILES["test_labels"]))
        # balanced 已经是 0-based；letters / byclass / bymerge 才是 1-based，
        # 这里按最小值判断，免得换 split 时悄悄错一位、准确率永远上不去。
        if ytr.min() >= 1:
            ytr, yte = ytr - 1, yte - 1
        if cache:
            np.savez_compressed(cache_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    if limit is not None and 0 < limit < len(Xtr):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(Xtr), limit, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]

    if verbose:
        print(f"训练集: X{Xtr.shape} y{ytr.shape}  像素范围[{Xtr.min():.2f},{Xtr.max():.2f}]")
        print(f"测试集: X{Xte.shape} y{yte.shape}")
        print(f"类别数: {ytr.max() + 1}   每类样本数: {np.bincount(ytr).min()}~{np.bincount(ytr).max()}")
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_emnist()
    chars = load_mapping()
    # 用字符画检查转置是否修好了：字母应该是正的，不是躺倒的
    chars_art = " .:-=+*#%@"
    for i in (0, 1, 2):
        img = Xtr[i].reshape(28, 28)
        print(f"\n第 {i} 张图，标签={chars[ytr[i]]} (id={ytr[i]})")
        for row in img[::2]:
            print("".join(chars_art[min(int(v * 9), 9)] for v in row))
