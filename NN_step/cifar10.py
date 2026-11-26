"""CIFAR-10 数据加载：官方 python 版 tar.gz -> 解包 pickle -> 缓存 npz。

和 MNIST / EMNIST 相比，这里有三个新问题，每一个都会让人误以为"模型不行"：

1. **不再是 (N, 784) 的灰度图，而是 (N, 32, 32, 3) 的彩色图**。
   像素在 pickle 里是按 (C, H, W) 存的（3072 = 3*32*32），必须 reshape 成
   (3, 32, 32) 再 transpose 到 (32, 32, 3)，否则你会得到一张"红绿蓝三块拼图"。
2. **32x32 太小，且物体位置/朝向每次都不一样**。原图直接训练，模型会死记
   "飞机总在正中间朝左"，于是测试集上掉一大截。所以训练时必须做随机裁剪 +
   水平翻转（见 random_crop_flip），这是 CIFAR 上最便宜也最有效的增强。
3. **三个通道的数值分布不一样，且相关性强**。不按通道做标准化，第一层卷积
   要花好几个 epoch 才把输入拉到同一量级；标准化之后 loss 从第一个 batch 就
   正常下降。标准化参数**只能用训练集统计**，否则测试集信息泄漏。

数据规模：训练 50,000 / 测试 10,000，10 类，每类 5,000 / 1,000，完全均衡。
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_NAME = "cifar-10-python.tar.gz"
DIR_NAME = "cifar-10-batches-py"
CACHE_NAME = "cifar10.npz"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def _ensure_files(data_dir, verbose=True):
    """把 tar.gz 下下来，并且只解出需要的那 7 个文件。"""
    os.makedirs(data_dir, exist_ok=True)
    need = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    missing = [f for f in need if
               not os.path.exists(os.path.join(data_dir, DIR_NAME, f))]
    if not missing:
        return

    tar_path = os.path.join(data_dir, TAR_NAME)
    if not os.path.exists(tar_path):
        print(f"下载 {URL} （163 MB，只需一次）...")
        urllib.request.urlretrieve(URL, tar_path)

    with tarfile.open(tar_path, "r:gz") as tf:
        for fname in missing:
            member = f"{DIR_NAME}/{fname}"
            if verbose:
                print(f"解压 {member} ...")
            tf.extract(member, data_dir)


def _read_batch(path):
    """一个 batch 文件 = 10000 张图 + 10000 个标签。"""
    with open(path, "rb") as f:
        # python3 的 pickle 里 key 是 bytes，必须指定 encoding，否则是 b'data'
        d = pickle.load(f, encoding="bytes")
    # (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return X, y


def _from_parquet(data_dir, verbose=True):
    """备选入口：从 HuggingFace 镜像的 parquet 转出同样格式的数据。

    官方 tar.gz 在海外，国内直连常常只有 10~20 KB/s（163 MB 要下几个钟头）。
    uoft-cs/cifar10 的 parquet 在镜像站能跑到几 MB/s，数据一模一样（同一批
    PNG 图），只是封装不同。这里把 PNG 解码成数组，落成和官方版相同的 npz 缓存，
    后面 load_cifar10 就完全不用关心数据是从哪来的。

    需要 pyarrow + Pillow，只在"第一次转换"时用到，转完就不需要了。
    """
    from io import BytesIO
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "这份 parquet 需要用 pyarrow 解：.venv/bin/python -m pip install pyarrow\n"
            "（只在这条镜像路径上需要；官方 pickle 版和 npz 缓存都不需要它）")
    from PIL import Image

    parts = []
    for split in ("train", "test"):
        path = os.path.join(data_dir, f"{split}.parquet")
        if verbose:
            print(f"从 {path} 解码 ...")
        table = pq.read_table(path, columns=["img", "label"])
        imgs, labels = [], []
        for row in table.to_pylist():
            im = Image.open(BytesIO(row["img"]["bytes"]))
            imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            labels.append(row["label"])
        parts.append((np.stack(imgs), np.asarray(labels, dtype=np.int64)))
    return parts[0], parts[1]


def load_cifar10(data_dir="data/cifar10", cache=True, verbose=True,
                 seed=None, limit=None):
    """返回 (X_train, y_train, X_test, y_test)。

    X: float32 (N, 32, 32, 3) ∈ [0, 1]，通道顺序 RGB
    y: int64   (N,)           ∈ [0, 9]

    注意：这里**不**做标准化。标准化参数要由训练集现算（见 normalize），
    把这一步留在训练脚本里，才能保证测试集用的是训练集的统计量。
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, CACHE_NAME)
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    else:
        base = os.path.join(data_dir, DIR_NAME)
        if os.path.exists(os.path.join(base, "data_batch_1")):
            parts = [_read_batch(os.path.join(base, f"data_batch_{i}"))
                     for i in range(1, 6)]
            Xtr = np.concatenate([p[0] for p in parts])
            ytr = np.concatenate([p[1] for p in parts])
            Xte, yte = _read_batch(os.path.join(base, "test_batch"))
        elif os.path.exists(os.path.join(data_dir, "train.parquet")):
            if verbose:
                print("没找到官方 pickle，改用 data/cifar10/*.parquet（HF 镜像）")
            (Xtr, ytr), (Xte, yte) = _from_parquet(data_dir, verbose)
        else:
            _ensure_files(data_dir, verbose)         # 会去下载官方 tar.gz
            base = os.path.join(data_dir, DIR_NAME)
            parts = [_read_batch(os.path.join(base, f"data_batch_{i}"))
                     for i in range(1, 6)]
            Xtr = np.concatenate([p[0] for p in parts])
            ytr = np.concatenate([p[1] for p in parts])
            Xte, yte = _read_batch(os.path.join(base, "test_batch"))
        if cache:
            # 存 uint8：50k*3072*1B ≈ 147 MB，float32 要 4 倍，没必要
            np.savez(cache_path, Xtr=Xtr, ytr=ytr.astype(np.int16),
                     Xte=Xte, yte=yte.astype(np.int16))

    Xtr = Xtr.astype(np.float32) / 255.0
    Xte = Xte.astype(np.float32) / 255.0

    if limit is not None and 0 < limit < len(Xtr):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(Xtr), limit, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]

    if verbose:
        print(f"训练集: X{Xtr.shape} y{ytr.shape}  像素范围[{Xtr.min():.2f},"
              f"{Xtr.max():.2f}]")
        print(f"测试集: X{Xte.shape} y{yte.shape}")
        print(f"类别数: {len(CLASSES)}   每类样本数: "
              f"{np.bincount(ytr).min()}~{np.bincount(ytr).max()}")
    return Xtr, ytr, Xte, yte


def normalize(Xtr, Xte):
    """按通道标准化：x = (x - mean) / std，统计量只从训练集算。

    **必须用 dtype=np.float64**。50000*32*32*3 = 1.5 亿个数，用 float32 累加会
    把有效位数耗在"加 0.5 到 7000 万"这种操作上，numpy 直接给出 0.32768 这种
    三通道一模一样的假均值（真实值是 0.4914 / 0.4822 / 0.4465）。症状是训练照样
    能跑，但输入没有被真正居中，前几轮 loss 下降变慢。1.5 亿个数用 float64 也就
    1.2 GB 的临时内存，求完均值立刻释放，不值得省。
    """
    mean = Xtr.mean(axis=(0, 1, 2), dtype=np.float64, keepdims=True)   # (1,1,1,3)
    std = Xtr.std(axis=(0, 1, 2), dtype=np.float64, keepdims=True) + 1e-6
    print(f"通道均值 {np.round(mean.ravel(), 4)}  "
          f"标准差 {np.round(std.ravel(), 4)}")
    return (Xtr - mean) / std, (Xte - mean) / std, mean, std


def random_crop_flip(xb, rng, pad=4):
    """随机裁剪 + 水平翻转：CIFAR 上最小可用的数据增强。

    pad=4 时先在四周补 4 像素，再从 40x40 里随机裁 32x32 —— 相当于让物体
    在画面里随机平移 ±4 像素；翻转则让"朝左的飞机"和"朝右的飞机"变成同一个样本。
    这两件事都不改变类别，却能把 50k 张图变成几乎无穷多的变体。
    """
    n, h, w, c = xb.shape
    padded = np.pad(xb, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(xb)
    for i in range(n):
        dy, dx = rng.integers(0, 2 * pad + 1, size=2)
        out[i] = padded[i, dy:dy + h, dx:dx + w]
    flip = rng.random(n) < 0.5
    out[flip] = out[flip][:, :, ::-1, :]                    # 左右镜像
    return np.ascontiguousarray(out)


def to_chw(x):
    """(N,32,32,3) -> (N,3,32,32)，给只想用 MLP 的地方展平前用。"""
    return x.transpose(0, 3, 1, 2)


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_cifar10()
    # 用彩色块自检：每类取一张，拼成 2 行 5 列的 RGB 缩略图，人眼一看就知道
    # 通道顺序对不对（飞机该是蓝天、青蛙该是绿的）
    chars = " .:-=+*#%@"
    print("\n第一张训练图的三个通道（左上角 8x8），检查 R/G/B 是否对得上：")
    for ch, name in enumerate("RGB"):
        print(f"{name} 通道:")
        for row in Xtr[0, :8, :8, ch]:
            print("  " + "".join(chars[min(int(v * 9), 9)] for v in row))
