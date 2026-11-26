# 01step

从零手写三个方向的学习项目，用 monorepo 管理（uv workspace，Python 3.12）。

| 子项目 | 内容 | 技术栈 |
|---|---|---|
| [NN_step](NN_step/) | 纯 numpy 手写 MLP / CNN，step1~step8 跑 MNIST / EMNIST / CIFAR-10 | numpy + matplotlib |
| [RAG_step](RAG_step/) | 从零实现 RAG 流水线，step1~step7：加载 → 切块 → 嵌入 → 索引 → 提示 → 回答 → 评估 | 标准库 + OpenAI 兼容 API |
| [LoRA_step](LoRA_step/) | 从零实现 LoRA，微调 ResNet18 / Swin-Tiny，含 Grad-CAM 可视化 | torch 2.2.2 |

每个子项目保留自己的 README，说明原理与步骤；根目录只负责环境与结构。

## 环境管理

整个仓库是一个 [uv](https://docs.astral.sh/uv/) workspace：根目录一个 `.venv`（Python 3.12）由三个子项目共享，锁文件统一在根目录 `uv.lock`。

```bash
# 在根目录：安装全部子项目的依赖
uv sync --all-packages

# 运行脚本：先 cd 进子目录（脚本里的 data/、models/、outputs/ 等路径相对于子目录）
cd NN_step   && uv run python step1_forward.py
cd RAG_step  && uv run python steps/step1_load.py
cd LoRA_step && uv run python train.py

# 只跑 LoRA 的单元测试
cd LoRA_step && uv run python -m unittest discover -s tests
```

在子目录里直接 `uv run` 也可以——uv 会自动识别它所属的 workspace 并同步该成员的依赖。

NN_step 的 CIFAR-10 第一次从 parquet 转换需要 pyarrow（`NN_step/data/` 里已有 `.npz` 缓存后不再需要）：

```bash
uv sync --all-packages --package nn-step --extra parquet
```

### 依赖变更

改对应子目录的 `pyproject.toml`，然后回根目录 `uv sync --all-packages`。

注意 LoRA_step 锁定 `torch==2.2.2`、`numpy<2`，所以共享环境里 numpy 固定在 1.26.x；NN_step 代码同时兼容 numpy 1.x / 2.x，不受影响。

## 约定

- 数据集、权重、索引、训练日志不进 git（见 `.gitignore`），图和可视化结果保留
- 每个子项目的 `.gitignore` 仍然生效，按需各自维护
