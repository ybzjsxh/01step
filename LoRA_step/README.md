# 从零实现 LoRA：ResNet-18 与 Swin Transformer 本地实战

这套教程不调用 Hugging Face 的 PEFT 库，而是直接用 PyTorch 实现 LoRA。具体来说，**PEFT 是参数高效微调方法的统称，LoRA 是其中一种具体算法，`peft` 是常用的现成工具箱**；第 2.1 节会先把这三个容易混在一起的名字讲清楚。完成后，你会理解并亲手跑通：

1. LoRA 为什么能减少可训练参数；
2. 如何给 `nn.Linear` 写 LoRA；
3. 如何把 LoRA 扩展到 ResNet 的 `nn.Conv2d`；
4. 如何选择注入位置、冻结主干、保留新分类头；
5. 如何训练、评估、保存、加载和合并 LoRA；
6. 如何在本地图片上执行推理；
7. 如何用 Grad-CAM 看懂模型关注区域，并用奇异值谱分析特征图的有效秩。

本项目已经在 **Apple M1 Pro / 16GB / PyTorch MPS** 上真实运行。使用 2,000 张 CIFAR-10 训练图片、500 张测试图片、2 个 epoch：

| 模型与方法 | 训练前准确率 | 最终测试准确率 | 可训练参数 | 可训练参数占比 | 适配器大小 |
|---|---:|---:|---:|---:|---:|
| ResNet-18，只训练分类头 | 9.40% | 32.20% | 5,130 | 0.0459% | 不适用 |
| ResNet-18，卷积 LoRA | 9.40% | **46.60%** | 148,822 | 1.3141% | 约 596KB |
| Swin-T，注意力 LoRA | 14.60% | **81.60%** | 113,674 | 0.4114% | 约 464KB |

> 这些结果用于验证“代码正确且 LoRA 能学习”，不是充分训练后的精度基准。测试集只有 500 张，准确率的统计波动也会比完整测试集大。

---

## 一、先用 5 分钟跑出结果

### 1. 进入项目并安装 uv

本机已经安装 uv 时可跳过安装。macOS 可用 Homebrew：

```bash
brew install uv
cd /Users/f/Desktop/scholar/01step/LoRA_step
```

### 2. 创建环境

本目录是 `01step` monorepo（uv workspace，Python 3.12）的一个成员，虚拟环境在仓库根目录 `01step/.venv` 统一维护：

```bash
cd /Users/f/Desktop/scholar/01step   # 仓库根目录
uv sync --all-packages              # 首次：一次装齐三个子项目的依赖
cd LoRA_step
```

依赖声明在本目录 `pyproject.toml`，统一锁定在根目录 `01step/uv.lock`，核心只有：

- Python 3.12；
- PyTorch 2.2.2；
- torchvision 0.17.2；
- NumPy 1.x；
- Matplotlib（只用于 Grad-CAM、特征秩诊断与曲线绘图）。

本文命令显式使用 `python3.12`，避免某些终端把未带版本号的 `python` 指向系统解释器。

### 3. 先验证实现是否正确

```bash
uv run --no-sync python3.12 -m unittest discover -s tests -v
```

应当看到 16 项测试全部 `OK`。测试覆盖：

- LoRA 初始化后不改变原输出；
- 原权重被冻结，A/B 能获得梯度；
- Linear 和 Conv2d 合并前后输出一致；
- ResNet 的 20 个卷积注入点；
- Swin 的 24 个注意力注入点和真实前向传播；
- 适配器保存、加载；
- 小数据集类别均衡采样；
- Grad-CAM 的标准 `[B,C,H,W]` 和 Swin `[B,H,W,C]` 特征布局；
- 特征矩阵的原始秩、有效秩、稳定秩与全零退化情形。

### 4. 训练 ResNet-18

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 \
  --epochs 2 \
  --train-limit 2000 \
  --test-limit 500 \
  --batch-size 32
```

首次运行会下载 CIFAR-10 和 ImageNet 预训练权重。输出在：

```text
outputs/resnet18_lora/
├── adapter.pt      # LoRA A/B + 新分类头 + 配置
├── metrics.json    # 参数量、准确率、合并误差等
└── train.log       # 完整训练日志
```

本机关键日志：

```text
参数统计：total=11325334, trainable=148822, lora=143692
训练前 test accuracy 9.40%
Epoch 1 test accuracy 38.60%
Epoch 2 test accuracy 46.60%
LoRA 合并前后最大 logits 误差：0.00000966
```

### 5. 训练 Swin-T

```bash
uv run --no-sync python3.12 train.py \
  --model swin_tiny \
  --epochs 2 \
  --train-limit 2000 \
  --test-limit 500 \
  --batch-size 32
```

本机关键日志：

```text
参数统计：total=27633028, trainable=113674, lora=105984
训练前 test accuracy 14.60%
Epoch 1 test accuracy 76.80%
Epoch 2 test accuracy 81.60%
LoRA 合并前后最大 logits 误差：0.00000000
```

### 6. 加载适配器并预测

预测 CIFAR-10 的第 1 张测试图片：

```bash
uv run --no-sync python3.12 predict.py \
  --checkpoint outputs/swin_tiny_lora/adapter.pt \
  --index 1
```

本机结果：真实类别为 `ship`，预测类别也为 `ship`，概率约为 `0.9904`。

预测自己的图片：

```bash
uv run --no-sync python3.12 predict.py \
  --checkpoint outputs/swin_tiny_lora/adapter.pt \
  --image /你的图片路径/example.jpg
```

注意：当前分类头只会输出 CIFAR-10 的 10 个类别，不是通用图片识别器。

### 7. 生成 Grad-CAM、特征秩与训练曲线

```bash
uv run --no-sync python3.12 visualize.py
```

一次生成两张模型 Grad-CAM 网格、一张特征有效秩图、一张训练曲线和一份 JSON 汇总：

```text
outputs/visualizations/
├── resnet18_gradcam.png
├── swin_tiny_gradcam.png
├── feature_rank.png
├── training_curves.png
└── gradcam_summary.json
```

默认会自动寻找 2 个正确、2 个错误预测，并尽量让真实类别不重复。也可以指定样本：

```bash
uv run --no-sync python3.12 visualize.py \
  --indices 0,1,3,4
```

---

## 二、LoRA 到底是什么

### 2.1 LoRA 和 PEFT 是什么关系

先把三个容易混在一起的名字分开：

| 名称 | 它到底是什么 | 和另外两个的关系 |
|---|---|---|
| **LoRA** | Low-Rank Adaptation，一种具体的低秩适配算法 | 属于 PEFT 方法家族，可以完全脱离 `peft` 库独立实现 |
| **PEFT** | Parameter-Efficient Fine-Tuning，参数高效微调的**方法类别统称** | 除了 LoRA，还包括 Prompt Tuning、Prefix Tuning、IA3 等；不等于 LoRA |
| **`peft`** | Hugging Face 提供的 Python 工具箱/库 | 把 LoRA、AdaLoRA 等多种 PEFT 方法做成现成接口；不等于算法本身 |

可以这样记：

```text
PEFT（方法大类）
├── LoRA          ← 本项目从零实现的就是它
├── Prompt Tuning
├── Prefix Tuning
├── IA3
└── ...
        ↑
   Hugging Face `peft` 是装这些方法的工具箱
```

因此，“用 LoRA”和“用 `peft` 库”不是同义词：

- **LoRA 可以不用 `peft` 库**。它只需要 PyTorch 的 `nn.Linear`、`nn.Conv2d` 和反向传播；
- **`peft` 库也不只做 LoRA**。它只是让多种 PEFT 方法更容易接入 Hugging Face 模型；
- **`peft` 不是一个模型，也不是一种新的微调算法**，它主要是模块替换、配置、保存与加载的上层封装。

本项目选择从零实现，不是因为 PEFT 不能用，而是因为它会把最关键的几步藏起来：哪些层被替换、原权重如何冻结、A/B 如何初始化、Swin 的 `qkv` 为什么没有走包装层 `forward`、合并前后的数值误差如何检查。自己写一遍后，再回头用 PEFT 会清楚它替你做了什么。

两套实现的概念对应关系：

| 本项目 `lora_core.py` | Hugging Face `peft` |
|---|---|
| `LoRAConfig` | `LoraConfig` |
| `inject_lora()` 按 `all_conv` / `attention` 规则替换层 | `get_peft_model()` + `target_modules` 指定模块名 |
| `save_adapter()` / `load_adapter()` | `save_pretrained()` / `from_pretrained()` |
| `merge_lora()` | `merge_and_unload()` |
| 自定义 `adapter.pt` | `adapter_config.json` + `adapter_model.safetensors`（或 `.bin`） |

最后一点很重要：**它们的数学原理相同，但文件格式不保证互通**。本项目的 `adapter.pt` 是自定义字典，不是 PEFT 的 adapter 目录结构，不能直接交给 `PeftModel.from_pretrained()` 加载。如果要迁移到 PEFT，需要转换权重键名并重建配置，或者直接用 PEFT 重新训练。

### 2.2 从普通微调开始

设某一层原来的权重为：

```text
W₀ ∈ R^(d_out × d_in)
```

普通全参数微调会直接更新 `W₀`：

```text
W = W₀ + ΔW
h = Wx = W₀x + ΔWx
```

如果 `d_in = d_out = 768`，单个权重矩阵就有：

```text
768 × 768 = 589,824 个参数
```

全参数微调需要为它们保存梯度，优化器通常还要保存更多状态。

### 2.3 LoRA 的核心假设

LoRA（Low-Rank Adaptation，低秩适配）认为：微调所需的权重变化 `ΔW` 可以用一个低秩矩阵近似。

于是不直接学习完整的 `ΔW`，而是学习两个小矩阵：

```text
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
ΔW = (α / r) · B · A
```

其中：

- `r` 是秩，通常远小于输入、输出维度；
- `α` 是缩放超参数；
- `α / r` 控制低秩分支对原输出的影响强度。

训练时冻结 `W₀`，只更新 A 和 B：

```text
h = W₀x + (α / r) · BAx
```

### 2.4 参数量为什么下降

原矩阵参数量：

```text
d_out × d_in
```

LoRA 参数量：

```text
r × d_in + d_out × r = r(d_in + d_out)
```

以 `d_in = d_out = 768，r = 4` 为例：

```text
原权重：589,824
LoRA：4 × (768 + 768) = 6,144
```

LoRA 参数量约为这个原矩阵的 **1.04%**。

需要注意：原模型权重仍然要加载到内存中。LoRA 主要减少的是**可训练参数、梯度和优化器状态**，并不意味着推理时原模型会消失，也不保证所有场景都显著减少显存。

### 2.5 为什么 B 初始化为 0

本项目采用：

```python
nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
nn.init.zeros_(self.lora_B.weight)
```

训练开始时：

```text
ΔW = (α / r) · B · A = 0
```

因此注入 LoRA 不会改变预训练模型的初始输出。随着训练进行，B 先获得梯度并脱离 0，随后 A、B 共同学习。

### 2.6 r、alpha、dropout 分别做什么

| 超参数 | 作用 | 常用起点 |
|---|---|---|
| `r` | 低秩分支容量；越大，参数越多、表达变化的能力越强 | 4 或 8 |
| `alpha` | 控制低秩增量幅度，实际缩放为 `alpha/r` | 常设为 `2*r` |
| `dropout` | 对低秩分支输入做正则化，缓解过拟合 | 0～0.1 |
| 注入位置 | 决定允许模型哪些部分适应新任务 | 注意力 qkv/proj，或卷积层 |

本项目默认 `r=4，alpha=8`，所以缩放系数为 2。

---

## 三、逐行理解 Linear LoRA

核心代码在 `lora_core.py` 的 `LoRALinear`。

### 3.1 保存原层，建立低秩分支

```python
self.base = base
self.r = r
self.scaling = alpha / r
self.lora_A = nn.Linear(base.in_features, r, bias=False)
self.lora_B = nn.Linear(r, base.out_features, bias=False)
```

注意 PyTorch 的 `nn.Linear` 权重形状是 `[out_features, in_features]`，所以：

- `lora_A.weight` 是 `[r, d_in]`；
- `lora_B.weight` 是 `[d_out, r]`；
- `B @ A` 恰好是 `[d_out, d_in]`。

### 3.2 前向传播

```python
def forward(self, x):
    original = self.base(x)
    update = self.lora_B(self.lora_A(self.lora_dropout(x)))
    return original + self.scaling * update
```

对应公式：

```text
original = W₀x + bias
update   = (α/r) · BAx
output   = original + update
```

原层 `base` 的参数被冻结，优化器只接收 `requires_grad=True` 的 A/B 和分类头。

### 3.3 合并 LoRA

训练完成后，可以把低秩增量加回原权重：

```python
delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
self.base.weight.add_(delta)
```

合并后：

```text
W = W₀ + (α/r)BA
```

模型不再需要有额外的 A/B 分支，因此不会增加推理路径。`merge_lora(model)` 会递归处理整个模型。

### 3.4 torchvision Swin 的特殊兼容性

torchvision 的 Swin 注意力层不是调用：

```python
x = self.qkv(x)
```

而是直接读取权重：

```python
x = F.linear(x, self.qkv.weight, self.qkv.bias)
```

如果只实现包装层的 `forward`，Swin 会绕过它。因此 `LoRALinear` 还暴露：

```python
@property
def weight(self):
    delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
    return self.base.weight + delta

@property
def bias(self):
    return self.base.bias
```

这样，无论模型调用 `qkv(x)`，还是直接读取 `qkv.weight`，都能得到 LoRA 调整后的结果。

由于外部 `F.linear` 无法执行包装层里的输入 dropout，本教学版要求 **Swin 的 LoRA dropout 为 0**。默认命令已自动这样设置。

---

## 四、ResNet 的卷积层如何做 LoRA

### 4.1 把卷积核看成一个矩阵

普通卷积核形状为：

```text
W ∈ R^(c_out × c_in × kh × kw)
```

把它展开：

```text
W_flat ∈ R^(c_out × (c_in·kh·kw))
```

就可以沿用相同的低秩思想：

```text
ΔW_flat = (α/r) · B_flat · A_flat
```

### 4.2 用两个卷积实现低秩分支

`LoRAConv2d` 中：

- A：与原卷积相同的 kernel size、stride、padding，但只输出 r 个通道；
- B：`1×1` 卷积，把 r 个通道映射回 `c_out`。

```python
self.lora_A = nn.Conv2d(
    base.in_channels,
    r,
    kernel_size=base.kernel_size,
    stride=base.stride,
    padding=base.padding,
    dilation=base.dilation,
    bias=False,
)
self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
```

前向传播仍然是：

```python
original = self.base(x)
update = self.lora_B(self.lora_A(self.lora_dropout(x)))
return original + self.scaling * update
```

### 4.3 卷积合并

合并时，把 A/B 都展开成矩阵：

```python
flat_a = self.lora_A.weight.reshape(r, -1)
flat_b = self.lora_B.weight.reshape(c_out, r)
delta = self.scaling * (flat_b @ flat_a)
base.weight += delta.reshape_as(base.weight)
```

需要说明：原始 LoRA 论文最常直接作用于 Linear。卷积 LoRA 有多种设计，本项目采用的是**展开卷积核后的低秩分解**，便于教学且能与 Linear 共用同一套数学理解。本实现仅处理 `groups=1` 的普通卷积，ResNet-18 满足这个条件。

---

## 五、完整的 LoRA 注入流程

无论换成什么模型，都按下面的顺序操作。

### 第 1 步：加载预训练模型

```python
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

预训练权重是迁移学习效果的重要来源。`--no-pretrained` 只用于调试流程，不建议用于正常实验。

### 第 2 步：替换任务分类头

CIFAR-10 有 10 类，而 ImageNet 有 1,000 类：

```python
model.fc = nn.Linear(model.fc.in_features, 10)  # ResNet
# 或
model.head = nn.Linear(model.head.in_features, 10)  # torchvision Swin
```

新分类头是随机初始化的，所以必须让它参与训练。否则主干即使学得再好，最后仍由随机分类器决定输出。

### 第 3 步：打印模块名，选择注入点

不要靠猜模块名，先打印：

```python
from torch import nn
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        print(name, type(module).__name__)
```

本项目选择：

- ResNet-18：`target=all_conv`，命中 20 个卷积；
- Swin-T：`target=attention`，只命中名字以 `qkv` 或 `proj` 结尾的 Linear，共 24 个。

也可以传入逗号分隔的模块名片段：

```bash
# 只给 Swin 的 qkv 注入，预期命中 12 个模块
uv run --no-sync python3.12 train.py \
  --model swin_tiny --target attn.qkv --epochs 2
```

### 第 4 步：注入 LoRA

```python
from lora_core import LoRAConfig, inject_lora

config = LoRAConfig(r=4, alpha=8, dropout=0.05, target="all_conv")
targets = inject_lora(model, config, head_names=("fc",))
```

`inject_lora` 会：

1. 遍历模块树；
2. 找到匹配层；
3. 用 `LoRALinear` 或 `LoRAConv2d` 包装原层；
4. 冻结整个模型；
5. 重新放开 A/B；
6. 重新放开新分类头。

### 第 5 步：检查真正可训练的参数

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name, parameter.shape)
```

再统计：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(trainable, total, 100 * trainable / total)
```

这是排查“LoRA 看似加了但实际没训练”的最直接方法。

### 第 6 步：优化器只接收可训练参数

```python
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3)
```

如果直接把 `model.parameters()` 全传入，虽然冻结参数不会有梯度，但会让调试和优化器状态管理更混乱。

### 第 7 步：正常训练

LoRA 不改变交叉熵和反向传播流程：

```python
logits = model(images)
loss = criterion(logits, labels)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

验证时：

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

### 第 8 步：保存、加载、合并

保存：

```python
save_adapter(model, "adapter.pt", config, targets, head_names=("fc",))
```

加载前，先创建**相同的预训练主干、相同的注入配置和相同的分类头**：

```python
payload = load_adapter(model, "adapter.pt")
```

`adapter.pt` 不包含完整主干，因此不能脱离原模型独立使用。它包含：

- LoRA A/B；
- 新分类头；
- LoRA 配置；
- 注入模块名；
- 类别名等元数据。

合并：

```python
from lora_core import merge_lora
merge_lora(model)
```

训练脚本每次都会自动比较合并前后的 logits。若差异很大，说明合并公式或前向路径存在问题。

---

## 六、两个模型分别做了什么

### 6.1 ResNet-18

模型构建：

```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
```

LoRA 配置：

```python
LoRAConfig(r=4, alpha=8, dropout=0.05, target="all_conv")
```

注入位置包括：

```text
conv1
layer1.0.conv1
layer1.0.conv2
...
layer4.1.conv2
以及 3 个 downsample.0
```

本机参数统计：

| 参数类型 | 数量 |
|---|---:|
| 注入 LoRA 后的总参数 | 11,325,334 |
| LoRA A/B | 143,692 |
| 新分类头 | 5,130 |
| 总可训练参数 | 148,822 |

只训练分类头的对照命令：

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 --method head \
  --epochs 2 --train-limit 2000 --test-limit 500
```

在相同小样本设置下，对照实验为 32.20%，卷积 LoRA 为 46.60%。这说明仅调整分类头不足以充分适应当前任务，低秩更新主干提供了额外适应能力。

### 6.2 Swin-T

模型构建：

```python
from torchvision.models import swin_t, Swin_T_Weights
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 10)
```

LoRA 配置：

```python
LoRAConfig(r=4, alpha=8, dropout=0.0, target="attention")
```

命中位置示例：

```text
features.1.0.attn.qkv
features.1.0.attn.proj
...
features.7.1.attn.qkv
features.7.1.attn.proj
```

本机参数统计：

| 参数类型 | 数量 |
|---|---:|
| 注入 LoRA 后的总参数 | 27,633,028 |
| LoRA A/B | 105,984 |
| 新分类头 | 7,690 |
| 总可训练参数 | 113,674 |

为了让 Swin-T 能在本地快速运行，教程将输入缩放到 `64×64`，不是标准的 `224×224` 完整benchmark设置。它仍然使用同一套 ImageNet 预训练权重和移位窗口注意力。

---

## 七、如何读实验结果

完整结果在各自的 `metrics.json` 中：

- `parameter_stats`：参数量；
- `target_modules`：所有实际注入位置；
- `initial`：训练前测试指标；
- `history`：每个 epoch 的训练/测试指标；
- `merge_max_logit_difference`：合并前后最大输出误差；
- `training_seconds`：纯训练与每轮验证的耗时，不含下载。

本机结果：

| 模型 | Epoch 1 测试准确率 | Epoch 2 测试准确率 | 合并误差 | 训练耗时* |
|---|---:|---:|---:|---:|
| ResNet-18 LoRA | 38.60% | 46.60% | 9.66e-6 | 15.79 秒 |
| Swin-T LoRA | 76.80% | 81.60% | 0 | 28.77 秒 |

\* 耗时只适用于本机当前设置，其他设备、batch size 和输入尺寸会不同。

不要仅凭 2 个 epoch 得出“某个模型永远更好”的结论。要继续提升，应增加数据和训练轮数，并系统调整学习率、秩和注入位置。

---

## 八、可视化：Grad-CAM、特征秩与训练曲线

这一章围绕同一批训练结果回答三个不同问题：

- **Grad-CAM（8.1～8.5）**：模型预测时主要看图片的哪些区域；
- **训练曲线（8.6）**：准确率和损失如何随 epoch 变化；
- **特征有效秩（8.8）**：目标层保留了多少种独立变化方向。

这些图都由同一个 `visualize.py` 生成，不需要重新训练。8.7 说明如何修改目标层，8.8 则是在 Grad-CAM 之外补充一个表示层面的诊断。

### 8.1 Grad-CAM 在计算什么

只看预测标签，我们不知道模型依据图片的哪个区域做决定。Grad-CAM 通过梯度回答这个问题：

```text
A      = 目标层特征图
y_c    = 类别 c 的分数
α_c    = 对 A 的空间位置求平均后的梯度
L_c    = ReLU(Σ α_c · A)
```

直观理解：

- 对某通道梯度越大，说明该通道对目标类别越重要；
- 用通道权重对特征图加权求和，得到类别激活图；
- ReLU 去掉“支持其他类别”的负响应；
- 最后上采样到输入分辨率并归一化到 `[0,1]`。

本实现没有调用第三方 CAM 库，核心在 `gradcam.py`。它同时处理：

| 模型 | 目标层 | 原始特征 | 上采样后 |
|---|---|---|---|
| ResNet-18 | `layer2.1.conv2` | `[1,128,4,4]` | `32×32` |
| Swin-T | `features.5` | `[1,4,4,384]` | `64×64` |

ResNet 的输入只有 `32×32`，到 `layer4` 时特征图只剩 `1×1`，不适合看空间位置；`layer2` 的 `4×4` 已能在小图上保留大致区域。Swin 的 `features.5` 是 `4×4×384`，最后一阶段 `features.7` 只有 `2×2×768`，热力图会过于粗糙，所以选择中间阶段。

### 8.2 一键生成

```bash
uv run --no-sync python3.12 visualize.py
```

脚本会自动扫描 CIFAR-10 测试集前 1,000 张图片，为每个模型挑选：

- 2 个预测正确样本；
- 2 个预测错误样本；
- 尽量不同的真实类别。

所以它不仅展示“成功案例”，也直接暴露模型的错误关注区域。若想固定样本，可以用：

```bash
uv run --no-sync python3.12 visualize.py --indices 0,1,3,4
```

### 8.3 ResNet-18 Grad-CAM

![ResNet-18 Grad-CAM](outputs/visualizations/resnet18_gradcam.png)

本次自动选择的样本为：

- 正确：`automobile → automobile`，置信度 93.4%；
- 错误：`cat → automobile`，置信度 77.5%；
- 正确：`airplane → airplane`，置信度 33.8%；
- 错误：`ship → automobile`，置信度 93.3%。

热区说明：

- 预测对时，红色区域落在车体、飞机主体附近；
- 预测错时，模型也会给出一个看似“集中”的热区，但区域与真实类别物体不一致；
- 由于目标层只有 `4×4`，再放大到 `32×32` 后必然呈块状。这是特征分辨率决定的，不代表代码错误。

### 8.4 Swin-T Grad-CAM

![Swin-T Grad-CAM](outputs/visualizations/swin_tiny_gradcam.png)

本次样本为：

- 正确：`cat → cat`，置信度 97.5%；
- 错误：`bird → horse`，置信度 79.2%；
- 正确：`ship → ship`，置信度 99.0%；
- 错误：`deer → cat`，置信度 80.9%。

Swin 使用 `[B,H,W,C]` 通道最后布局，`gradcam.py` 会先转换为 `[B,C,H,W]` 再做通道加权。图中可以看到正确样本的热区集中在猫的脸部、船体结构附近；错误样本则偏向背景纹理或局部形状。

### 8.5 如何正确解读

1. **Grad-CAM 的目标是预测类别**。红色表示“支持当前预测”，不代表红色区域一定是该物体。
2. **热力图不是目标检测框**。它分辨率低，只提供粗略证据区域。
3. **梯度显著不等于因果解释**。它是有用的诊断工具，但不能证明模型内部唯一依赖该区域。
4. **错误样本同样有价值**。如果高置信度错误样本的热区落在背景，通常说明数据偏置、训练不足或模型学到了捷径。
5. **层选择会影响结果**。浅层保留更多空间细节，深层语义更强但分辨率更低。
6. **低分辨率输入需要谨慎**。`32×32` 与 `64×64` 的 CIFAR-10 实验中，`4×4` 热图只能看大致区域。

如果想看“真实类别对应的证据”，可以把 `GradCAM.__call__` 的 `target_class` 传为真实标签。本项目可视化默认使用模型自己的预测类别，因为这样更能解释“模型为什么做出这个判断”。

### 8.6 训练曲线与参数效率

![训练曲线](outputs/visualizations/training_curves.png)

四张子图分别是：

- **Test accuracy**：三种方法的测试准确率随 epoch 变化；
- **Test loss**：测试损失，Swin-T 下降最快；
- **Train vs test accuracy**：两个 LoRA 实验的训练/测试差距；
- **Trainable parameters**：绝对可训练参数与占模型总参数的比例。

ResNet 在这个小样本设置下训练准确率低于测试准确率，主要因为训练时启用了随机裁剪、翻转和 dropout，而测试时关闭了这些随机性；它还不能说明过拟合。

### 8.7 自定义目标层和输出

目标层配置在 `visualize.py` 的 `VISUALS` 字典中：

```python
VISUALS = {
    "resnet18": {
        "layer": "layer2.1.conv2",
        "channels_last": False,
    },
    "swin_tiny": {
        "layer": "features.5",
        "channels_last": True,
    },
}
```

修改 `layer` 为任意 `model.named_modules()` 中存在的层即可。Transformer 特征若是 `[B,N,C]`，Grad-CAM 会把完全平方数的 token 数还原成方形网格；若是 `[B,H,W,C]`，则设置 `channels_last=True`。这个配置不只影响热图，下一节的特征秩分析也会使用同一个目标层。

### 8.8 特征有效秩：看模型保留了几种独立方向

Grad-CAM 回答的是“模型看哪里”，但它回答不了另一个问题：**这个目标层把输入压缩成了多少种彼此独立的变化方向？** 这一节用同一个目标层的特征做谱分析，作为 Grad-CAM 的补充诊断，不是另一套训练或另一套模型。

先区分两个容易混淆的“秩”：

- **LoRA 的秩 \(r\)** 是人为设定的超参数，限制权重更新 \(\Delta W=BA\) 的秩，本项目固定为 4；
- **特征有效秩** 是从前向特征中计算出来的连续指标，不受 LoRA 的 \(r\) 直接限制，也不等于 LoRA 的秩。

所以这一节不是“证明 LoRA 的 \(r=4\)”，而是观察训练后的模型实际把 CIFAR-10 输入表示成了什么样的谱结构。

**第一步：把单个样本的目标层特征变成矩阵**

记目标层特征为：

\[
F \in \mathbb{R}^{C \times HW}
\]

本项目两个目标层的实际形状是：

| 模型 | 原始特征图 | 展平后的矩阵 \(F\) | 原始秩上限 \(\min(C,HW)\) |
|---|---|---:|---:|
| ResNet-18 | `[1,128,4,4]` | `128×16` | 16 |
| Swin-T | `[1,4,4,384]` | `384×16` | 16 |

这里故意使用 \(F\)，避免和 LoRA 里的低秩矩阵 \(A\)、\(B\) 混在一起。

**第二步：为什么原始 rank 几乎没有区分度**

矩阵的原始秩满足：

\[
\operatorname{rank}(F) \leq \min(C, HW)
\]

两个模型展平后都只有 16 个空间位置，所以原始秩最多是 16。本次 8 个可视化样本的原始秩全部是 16。它只能说明矩阵在数值上没有退化，不能说明特征包含多少有效方向，也会让 8 个面板看起来完全一样。因此图里保留 `raw rank` 作为参考，但不把它当作主要结论。

**第三步：用奇异值谱衡量有效维度**

对 \(F\) 做 SVD：

\[
F=U\Sigma V^\top
\]

得到奇异值 \(\sigma_1\geq\sigma_2\geq\cdots\)。把平方后的能量归一化：

\[
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2}
\]

再计算两个连续指标：

\[
\text{effective rank}=\exp\left(-\sum_i p_i\log p_i\right)
\]

\[
\text{stable rank}=\frac{\sum_i\sigma_i^2}{\sigma_1^2}
\]

它们的含义不同：

- **奇异值谱**：横轴是奇异值序号，纵轴是 \(\sigma_i/\sigma_1\)。曲线下降越快，能量越集中在少数方向；
- **effective rank**：基于奇异值能量分布的熵，越分散越大，越集中越接近 1；
- **stable rank**：更保守，直接受最大奇异值支配，适合快速看是否被单一方向主导；
- **spectral entropy**：熵值本身，等于 \(\log(\text{effective rank})\)，图中和 JSON 里都保留。

**第四步：如何生成**

不需要额外命令。运行第 8.2 节的 `visualize.py` 时，脚本会为每个 Grad-CAM 样本保留目标层特征，并自动生成：

```text
outputs/visualizations/
├── feature_rank.png
└── gradcam_summary.json   # 每个样本的 rank 指标和奇异值谱
```

核心调用关系是：

```text
GradCAM 捕获目标层特征
        ↓
GradCAMResult.feature_map
        ↓
feature_rank_stats() 计算奇异值谱 / 有效秩 / 稳定秩
        ↓
plot_feature_ranks() 生成紫色主题图
```

![特征图有效秩](outputs/visualizations/feature_rank.png)

**第五步：这张图怎么读**

左图是每个样本的有效秩：

- 深紫色是 ResNet-18，洋红紫是 Swin-T；
- 实心柱表示预测正确，斜线柱表示预测错误；
- 柱末还标了 `eff`、`stable` 和 `raw` 三个数值，方便核对。

右图是归一化奇异值谱：

- 每条线是一个样本，横轴是奇异值序号，纵轴是 \(\sigma_i/\sigma_1\)；
- 实线圆点表示预测正确，虚线叉号表示预测错误；
- 曲线越平，说明能量分布越分散，有效秩通常越高；曲线越陡，说明能量越集中在少数方向。

本次 8 个样本的完整数值：

| 模型 | 样本 | 预测 | 原始秩 | 有效秩 | 稳定秩 |
|---|---|---:|---:|---:|---:|
| ResNet-18 | `automobile` | `automobile` ✓ | 16 | 5.9 | 1.9 |
| ResNet-18 | `cat` | `automobile` ✗ | 16 | 6.7 | 2.1 |
| ResNet-18 | `airplane` | `airplane` ✓ | 16 | 6.3 | 2.1 |
| ResNet-18 | `ship` | `automobile` ✗ | 16 | 5.9 | 2.1 |
| Swin-T | `cat` | `cat` ✓ | 16 | 4.2 | 1.7 |
| Swin-T | `bird` | `horse` ✗ | 16 | 4.1 | 1.5 |
| Swin-T | `ship` | `ship` ✓ | 16 | 2.9 | 1.3 |
| Swin-T | `deer` | `cat` ✗ | 16 | 4.0 | 1.5 |

**第六步：正确解读，避免过度结论**

1. **原始秩饱和不等于没有结构。** 原始秩是硬上限，奇异值谱才是连续结构；本项目 16 个空间位置把所有样本的原始秩都顶到了 16。
2. **有效秩不是准确率指标。** ResNet 的 `cat → automobile` 错误样本有效秩为 6.7，反而高于两个正确样本。不能得出“秩高一定错”或“秩低一定对”。
3. **它适合诊断表示是否塌缩。** 有效秩接近 1，说明特征几乎只沿一个方向变化；数值越接近当前上限 16，说明能量分布越分散、独立方向越多。
4. **不同空间尺寸不能直接横比。** 有效秩受 \(HW\) 上限约束，`4×4` 和 `28×28` 的层不能只拿数值大小下结论。本项目只比较同一目标层内的样本。
5. **8 个样本只能做案例演示。** 要判断“正确样本的有效秩是否普遍更高”，需要在几百到上千个测试样本上收集指标，按正确/错误分组画箱线图或做显著性检验。
6. **它与 LoRA 的秩不是一回事。** LoRA 的 \(r\) 限制的是 \(\Delta W\) 的秩；这里测的是整个模型前向得到的特征秩。若要看 LoRA 本身的秩，应单独对 \(BA\) 做 SVD，而不是看 `feature_map`。

这项指标最适合做**同层、同数据、不同 LoRA 配置的对照**：例如固定 ResNet 的 `layer2.1.conv2`，分别训练 `r=2`、`r=4`、`r=8`，在相同测试集上统计有效秩分布和准确率。这样既能看参数效率，也能看低秩容量是否让特征表示发生塌缩。

实现分别在两个文件：

- `gradcam.py`：`feature_rank_stats()` 负责 SVD、原始秩、有效秩、稳定秩和归一化谱；
- `visualize.py`：`plot_feature_ranks()` 负责紫色主题图，样本指标同时写入 `gradcam_summary.json`。

---

## 九、继续实验：从跑通到跑好

### 9.1 增加训练轮数和数据量

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 \
  --epochs 10 \
  --train-limit 10000 \
  --test-limit 2000 \
  --output-dir outputs/resnet18_r4_10epochs
```

`--train-limit 0 --test-limit 0` 表示使用完整 CIFAR-10：

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 --epochs 10 \
  --train-limit 0 --test-limit 0
```

### 9.2 改变秩和 alpha

```bash
uv run --no-sync python3.12 train.py \
  --model swin_tiny \
  --rank 8 --alpha 16 \
  --epochs 5 \
  --output-dir outputs/swin_r8_alpha16
```

建议先保持 `alpha = 2*r`，单独观察 r 的影响，再调整 alpha，避免两个因素混在一起。

### 9.3 对照全参数微调

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 --method full \
  --epochs 2 --lr 1e-4
```

全参数微调通常需要更小学习率，也会消耗更多内存。不要把全参数微调的学习率机械地设成与 LoRA 相同。

### 9.4 保存完整合并模型

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 --epochs 2 --save-merged
```

会额外生成 `merged_model.pt`。它包含完整模型权重，因此文件远大于适配器。

---

## 十、换成你自己的模型时的检查清单

1. **原模型能正常前向传播**：先不加入 LoRA，确认输出 shape。
2. **正确替换任务头**：类别数与数据集一致，分类头可训练。
3. **打印模块名**：确认实际叫 `qkv`、`query`、`proj`，还是其他名称。
4. **先选少量注入点**：Transformer 可先选注意力 Linear；CNN 可先选 1×1 或普通卷积。
5. **检查参数冻结**：打印所有 `requires_grad=True` 的名称。
6. **验证初始等价性**：B=0 时，注入前后同一输入的输出应一致。
7. **只把可训练参数交给优化器**。
8. **先跑一个小 batch**：确认 forward、backward、optimizer.step 都能执行。
9. **检查 loss 是否下降**：不下降时先查标签、学习率、分类头和数据预处理。
10. **保存配置与类别映射**：只存 A/B 不足以完整复现推理。
11. **验证保存后加载**：重新构建模型，加载适配器，比较输出。
12. **验证合并等价性**：合并前后 logits 应仅有浮点误差。

---

## 十一、常见问题

### Q1：报错“没有模块匹配 target”

先运行 `named_modules()` 打印名称。目标名必须匹配真实模块名。例如 torchvision Swin 使用 `features...attn.qkv`，其他库可能叫 `attn.query`。

### Q2：报错 `LoRALinear object has no attribute weight`

这通常意味着模型像 torchvision Swin 一样，直接使用 `F.linear(x, layer.weight, ...)`。本项目已经通过 `LoRALinear.weight` 和 `bias` 属性解决；同时请保持 Swin 的 LoRA `dropout=0`。

### Q3：准确率一直接近随机猜测

按顺序检查：

- 是否误用了 `--no-pretrained`；
- 新分类头是否 `requires_grad=True`；
- 标签是否为 `0～9`；
- 是否使用了与预训练模型匹配的归一化；
- 学习率是否过大或过小；
- 训练轮数是否太少。

### Q4：为什么 LoRA 加了，但 A 一开始没有梯度

B 初始化为 0，因此最开始 A 的梯度可能为 0，而 B 可以学习。B 脱离 0 后，A 也会获得有效梯度。这是预期行为。

### Q5：LoRA 是否一定减少显存

不一定等比例减少。主干权重和中间激活仍在，尤其大 batch、大分辨率时激活仍占很多内存。LoRA 最明确减少的是可训练参数对应的梯度、优化器状态和适配器存储。

### Q6：MPS/CUDA 内存不足

优先减小 batch：

```bash
uv run --no-sync python3.12 train.py \
  --model swin_tiny --batch-size 8
```

也可以先用 CPU 验证：

```bash
uv run --no-sync python3.12 train.py \
  --model resnet18 --device cpu --train-limit 128 --test-limit 64 --epochs 1
```

### Q7：CIFAR-10 官方下载太慢

可使用公开镜像下载到本项目，再运行训练：

```bash
mkdir -p data
curl -L --fail --retry 3 \
  -o data/cifar-10-python.tar.gz \
  https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz
md5 data/cifar-10-python.tar.gz
```

正确文件的 MD5 应为：

```text
c58f30108f718f92721af3b95e74349a
```

torchvision 会自动识别并解压已下载文件。

### Q8：uv 创建环境后仍使用了系统 Python

本文统一使用：

```bash
uv run --no-sync python3.12 ...
```

也可以检查：

```bash
uv python find 3.12
uv run --no-sync python3.12 -c "import sys; print(sys.version); print(sys.executable)"
```

### Q9：Grad-CAM 没有梯度或热图全黑

- 确认调用时没有包在 `torch.no_grad()` 中；
- 目标层必须在当前前向路径上；
- 目标层输出必须能通过可训练 LoRA/其他参数连接到类别分数；
- 若热图全黑，可换一个更靠近网络输出、梯度更稳定的层；
- `channels_last` 必须与特征布局一致：CNN 通常是 `False`，torchvision Swin 阶段输出是 `True`。

本项目的 `gradcam.py` 会在目标层输出没有梯度时直接报错，而不是静默生成一张错误的图。

---

1. **秩对照**：分别运行 `r=2、4、8`，记录 LoRA 参数量和准确率，不只看精度。
2. **注入位置对照**：Swin 分别使用 `--target attn.qkv` 和 `--target attention`，观察 12 与 24 个注入点的差异。
3. **合并实验**：在测试文件中随机修改 A/B，比较合并前后输出，理解为什么必须 eval 后再比较。
4. **小数据到过拟合**：只用 200 张训练图片并训练 20 轮，观察训练准确率与测试准确率之间的差距，再尝试调整 dropout（ResNet）或数据增强。
5. **Grad-CAM 层对照**：把 ResNet 目标层从 `layer2.1.conv2` 换到 `layer3.1.conv2`，观察高语义、低分辨率特征图如何改变热区；再尝试用真实标签而不是预测标签作为 Grad-CAM 目标。

完成这些后，你应该能把 LoRA 概括为一句话：

> **冻结原模型，用一个低秩矩阵乘积学习任务所需的权重变化；训练只更新小矩阵，推理时还能把它们合并回原模型。**
