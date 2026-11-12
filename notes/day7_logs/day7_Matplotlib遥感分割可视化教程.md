# Matplotlib 遥感分割可视化完整教程（补全"方法作用"版）

> 核心认知：Matplotlib 是 Python 中最常用的绘图库。它可以绘制曲线、散点图、柱状图、图像、混淆矩阵和训练过程可视化结果。
>
> 在遥感分割项目中，Matplotlib 主要用于观察训练是否正常、比较真实标签与预测结果、分析类别混淆情况，以及生成论文中的实验图。
>
> 推荐学习路线：基础概念 → 基础绘图 → 图像显示 → Tensor 可视化 → 训练曲线 → 混淆矩阵 → 独立 `visualize.py` 文件 → 接入 `engine.py`。

---

## 一、Matplotlib 的基本组成

Matplotlib 的绘图过程可以理解为：

```text
创建画布 Figure
        ↓
创建坐标轴 Axes
        ↓
在 Axes 上绘图
        ↓
设置标题、坐标轴、图例
        ↓
保存或显示图片
```

### 1. Figure 是什么

`Figure` 是整张画布，可以理解为一张完整的纸。

```python
fig = plt.figure(figsize=(6, 4))
```

`figsize=(6, 4)` 表示画布宽 6 英寸，高 4 英寸。

**方法作用**：`plt.figure()` 只创建一张**空白画布**，不画任何东西。

| 参数 | 作用 | 建议值 |
| :--- | :--- | :--- |
| `figsize=(w, h)` | 画布宽高（英寸） | 曲线 `(8, 5)`；2×2 子图 `(12, 8)` |
| `dpi` | 分辨率 | 保存用 `150~300`，屏幕看用 `100` |

**常见坑**：只 `figure()` 不 `savefig()` 或 `show()`，画布不显示也不保存——它只是"准备了一张纸"。

### 2. Axes 是什么

`Axes` 是画布中的一个具体绘图区，包含坐标轴、刻度和图形。

```python
ax = fig.add_subplot()
```

一张 Figure 可以包含多个 Axes，因此可以在同一张画布中放置多张子图。

**方法作用**：`fig.add_subplot()` 在 Figure 上**切出一块绘图区**，后续 `ax.plot(...)` 都画在这块里。

**常见坑**：`fig.add_subplot()` 无参数时默认 1×1（整张画布一块）；多子图要 `fig.add_subplot(2, 2, 1)` 或直接用 `plt.subplots(2, 2)`。

### 3. Pyplot 状态机写法与面向对象写法

快速调试时可以使用 `plt` 状态机写法：

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [2, 4, 3])
plt.title("A Simple Curve")
plt.show()
```

工程项目推荐使用 `fig, ax` 面向对象写法：

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3], [2, 4, 3])
ax.set_title("A Simple Curve")
fig.tight_layout()
plt.show()
```

**两种写法的作用对比**：

| 写法 | 作用 | 适用 |
| :--- | :--- | :--- |
| `plt.plot(...)` | 状态机：对"当前激活的 Axes"画图 | 快速调试、单图 |
| `ax.plot(...)` | 面向对象：明确指定画在哪个 Axes | 工程 / 多子图，推荐 |

**常见坑**：`plt.plot` 在多子图时容易"画错图"——你以为是第 2 个子图，其实画到了第 1 个。`ax.plot` 永远指向明确。

---

## 二、最基础的绘图方法

### 1. 绘制折线图

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 6, 5]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y)
ax.set_title("Line Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(alpha=0.3)

plt.show()
```

**`ax.plot(x, y, label=, color=, linestyle=, linewidth=, marker=)` 的作用**：画**折线图**，`x`/`y` 是等长序列（`list` / `np.ndarray` 都行）。

| 参数 | 作用 | 例子 |
| :--- | :--- | :--- |
| `label` | 图例文字（必须配 `ax.legend()` 才显示） | `"Train Loss"` |
| `color` | 线色 | `"blue"` / `"tab:blue"` / `"#FF0000"` |
| `linestyle` | 线型 | `"-"` 实线 / `"--"` 虚线 / `":"` 点线 |
| `linewidth` | 线宽 | `2` |
| `marker` | 数据点标记 | `"o"` 圆 / `"s"` 方 / `"^"` 三角 |
| `markersize` | 标记大小 | `3~5`（点太大会糊） |

**常见坑**：

- `label` 不加 `legend()` 不显示——很多人写了 `label` 却忘了 `ax.legend()`。
- `x`/`y` 长度必须相等，否则 `ValueError: x and y must have same first dimension`。
- `x` 不传时默认 `range(len(y))`——但画 loss 曲线时 x 是 epoch，必须传。

折线图适合表示：

- loss 随 epoch 的变化
- mIoU 随 epoch 的变化
- 学习率变化
- 时间序列数据

### 2. 绘制多条曲线

```python
epochs = [1, 2, 3, 4, 5]
train_loss = [1.2, 0.9, 0.7, 0.5, 0.4]
val_loss = [1.3, 1.0, 0.8, 0.7, 0.65]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    epochs,
    train_loss,
    label="Train Loss",
    color="blue",
    linewidth=2,
)

ax.plot(
    epochs,
    val_loss,
    label="Val Loss",
    color="red",
    linestyle="--",
    linewidth=2,
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")
ax.legend()
ax.grid(alpha=0.3)

plt.show()
```

**作用**：多条曲线画在同一 Axes 上，靠 `color` / `linestyle` 区分，靠 `legend()` 说明——能直观看出"过拟合拐点"（train 降、val 升）。

### 3. 散点图

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 6, 5]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, color="green", s=60)
ax.set_title("Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
```

**`ax.scatter(x, y, color=, s=)` 的作用**：画散点图，每个点独立，不连线。

| 参数 | 作用 |
| :--- | :--- |
| `s` | 点的大小（`s=60` 比默认大） |
| `color` | 点颜色 |

**用途**：看两个指标的分布关系（如"预测置信度 vs 实际 IoU"），不适合看趋势（趋势用 `plot`）。

### 4. 柱状图

```python
classes = ["Building", "Road", "Water", "Forest"]
iou = [0.75, 0.68, 0.81, 0.63]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(classes, iou, color="steelblue")
ax.set_ylim(0, 1)
ax.set_ylabel("IoU")
ax.set_title("Per-class IoU")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
```

**`ax.bar(x, height, color=)` 的作用**：画柱状图，适合离散类别对比（如 per-class IoU）。

| 参数 | 作用 |
| :--- | :--- |
| `x` | 类别标签（`list[str]` 或数字） |
| `height` | 柱高（`list[float]`） |
| `color` | 柱色 |

**常见坑**：类别名太长会重叠——加 `plt.xticks(rotation=20)` 旋转，或 `fig.tight_layout()` 自动调间距。

柱状图适合绘制逐类别 IoU、逐类别准确率和不同模型对比结果。

---

## 三、子图和布局

### 1. 创建一行两列子图

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 4),
)

axes[0].plot([1, 2, 3], [1, 4, 2])
axes[0].set_title("Curve")

axes[1].bar(["A", "B", "C"], [3, 5, 2])
axes[1].set_title("Bar")

fig.tight_layout()
plt.show()
```

**`plt.subplots(nrows, ncols, figsize=)` 的作用**：一次性创建 Figure + 多个 Axes，返回 `(fig, axes)`。

**关键**：

- `nrows=1, ncols=2` → `axes` 是 1D 数组，用 `axes[0]` / `axes[1]`
- `nrows=2, ncols=2` → `axes` 是 2D 数组，用 `axes[0, 0]` / `axes[0, 1]`
- `nrows=1, ncols=1` → `axes` 是单个 Axes（不是数组），直接 `axes.plot(...)`

**常见坑**：1×1 时 `axes[0]` 会报错——因为 `axes` 不是数组。`squeeze=False` 可强制返回 2D。

### 2. 创建两行两列子图

```python
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
)

axes[0, 0].plot([1, 2, 3])
axes[0, 0].set_title("Plot 1")

axes[0, 1].plot([3, 2, 1])
axes[0, 1].set_title("Plot 2")

axes[1, 0].bar(["A", "B"], [2, 5])
axes[1, 0].set_title("Plot 3")

axes[1, 1].scatter([1, 2, 3], [3, 1, 4])
axes[1, 1].set_title("Plot 4")

fig.tight_layout()
plt.show()
```

### 3. `tight_layout()` 的作用

```python
fig.tight_layout()
```

**作用**：自动调整子图间距，减少标题、坐标轴标签和图像之间的重叠。

**常见坑**：

- 必须在 `savefig` 之前调——`savefig` 后再 `tight_layout` 无效。
- `tight_layout` 和 `colorbar` 偶尔冲突——这时用 `bbox_inches="tight"` 兜底。

保存图片时建议先调用：

```python
fig.tight_layout()
fig.savefig("result.png", dpi=300, bbox_inches="tight")
```

---

## 四、Matplotlib 显示图像

### 1. `imshow()` 的输入格式

`imshow()` 常见输入格式如下：

| 图像类型 | 数组形状 |
|---|---|
| 灰度图 | `[H, W]` |
| RGB 图像 | `[H, W, 3]` |
| RGBA 图像 | `[H, W, 4]` |

PyTorch 图像通常是：

```text
[C, H, W]
```

因此送入 `imshow()` 前需要转换为：

```text
[H, W, C]
```

### 2. `imshow` 方法作用与参数

**`ax.imshow(data, cmap=, vmin=, vmax=, interpolation=, alpha=)` 的作用**：把二维数组 / 三维数组渲染成图像。

| 参数 | 作用 | 关键点 |
| :--- | :--- | :--- |
| `data` | 图像数组 | `[H,W]` 灰度 / `[H,W,3]` RGB / `[H,W,4]` RGBA |
| `cmap` | 颜色映射 | `"gray"` 灰度 / `"tab10"` 离散分类 / `"Blues"` 热力图 |
| `vmin`/`vmax` | 颜色映射的数值范围 | mask 必须固定，否则同类不同色 |
| `interpolation` | 插值方式 | mask 必须 `"nearest"`，否则类别边缘被平滑成"渐变" |
| `alpha` | 透明度 | `0.5` 半透明叠加 |

**常见坑 1：mask 不设 `vmin`/`vmax`** → 同一类在不同图里颜色不同——因为 matplotlib 按当前图的最小/最大值自动映射。必须固定 `vmin=0, vmax=num_classes-1`。

**常见坑 2：mask 用默认插值** → 类别边界模糊——`imshow` 默认 `interpolation="antialiased"`，会把"0/1 边界"插值成 `0.3, 0.7`，看起来像渐变。mask 必须 `interpolation="nearest"`。

**常见坑 3：PyTorch tensor 直接 `imshow`** → 报错或颜色错——`[C,H,W]` 不是 `[H,W,C]`，必须先 `transpose(1,2,0)`。

### 3. 显示普通 RGB 图像

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("image.png").convert("RGB")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
ax.axis("off")
plt.show()
```

**作用**：`ax.axis("off")` 隐藏坐标轴刻度和边框，让图像"干净"显示。

### 4. 显示灰度图

```python
ax.imshow(gray_image, cmap="gray")
```

`cmap="gray"` 表示使用灰度颜色映射。

### 5. 显示 mask

mask 是离散类别图，不应该使用平滑插值。

```python
ax.imshow(
    mask,
    cmap="tab10",
    vmin=0,
    vmax=6,
    interpolation="nearest",
)
```

参数含义：

- `cmap="tab10"` 表示使用离散分类颜色
- `vmin=0` 表示最小类别编号
- `vmax=6` 表示最大类别编号
- `interpolation="nearest"` 防止类别边缘被平滑

### 6. 为什么 mask 要设置 `vmin` 和 `vmax`

如果不设置范围，Matplotlib 会根据当前图片中的最小值和最大值自动调整颜色。不同图像中出现的类别可能不同，导致同一个类别在不同图片中显示成不同颜色。

```python
ax.imshow(
    mask,
    cmap="tab10",
    vmin=0,
    vmax=num_classes - 1,
    interpolation="nearest",
)
```

在批量对比真实标签和预测结果时，应固定这个范围。

### 7. 原图、标签和预测结果对比

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_rgb)
axes[0].set_title("Image")
axes[0].axis("off")

axes[1].imshow(
    mask,
    cmap="tab10",
    vmin=0,
    vmax=6,
    interpolation="nearest",
)
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(
    pred_mask,
    cmap="tab10",
    vmin=0,
    vmax=6,
    interpolation="nearest",
)
axes[2].set_title("Prediction")
axes[2].axis("off")

fig.tight_layout()
plt.show()
```

### 8. 叠加显示 mask

```python
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img_rgb)
ax.imshow(
    mask,
    cmap="tab10",
    alpha=0.5,
    vmin=0,
    vmax=6,
    interpolation="nearest",
)

ax.axis("off")
plt.show()
```

`alpha` 控制透明度。`alpha=0.5` 表示 mask 半透明显示。

**作用**：两次 `imshow` 叠加——第一次画原图，第二次画半透明 mask，**能看出"分割结果和原图是否对齐"**。

---

## 五、Tensor 图像如何转换后显示

### 1. 从 `[C, H, W]` 转换为 `[H, W, C]`

```python
img_tensor = images[0]

img_numpy = img_tensor.detach().cpu().numpy()
img_numpy = img_numpy.transpose(1, 2, 0)

plt.imshow(img_numpy)
plt.show()
```

转换步骤：

```text
GPU Tensor
    ↓ cpu()
CPU Tensor
    ↓ detach()
脱离计算图
    ↓ numpy()
NumPy 数组
    ↓ transpose(1, 2, 0)
HWC 图像
```

**`tensor.detach().cpu().numpy()` 的作用**：把 GPU tensor 转成 numpy 数组。

| 步骤 | 作用 | 不做的后果 |
| :--- | :--- | :--- |
| `.detach()` | 脱离计算图 | `requires_grad=True` 时 `.numpy()` 报错 |
| `.cpu()` | 搬到 CPU | GPU tensor `.numpy()` 报错 |
| `.numpy()` | 转 numpy | — |

**顺序不能反**：`.cpu()` 必须在 `.numpy()` 前，`.detach()` 必须在 `.numpy()` 前（若 `requires_grad`）。

**`transpose(1, 2, 0)` 的作用**：`[C, H, W]` → `[H, W, C]`——matplotlib 只认 HWC。

**常见坑**：`transpose` 不复制数据，只换"视图"——`img.transpose(1,2,0)` 后 `.shape` 变了，但内存布局没变，`imshow` 能正确读。

### 2. 反归一化

如果图像经过：

```python
Normalize(mean=0.5, std=0.5)
```

则显示前需要反归一化：

```python
img_numpy = img_numpy * 0.5 + 0.5
img_numpy = np.clip(img_numpy, 0, 1)
```

完整示例：

```python
import numpy as np
import matplotlib.pyplot as plt

img_tensor = images[0]
img_numpy = img_tensor.detach().cpu().numpy()
img_numpy = img_numpy.transpose(1, 2, 0)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
img_numpy = img_numpy * std + mean
img_numpy = np.clip(img_numpy, 0, 1)

plt.imshow(img_numpy)
plt.axis("off")
plt.show()
```

**反归一化 `img * std + mean` 的作用**：把 `Normalize(mean=0.5, std=0.5)` 处理过的图像还原到 [0,1]。

**数学**：`Normalize` 做的是 `(x - mean) / std`，逆运算就是 `x * std + mean`。

**常见坑**：忘了 `np.clip(0, 1)` → 反归一化后可能 >1 或 <0，`imshow` 显示异常颜色（或警告）。

---

## 六、保存图片和释放画布

### 1. 保存图片

```python
fig.savefig(
    "result.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
```

**`fig.savefig(path, dpi=, bbox_inches=, facecolor=)` 的作用**：把当前 Figure 写盘为图片文件。

| 参数 | 作用 | 建议 |
| :--- | :--- | :--- |
| `dpi` | 分辨率 | 屏幕 `100`；论文 `300` |
| `bbox_inches="tight"` | 裁剪多余白边 | 必加 |
| `facecolor="white"` | 背景色 | 论文图用白色 |

**常见坑**：`savefig` 后不 `close` → 内存泄漏——100 epoch 后 100 个 Figure 全在内存，最终 OOM。

### 2. `show()` 和 `savefig()` 的区别

```python
plt.show()
```

**作用**：弹窗显示，阻塞直到关闭。适合交互式调试。

```python
fig.savefig("result.png")
```

**作用**：写盘，不阻塞。适合训练脚本和服务器环境。

**常见坑**：训练循环里调 `plt.show()` → 每轮弹窗，训练卡住。训练中只 `savefig`。

训练过程中主要使用 `savefig()`，因为服务器通常没有图形界面。

### 3. 关闭画布

循环绘图时必须关闭画布：

```python
plt.close(fig)
```

**作用**：释放 Figure 占用的内存。

**常见坑**：循环里 `subplots` 不 `close` → matplotlib 警告 `More than 20 figures have been opened`，最终 OOM。每次 `savefig` 后必须 `close`。

---

## 七、绘制训练曲线

你的 `history` 中保存了以下内容：

```python
history = [
    {
        "epoch": 1,
        "train_avg_loss": 1.2,
        "val_avg_loss": 1.0,
        "miou": 0.35,
        "mean_dice": 0.48,
        "pixel_acc": 0.72,
        "lr": 0.001,
        "confmat": confmat,
    }
]
```

从 `history` 中提取曲线数据：

```python
epochs = [h["epoch"] for h in history]
train_loss = [h["train_avg_loss"] for h in history]
val_loss = [h["val_avg_loss"] for h in history]
miou = [h["miou"] for h in history]
dice = [h["mean_dice"] for h in history]
pixel_acc = [h["pixel_acc"] for h in history]
lr = [h["lr"] for h in history]
```

**列表推导 `[h["train_avg_loss"] for h in history]` 的作用**：从 `list[dict]` 中提取某一列成 `list`。

**关键**：字段名必须和 `epoch_metrics` 的 key 完全一致——`"train_avg_loss"` 写成 `"train_loss"` 就 `KeyError`。

最小绘图示例：

```python
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(epochs, train_loss, label="Train Loss")
ax.plot(epochs, val_loss, label="Val Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss Curve")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("loss_curve.png", dpi=200)
plt.close(fig)
```

**双 y 轴 / 对数轴的作用**：

```python
ax.set_yscale("log")   # lr 跨数量级，必须对数轴
```

**作用**：压缩数量级差距——`lr` 从 `1e-3` 降到 `1e-6`，线性轴上前 90% 是"一条平线"，对数轴才看得出"阶梯下降"。

**常见坑**：`lr` 有 0 时对数轴报错——`log(0)` 无定义。`lr` 最低别设 0（`eta_min=1e-6`）。

---

## 八、绘制混淆矩阵

### 1. 混淆矩阵的含义

对于语义分割任务：

- 行表示真实类别
- 列表示预测类别
- 对角线表示预测正确的像素
- 非对角线表示类别混淆

例如：

```text
confmat[2, 4]
```

表示真实类别为 2，但模型预测为类别 4 的像素数量。

### 2. 使用 `imshow()` 显示混淆矩阵

```python
import numpy as np
import matplotlib.pyplot as plt

confmat = np.array([
    [90, 5, 2],
    [8, 80, 7],
    [3, 6, 91],
])

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(confmat, cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Confusion Matrix")

plt.show()
```

**`ax.imshow(confmat, cmap="Blues")` 的作用**：把 C×C 矩阵渲染成热力图。

**关键**：`imshow` 默认 `origin="upper"`——第 0 行在顶部。混淆矩阵行=真值，所以"真值 0"在顶部，符合直觉。若想"真值 0 在底部"，加 `origin="lower"`。

**`fig.colorbar(image, ax=ax)` 的作用**：在 Axes 旁边**添加颜色条**，说明"数值 → 颜色"的映射关系。

### 3. 在格子中标出数值

```python
for row in range(confmat.shape[0]):
    for col in range(confmat.shape[1]):
        ax.text(
            col,
            row,
            str(confmat[row, col]),
            ha="center",
            va="center",
        )
```

**`ax.text(col, row, text, ha=, va=, color=)` 的作用**：在格子中央写数值。

| 参数 | 作用 |
| :--- | :--- |
| `col, row` | x=列，y=行（注意顺序！`text(x, y)` 不是 `text(row, col)`） |
| `ha="center"` | 水平居中 |
| `va="center"` | 垂直居中 |
| `color` | 深色格子写白字，浅色格子写黑字（对比度） |

**常见坑**：`ax.text(col, row)` 和 `confmat[row, col]` 的 `row/col` 顺序相反——`text` 是 `(x, y)` = `(col, row)`，写反了数字全错位。

### 4. 归一化混淆矩阵

按真实类别进行归一化：

```python
row_sum = confmat.sum(axis=1, keepdims=True)
normalized = np.divide(
    confmat,
    row_sum,
    out=np.zeros_like(confmat, dtype=float),
    where=row_sum != 0,
)
```

**作用**：每行除以该行总和 → 对角线 = 每类召回率。

**`where=row_sum != 0` 的作用**：防止"某类完全没出现"时除 0——`out=np.zeros_like` 保证除 0 的位置填 0，不报 `nan`。

**常见坑**：不用 `where` 直接除 → `RuntimeWarning: invalid value encountered`，结果含 `nan`，`imshow` 显示异常。

归一化后，每一行表示某个真实类别被预测到各个类别的比例。

---

## 九、建立独立的 `visualize.py`

训练、验证和绘图应当分开组织：

```text
rsunet/
├── engine.py       # 训练和验证
├── metrics.py      # 指标计算
└── visualize.py    # 所有可视化函数
```

`visualize.py` 不负责训练，它只接收 `history` 和保存路径，然后生成图片。

### 1. 导入依赖

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
```

**`matplotlib.use("Agg")` 的作用**：切换无 GUI 后端——不弹窗、不依赖显示器，适合服务器 / 训练脚本。

**常见坑**：`use("Agg")` 必须在 `import matplotlib.pyplot` 之前——顺序反了后端已初始化，`use` 无效。

```python
# 正确顺序
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 顺序反了
import matplotlib.pyplot as plt
matplotlib.use("Agg")   # 已初始化，无效
```

### 2. 统一数值转换函数

`history` 中的指标有时是 Python 数字，有时是 GPU Tensor，因此需要统一转换：

```python
def _to_float(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().item()
    return float(value)
```

**`_to_float(value)` 的作用**：统一把 tensor / numpy / Python 数字转成 Python `float`。

**为什么需要**：`history` 里有的字段是 tensor（`miou`）、有的是 float（`train_loss`）——`plot` 对 GPU tensor 会报错，统一转 float 最稳。

**常见坑**：`float(tensor)` 在 GPU tensor 上会报错——必须先 `.cpu()`，所以 `.detach().cpu().item()` 三步缺一不可。

### 3. 训练曲线函数

```python
def plot_training_curves(history, save_path):
    if not history:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [_to_float(h["train_avg_loss"]) for h in history]
    val_loss = [_to_float(h["val_avg_loss"]) for h in history]
    miou = [_to_float(h["miou"]) for h in history]
    dice = [_to_float(h["mean_dice"]) for h in history]
    pixel_acc = [_to_float(h["pixel_acc"]) for h in history]
    lr = [_to_float(h["lr"]) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss, label="Train Loss")
    axes[0, 0].plot(epochs, val_loss, label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, miou, label="mIoU")
    axes[0, 1].set_title("mIoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, dice, label="Mean Dice")
    axes[1, 0].plot(epochs, pixel_acc, label="Pixel Accuracy")
    axes[1, 0].set_title("Segmentation Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, lr, label="Learning Rate")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("RS-Unet Training Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
```

**`save_path.parent.mkdir(parents=True, exist_ok=True)` 的作用**：自动创建保存目录——目录不存在时 `savefig` 报错。

- `parents=True`：多级目录一起建（`a/b/c` 不存在时全建）。
- `exist_ok=True`：目录已存在不报错。

**常见坑**：不写这两行 → `FileNotFoundError: No such file or directory`——训练第 1 轮就崩。

**`axes[0, 1].set_ylim(0, 1)` 的作用**：固定 y 轴范围到 `[0,1]`——mIoU / Dice / Pixel Acc 都是比例，**固定范围才能"跨 epoch 对比"**。不设的话，matplotlib 按当前值自动缩放，**第 1 轮 0.3 可能占满整个 y 轴，看起来像"0.9"**。

### 4. 混淆矩阵函数

```python
def plot_confusion_matrix(
    history,
    save_path,
    class_names=None,
    normalize=False,
):
    if not history:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confmat = np.asarray(
        history[-1]["confmat"],
        dtype=np.float64,
    )

    if normalize:
        row_sum = confmat.sum(axis=1, keepdims=True)
        display_mat = np.divide(
            confmat,
            row_sum,
            out=np.zeros_like(confmat),
            where=row_sum != 0,
        )
        value_format = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        display_mat = confmat
        value_format = ".0f"
        title = "Confusion Matrix"

    num_classes = confmat.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(display_mat, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(
        f"{title} - Epoch {history[-1]['epoch']}"
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = display_mat.max() / 2

    for row in range(num_classes):
        for col in range(num_classes):
            color = (
                "white"
                if display_mat[row, col] > threshold
                else "black"
            )
            ax.text(
                col,
                row,
                format(display_mat[row, col], value_format),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
```

**`history[-1]["confmat"]` 的作用**：取最新一轮的混淆矩阵——实时更新时，每轮覆盖显示最新状态。

**常见坑**：`history[-1]` 在 `history` 为空时 `IndexError`——`plot_confusion_matrix` 开头必须 `if not history: return`。

**`np.asarray(..., dtype=np.float64)` 的作用**：把 int64 混淆矩阵转 float——**后面要归一化（除法），int 除 int 会截断**。

**`threshold = display_mat.max() / 2` 的作用**：**自动判断"深色格子写白字，浅色格子写黑字"**——数值 > 一半最大值时格子是深色，写白字才看得清。

**`ax.set_xticks` + `ax.set_xticklabels` 的作用**：把默认的 `0,1,2...` 刻度换成**类别名**——`imshow` 默认按像素索引显示刻度，**不换的话横轴是 0~C-1 的数字**，不直观。

### 5. 统一调用函数

```python
def visualize_history(
    history,
    save_dir,
    class_names=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        history,
        save_dir / "training_curves.png",
    )

    plot_confusion_matrix(
        history,
        save_dir / "confusion_matrix.png",
        class_names=class_names,
        normalize=False,
    )

    plot_confusion_matrix(
        history,
        save_dir / "confusion_matrix_normalized.png",
        class_names=class_names,
        normalize=True,
    )
```

**`visualize_history(...)` 的作用**：**统一入口**——一次调用画 3 张图（曲线 + 原始混淆矩阵 + 归一化混淆矩阵），**`fit` 里只调一个函数**，代码干净。

---

## 十、在 `engine.py` 中接入可视化

在 `engine.py` 顶部导入：

```python
from rsunet.visualize import visualize_history
```

在 `fit()` 中完成本轮指标整理后：

```python
history.append(epoch_metrics)

visualize_history(
    history=history,
    save_dir=save_dir / "figures",
    class_names=[
        "background",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
        "class_5",
        "class_6",
    ],
)
```

建议放在：

```python
history.append(epoch_metrics)
```

之后。

**为什么放 `history.append(epoch_metrics)` 之后**：保证当前轮的指标已经进 `history`——`visualize_history(history)` 才能画到最新点。

**常见坑**：放 `append` 之前 → 图里少一个点——第 1 轮画出来是"空图"（`len(history)==0` 直接 return）。

**为什么 `save_dir / "figures"`**：图和权重分开存——`weights/` 存 `.pth`，`figures/` 存 `.png`，目录清晰。

**常见坑**：图和权重混在 `weights/` → 找图困难，且图文件被误当权重加载。

训练过程将不断覆盖同名文件：

```text
第 1 轮 → figures/training_curves.png
第 2 轮 → 覆盖 figures/training_curves.png
第 3 轮 → 覆盖 figures/training_curves.png
```

最终目录：

```text
runs/
└── unet_r50_512_bs8/
    ├── figures/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   └── confusion_matrix_normalized.png
    └── weights/
        ├── last.pth
        └── best_iou.pth
```

---

## 十一、为什么混淆矩阵使用最后一个 epoch

你的 `history` 每个元素都保存了一个 epoch 的混淆矩阵：

```python
history[-1]["confmat"]
```

表示当前最新一轮的混淆矩阵。因此实时更新时，绘图函数使用：

```python
confmat = history[-1]["confmat"]
```

每轮训练结束后，它都会被新的验证结果覆盖显示。

**常见坑**：`history[-1]` 在 `history` 为空时 `IndexError`——必须 `if not history: return`。

如果想查看某一轮的混淆矩阵，可以使用：

```python
epoch_index = 9
confmat = history[epoch_index]["confmat"]
```

---

## 十二、实时绘图中的注意事项

### 1. 使用同一个文件名

```python
save_dir / "training_curves.png"
```

**作用**：`savefig` 自动覆盖——不会每轮生成一堆 `epoch_1.png` / `epoch_2.png`。

**常见坑**：文件名带 epoch → 磁盘爆炸——100 epoch × 3 张图 = 300 张图。

不要每轮把 epoch 写进文件名，否则会得到很多图片：

```python
# 会生成很多图片，不适合实时覆盖
f"training_curves_epoch_{epoch}.png"
```

### 2. 保存后关闭画布

```python
fig.savefig(save_path)
plt.close(fig)
```

**作用**：释放 Figure 内存——不 close 的话 100 epoch 后内存爆。

训练 100 个 epoch 时，如果不关闭 Figure，可能会不断增加内存占用。

### 3. 不要在训练循环的每个 batch 画图

**作用**：每个 epoch 画一次——batch 级画图会**磁盘 IO 爆炸**，训练慢 10 倍。

建议每个 epoch 画一次。每个 batch 画图会：

- 降低训练速度
- 频繁读写磁盘
- 占用更多内存
- 让图片更新过于频繁

**常见坑**：在 `for batch` 里 `savefig` → 训练从 1 小时变 10 小时。

### 4. 指标需要转换成普通数值

验证函数中建议返回：

```python
metrics = {
    "miou": miou.item(),
    "mean_dice": mean_dice.item(),
    "pixel_acc": pixel_acc.item(),
}
```

**作用**：`metrics` 里只放 float / list / ndarray——`plot`、`json.dump`、`pd.DataFrame` 都正常。

**常见坑**：`metrics` 里留 GPU tensor → `plot` 报错、`json.dump` 报错、`print` 显示 `tensor(...)`。

这样保存、绘图和日志记录都更稳定。

---

## 十三、训练曲线应该如何解读

### 1. 训练 loss 和验证 loss

**作用**：判断过拟合——`train` 降、`val` 升 = 过拟合；两者都降 = 正常。

理想情况：

```text
train loss 下降
val loss 下降并逐渐稳定
```

如果训练 loss 继续下降，但验证 loss 开始上升，说明可能出现过拟合。

**常见坑**：只看 train loss → 以为训练很好，实际早已过拟合。

### 2. mIoU 和 Dice

mIoU 和 Dice 越高越好。二者都反映预测区域和真实区域的重叠程度。

**作用**：诊断"loss 降但指标不涨"——可能是 mask 编号错 / `ignore_index` 错 / 输出通道数 ≠ 类别数 / 全预测背景。

如果 loss 下降，但 mIoU 长时间不提升，需要检查：

- mask 类别编号是否正确
- `ignore_index` 是否设置正确
- 模型输出通道数是否等于类别数
- 预测结果是否全部偏向背景

**常见坑**：只看 loss 不看 mIoU → 训练"正常"但模型学了个"全背景"。

### 3. Pixel Accuracy

Pixel Accuracy 可能受到背景类别像素数量影响。遥感图像中背景占比很大时，Pixel Accuracy 较高并不一定代表小目标类别分割良好，因此需要结合 mIoU 和逐类别 IoU 一起观察。

**作用**：背景占 90% 时，全预测背景也有 90% Pixel Acc——必须结合 mIoU 看。

**常见坑**：只看 Pixel Acc → 被"90% 准确率"骗了，实际小类全错。

### 4. 学习率

CosineAnnealingLR 的学习率通常会逐渐下降。绘图时使用对数坐标：

```python
ax.set_yscale("log")
```

**作用**：压缩数量级差距——`lr` 从 `1e-3` 降到 `1e-6`，线性轴上前 90% 是"一条平线"，对数轴才看得出"阶梯下降"。

**常见坑**：`lr` 有 0 时对数轴报错——`log(0)` 无定义。

可以更清楚地观察学习率变化。

---

## 十四、完整调用关系

```text
train.py
    ↓
fit()
    ├── train_one_epoch()
    ├── validate_one_epoch()
    ├── history.append(epoch_metrics)
    ├── visualize_history()
    │     ├── plot_training_curves()
    │     ├── plot_confusion_matrix()
    │     └── plot_confusion_matrix(normalize=True)
    ├── save last.pth
    └── save best_iou.pth
```

`engine.py` 负责计算和组织数据，`visualize.py` 负责绘图，二者通过 `history` 连接。

---

## 十五、最终最小使用示例

```python
from rsunet.visualize import visualize_history

history.append(epoch_metrics)

visualize_history(
    history=history,
    save_dir=save_dir / "figures",
    class_names=[
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
    ],
)
```

只要 `epoch_metrics` 中包含以下字段，绘图文件就可以正常工作：

```python
{
    "epoch": epoch,
    "train_avg_loss": train_avg_loss,
    "val_avg_loss": val_avg_loss,
    "miou": miou,
    "mean_dice": mean_dice,
    "pixel_acc": pixel_acc,
    "lr": current_lr,
    "confmat": confmat,
}
```

这套组织方式适合你的 RS-Unet 项目。后续增加预测样例、类别 IoU 柱状图、误分类图时，可以继续在 `visualize.py` 中增加函数。

---

## 附录：方法作用速查表

| 方法 | 核心作用 | 最易踩的坑 |
| :--- | :--- | :--- |
| `figure` | 创建画布 | 不 `savefig` 不 `show` = 白建 |
| `subplots` | 一次建 fig+axes | 1×1 时 `axes` 不是数组 |
| `plot` | 折线图 | `label` 不加 `legend()` 不显示 |
| `scatter` | 散点图 | 不适合看趋势 |
| `bar` | 柱状图 | 类别名长会重叠 |
| `tight_layout` | 调间距 | 必须在 `savefig` 前 |
| `imshow` | 显示图像/矩阵 | mask 必须 `vmin/vmax/nearest` |
| `detach().cpu().numpy()` | tensor→numpy | 顺序不能反 |
| `transpose(1,2,0)` | CHW→HWC | — |
| `savefig` | 写盘 | 不 `close` 会 OOM |
| `close(fig)` | 释放内存 | 循环里必须调 |
| `text(col, row)` | 格子写数 | `(x,y)` 不是 `(row,col)` |
| `np.divide(where=)` | 防除 0 | 不加 `where` 出 `nan` |
| `use("Agg")` | 无 GUI 后端 | 必须在 `import pyplot` 前 |
| `_to_float` | 统一转 float | GPU tensor 先 `.cpu()` |
| `mkdir(parents=True, exist_ok=True)` | 建目录 | 不写就 `FileNotFoundError` |
| `axis("off")` | 隐藏坐标轴 | — |
| `colorbar` | 加颜色条 | 和 `tight_layout` 偶尔冲突 |
| `set_ylim(0,1)` | 固定比例范围 | 不设会"跨 epoch 不可比" |
| `threshold = max/2` | 深色格子写白字 | 不设会"看不清数字" |