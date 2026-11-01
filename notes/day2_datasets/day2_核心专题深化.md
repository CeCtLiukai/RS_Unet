# 从数据到张量：核心专题笔记

- **学习主题**：NumPy 图像操作 / Tensor 与常用张量格式 /
  image 与 mask 的处理差异 / Dataset 写法 / img-mask 配对
- **定位**：day2 核心专题笔记，与 loveda_dataset.py 精读笔记互补——
  精读解决"看懂别人的"，这份解决"自己写得出"
- **前置**：day1 数据校验笔记（`scripts/data_check.py`）
- **使用方式**：每个专题自成一个知识块，先讲清"为什么"，
  再给"可以抄的例子"；例子均已在本机 rs_task 环境实测通过

> 文档格式说明：本文按 A4 排版优化，行宽控制在 80 字符以内，表格不超过 4 列，
> 不依赖 Mermaid 等需要插件渲染的语法，可直接用 pandoc 或 Markdown PDF 插件导出。

## 目录

- 专题 1：NumPy 图像操作必备
- 专题 2：PyTorch Tensor 操作与常用张量格式
- 专题 3：image 与 mask —— 为什么处理完全不同
- 专题 4：写 Dataset 的完整思路
- 专题 5：img 与 mask 配对方法论
- 附录：NumPy / Tensor 速查卡

---

> 这份笔记针对我自己暴露出的薄弱点集中补强，每个专题自成一个
> 知识块，先讲清"为什么"，再给"可以抄的例子"。

## 专题 1：NumPy 图像操作必备

一句话定位：**NumPy 是所有图像库的"通用货币"**——OpenCV 读进来是
ndarray，PIL 转出来是 ndarray，喂给 PyTorch 前还是 ndarray。
图像在 NumPy 世界里只有一种身份：带形状、带 dtype 的数字表格。

### 1.1 形状约定：HWC 与行列顺序

```python
img = cv2.imread(str(p), cv2.IMREAD_COLOR)       # (H, W, 3)
mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # (H, W)

img.shape    # (1024, 1024, 3) —— 高在前，宽在后
mask.shape   # (1024, 1024)
```

- **NumPy / OpenCV / PIL 生态统一用 HWC**（通道在最后）；
- **PyTorch 统一用 CHW**（通道在最前），这是两边互转必须 transpose
  的根本原因；
- 第一个下标是**行（y，从上往下）**，第二个是**列（x，从左往右）**，
  所以 `img[100, 200]` 是"第 100 行、第 200 列"的像素，即 y=100、x=200。

### 1.2 索引与切片：裁剪就是切片

```python
img[0, 0]              # 左上角像素，返回 (3,) 的 BGR 三元组
img[:, :, 0]           # 取蓝色通道，(H, W)
img[:, :, ::-1]        # BGR → RGB 的切片写法
mask[512, 512]         # 中心点的类别编号

# 裁剪：取左上角起 512×512 的区域（训练随机裁剪的基本功）
crop = img[0:512, 0:512]        # [y1:y2, x1:x2]
```

### 1.3 dtype 转换与 uint8 溢出（大坑）

NumPy 不会自动"帮你扩宽类型"，`uint8` 算着算着就回绕：

```python
a = np.array([200, 250], dtype=np.uint8)
a + 100      # 期望 [300, 350]，实际 [44, 94] —— 溢出取模，不报错
```

所以**任何算术前先转换 dtype**：

```python
a.astype(np.int32) + 100              # 扩宽后再算
img.astype(np.float32) / 255.0        # 归一化的标准写法
```

| 转换 | 写法 | 注意 |
|---|---|---|
| uint8 → float32 | `img.astype(np.float32)` | 再除 255 |
| float → uint8 | `arr.astype(np.uint8)` | 超过 255 会回绕，先 `np.clip` |
| uint8 → int64 | `mask.astype(np.int64)` | 扩宽永远安全 |

### 1.4 布尔掩码操作：分割任务的核心武器

**比较运算返回同形状的 bool 数组**，它本身就是"下标筛选器"：

```python
mask = np.zeros((4, 4), dtype=np.uint8)
mask[1:3, 1:3] = 3                  # 中间 2×2 是类别 3

mask == 3                # 4×4 的 bool 阵，True 即类别 3 的位置
(mask == 3).sum()        # 4 —— 数一类像素有多少个
mask == 3                # ↑ 统计类别占比就是这么来的
```

bool 阵可以直接当索引用（布尔索引），也可以做逻辑组合：

```python
sel = (mask == 3) | (mask == 5)      # 或；& 是与；~ 是非
mask[sel]                            # 取出所有满足条件的值
mask[mask == 0] = 255                # 原地替换：0 → ignore_index
```

三件套（对应 data_check.py 里的实际用法）：

```python
# ① 统计：每个类别多少像素
vals, cnts = np.unique(mask, return_counts=True)

# ② 替换：把 0 改成 255（np.where 是"逐元素三目"）
mask = np.where(mask == 0, 255, mask)

# ③ 可视化：把"是否等于 0"画成黑白图
zero_bin = (mask == 0).astype(np.uint8) * 255
```

### 1.5 广播：小数组自动对齐大数组

形状不同的数组运算时，NumPy 会把**末尾对齐**的维度自动扩展：

```python
img01 = img.astype(np.float32) / 255.0       # (H, W, 3) / 标量
mean = np.array([0.485, 0.456, 0.406])       # (3,)
std  = np.array([0.229, 0.224, 0.225])       # (3,)

norm = (img01 - mean) / std
# (H, W, 3) 与 (3,) 运算：mean 自动"广播"到每个像素位置
```

这就是 `transforms.Normalize` 背后发生的事——它不是魔法，只是一条
广播规则。**注意 `mean` 的长度必须等于通道数**，顺序对应 R、G、B。

### 1.6 统计与去重

```python
np.unique(mask)                     # 出现过哪些类别值（升序）
np.unique(mask, return_counts=True) # 类别值 + 像素个数
mask.sum(), mask.mean(), mask.max()
np.bincount(mask.ravel())           # 0~最大值的完整计数向量
```

`ravel()` 拉平成 1 维（视图）；`bincount` 只吃非负整数，
但输出"每个值各多少个"一步到位，比 `unique` 更适合画直方图。

### 1.7 拼接：`stack` vs `concatenate`

```python
a = np.zeros((256, 256))
b = np.ones((256, 256))

np.stack([a, b])          # (2, 256, 256) —— 新增一个维度
np.stack([a, b], axis=2)  # (256, 256, 2) —— 维度加在末尾
np.concatenate([a, b])    # (512, 256)    —— 沿已有维度拼
```

记法：**stack 堆出新维度，concatenate 沿老维度接长**。
"把两个类别图叠成双通道"用 stack，"上下拼图"用 concatenate。

### 1.8 综合示例：img / mask 同步随机裁剪

增强的核心纪律：**几何变换必须用同一组随机数同时作用于 img 和 mask**。

```python
import numpy as np

def random_crop(img, mask, size=512):
    """img: (H, W, 3) uint8;  mask: (H, W) uint8"""
    h, w = img.shape[:2]
    top  = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    return (img[top:top + size, left:left + size],
            mask[top:top + size, left:left + size])

img_c, mask_c = random_crop(img, mask)
# img_c.shape == (512, 512, 3), mask_c.shape == (512, 512)
# 两次切片用的是同一个 (top, left)，空间位置严格对齐
```

---

## 专题 2：PyTorch Tensor 操作与常用张量格式

### 2.1 训练链路上的常用张量格式（总表）

这张表是**本专题最重要的一张表**，先背这张，再学操作：

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| 单张图像 | `(3, H, W)` | float32 | Dataset 返回，值 [0,1] |
| 单张掩码 | `(H, W)` | int64 | 值 ∈ {0..6, 255} |
| batch 图像 | `(N, 3, H, W)` | float32 | DataLoader 拼出 |
| batch 掩码 | `(N, H, W)` | int64 | 与图像同 N、H、W |
| 模型输出 | `(N, 7, H, W)` | float32 | 每个类别一个通道 |
| 预测图 | `(N, H, W)` | int64 | `argmax(dim=1)` 的结果 |
| one-hot 标签 | `(N, 7, H, W)` | float32 | 值只有 0/1 |
| 损失值 | `()` 标量 | float32 | `loss.item()` 取出 |

逐条解释"为什么是这个形状"：

- **图像有通道维、掩码没有**：网络输入要求 CHW；而掩码是"每个像素
  一个类别编号"，不需要通道维——类别信息在模型输出端由 7 个通道
  表达（每个通道是该类的得分）。
- **float32**：模型权重就是 float32，输入必须同型，否则报
  `expected Float but found Byte/Double`。
- **int64（Long）**：`CrossEntropyLoss` 拿掩码值当"通道下标"去取
  logits，下标必须是整数。
- **模型输出 (N, 7, H, W) 叫 logits**：每个像素在 7 个类别上的
  未归一化得分，可为任意实数；`softmax(dim=1)` 后变成概率；
  `argmax(dim=1)` 后变成 (N, H, W) 的类别图。
- **one-hot**：把掩码"每个位置存类别号"改写成"每个类别存 0/1"，
  形状就变成 (N, 7, H, W)。Dice / Focal Loss 的部分实现需要它。
- **损失是 0 维标量**：不是 python 数，进日志必须 `loss.item()`。

### 2.2 创建与互转：`from_numpy` 共享内存

```python
import numpy as np
import torch

arr = np.zeros(3, dtype=np.float32)

t1 = torch.from_numpy(arr)     # 零拷贝，与 arr 共享同一块内存
t2 = torch.tensor(arr)         # 拷贝，独立一份

arr[0] = 99
print(t1)      # tensor([99.,  0.,  0.])  —— 跟着变了！
print(t2)      # tensor([0., 0., 0.])     —— 不受影响
```

| 写法 | 语义 | 适用 |
|---|---|---|
| `torch.from_numpy(arr)` | 零拷贝共享内存 | 数组用完即弃（Dataset 返回前） |
| `torch.tensor(arr)` | 总是拷贝 | 需要"快照"隔离时 |
| `t.numpy()` | ndarray（仅 CPU 张量） | 保存预测图前 |

反向同样成立：改 tensor，numpy 数组也变。要隔离就 `.clone()`。

### 2.3 形状操作四件套

```python
img = torch.rand(3, 1024, 1024)        # 假装是单张图

img.unsqueeze(0)                       # (1, 3, 1024, 1024) 加 batch 维
img.unsqueeze(0).squeeze(0)            # 变回去，去掉长度 1 的维

hwc = torch.rand(1024, 1024, 3)
chw = hwc.permute(2, 0, 1)             # HWC → CHW（对应 np.transpose）

batch = torch.stack([img, img, img])   # (3, 3, 1024, 1024) 组 batch
big = torch.cat([batch, batch], dim=0) # (6, 3, 1024, 1024) 沿 N 拼
```

- `unsqueeze(0)` 加在第 0 维 → "单张变单 batch"，**推理时必用**；
- `permute` 与 NumPy 的 `transpose` 一一对应；
- `stack` / `cat` 的区别同 NumPy（专题 1.7）；
- DataLoader 自动帮我们做 `stack`，所以 `__getitem__` 只返回单样本。

### 2.4 dtype 转换

```python
t.float()              # → float32
t.long()               # → int64
t.to(torch.uint8)      # 通用写法
t.half()               # → float16（混合精度用）
```

NumPy 的 `astype` 是函数带括号传类型；PyTorch 是方法名即类型，
**`.long()` 不是"变长"，是 LongTensor = int64**——名字来源于 C 时代。

### 2.5 训练循环常用操作

```python
with torch.no_grad():              # 推理/评估关梯度
    logits = model(imgs)           # (N, 7, H, W)
    prob   = logits.softmax(dim=1) # (N, 7, H, W) 每像素概率和为 1
    pred   = logits.argmax(dim=1)  # (N, H, W) int64

one = pred[0].cpu().numpy().astype(np.uint8)   # → ndarray 才能存图
Image.fromarray(one).save("pred.png")

loss.item()                        # 标量张量 → python float
imgs = imgs.to(device)             # 搬 GPU；mask 也要搬！
t.detach()                         # 从计算图摘下来
```

### 2.6 一个最小训练循环（把上面的格式全部串起来）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(num_classes=7).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

for imgs, masks in loader:             # imgs (N,3,H,W) f32
    imgs = imgs.to(device)             # masks (N,H,W) i64
    masks = masks.to(device)           # 标签也要上同一设备

    logits = model(imgs)               # (N, 7, H, W)
    loss = criterion(logits, masks)    # 标量 f32

    opt.zero_grad()                    # 清上一次的梯度
    loss.backward()                    # 反向传播
    opt.step()                         # 更新权重

    print(loss.item())                 # → python 数才能 print/log
```

对照 2.1 的总表逐行核对一遍形状与 dtype，训练循环就不再神秘。

---

## 专题 3：image 与 mask —— 为什么处理完全不同

### 3.1 一句话本质

> **image 存的是"这里看到了什么"——是传感器测出来的连续物理量；
> mask 存的是"这里是什么"——是人打的离散类别编号。**
> 对连续量做算术（插值、平均、缩放、归一化）都有意义；
> 对离散编码做任何算术都是在破坏编码本身。

这一句话能推导出下面整张表，遇到拿不准的操作就回到这句话问自己：
"这个操作对'编号'有意义吗？"

### 3.2 全流程对比总表

| 环节 | image | mask | 原因 |
|---|---|---|---|
| 读取 | `IMREAD_COLOR` | `IMREAD_GRAYSCALE` | 要颜色 / 要单通道索引 |
| 转 float | 要，`/255` | **绝不能** | 编号不能缩放 |
| normalize | mean / std | 不动 | 同上 |
| 通道维 | 转成 `(3,H,W)` | 保持 `(H,W)` | 网络要 CHW / CE 要索引 |
| 缩放插值 | BILINEAR 等 | **NEAREST** | 插值会造出类别 |
| 颜色增强 | 允许 | **严禁** | 会改类别编号 |
| 几何增强 | 随机翻转裁剪 | 同参数同步做 | 空间必须对齐 |
| 最终 dtype | float32 | int64 | 权重 / CE 的要求 |
| 读后检查 | 值域 [0,1] | 值 ∈ 合法类别集 | 见专题 4.4 |

### 3.3 原理一：为什么插值会毁掉 mask

双线性插值 = 周围 4 个像素的加权平均。看这个可复现的小实验：

```python
import cv2
import numpy as np

small = np.array([[3, 3, 5, 5],
                  [3, 3, 5, 5],
                  [3, 3, 5, 5],
                  [3, 3, 5, 5]], dtype=np.uint8)

wrong = cv2.resize(small, (8, 8))                      # 默认双线性
right = cv2.resize(small, (8, 8),
                   interpolation=cv2.INTER_NEAREST)

print(np.unique(wrong))   # [3 4 5] —— 多出了 4！
print(np.unique(right))   # [3 5]   —— 类别完整保留
```

原图里只有"水体(3)"和"森林(5)"，双线性插值在它们的边界上
平均出了 `4`——**"荒地"这个类别凭空出现了**。对 image，相邻颜色
平均出一个中间色完全合理；对 mask，平均编号等于伪造标签。

### 3.4 原理二：为什么 mask 不能归一化

```python
mask = np.array([6], dtype=np.uint8)      # 类别 6：农业用地
mask.astype(np.float32) / 255.0           # [0.0235...] —— 这是啥？
```

`/255` 对 image 是**单位换算**（0~255 → 0~1，信息无损）；
对 mask 是**编码破坏**（类别 6 变成 0.0235，既不是 6 也不是任何
合法类别）。同样道理，mean/std normalize、亮度抖动、色相偏移
这些"像素算术"都只属于 image。

### 3.5 原理三：为什么 CE 要求 int64

```
logits:  (N, 7, H, W)      每个像素 7 个类得分
mask:    (N, H, W)         每个像素 1 个类别编号

CrossEntropyLoss 内部：
    对每个像素 (n, y, x)：
        编号 c = mask[n, y, x]          ← 拿来当"下标"用
        取   logits[n, c, y, x]         ← 索引第 c 个通道
```

掩码值在 CE 眼里不是"数"，是**下标**。下标必须是整数，且必须在
`0~6` 范围内（255 是被 `ignore_index` 特赦的哨兵）——这正是专题 4.4
冒烟测试第三条 assert 要拦的东西。

### 3.6 三组"错误 vs 正确"代码对照

```python
# ① 归一化
mask = mask.astype(np.float32) / 255.0        # ✗ 类别 6 → 0.0235
mask = mask.astype(np.int64)                  # ✓

# ② 缩放
mask = cv2.resize(mask, (512, 512))           # ✗ 默认双线性，造新类别
mask = cv2.resize(mask, (512, 512),
                  interpolation=cv2.INTER_NEAREST)   # ✓

# ③ 翻转增强
if random.random() < 0.5:
    img = img[:, ::-1]                        # ✗ 只翻图，空间错位
if random.random() < 0.5:                     # ✓ 同一随机数，同步翻
    img, mask = img[:, ::-1], mask[:, ::-1]
```

### 3.7 增强同步的两条纪律

1. **几何类**（crop / flip / rotate / 平移）：一次采样随机参数，
   同一份参数同时作用于 img 和 mask——位置对不齐，等于教模型
   "看着水体说是森林"；
2. **颜色类**（亮度 / 对比度 / 色相 / 模糊）：**只作用 image**，
   mask 碰都不能碰。

写增强时先问自己：这个变换是"改颜色"还是"改位置"？改颜色的，
mask 免死；改位置的，两个一起。

---

## 专题 4：写 Dataset 的完整思路

### 4.1 总原则：先定签名，再倒推每一步

新手常见错误是从"怎么读文件"开始写。正确顺序是反过来的：

```
第 1 步  写死 __getitem__ 的返回签名（终点）
         img:  (3, H, W) float32
         mask: (H, W)    int64
           ↓ 由此倒推
第 2 步  __init__  只建"地址簿"（路径对），不读像素
第 3 步  __len__   返回地址簿长度
第 4 步  __getitem__ 按 6 步数据流填空
第 5 步  用 assert 冒烟测试收尾
```

**为什么 __init__ 不读像素**：3 万张图 × 每张 3 MB，读进内存就是
90 GB。Dataset 的设计哲学是"__init__ 存地址，__getitem__ 用到才读"
（惰性加载），DataLoader 的多进程预取才能按需工作。

### 4.2 `__getitem__` 内部的 6 步数据流

```
① 读      image 走 convert('RGB')；mask 保住单通道索引
② 改值    需要的话：np.where(mask==0, 255, mask)
③ 增强    （可选）几何同步、颜色只给 image
④ 换轴    image: HWC → CHW（transpose/permute）
⑤ 归一化  image: astype(float32)/255（mask 跳过此步）
⑥ 转张量  image → float32；mask → int64
```

第 ①③ 步两类对象分开处理（专题 3），第 ④⑤⑥ 步只对 image 做
"换轴 + 缩放"，mask 原地直达 int64。

### 4.3 完整可运行示例：精简版 LoveDADataset

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LoveDADataset(Dataset):
    def __init__(self, root, split="Train",
                 ignore_index=255, remap_zero=False):
        self.ignore_index = ignore_index
        self.remap_zero = remap_zero
        self.samples = []                       # 地址簿

        for domain in ("Urban", "Rural"):
            img_dir = Path(root) / split / domain / "images_png"
            mask_dir = Path(root) / split / domain / "masks_png"

            imgs = {p.stem: p for p in img_dir.glob("*.png")}
            masks = {p.stem: p for p in mask_dir.glob("*.png")}
            only_img = imgs.keys() - masks.keys()
            only_mask = masks.keys() - imgs.keys()
            if only_img or only_mask:           # 启动即崩，不静默丢
                raise RuntimeError(
                    f"配对失败: 仅图{sorted(only_img)[:3]} "
                    f"仅掩码{sorted(only_mask)[:3]}")

            for stem in sorted(imgs.keys() & masks.keys()):
                self.samples.append((imgs[stem], masks[stem]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ① 读：image 要颜色统一 RGB；mask 只要索引
        with Image.open(img_path) as im:
            img = np.array(im.convert("RGB"))      # (H, W, 3) uint8
        with Image.open(mask_path) as im:
            mask = np.array(im)                    # (H, W) uint8

        # ② 改值（可选）：0（背景/NoData）→ 255（ignore）
        if self.remap_zero:
            mask = np.where(mask == 0,
                            self.ignore_index, mask)
        mask = mask.astype(np.uint8)

        # ③ 增强（预留位置）：几何变换 img/mask 必须同步

        # ④+⑤ image 专用：换轴 + 归一化，一次到位
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1)))

        # ⑥ mask 专用：原地升 int64，无换轴无缩放
        mask = torch.from_numpy(mask.astype(np.int64))

        return img, mask
```

对照 6 步数据流逐行看一遍：每个步骤在代码里都有注释锚点。
`ascontiguousarray` 的原因见 day2 精读笔记（transpose 只换视图
不换内存，`from_numpy` 要求内存连续）。

### 4.4 第 5 步：冒烟测试（进训练前的最后一道闸）

```python
from torch.utils.data import DataLoader

ds = LoveDADataset("D:/Postgraduate_Study/Data/loveda",
                   split="Train", remap_zero=True)
loader = DataLoader(ds, batch_size=2, num_workers=0)   # Win 先 0
imgs, masks = next(iter(loader))

assert imgs.dtype == torch.float32                      # 喂模型
assert imgs.shape == (2, 3, 1024, 1024)                 # N C H W
assert masks.dtype == torch.int64                       # 喂 CE
assert masks.shape == (2, 1024, 1024)                   # 无通道维
legal = set(range(7)) | {255}
assert set(masks.unique().tolist()) <= legal            # 无脏值
print("冒烟测试通过：", len(ds), "个样本")
```

任何一条 assert 失败，都能直接定位到 4.2 数据流的哪一步出了问题——
这就是"检查前置"的价值：错误在训练开始前 1 秒暴露，而不是
loss 变 NaN 之后半天排查。

### 4.5 DataLoader 常用参数速记

| 参数 | 建议值 | 说明 |
|---|---|---|
| `batch_size` | 2~4 起步 | 显存不够就减半 |
| `shuffle` | 训练 True / 验证 False | 打乱样本顺序 |
| `num_workers` | Windows 先 0 | 多进程读图，Win 下调试期易踩坑 |
| `pin_memory` | 有 GPU 则 True | 加速 H2D 拷贝 |
| `drop_last` | 训练 True | 丢不满的尾批，防 BN 报错 |

### 4.6 Dataset 常见坑

| 坑 | 原因 | 解法 |
|---|---|---|
| mask 也 `/255` | 和 image 写成对称了 | mask 只升 int64 |
| `__init__` 里读像素 | 内存爆炸 | 只存路径对 |
| mask 被 `convert('RGB')` | 复制了 image 的读法 | mask 不 convert，直接 `np.array` |
| 忘了同步增强 | 翻转只作用 image | 随机参数采样一次，两边共用 |
| 悄悄丢样本 | zip 截断 / 差集没检查 | 配对差集必须 raise |
| Windows 多进程卡死 | `num_workers>0` + 脚本无 `main` 保护 | 调试期 `num_workers=0` |

---

## 专题 5：img 与 mask 配对方法论

### 5.1 配对的本质：找公共键，而不是对顺序

"配对"要回答的问题是：**哪两个文件描述的是同一块地面？**
做法是把两边各自建成 `{键: 路径}` 字典，再用**键的交集**对齐。
绝对不要用"列表下标对齐"：

```python
imgs = sorted(img_dir.glob("*.png"))
masks = sorted(mask_dir.glob("*.png"))
pairs = list(zip(imgs, masks))     # ✗ 危险
```

原因：`glob` 的返回顺序依赖文件系统，两边数量不等时 `zip` 会
**静默截断**，少 10 个样本没有报错、指标悄悄失真——这是最阴险的
一类 bug。字典 + 差集的做法能把问题在启动阶段炸出来。

### 5.2 六步标准流程

```
① 两边各自 glob，转成 {stem: path} 字典
② 键选 stem（剥掉扩展名，兼容 .png/.PNG/.jpg）
③ 交集 imgs.keys() & masks.keys()  —— 合法样本
④ 差集只报警不吞：任意一边非空 → raise
⑤ sorted(交集)  —— 顺序与文件系统无关，跨运行可复现
⑥ 逐对二次校验尺寸（软配对陷阱的最后一道保险）
```

**为什么键选 `stem` 而不是 `name`**：image 和 mask 常常扩展名不同
（`42.png` 万一被转成 `42.jpg`），`stem` 剥掉后缀，天然容错。
`name` 一旦后缀不同就全部配对失败。

### 5.3 完整示例：通用配对函数

```python
from pathlib import Path

def scan_pairs(img_dir: Path, mask_dir: Path):
    """返回 [(img_path, mask_path), ...]，配不齐直接抛异常"""
    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}

    only_img = sorted(imgs.keys() - masks.keys())
    only_mask = sorted(masks.keys() - imgs.keys())
    if only_img or only_mask:
        raise RuntimeError(
            f"仅图无掩码: {only_img[:5]} | "
            f"仅掩码无图: {only_mask[:5]}")

    pairs = [(imgs[s], masks[s])
             for s in sorted(imgs.keys() & masks.keys())]
    return pairs

pairs = scan_pairs(
    Path("D:/Postgraduate_Study/Data/loveda/Train/Urban/images_png"),
    Path("D:/Postgraduate_Study/Data/loveda/Train/Urban/masks_png"))
print(f"配对成功 {len(pairs)} 对，前 3 对：")
for img_p, mask_p in pairs[:3]:
    print(f"  {img_p.name}  ↔  {mask_p.name}")
```

两个设计点：

- **报错信息带 `[:3]` 样例**：启动失败时直接看到是哪几个文件
  没配上，而不是一个"长度不等"的抽象报错；
- **`sorted(交集)`**：即使文件系统返回顺序变化，样本顺序也稳定，
  训练可复现。

### 5.4 变体场景速查

| 场景 | 键的写法 |
|---|---|
| 两边扩展名不同 | `p.stem`（默认方案） |
| 大小写混乱 | `p.stem.lower()`（需防键冲突） |
| 跨多级子目录 | 拼相对路径字符串做键 |
| 文件名带域前缀 | 剥前缀后再比较 |

大小写方案注意：如果同时存在 `A.png` 和 `a.png`，
`p.stem.lower()` 会让两个文件抢同一个键，后者覆盖前者。严谨做法是
建字典前先检查重复键。

### 5.5 配对之后：尺寸二次校验

"文件名对上了"不等于"内容对上了"——可能存在重采样过的半成品图：

```python
mismatched = []
for stem in sorted(imgs.keys() & masks.keys()):
    with Image.open(imgs[stem]) as a, Image.open(masks[stem]) as b:
        if a.size != b.size:
            mismatched.append((stem, a.size, b.size))

# img.size 是 (宽, 高)，mask 同为 (1024, 1024) 才合法
```

这正是 data_check.py 里 `check_pairing` 的第三项输出。
数据检查的完整闭环：
**键配对（结构层） → 尺寸校验（形状层） → 像素直方图（数值层）**，
一层比一层深入。

---

## 附录：NumPy / Tensor 速查卡

同一件事的两个写法，混用记不住时查这张表：

| 操作 | NumPy | PyTorch |
|---|---|---|
| 通道转置 | `arr.transpose(2, 0, 1)` | `t.permute(2, 0, 1)` |
| 增加维度 | `arr[None]` | `t.unsqueeze(0)` |
| 去掉维度 | `arr.squeeze()` | `t.squeeze()` |
| 沿维拼接 | `np.concatenate(...)` | `torch.cat(...)` |
| 堆叠成批 | `np.stack(...)` | `torch.stack(...)` |
| 复制 | `arr.copy()` | `t.clone()` |
| 类型转换 | `arr.astype(...)` | `t.float() / t.long()` |
| 最大值位置 | `arr.argmax()` | `t.argmax(dim=)` |
| 内存连续 | `np.ascontiguousarray` | `t.contiguous()` |
| ndarray ↔ tensor | — | `from_numpy` / `t.numpy()` |

三个"形同意不同"的陷阱：

| 词汇 | NumPy 语义 | PyTorch 语义 |
|---|---|---|
| `transpose` | 可传轴序列 `transpose(2,0,1)` | 只换两轴，多轴用 `permute` |
| `view` | 不存在此方法 | 重排维度（要求连续），不是"看一眼" |
| `long()` | 不存在 | 转 int64，不是"变长" |
