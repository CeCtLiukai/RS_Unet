# 遥感语义分割 Dataset 编写学习路线

## 1. 学习目标

本教程的目标不是直接背诵一个复杂的 `LoveDADataset`，而是理解一个遥感语义分割数据集如何从最小版本逐步完善为可以支撑论文实验的成熟版本。

最终希望能够独立完成以下工作：

- 根据目录结构找到原图和标签；
- 判断原图与标签是否一一对应；
- 正确读取 RGB 原图和单通道掩码；
- 检查图像尺寸、通道数和标签值；
- 将图像和掩码转换为 PyTorch Tensor；
- 对原图和掩码执行同步几何增强；
- 区分训练集、验证集和测试集；
- 使用 `DataLoader` 批量加载；
- 在训练前完成数据质量检查；
- 读懂公开的 MMSegmentation、TorchGeo 等大型项目中的 Dataset 设计。

你上传的 `Path_practice.py`、`Image_practice.py`、`PLT_practice.py`、`data_check.py`、`transform.py` 和 `loveda_dataset.py`，正好可以作为这条学习路线的配套材料。

---

## 2. 语义分割 Dataset 的基本概念

### 2.1 分类、目标检测和语义分割的区别

图像分类通常返回：

```text
image -> 一个类别
```

目标检测通常返回：

```text
image -> 多个目标框和类别
```

语义分割返回：

```text
image -> 每一个像素的类别
```

遥感语义分割中的一个样本通常包含：

```text
原图 image：H × W × 3
标签 mask：H × W
```

例如，一张遥感图像中：

```text
0：背景
1：建筑
2：道路
3：水体
4：森林
```

标签图中的像素值不是普通的颜色，而是类别编号：

```python
mask[100, 200] == 2
```

表示坐标 `(100, 200)` 的像素属于第 2 类。

### 2.2 为什么原图和掩码必须同步处理

如果原图裁剪了左上角区域，而掩码裁剪了右下角区域，二者虽然尺寸相同，但内容已经错位。模型会把错误的像素和错误的类别对应起来。

因此必须满足：

```text
相同的裁剪框
相同的缩放比例
相同的翻转方向
相同的旋转角度
```

但颜色增强只能作用于原图：

```text
几何增强：image 和 mask 都做
颜色增强：只对 image 做
```

### 2.3 为什么掩码不能使用普通图像处理方式

原图可以使用双线性插值：

```python
Image.Resampling.BILINEAR
```

掩码必须使用最近邻插值：

```python
Image.Resampling.NEAREST
```

因为掩码的 `1` 和 `2` 是类别编号。若对掩码使用双线性插值，可能产生 `1.5` 之类的非法类别。

---

## 3. Dataset 的最小组成

PyTorch 中最基本的 Dataset 只需要实现两个方法：

```python
__len__()
__getitem__(index)
```

其中：

- `__len__()`：返回数据集样本数量；
- `__getitem__()`：根据索引返回一个样本。

最小数据目录如下：

```text
data/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── 0003.png
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── 0003.png
```

原图和掩码通过相同的文件名配对。

---

## 4. 第一个基础例子：读取一个样本

先不要急着写 Dataset，先理解一个样本是如何读取的。

```python
from pathlib import Path
from PIL import Image
import numpy as np

image_path = Path("data/images/0001.png")
mask_path = Path("data/masks/0001.png")

with Image.open(image_path) as image:
    image = image.convert("RGB")
    image_array = np.array(image)

with Image.open(mask_path) as mask:
    mask = mask.convert("L")
    mask_array = np.array(mask)

print("image shape:", image_array.shape)
print("mask shape:", mask_array.shape)
print("image dtype:", image_array.dtype)
print("mask dtype:", mask_array.dtype)
print("mask values:", np.unique(mask_array))
```

典型输出如下：

```text
image shape: (1024, 1024, 3)
mask shape: (1024, 1024)
image dtype: uint8
mask dtype: uint8
mask values: [0 1 2 3 4 5 6]
```

这里需要掌握：

- PIL 的 `size` 顺序是 `(width, height)`；
- NumPy 的 `shape` 顺序是 `(height, width, channel)`；
- 原图通常是三维数组；
- 掩码通常是二维数组；
- 掩码必须保留整数类别值。

---

## 5. 第二个基础例子：最小 Dataset

```python
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MiniSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask
```

测试：

```python
dataset = MiniSegDataset("data/images", "data/masks")

print("dataset length:", len(dataset))
image, mask = dataset[0]
print("image:", image.shape, image.dtype)
print("mask:", mask.shape, mask.dtype)
```

期望结果：

```text
image: torch.Size([3, 1024, 1024]) torch.float32
mask: torch.Size([1024, 1024]) torch.int64
```

此时已经可以被模型使用，但仍然不够可靠，因为它没有检查文件是否配对，也没有检查尺寸是否一致。

---

## 6. 从最小 Dataset 到成熟 Dataset 的演进

### 6.1 第一步：规范路径

使用 `pathlib.Path` 管理路径：

```python
root = Path("data/loveda")
image_dir = root / "Train" / "Urban" / "images_png"
mask_dir = root / "Train" / "Urban" / "masks_png"
```

检查目录：

```python
if not image_dir.is_dir():
    raise FileNotFoundError(f"目录不存在：{image_dir}")
```

注意，下面的写法不可靠：

```python
if not image_dir:
    ...
```

`Path` 对象本身不等于“路径存在”。应使用 `.exists()` 或 `.is_dir()`。

### 6.2 第二步：兼容大小写后缀

不要只写：

```python
directory.glob("*.PNG")
```

更稳妥的写法：

```python
image_paths = [
    p for p in directory.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
]
```

这可以同时识别：

```text
image.png
image.PNG
image.jpg
image.JPG
```

### 6.3 第三步：通过 stem 建立配对

```python
images = {
    p.stem: p
    for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() == ".png"
}

masks = {
    p.stem: p
    for p in mask_dir.iterdir()
    if p.is_file() and p.suffix.lower() == ".png"
}

only_images = sorted(set(images) - set(masks))
only_masks = sorted(set(masks) - set(images))

if only_images or only_masks:
    raise RuntimeError(
        f"图像和掩码不匹配：只存在图像={only_images[:5]}，"
        f"只存在掩码={only_masks[:5]}"
    )

samples = [
    (images[stem], masks[stem])
    for stem in sorted(images)
]
```

这里的关键是：

```python
p.stem
```

它只表示不带扩展名的文件名，例如：

```text
0001.PNG -> 0001
```

### 6.4 第四步：检查原图与掩码尺寸

```python
from PIL import Image

for image_path, mask_path in samples:
    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as mask:
        mask_size = mask.size

    if image_size != mask_size:
        raise ValueError(
            f"尺寸不一致：{image_path.name}，"
            f"image={image_size}，mask={mask_size}"
        )
```

建议在 Dataset 初始化时检查，而不是等到训练中才发现问题。

### 6.5 第五步：检查标签值

```python
valid_labels = {0, 1, 2, 3, 4, 5, 6, 255}

values = set(np.unique(mask_array).tolist())
invalid = values - valid_labels

if invalid:
    raise ValueError(f"发现非法标签值：{invalid}")
```

如果数据集规定 `0` 是未标注区域，也可以将它转换为损失函数使用的忽略值：

```python
mask = np.where(mask == 0, 255, mask).astype(np.int64)
```

但是否转换必须根据数据集官方定义决定，不能机械套用。

---

## 7. 同步数据增强

### 7.1 同步水平翻转

```python
import random
from PIL import Image


def random_flip(image, mask, hflip=0.5, vflip=0.5):
    if random.random() < hflip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < vflip:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image, mask
```

注意：虽然水平翻转和垂直翻转分别生成随机数，但每一次决定都必须同时作用于 image 和 mask。

### 7.2 同步随机裁剪

```python
import random


def random_crop(image, mask, crop_size):
    if image.size != mask.size:
        raise ValueError("原图和掩码尺寸不一致")

    width, height = image.size

    if width < crop_size or height < crop_size:
        raise ValueError(
            f"图像尺寸 {width}×{height} 小于裁剪尺寸 {crop_size}"
        )

    x1 = random.randint(0, width - crop_size)
    y1 = random.randint(0, height - crop_size)
    box = (x1, y1, x1 + crop_size, y1 + crop_size)

    return image.crop(box), mask.crop(box)
```

核心原则是：只生成一次 `box`。

错误写法是分别对原图和掩码调用两个独立的随机裁剪函数，因为它们可能得到不同的位置。

### 7.3 缩放时区分插值方式

```python
image = image.resize(
    (512, 512),
    Image.Resampling.BILINEAR
)

mask = mask.resize(
    (512, 512),
    Image.Resampling.NEAREST
)
```

### 7.4 图像颜色增强

颜色增强不能作用于掩码：

```python
from PIL import ImageEnhance

image = ImageEnhance.Brightness(image).enhance(1.1)
```

掩码不能执行亮度、对比度或颜色抖动，否则类别编号会被改变。

---

## 8. 训练、验证和测试模式

训练集通常允许：

```text
随机缩放
随机裁剪
随机翻转
颜色增强
归一化
```

验证集通常只执行：

```text
固定缩放或固定裁剪
归一化
```

一个简单的 Transform 结构如下：

```python
class SegTransform:
    def __init__(self, crop_size=512, training=True):
        self.crop_size = crop_size
        self.training = training

    def __call__(self, image, mask):
        if self.training:
            image, mask = random_crop(
                image, mask, self.crop_size
            )
            image, mask = random_flip(image, mask)
        else:
            image = image.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.NEAREST
            )

        return image, mask
```

你上传的 `transform.py` 已经进一步加入了：

- 随机缩放；
- 随机裁剪；
- 水平翻转；
- 垂直翻转；
- 亮度和对比度增强；
- 均值方差归一化。

---

## 9. 从单域 Dataset 到 LoveDA 多域 Dataset

最简单的数据集只有：

```text
image_dir
mask_dir
```

LoveDA 还包含：

```text
Train / Val
Urban / Rural
```

因此一个样本可以表示为：

```python
(image_path, mask_path, domain)
```

目录结构：

```text
root/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Val/
    ├── Urban/
    └── Rural/
```

成熟 Dataset 的扫描函数可以写成：

```python
def scan(root, split, domains):
    samples = []

    for domain in domains:
        image_dir = root / split / domain / "images_png"
        mask_dir = root / split / domain / "masks_png"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue

        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        }

        masks = {
            p.stem: p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        }

        if set(images) != set(masks):
            raise RuntimeError(f"{split}/{domain} 图文不匹配")

        for stem in sorted(images):
            samples.append((images[stem], masks[stem], domain))

    return samples
```

你现在的 `loveda_dataset.py` 中的 `_scan()` 就是这一思路的完整实现。

---

## 10. 一个成熟版 Dataset 的核心框架

```python
class RemoteSensingSegDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        domains=("Urban", "Rural"),
        transform=None,
        ignore_index=255,
        remap_zero_to_ignore=False,
    ):
        self.root = Path(root)
        self.split = split
        self.domains = list(domains)
        self.transform = transform
        self.ignore_index = int(ignore_index)
        self.remap_zero_to_ignore = bool(remap_zero_to_ignore)
        self.samples = self._scan()

        if not self.samples:
            raise RuntimeError("没有找到任何有效样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, domain = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        with Image.open(mask_path) as mask:
            mask = np.array(mask.convert("L"), dtype=np.uint8)

        if self.remap_zero_to_ignore:
            mask = np.where(
                mask == 0,
                self.ignore_index,
                mask
            ).astype(np.int64)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
            mask = torch.from_numpy(mask.astype(np.int64))

        meta = {
            "stem": image_path.stem,
            "domain": domain,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

        return image, mask, meta
```

成熟 Dataset 的 `__getitem__()` 不只是“读图”，还需要保证：

```text
文件可读
原图与掩码对应
尺寸正确
标签合法
增强同步
数据类型正确
形状正确
元信息可追踪
```

---

## 11. DataLoader 与 batch 检查

Dataset 负责单个样本，DataLoader 负责批量读取：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
```

训练前必须先检查一个 batch：

```python
images, masks, metas = next(iter(loader))

print(images.shape)
print(masks.shape)
print(images.dtype)
print(masks.dtype)
```

典型结果：

```text
images: torch.Size([4, 3, 512, 512])
masks: torch.Size([4, 512, 512])
images.dtype: torch.float32
masks.dtype: torch.int64
```

验证集通常使用：

```python
shuffle=False
drop_last=False
```

验证集不能随意丢弃最后一个 batch，否则可能影响评价结果。

---

## 12. 数据质量检查流程

一个成熟项目通常将数据检查单独写成脚本，而不是全部塞进 Dataset。

推荐检查顺序：

```text
目录是否存在
    ↓
文件是否可读
    ↓
原图和掩码是否配对
    ↓
尺寸是否一致
    ↓
标签值是否合法
    ↓
原图和掩码是否可视化对齐
    ↓
DataLoader 是否能正常取 batch
```

### 12.1 文件可读性检查

```python
from PIL import Image


def check_file(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True, "正常"
    except Exception as error:
        return False, repr(error)
```

调用 `verify()` 后，如果还要读取像素，必须重新打开：

```python
with Image.open(path) as image:
    image.verify()

with Image.open(path) as image:
    array = np.array(image)
```

### 12.2 标签统计

```python
from collections import Counter

counter = Counter()

for mask_path in mask_paths:
    mask = np.array(Image.open(mask_path).convert("L"))
    values, counts = np.unique(mask, return_counts=True)
    counter.update(dict(zip(values.tolist(), counts.tolist())))

print(counter)
```

### 12.3 可视化抽检

最少展示三类图：

```text
原图
掩码
原图叠加掩码
```

```python
plt.imshow(image)
plt.imshow(mask, cmap="tab10", alpha=0.4)
plt.axis("off")
plt.show()
```

你上传的 `data_check.py` 已经实现了：

- 文件可读性检查；
- 图像与掩码配对检查；
- 尺寸一致性检查；
- 标签像素值统计；
- 原图、掩码和零标签区域的可视化。

---

## 13. 常见错误及原因

### 错误 1：使用 `if not path` 判断路径是否存在

错误：

```python
if not img_dir:
    raise FileNotFoundError(...)
```

正确：

```python
if not img_dir.exists():
    raise FileNotFoundError(...)
```

### 错误 2：直接比较 Path 对象和字符串

错误：

```python
set(image_paths) - set(mask_paths)
```

如果两个目录不同，即使文件名相同，也会被认为不同。

正确：

```python
image_names = {p.stem for p in image_paths}
mask_names = {p.stem for p in mask_paths}
```

### 错误 3：将掩码转换为 RGB

掩码一般应使用：

```python
mask = Image.open(mask_path).convert("L")
```

不要把标签图当作普通彩色照片处理。

### 错误 4：掩码使用双线性插值

错误：

```python
mask.resize(size, Image.Resampling.BILINEAR)
```

正确：

```python
mask.resize(size, Image.Resampling.NEAREST)
```

### 错误 5：原图和掩码分别随机裁剪

必须共享同一个随机裁剪框。

### 错误 6：验证集使用随机增强

验证集应该保持稳定的预处理流程，否则每轮验证结果可能不同。

### 错误 7：`verify()` 后继续使用同一个 PIL 对象

`verify()` 用于检查文件结构，之后需要重新打开文件。

### 错误 8：忘记转换通道顺序

OpenCV 默认读取为 BGR：

```python
image = cv2.imread(str(path), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

PIL 默认读取为 RGB，二者不能混用而不检查通道顺序。

---

## 14. 编程训练题

建议按顺序完成，不要直接跳到最后一道。

### 题目 1：读取单个样本

编写：

```python
def load_single_sample(image_path, mask_path):
    ...
```

要求：

1. 使用 PIL 读取原图和掩码；
2. 原图转换为 RGB；
3. 掩码转换为灰度；
4. 转换为 NumPy 数组；
5. 返回数组和标签唯一值。

### 题目 2：检查文件完整性

编写：

```python
def check_image_mask(image_path, mask_path):
    ...
```

要求：

- 检查文件是否存在；
- 使用 `verify()`；
- 检查图像尺寸；
- 返回 `valid`、尺寸和错误信息。

### 题目 3：获取所有文件 stem

编写：

```python
def collect_stems(directory):
    ...
```

要求支持 `.png`、`.PNG`、`.jpg` 和 `.JPG`，返回排序后的不带扩展名文件名。

### 题目 4：查找未配对文件

编写：

```python
def find_unpaired_files(image_dir, mask_dir):
    ...
```

返回：

```python
only_images, only_masks
```

### 题目 5：实现最小 Dataset

实现 `MiniSegDataset`，要求：

- 支持 `__len__()`；
- 支持 `__getitem__()`；
- 图像返回 `[C, H, W]` 的 `float32` Tensor；
- 掩码返回 `[H, W]` 的 `long` Tensor。

### 题目 6：初始化时检查尺寸

在 Dataset 初始化阶段检查所有原图与掩码的尺寸，如果不一致立即报错，并指出文件名。

### 题目 7：同步翻转

实现：

```python
def random_flip(image, mask, hflip=0.5, vflip=0.5):
    ...
```

要求原图和掩码的翻转方向始终一致。

### 题目 8：同步随机裁剪

实现：

```python
def random_crop(image, mask, crop_size):
    ...
```

要求原图和掩码共享同一个裁剪框。

### 题目 9：训练和验证 Transform

实现一个 `SegTransform`，训练时随机裁剪和翻转，验证时固定缩放。

### 题目 10：标签像素统计

实现：

```python
def count_mask_values(mask_paths, sample=None):
    ...
```

返回各标签的总像素数和占比。

### 题目 11：非法标签检查

实现：

```python
def check_label_values(mask_path, valid_labels):
    ...
```

找出所有非法标签值，并返回其文件路径。

### 题目 12：完整遥感分割 Dataset

综合实现：

```python
class RemoteSensingSegDataset(Dataset):
    ...
```

至少支持：

```python
root
split
domains
transform
ignore_index
remap_zero_to_ignore
```

返回：

```python
image, mask, meta
```

`meta` 至少包含：

```python
{
    "stem": ..., 
    "domain": ..., 
    "image_path": ..., 
    "mask_path": ...
}
```

---

## 15. 开源项目代码学习路线

下面的项目适合在完成前 12 道练习后阅读。建议先看目录结构和 Dataset 注册方式，再看数据增强和数据检查，不要一开始就通读整个项目。

### 15.1 LoveDA 官方代码：NeurIPS 2021

LoveDA 是发表在 NeurIPS 2021 的遥感土地覆盖域自适应语义分割数据集。官方仓库：

- [Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [LoveDA 语义分割代码说明](https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation)

建议重点阅读：

- 数据目录结构；
- Urban/Rural 域划分；
- 训练和验证数据准备；
- 标签类别定义；
- 与 MMSegmentation 或 segmentation_models.pytorch 的衔接方式。

它最适合帮助你理解：

```text
一个具体遥感数据集如何进入完整实验工程
```

### 15.2 MMSegmentation：大型语义分割框架

官方仓库：

- [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMSegmentation Dataset 文档](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/datasets.html)

MMSegmentation 中 Dataset 的基本职责包括：

1. 加载样本信息；
2. 将图像路径和标注路径组织成统一的数据结构；
3. 将样本送入 transform pipeline；
4. 对类别映射、忽略标签和评估格式进行统一管理。

MMSegmentation 的典型思想是：

```text
Dataset 类：管理样本信息
Transform pipeline：管理数据处理
DataLoader：管理批量读取
Evaluator：管理指标计算
```

这比把所有逻辑都写在 `__getitem__()` 里更适合大型项目。

MMSegmentation 的 Dataset 文档还展示了常见目录结构：

```text
my_dataset/
├── img_dir/
│   ├── train/
│   └── val/
└── ann_dir/
    ├── train/
    └── val/
```

建议重点看：

- `BaseSegDataset`；
- `load_data_list()`；
- `img_suffix`；
- `seg_map_suffix`；
- `reduce_zero_label`；
- 自定义数据集注册机制。

### 15.3 TorchGeo：面向地理空间数据的框架

官方仓库：

- [torchgeo/torchgeo](https://github.com/microsoft/torchgeo)
- [TorchGeo Dataset 文档](https://docs.torchgeo.org/en/stable/api/datasets.html)

TorchGeo 不只处理普通 PNG 文件，还考虑：

- 地理坐标参考系统 CRS；
- 空间范围 bounding box；
- 栅格影像；
- 矢量标签；
- 空间采样；
- 不同影像之间的地理对齐。

它适合在你掌握普通 `Path + PIL + Dataset` 后学习，因为它说明了成熟遥感数据集还会进一步处理：

```text
像素坐标之外的地理坐标
```

TorchGeo 的语义分割任务文档：

- [TorchGeo Semantic Segmentation Task](https://torchgeo.readthedocs.io/en/stable/api/tasks.html)

建议重点观察：

- Dataset 与 Task 如何分离；
- 训练、验证、测试划分如何定义；
- 空间采样器如何生成 patch；
- 多源遥感数据如何统一输入。

### 15.4 公开研究代码的阅读方法

阅读顶会或高水平论文代码时，不要从模型开始。建议顺序是：

```text
README
  ↓
数据准备脚本
  ↓
Dataset 类
  ↓
Transform
  ↓
DataLoader
  ↓
训练入口
  ↓
模型
```

重点记录下面这些问题：

| 问题 | 需要找到的代码 |
|---|---|
| 数据从哪里读取 | `Dataset` 或数据准备脚本 |
| 原图和标签如何配对 | `stem`、文件名或索引文件 |
| 标签是否重映射 | `label_map`、`reduce_zero_label` |
| 训练增强在哪里 | transform pipeline |
| 验证是否随机 | train/val pipeline 配置 |
| patch 如何生成 | crop、tiling 或 sampler |
| 预测结果如何恢复 | sliding window、merge、stitch |
| 指标如何计算 | evaluator 或 metric 类 |

---

## 16. 顶会和大型开源代码中的共同设计思想

不同项目的语法可能不同，但核心思想高度一致：

### 16.1 数据读取与数据增强分离

不要把所有内容写成一个巨大函数：

```text
Dataset：读取和组织样本
Transform：处理样本
DataLoader：批量读取
Evaluator：计算指标
```

### 16.2 初始化阶段完成静态检查

以下内容不应在每个训练 batch 中重复检查：

- 目录是否存在；
- 文件名是否配对；
- 数据集是否为空；
- 静态尺寸是否合理。

这些内容应在初始化或训练前完成。

### 16.3 动态处理放在 `__getitem__()` 或 Transform

以下内容可以在每次取样时执行：

- 随机裁剪；
- 随机翻转；
- 随机缩放；
- 颜色增强。

### 16.4 元信息必须保留

成熟项目通常不会只返回图像和标签，还会保留：

```python
{
    "filename": ...,
    "ori_shape": ...,
    "domain": ...,
    "geospatial_info": ...
}
```

这样预测结果保存、错误样本定位和跨域分析都会更方便。

### 16.5 训练前必须可视化

单元测试只能说明代码能运行，不能说明图像和掩码对齐。

因此每次改变以下内容后，都应重新可视化：

- resize；
- crop；
- flip；
- label remap；
- mask colorization；
- patch 拼接。

---

## 17. 推荐学习安排

### 第一阶段：基础读取

完成题目 1～4，重点掌握：

```text
Path
PIL.Image
NumPy
文件配对
```

### 第二阶段：最小 Dataset

完成题目 5～6，重点掌握：

```text
Dataset
__len__
__getitem__
Tensor 形状
Tensor 数据类型
```

### 第三阶段：同步增强

完成题目 7～9，重点掌握：

```text
共同随机参数
图像插值
掩码插值
训练/验证模式
```

### 第四阶段：数据质量控制

完成题目 10～11，重点掌握：

```text
标签统计
非法标签检测
可视化检查
```

### 第五阶段：成熟工程实现

完成题目 12，再阅读：

```text
LoveDA 官方代码
MMSegmentation Dataset
TorchGeo Dataset
```

---

## 18. 最终检查清单

在开始遥感分割训练前，至少检查以下内容：

- [ ] 数据根目录存在；
- [ ] Train、Val、Test 目录名称正确；
- [ ] 原图和掩码数量一致；
- [ ] 原图和掩码文件名能够配对；
- [ ] 所有文件都可以正常读取；
- [ ] 原图和掩码尺寸一致；
- [ ] 原图是 RGB 或正确的多波段格式；
- [ ] 掩码是单通道类别图；
- [ ] 标签值全部合法；
- [ ] 掩码缩放使用最近邻插值；
- [ ] 原图和掩码使用相同的裁剪和翻转参数；
- [ ] 图像 Tensor 是 `[C, H, W]`；
- [ ] 掩码 Tensor 是 `[H, W]`；
- [ ] 图像是 `float32`；
- [ ] 掩码是 `long`；
- [ ] 一个 batch 能正常取出；
- [ ] 原图、掩码和叠加图已经可视化确认；
- [ ] 训练集和验证集处理逻辑不同；
- [ ] 预测结果能够追溯到原始文件名。

当这些检查全部通过后，Dataset 才可以认为达到了比较可靠的实验使用标准。
