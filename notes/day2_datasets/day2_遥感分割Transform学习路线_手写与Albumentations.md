# 遥感语义分割 Transform 学习路线：先手写，再使用 Albumentations

## 1. 学习目标

在遥感语义分割中，Dataset 负责找到并读取样本，Transform 负责把原始样本处理成模型能够使用的输入。

完整流程是：

```text
原始 image + mask
        ↓
同步几何增强
        ↓
图像专用增强
        ↓
尺寸调整与归一化
        ↓
转换为 Tensor
        ↓
送入模型
```

本教程采用和 Dataset 相同的学习方式：

```text
先理解原理
    ↓
手写最小版本
    ↓
手写完整版本
    ↓
接入 Dataset
    ↓
使用 Albumentations 调包
    ↓
完成训练前检查
```

这里使用的库是 **Albumentations**。它专门用于图像增强，并且能够自动保证图像和分割掩码使用同一组随机几何参数。

官方语义分割文档：

- [Albumentations Semantic Segmentation](https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/)
- [Albumentations Additional Targets](https://albumentations.ai/docs/4-advanced-guides/additional-targets/)

---

## 2. Transform 的基础知识

### 2.1 什么是几何增强

几何增强会改变像素的空间位置，例如：

- 缩放；
- 裁剪；
- 水平翻转；
- 垂直翻转；
- 旋转；
- 仿射变换。

这些操作必须同时应用于原图和掩码：

```text
image 和 mask 使用同一个裁剪框
image 和 mask 使用同一个翻转方向
image 和 mask 使用同一个旋转角度
```

### 2.2 什么是图像专用增强

图像专用增强只改变原图的像素值，不改变掩码的类别编号，例如：

- 亮度变化；
- 对比度变化；
- 色彩抖动；
- 高斯噪声；
- 模糊；
- JPEG 压缩模拟。

例如：

```text
原图：可以改变亮度
掩码：不能改变亮度
```

因为掩码中的 `0、1、2、3` 是类别编号，不是普通颜色。

### 2.3 为什么掩码必须使用最近邻插值

原图可以使用双线性插值：

```python
Image.Resampling.BILINEAR
```

掩码必须使用最近邻插值：

```python
Image.Resampling.NEAREST
```

假设掩码相邻像素为：

```text
0 和 2
```

如果使用双线性插值，边界可能产生：

```text
0.5、1.0、1.5
```

这些值不一定代表合法类别。最近邻插值只会保留原有类别值。

---

## 3. 第一阶段：手写 Transform

## 3.1 第一个基础例子：只转换 Tensor

```python
import numpy as np
import torch


class BasicTransform:
    def __call__(self, image, mask):
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )
        mask = torch.from_numpy(mask)

        return image, mask
```

输入和输出：

```text
原图：H × W × 3
        ↓
Tensor：3 × H × W

掩码：H × W
        ↓
Tensor：H × W
```

注意：

- 原图通常是 `float32`；
- 掩码通常是 `int64` 或 PyTorch 中的 `long`；
- 原图需要从 `[0, 255]` 转换到 `[0, 1]`；
- 掩码不能除以 255。

---

## 3.2 第二个基础例子：图像和掩码缩放

```python
from PIL import Image


def resize_image_and_mask(image, mask, size):
    image = image.resize(
        size,
        Image.Resampling.BILINEAR
    )

    mask = mask.resize(
        size,
        Image.Resampling.NEAREST
    )

    return image, mask
```

测试：

```python
image, mask = resize_image_and_mask(
    image,
    mask,
    (512, 512)
)

print(image.size)
print(mask.size)
```

这里原图和掩码的目标尺寸相同，但插值方法不同。

---

## 3.3 第三个基础例子：同步水平翻转

```python
from PIL import Image


def horizontal_flip(image, mask):
    image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    mask = mask.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    return image, mask
```

不能只翻转原图：

```python
image = image.transpose(...)
```

因为这样会造成图像和标签错位。

---

## 3.4 第四个基础例子：同步随机翻转

```python
import random
from PIL import Image


def random_flip(image, mask, hflip=0.5, vflip=0.5):
    if random.random() < hflip:
        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
        mask = mask.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    if random.random() < vflip:
        image = image.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        mask = mask.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

    return image, mask
```

重要的是每一次随机结果都同时用于 image 和 mask。

错误写法：

```python
if random.random() < 0.5:
    image = flip(image)

if random.random() < 0.5:
    mask = flip(mask)
```

这两个随机数可能不同。

---

## 3.5 第五个基础例子：同步随机裁剪

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

    box = (
        x1,
        y1,
        x1 + crop_size,
        y1 + crop_size,
    )

    image = image.crop(box)
    mask = mask.crop(box)

    return image, mask
```

核心思想：

```python
只生成一次 box
image 和 mask 共用同一个 box
```

---

## 3.6 第六个基础例子：图像颜色增强

```python
import random
from PIL import ImageEnhance


def color_jitter(image, mask, strength=0.2):
    brightness_factor = random.uniform(
        1 - strength,
        1 + strength
    )
    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness_factor)

    contrast_factor = random.uniform(
        1 - strength,
        1 + strength
    )
    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast_factor)

    return image, mask
```

颜色增强只对 image 执行。mask 只是原样返回。

---

## 3.7 手写一个完整 Transform

```python
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance


class HandWrittenSegTransform:
    def __init__(
        self,
        crop_size=512,
        training=True,
        hflip=0.5,
        vflip=0.5,
        color_jitter=0.0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.crop_size = int(crop_size)
        self.training = bool(training)
        self.hflip = float(hflip)
        self.vflip = float(vflip)
        self.color_jitter = float(color_jitter)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, mask):
        if self.training:
            image, mask = self.random_crop(image, mask)
            image, mask = self.random_flip(image, mask)

            if self.color_jitter > 0:
                image = self.jitter(image)
        else:
            image = image.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.BILINEAR,
            )
            mask = mask.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.NEAREST,
            )

        return self.to_tensor(image, mask)

    def random_crop(self, image, mask):
        if image.size != mask.size:
            raise ValueError("原图与掩码尺寸不一致")

        width, height = image.size
        size = self.crop_size

        if width < size or height < size:
            raise ValueError("图像尺寸小于 crop_size")

        x1 = random.randint(0, width - size)
        y1 = random.randint(0, height - size)
        box = (x1, y1, x1 + size, y1 + size)

        return image.crop(box), mask.crop(box)

    def random_flip(self, image, mask):
        if random.random() < self.hflip:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if random.random() < self.vflip:
            image = image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        return image, mask

    def jitter(self, image):
        factor = random.uniform(
            1 - self.color_jitter,
            1 + self.color_jitter,
        )
        image = ImageEnhance.Brightness(image).enhance(factor)

        factor = random.uniform(
            1 - self.color_jitter,
            1 + self.color_jitter,
        )
        image = ImageEnhance.Contrast(image).enhance(factor)

        return image

    def to_tensor(self, image, mask):
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std

        image = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        )

        mask = torch.from_numpy(
            np.array(mask, dtype=np.int64)
        )

        return image, mask
```

使用：

```python
train_transform = HandWrittenSegTransform(
    crop_size=512,
    training=True,
    hflip=0.5,
    vflip=0.5,
    color_jitter=0.2,
)

val_transform = HandWrittenSegTransform(
    crop_size=512,
    training=False,
)
```

---

## 4. 手写 Transform 接入 Dataset

如果 Dataset 的 `__getitem__()` 中写有：

```python
if self.transform is not None:
    image, mask = self.transform(image, mask)
```

那么可以直接传入手写 Transform：

```python
train_dataset = LoveDADataset(
    root=data_root,
    split="Train",
    domains=("Urban", "Rural"),
    transform=train_transform,
)
```

Dataset 和 Transform 的职责应保持清晰：

```text
Dataset：找到文件、读取文件、返回元信息
Transform：裁剪、翻转、归一化、Tensor 化
```

不要把所有增强代码都塞到 Dataset 的路径扫描函数中。

---

## 5. 第二阶段：使用 Albumentations 调包

## 5.1 安装

```bash
pip install -U albumentations
```

如果需要转换为 PyTorch Tensor：

```python
from albumentations.pytorch import ToTensorV2
```

如果环境中没有 PyTorch 相关依赖，可以先只使用 NumPy 输出。

---

## 5.2 Albumentations 的基本思想

Albumentations 使用 `A.Compose` 组合多个变换：

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=512, width=512),
])
```

调用时必须把原图和掩码作为关键字参数传入：

```python
result = transform(
    image=image,
    mask=mask,
)

image = result["image"]
mask = result["mask"]
```

Albumentations 会对几何变换使用同一组随机参数。

---

## 5.3 Albumentations 的最小例子

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
])

result = transform(
    image=image,
    mask=mask,
)

image = result["image"]
mask = result["mask"]
```

这相当于手写版本中的：

```python
if random.random() < 0.5:
    image = flip(image)
    mask = flip(mask)
```

---

## 5.4 Albumentations 训练 Transform

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.SmallestMaxSize(
        max_size=1024,
        p=1.0,
    ),
    A.RandomCrop(
        height=512,
        width=512,
        p=1.0,
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

其中：

| 操作 | 作用对象 |
|---|---|
| `SmallestMaxSize` | image 和 mask |
| `RandomCrop` | image 和 mask |
| `HorizontalFlip` | image 和 mask |
| `VerticalFlip` | image 和 mask |
| `RandomBrightnessContrast` | 只处理 image |
| `Normalize` | 只处理 image |
| `ToTensorV2` | 转换输出格式 |

Albumentations 对传入的 `mask` 默认采用适合类别标签的最近邻插值。

---

## 5.5 Albumentations 验证 Transform

验证集不应使用随机裁剪和随机颜色增强：

```python
val_transform = A.Compose([
    A.Resize(
        height=512,
        width=512,
        p=1.0,
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

验证 Transform 的目标是：

```text
同一张图每次得到相同的处理结果
```

---

## 5.6 Albumentations Transform 包装类

你的 Dataset 当前调用方式是：

```python
image, mask = self.transform(image, mask)
```

而 Albumentations 的输出是字典，因此可以写一个包装类：

```python
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsTransform:
    def __init__(self, training=True, crop_size=512):
        if training:
            self.transform = A.Compose([
                A.SmallestMaxSize(
                    max_size=crop_size * 2,
                    p=1.0,
                ),
                A.RandomCrop(
                    height=crop_size,
                    width=crop_size,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(
                    height=crop_size,
                    width=crop_size,
                    p=1.0,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])

    def __call__(self, image, mask):
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        result = self.transform(
            image=image,
            mask=mask,
        )

        image = result["image"]
        mask = result["mask"].long()

        return image, mask
```

然后可以直接接入现有 Dataset：

```python
train_transform = AlbumentationsTransform(
    training=True,
    crop_size=512,
)

val_transform = AlbumentationsTransform(
    training=False,
    crop_size=512,
)

train_dataset = LoveDADataset(
    root=data_root,
    split="Train",
    domains=("Urban", "Rural"),
    transform=train_transform,
)
```

这样只替换 Transform，Dataset 的扫描和读取逻辑不需要改变。

---

## 6. 手写版本和 Albumentations 版本对照

| 功能 | 手写实现 | Albumentations |
|---|---|---|
| 缩放 | `image.resize()`、`mask.resize()` | `A.Resize()` |
| 随机裁剪 | 手动生成 `box` | `A.RandomCrop()` |
| 水平翻转 | `Image.Transpose.FLIP_LEFT_RIGHT` | `A.HorizontalFlip()` |
| 垂直翻转 | `Image.Transpose.FLIP_TOP_BOTTOM` | `A.VerticalFlip()` |
| 亮度变化 | `ImageEnhance.Brightness` | `A.RandomBrightnessContrast()` |
| 归一化 | NumPy 手动计算 | `A.Normalize()` |
| Tensor 转换 | `torch.from_numpy()` | `ToTensorV2()` |
| 同步随机参数 | 手动保证 | `A.Compose()` 自动保证 |

学习顺序不能反过来。先手写，才能理解 Albumentations 解决了什么问题；直接调包容易只会记函数名称，不理解 mask 为什么能够同步变换。

---

## 7. Albumentations 中最容易出错的地方

### 7.1 忘记同时传入 image 和 mask

错误：

```python
result = transform(image=image)
```

正确：

```python
result = transform(
    image=image,
    mask=mask,
)
```

### 7.2 把 mask 当成 image 传入

分割标签应该使用：

```python
result = transform(image=image, mask=mask)
```

不要把掩码放到：

```python
image=mask
```

否则颜色增强、归一化等图像操作可能作用于掩码。

### 7.3 将 mask 归一化

`A.Normalize()` 默认只作用于 image，不应把掩码转换成浮点类别值。

### 7.4 忘记 `ToTensorV2()` 的输出类型

使用后应检查：

```python
print(image.shape, image.dtype)
print(mask.shape, mask.dtype)
```

训练交叉熵损失时，掩码通常需要：

```python
mask = mask.long()
```

### 7.5 忽略图像尺寸

`A.RandomCrop(height=512, width=512)` 要求输入图像至少能够裁剪出对应区域。如果原图更小，需要先缩放或改用填充操作。

---

## 8. 手写版本与调包版本的测试

无论哪种写法，都必须检查：

```python
image, mask = transform(image=image, mask=mask)
```

对于手写版本：

```python
image, mask = hand_transform(image, mask)
```

检查：

```python
print("image shape:", image.shape)
print("mask shape:", mask.shape)
print("image dtype:", image.dtype)
print("mask dtype:", mask.dtype)
print("mask labels:", torch.unique(mask))
```

预期：

```text
image shape: [3, 512, 512]
mask shape: [512, 512]
image dtype: torch.float32
mask dtype: torch.int64
```

还要检查标签值是否变化：

```python
before = set(np.unique(original_mask).tolist())
after = set(torch.unique(mask).tolist())

print("before:", before)
print("after:", after)
```

增强后不应该出现非法标签值。

---

## 9. Transform 可视化检查

```python
import matplotlib.pyplot as plt
import torch


def show_transform(image, mask):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()

    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab10")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(mask, cmap="tab10", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
```

注意：如果图像已经经过 ImageNet 归一化，直接显示可能颜色异常。可视化前需要反归一化，或者在归一化之前保存一份图像。

---

## 10. 训练题

### 题目 1：手写缩放

编写：

```python
def resize_pair(image, mask, size):
    ...
```

要求：

- 原图使用双线性插值；
- 掩码使用最近邻插值；
- 输出尺寸一致；
- 检查掩码标签值是否发生异常。

### 题目 2：手写同步翻转

编写：

```python
def paired_flip(image, mask, probability):
    ...
```

要求：

- 使用一个随机数；
- 图像和掩码共同决定是否翻转；
- 返回增强后的图像和掩码。

### 题目 3：手写同步裁剪

编写：

```python
def paired_crop(image, mask, crop_size):
    ...
```

要求：

- 检查尺寸；
- 只生成一次裁剪框；
- 同时作用于 image 和 mask。

### 题目 4：手写完整 Transform

实现：

```python
class MyTransform:
    ...
```

训练模式包括：

```text
随机缩放
随机裁剪
随机翻转
亮度增强
归一化
Tensor 化
```

验证模式包括：

```text
固定缩放
归一化
Tensor 化
```

### 题目 5：用 Albumentations 重写题目 4

要求使用：

```python
A.Resize
A.RandomCrop
A.HorizontalFlip
A.VerticalFlip
A.RandomBrightnessContrast
A.Normalize
ToTensorV2
```

### 题目 6：接入 LoveDADataset

将你写的 Albumentations Transform 传入：

```python
LoveDADataset(
    root=...,
    split="Train",
    domains=("Urban", "Rural"),
    transform=...
)
```

测试：

```python
image, mask, meta = dataset[0]
print(image.shape)
print(mask.shape)
print(meta)
```

### 题目 7：训练和验证结果对比

分别创建：

```python
train_transform
val_transform
```

对同一张图像分别运行 5 次：

- 训练 Transform 的结果应可能不同；
- 验证 Transform 的结果应保持一致。

### 题目 8：增强结果可视化

随机取 8 个样本，每个样本显示：

```text
增强原图
增强掩码
叠加结果
```

检查：

- 建筑、道路等目标是否仍然对齐；
- mask 是否出现非法类别；
- 颜色增强是否只改变图像；
- 图像尺寸是否统一。

---

## 11. 手写到调包的正确学习路线

```text
1. 学会读取 image 和 mask
        ↓
2. 手写 resize
        ↓
3. 手写 flip
        ↓
4. 手写 crop
        ↓
5. 手写颜色增强
        ↓
6. 手写 Normalize 和 Tensor 化
        ↓
7. 组合成手写 Transform
        ↓
8. 接入 Dataset
        ↓
9. 用 Albumentations 重写
        ↓
10. 对比两个版本的输出
        ↓
11. 接入 DataLoader 和模型训练
```

最终你应当能够回答：

- 为什么 mask 需要单独处理？
- 为什么 mask 不能使用双线性插值？
- 为什么颜色增强不能作用于 mask？
- Albumentations 如何保证 image 和 mask 同步？
- `A.Compose()` 的作用是什么？
- `ToTensorV2()` 转换了什么？
- 训练和验证 Transform 为什么不同？
- 为什么必须对增强结果进行可视化？

当这些问题都能解释清楚时，你就不仅是在“调用增强库”，而是真正理解了遥感分割 Transform 的设计逻辑。
