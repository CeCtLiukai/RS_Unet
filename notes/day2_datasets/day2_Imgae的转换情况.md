
因为原图和掩码表达的信息不同，所以读取时要采用不同模式。

## 1. 为什么原图转换为 `RGB`

```python
image = Image.open(image_path).convert("RGB")
```

普通遥感可视化图像通常由三个颜色通道组成：

```text
R：红色
G：绿色
B：蓝色
```

转换为 `RGB` 可以保证：

```python
image.shape == (H, W, 3)
```

例如：

```text
(1024, 1024, 3)
```

这样后续才能按照普通三通道图像处理：

```python
image = image.transpose(2, 0, 1)
```

转换后变成 PyTorch 常用格式：

```text
(H, W, 3) → (3, H, W)
```

### 为什么不直接使用原始模式？

不同图片可能有不同模式：

```text
RGB：三通道彩色图
L：单通道灰度图
P：调色板图
RGBA：四通道图，包含透明度
```

如果不统一，可能出现：

```text
有的图像是 (H, W, 3)
有的图像是 (H, W, 4)
有的图像是 (H, W)
```

模型就无法组成统一的 batch。

因此，对于普通三通道遥感图像，统一转换成：

```python
RGB
```

可以保证通道数量一致。

### 注意：多光谱遥感影像不能简单转 RGB

如果你的影像包含：

```text
红、绿、蓝、近红外、短波红外……
```

那么不能直接：

```python
convert("RGB")
```

因为这样会丢失额外波段。多光谱数据通常使用：

```python
rasterio
GDAL
```

读取，并保留为：

```text
(C, H, W)
```

例如：

```text
(8, 1024, 1024)
```

所以 `RGB` 主要适用于普通三通道 PNG、JPG 图像。

---

## 2. 为什么掩码转换为 `L`

```python
mask = Image.open(mask_path).convert("L")
```

`L` 表示单通道灰度图，像素值通常是：

```text
0、1、2、3、4……
```

这些数字代表类别：

```text
0：背景
1：建筑
2：道路
3：水体
```

掩码不是普通彩色图，而是一张“类别编号表”。

例如：

```python
mask[100, 200] == 2
```

表示这个像素属于第 2 类。

因此掩码需要读取成：

```text
(H, W)
```

而不是：

```text
(H, W, 3)
```

### 为什么不能转为 RGB？

如果掩码原本是：

```text
0：背景
1：建筑
2：道路
```

转换成 RGB 后，每个类别可能变成三个通道：

```text
0 → [0, 0, 0]
1 → [1, 1, 1]
2 → [2, 2, 2]
```

这样会让掩码变成 `(H, W, 3)`，失去语义分割通常需要的单通道类别格式。

模型的交叉熵损失一般要求：

```text
预测结果：[N, C, H, W]
标签：    [N, H, W]
```

所以掩码必须是二维类别索引。

### 为什么常用 `L` 而不是 `1`

因为很多 PNG 标签虽然视觉上是灰度图，但实际模式可能是：

```text
P：调色板模式
L：灰度模式
```

`convert("L")` 可以把它统一成单通道格式，便于后续处理。

不过如果掩码是调色板标签，有时更安全的做法是直接保留类别索引：

```python
with Image.open(mask_path) as mask:
    if mask.mode == "P":
        mask_array = np.array(mask)
    else:
        mask_array = np.array(mask.convert("L"))
```

你当前 `LoveDADataset` 中的写法就是这种更稳妥的处理方式。

---

## 3. 为什么还要转换成 NumPy 数组

PIL 的 `Image` 对象适合：

```text
打开图片
保存图片
裁剪图片
缩放图片
翻转图片
```

但是模型训练和数值处理需要数组或 Tensor。

```python
image_array = np.array(image)
mask_array = np.array(mask)
```

转换后才能进行：

### 查看尺寸

```python
print(image_array.shape)
print(mask_array.shape)
```

### 查看数据类型

```python
print(image_array.dtype)
print(mask_array.dtype)
```

### 查看标签类别

```python
print(np.unique(mask_array))
```

### 执行标签映射

```python
mask_array = np.where(
    mask_array == 0,
    255,
    mask_array
)
```

### 转换为 PyTorch Tensor

```python
image_tensor = torch.from_numpy(image_array)
mask_tensor = torch.from_numpy(mask_array)
```

---

## 4. 原图和掩码转换过程对比

### 原图

```python
with Image.open(image_path) as image:
    image = image.convert("RGB")

image = np.array(image, dtype=np.float32)
image = image / 255.0
image = image.transpose(2, 0, 1)
image = torch.from_numpy(image)
```

数据变化：

```text
PIL RGB
  ↓
NumPy：(H, W, 3)
  ↓
归一化：float32
  ↓
转置：(3, H, W)
  ↓
Tensor
```

### 掩码

```python
with Image.open(mask_path) as mask:
    mask = mask.convert("L")

mask = np.array(mask, dtype=np.int64)
mask = torch.from_numpy(mask)
```

数据变化：

```text
PIL L
  ↓
NumPy：(H, W)
  ↓
保留类别整数
  ↓
Tensor：[H, W]
```

---

## 5. 最终为什么二者形状不同

模型输入通常是：

```text
image：[N, 3, H, W]
mask： [N, H, W]
```

例如：

```text
image：[4, 3, 512, 512]
mask： [4, 512, 512]
```

原因是：

- 原图有 3 个颜色通道；
- 掩码每个像素只需要一个类别编号。

可以记住：

```text
RGB 是为了保留图像的颜色通道；
L 是为了保留掩码的类别编号；
NumPy 是为了进行数值处理；
Tensor 是为了送入 PyTorch 模型。
```

有一个例外：如果你的遥感数据是多光谱或高光谱影像，就不能简单按照 `RGB` 读取，而应保留所有波段。