# 深度学习损失函数学习路线

> 本文用于持续记录遥感语义分割和深度学习中的损失函数知识。后续学习到新的损失函数、公式、代码、实验现象或问题时，继续追加到对应章节即可。

---

## 0. 当前学习背景

当前任务是使用 U-Net 完成 LoveDA 遥感语义分割。

模型输入：

```text
[N, 3, H, W]
```

U-Net 输出：

```text
[N, C, H, W]
```

如果 LoveDA 经过标签重映射后有 7 个有效类别，则：

```text
[N, 7, H, W]
```

标签掩码：

```text
[N, H, W]
```

标签值：

```text
0～6：有效类别
255：忽略区域
```

当前计划使用多类别交叉熵损失：

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=255
)
```

这个计划是可行的。

---

# 第一部分：损失函数基础

## 1. 什么是损失函数

损失函数用于衡量模型预测结果与真实标签之间的差异。

```text
模型输出预测结果
        ↓
损失函数比较预测和真实值
        ↓
得到一个标量 loss
        ↓
反向传播计算梯度
        ↓
更新模型参数
```

训练目标是让损失函数逐渐变小。

需要注意：

```text
loss 越小，说明当前批次的预测结果通常越接近标签
```

但 loss 不是最终评价指标。语义分割还需要关注：

- Pixel Accuracy；
- Mean IoU；
- Dice；
- F1-score；
- 各类别 IoU。

---

## 2. 损失函数在训练中的位置

```python
logits = model(images)
loss = criterion(logits, masks)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

四个关键对象的作用：

| 对象 | 作用 |
|---|---|
| `model` | 根据图像生成预测结果 |
| `logits` | 模型输出的原始类别分数 |
| `criterion` | 比较预测和真实标签 |
| `optimizer` | 根据梯度更新模型参数 |

---

# 第二部分：当前任务中的张量格式

## 3. U-Net 的输出格式

你的 U-Net 最后一层是：

```python
self.outc = nn.Conv2d(chs[0], num_classes, 1)
```

当：

```python
num_classes = 7
```

模型输出：

```python
logits.shape == [N, 7, H, W]
```

含义是：

```text
N：一批图像的数量
7：每个像素的 7 个类别分数
H：图像高度
W：图像宽度
```

对于某一个像素 `(h, w)`，模型输出：

```python
logits[:, :, h, w]
```

它包含该像素属于 7 个类别的原始分数。

---

## 4. 交叉熵要求的标签格式

多类别语义分割中，交叉熵要求：

```text
logits：[N, C, H, W]
target：[N, H, W]
```

标签必须是：

```python
torch.long
```

也就是 `int64`。

标签值必须满足：

```text
0 ≤ target < C
```

如果 `C=7`，有效标签只能是：

```text
0、1、2、3、4、5、6
```

另外可以设置一个特殊值表示忽略区域：

```text
255：ignore_index
```

不能把分割标签写成：

```text
[N, 1, H, W]
```

也不需要将标签转换成 one-hot 后再传给普通 `CrossEntropyLoss`。

---

# 第三部分：LoveDA 标签映射

## 5. 为什么需要标签重映射

LoveDA 原始标签通常是：

```text
0：忽略区域
1～7：七个语义类别
```

而 U-Net 的输出通道数是：

```python
num_classes = 7
```

交叉熵要求有效类别编号是：

```text
0～6
```

所以必须映射为：

```text
原始 0 → 255
原始 1 → 0
原始 2 → 1
原始 3 → 2
原始 4 → 3
原始 5 → 4
原始 6 → 5
原始 7 → 6
```

如果只把 `0` 改成 `255`，保留标签 `1～7`，那么标签 `7` 会超出模型类别范围，可能报错：

```text
Target 7 is out of bounds
```

## 6. 手写标签映射

```python
import numpy as np


def remap_loveda_mask(mask, ignore_index=255):
    mask = mask.astype(np.int64)

    new_mask = np.full(
        mask.shape,
        fill_value=ignore_index,
        dtype=np.int64,
    )

    valid = (mask >= 1) & (mask <= 7)
    new_mask[valid] = mask[valid] - 1

    return new_mask
```

也可以使用：

```python
mask = np.where(
    (mask >= 1) & (mask <= 7),
    mask - 1,
    self.ignore_index,
).astype(np.int64)
```

检查结果：

```python
print(np.unique(mask))
```

期望得到：

```text
[0, 1, 2, 3, 4, 5, 6, 255]
```

参数名建议使用：

```python
remap_loveda_labels
```

因为实际操作不仅是把零值映射为忽略标签，还把 `1～7` 映射为 `0～6`。

---

# 第四部分：交叉熵损失

## 7. 交叉熵适合当前任务吗

适合。当前任务是多类别语义分割，每个像素属于一个类别，因此可以使用：

```python
nn.CrossEntropyLoss()
```

它适合：

- 多类别分类；
- 多类别语义分割；
- 每个样本或每个像素只有一个真实类别。

当前 U-Net 输出：

```text
[N, 7, H, W]
```

当前标签：

```text
[N, H, W]
```

两者正好匹配。

---

## 8. 交叉熵的数学含义

对于一个像素，模型输出各类别的 logits：

```text
z = [z0, z1, ..., zC-1]
```

Softmax 概率为：

[
 p_c = \frac{e^{z_c}}{\sum_j e^{z_j}}
]

如果真实类别为 `y`，交叉熵为：

[
 L = -\log p_y
]

含义是：

```text
真实类别概率越大，loss 越小
真实类别概率越小，loss 越大
```

例如：

```text
真实类别概率 = 0.9 → loss 较小
真实类别概率 = 0.1 → loss 较大
```

---

## 9. 为什么模型最后不能加 Softmax

交叉熵内部已经包含了类似：

```text
LogSoftmax + NLLLoss
```

因此模型应该直接输出 logits：

```python
return self.outc(x)
```

不要在 U-Net 最后增加：

```python
nn.Softmax(dim=1)
```

正确写法：

```python
logits = model(images)
loss = criterion(logits, masks)
```

预测类别时才使用：

```python
pred = torch.argmax(logits, dim=1)
```

`argmax` 只用于预测和评价，不能用于计算训练损失，因为它会破坏梯度。

---

# 第五部分：`log_softmax` 的含义

## 10. `logits` 是什么

`logits` 是模型最后一层输出的原始分数。

例如某个像素的 logits：

```python
[2.0, 1.0, 0.1]
```

它们不是概率，因为：

- 可以是负数；
- 不要求总和为 1；
- 数值大小只表示模型对类别的相对偏好。

## 11. `softmax` 的作用

```python
probs = torch.softmax(logits, dim=1)
```

Softmax 将原始分数转换为概率：

```text
每个概率在 0～1 之间
所有类别概率之和为 1
```

## 12. `log_softmax` 的作用

```python
log_probs = torch.log_softmax(logits, dim=1)
```

它等价于：

```python
log_probs = torch.log(
    torch.softmax(logits, dim=1)
)
```

也就是先求 Softmax 概率，再取自然对数。

在交叉熵中，如果真实类别为 `y`：

```python
loss = -log_probs[真实类别]
```

因此交叉熵流程是：

```text
logits
  ↓
log_softmax
  ↓
取真实类别的 log 概率
  ↓
加负号
  ↓
求平均
```

## 13. `dim=1` 为什么表示类别维度

U-Net 输出为：

```text
[N, C, H, W]
```

各维含义：

```text
0：batch
1：类别
2：高度
3：宽度
```

因此：

```python
torch.log_softmax(logits, dim=1)
```

表示对每个像素位置，在所有类别之间计算 Softmax。

不能写成：

```python
torch.log_softmax(logits, dim=0)
```

因为 `dim=0` 是在不同图像之间计算，含义错误。

## 14. 为什么不写成 `log(softmax())`

数学上二者等价：

```python
torch.log(torch.softmax(logits, dim=1))
```

但直接这样计算可能出现数值不稳定，例如概率过小时变成 0，随后出现：

```text
log(0)
```

`torch.log_softmax()` 使用了更加稳定的计算方式，因此更适合神经网络训练。

---

# 第六部分：手写交叉熵

## 15. 手写普通分类交叉熵

```python
import torch


def cross_entropy_manual(logits, target):
    """
    logits: [N, C]
    target: [N]
    """
    log_probs = torch.log_softmax(logits, dim=1)

    loss = -log_probs[
        torch.arange(logits.size(0), device=logits.device),
        target,
    ]

    return loss.mean()
```

## 16. 手写语义分割交叉熵

分割任务的 logits 是四维，因此需要先把每个像素展开成一个分类样本：

```python
def segmentation_cross_entropy(
    logits,
    target,
    ignore_index=255,
):
    """
    logits: [N, C, H, W]
    target: [N, H, W]
    """
    n, c, h, w = logits.shape

    # [N, C, H, W] -> [N, H, W, C]
    logits = logits.permute(0, 2, 3, 1)

    # [N, H, W, C] -> [N*H*W, C]
    logits = logits.reshape(-1, c)

    # [N, H, W] -> [N*H*W]
    target = target.reshape(-1).long()

    # 忽略 ignore_index 区域
    valid = target != ignore_index

    logits = logits[valid]
    target = target[valid]

    if target.numel() == 0:
        return logits.sum() * 0.0

    log_probs = torch.log_softmax(logits, dim=1)

    row_index = torch.arange(
        target.numel(),
        device=target.device,
    )

    selected_log_probs = log_probs[
        row_index,
        target,
    ]

    loss = -selected_log_probs.mean()

    return loss
```

核心操作：

```text
[N, C, H, W]
        ↓ 展开
[N×H×W, C]
        ↓ 每个像素变成一个分类样本
[N×H×W]
        ↓ 取出真实类别
求平均
```

---

## 17. 使用 `logsumexp` 手写更稳定的交叉熵

交叉熵还可以写成：

[
L_i = \log\left(\sum_c e^{z_{i,c}}\right)-z_{i,y_i}
]

代码：

```python
def segmentation_cross_entropy_v2(
    logits,
    target,
    ignore_index=255,
):
    n, c, h, w = logits.shape

    logits = logits.permute(0, 2, 3, 1)
    logits = logits.reshape(-1, c)
    target = target.reshape(-1).long()

    valid = target != ignore_index
    logits = logits[valid]
    target = target[valid]

    if target.numel() == 0:
        return logits.sum() * 0.0

    log_sum_exp = torch.logsumexp(
        logits,
        dim=1,
    )

    row_index = torch.arange(
        target.numel(),
        device=target.device,
    )

    true_class_logits = logits[
        row_index,
        target,
    ]

    loss = log_sum_exp - true_class_logits

    return loss.mean()
```

这个版本没有显式先计算 Softmax，数值稳定性更好。

---

# 第七部分：`ignore_index` 的作用

## 18. 为什么需要忽略区域

有些像素没有可靠标签，或者数据集规定某些像素不参与训练。

当前 LoveDA 标签可以设置为：

```text
255：忽略区域
```

损失函数：

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=255
)
```

含义是：

```text
标签等于 255 的像素不参与损失计算
```

## 19. 手写损失中如何忽略 255

```python
valid = target != ignore_index

logits = logits[valid]
target = target[valid]
```

如果所有像素都是忽略区域，需要返回一个仍然可以反向传播的零损失：

```python
if target.numel() == 0:
    return logits.sum() * 0.0
```

---

# 第八部分：官方交叉熵和手写交叉熵对照

```python
manual_loss = segmentation_cross_entropy(
    logits,
    masks,
    ignore_index=255,
)

official_loss = torch.nn.CrossEntropyLoss(
    ignore_index=255
)(logits, masks)

print(manual_loss)
print(official_loss)
print(torch.allclose(manual_loss, official_loss))
```

如果实现正确，通常会输出接近：

```text
True
```

手写版本用于理解原理，正式训练建议使用 PyTorch 官方实现：

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=255,
)
```

---

# 第九部分：完整训练中的损失函数流程

```python
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=255,
)

for images, masks, metas in train_loader:
    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)

    loss = criterion(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

训练前检查：

```python
print(images.shape)
print(masks.shape)
print(logits.shape)
print(masks.dtype)
print(torch.unique(masks))
```

期望：

```text
images： [N, 3, H, W]
masks：  [N, H, W]
logits： [N, 7, H, W]
masks.dtype：torch.int64
torch.unique(masks)：0～6 和 255
```

---

# 第十部分：交叉熵相关常见错误

## 错误 1：模型最后加 Softmax

错误：

```python
self.softmax = nn.Softmax(dim=1)
```

然后再传入 `CrossEntropyLoss`。

正确：

```python
model 输出 logits
CrossEntropyLoss 内部完成 log-softmax
```

## 错误 2：mask 是 float32

错误：

```python
mask = mask.float()
```

正确：

```python
mask = mask.long()
```

## 错误 3：mask 形状为 `[N, 1, H, W]`

交叉熵通常要求：

```text
[N, H, W]
```

如果是 `[N, 1, H, W]`，需要去掉通道维：

```python
mask = mask.squeeze(1)
```

## 错误 4：标签编号从 1 开始但模型类别从 0 开始

例如模型有 7 类，但标签为：

```text
1、2、3、4、5、6、7
```

需要映射为：

```text
0、1、2、3、4、5、6
```

## 错误 5：标签 255 没有设置 `ignore_index`

如果标签中存在 255，但损失函数没有设置：

```python
ignore_index=255
```

就可能出现类别越界错误。

## 错误 6：使用 `argmax` 后再计算损失

错误：

```python
pred = torch.argmax(logits, dim=1)
loss = criterion(pred, masks)
```

正确：

```python
loss = criterion(logits, masks)
pred = torch.argmax(logits, dim=1)
```

## 错误 7：把 mask 做 Normalize

图像可以归一化：

```python
image = (image - mean) / std
```

掩码不能进行这种归一化，因为掩码中的数字是类别编号。

---

# 第十一部分：建议的损失函数学习路线

## 阶段 1：单样本分类交叉熵

掌握：

- logits；
- softmax；
- log 概率；
- 真实类别索引；
- 负对数似然。

练习：手写 `[N, C]` 分类交叉熵。

## 阶段 2：语义分割交叉熵

掌握：

- `[N, C, H, W]`；
- `[N, H, W]`；
- 每个像素作为一个分类样本；
- `permute`；
- `reshape`；
- `ignore_index`。

## 阶段 3：验证手写实现

掌握：

- 与 `nn.CrossEntropyLoss` 对比；
- `torch.allclose`；
- 梯度是否正常；
- 空有效像素情况。

## 阶段 4：类别不平衡

后续学习：

- 加权交叉熵；
- 类别频率统计；
- `weight` 参数；
- 为什么背景类别过多会影响模型。

## 阶段 5：Dice Loss 和 IoU Loss

后续学习：

- Dice 系数；
- Dice Loss；
- 多类别 Dice；
- 忽略区域处理；
- 交叉熵与 Dice 的组合。

## 阶段 6：Focal Loss

后续学习：

- 难分类样本；
- 调节因子；
- `gamma`；
- 类别不平衡。

## 阶段 7：二分类与多标签损失

后续学习：

- `BCEWithLogitsLoss`；
- Sigmoid；
- 二分类分割；
- 多标签分割；
- BCE 与 Cross Entropy 的区别。

## 阶段 8：组合损失

后续学习：

```python
loss = ce_loss + lambda_dice * dice_loss
```

需要研究：

- 两个损失的量纲和数量级；
- 权重如何设置；
- 是否出现某个损失主导训练；
- 验证集指标是否真正改善。

## 阶段 9：论文和工程中的损失函数

后续学习：

- Lovász-Softmax Loss；
- Tversky Loss；
- Boundary Loss；
- Hausdorff Loss；
- OHEM；
- 深监督损失；
- 多尺度输出损失。

---

# 第十二部分：后续追加记录区

## 追加记录 1：

> 日期：
>
> 学习主题：
>
> 关键问题：
>
> 结论：
>
> 代码：
>
> 实验现象：
>
> 仍未解决的问题：

## 追加记录 2：

> 日期：
>
> 学习主题：
>
> 关键问题：
>
> 结论：
>
> 代码：
>
> 实验现象：
>
> 仍未解决的问题：

---

# 第十三部分：当前阶段检查清单

- [ ] 能解释损失函数在训练中的作用；
- [ ] 能解释 logits 的含义；
- [ ] 能解释 softmax 和 log_softmax；
- [ ] 能解释 `dim=1` 为什么是类别维；
- [ ] 能说出交叉熵的输入形状；
- [ ] 能说出标签必须是 `long` 类型；
- [ ] 能完成 LoveDA 标签重映射；
- [ ] 能解释 `ignore_index=255`；
- [ ] 能手写二维分类交叉熵；
- [ ] 能手写语义分割交叉熵；
- [ ] 能与官方 `CrossEntropyLoss` 对比；
- [ ] 能区分训练损失和预测 `argmax`；
- [ ] 能检查一个 batch 的 logits、mask 和 loss；
- [ ] 能继续学习 Dice、Focal 和组合损失。
