# 遥感语义分割关键评价指标学习路线

本文记录遥感语义分割中常用评价指标的数学原理和代码实现。当前先学习三个最重要的指标：

1. Mean Intersection over Union，mIoU；
2. Dice coefficient / F1-score；
3. Pixel Accuracy / Overall Accuracy，像素准确率。

后续可以继续追加：Precision、Recall、Fβ、频率加权 IoU、边界 IoU、Boundary F1、混淆矩阵和类别不平衡分析。

---

## 1. 为什么需要多个指标

语义分割模型对每个像素进行分类，因此不能只观察训练损失。

例如，一张遥感图像中 90% 都是背景，模型即使把所有像素都预测为背景，也可能获得较高的像素准确率，但建筑、道路和水体的分割结果可能完全不可用。

不同指标关注的内容不同：

| 指标 | 主要关注点 |
|---|---|
| Pixel Accuracy | 总体像素预测是否正确 |
| IoU | 预测区域与真实区域的重叠程度 |
| Dice/F1 | 对前景区域重叠程度更敏感 |
| mIoU | 各类别的平均分割质量 |

在遥感分割中，通常优先关注 mIoU，同时报告 Dice 和 Pixel Accuracy。

---

# 2. 混淆矩阵基础

对于某一个类别，可以将像素分成四类：

- True Positive，TP：真实属于该类，预测也属于该类；
- False Positive，FP：真实不属于该类，但预测成该类；
- False Negative，FN：真实属于该类，但预测成其他类；
- True Negative，TN：真实不属于该类，预测也不属于该类。

在多类别语义分割中，可以为每个类别分别计算 TP、FP 和 FN。

对于类别 `k`：

```text
TP_k = 真实为 k 且预测为 k
FP_k = 真实不是 k 但预测为 k
FN_k = 真实为 k 但预测不是 k
```

`ignore_index` 区域不应该参与任何指标计算。

---

# 3. 指标一：IoU 和 mIoU

## 3.1 IoU 的数学定义

IoU 又称交并比，定义为预测区域和真实区域的交集除以并集：

\[
IoU_k = \frac{TP_k}{TP_k + FP_k + FN_k}
\]

其中：

- 分子 `TP_k` 是预测区域与真实区域的交集；
- 分母 `TP_k + FP_k + FN_k` 是预测区域与真实区域的并集。

如果：

```text
TP = 80
FP = 10
FN = 20
```

则：

\[
IoU = \frac{80}{80+10+20}=0.7273
\]

## 3.2 mIoU 的数学定义

对于 `C` 个类别，先分别计算每个类别的 IoU，再求平均：

\[
mIoU = \frac{1}{C}\sum_{k=1}^{C} IoU_k
\]

例如七个类别的 IoU 为：

```text
[0.80, 0.70, 0.60, 0.50, 0.75, 0.65, 0.55]
```

则：

```text
mIoU = 这些类别 IoU 的平均值
```

## 3.3 为什么 mIoU 重要

mIoU 对每个类别一视同仁。即使某个类别像素很少，只要该类别分割很差，也会拉低 mIoU。

因此 mIoU 能够反映模型是否真正学会了不同地物类别，而不是只依赖像素数量最多的背景类别。

在遥感场景中，建筑、道路、水体、林地等类别往往像素数量差异很大，所以 mIoU 通常比单纯的 Pixel Accuracy 更有参考价值。

## 3.4 手写单类别 IoU

```python
def binary_iou(pred, target, class_id, ignore_index=255):
    """
    pred:   [H, W]
    target: [H, W]
    """
    valid = target != ignore_index

    pred_class = (pred == class_id) & valid
    target_class = (target == class_id) & valid

    intersection = (pred_class & target_class).sum()
    union = (pred_class | target_class).sum()

    if union == 0:
        return None

    return intersection / union
```

## 3.5 手写多类别 mIoU

```python
import torch


def mean_iou(
    pred,
    target,
    num_classes,
    ignore_index=255,
):
    """
    pred:   [N, H, W]
    target: [N, H, W]
    """
    valid = target != ignore_index

    ious = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id) & valid
        target_class = (target == class_id) & valid

        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()

        if union == 0:
            # 当前 batch 中没有真实或预测的该类别
            continue

        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(ious).mean()
```

调用：

```python
logits = model(images)
pred = torch.argmax(logits, dim=1)

miou = mean_iou(
    pred=pred,
    target=masks,
    num_classes=7,
    ignore_index=255,
)

print("mIoU:", miou.item())
```

注意：`argmax` 可以用于计算指标，因为指标不需要反向传播；训练损失仍然必须使用原始 logits。

## 3.6 使用混淆矩阵计算 mIoU

逐像素使用布尔运算可以直观理解，但训练时更推荐先累计混淆矩阵。

```python
import torch


def update_confusion_matrix(
    confmat,
    pred,
    target,
    num_classes,
    ignore_index=255,
):
    """
    confmat: [C, C]
    pred:    [N, H, W]
    target:  [N, H, W]
    """
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    valid_range = (
        (target >= 0)
        & (target < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    pred = pred[valid_range]
    target = target[valid_range]

    index = target * num_classes + pred
    hist = torch.bincount(
        index,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confmat += hist
    return confmat
```

混淆矩阵中：

```text
行：真实类别
列：预测类别
```

因此：

```python
tp = torch.diag(confmat)
fp = confmat.sum(dim=0) - tp
fn = confmat.sum(dim=1) - tp
union = tp + fp + fn
ious = tp / union
miou = ious[union > 0].mean()
```

---

# 4. 指标二：Dice coefficient / F1-score

## 4.1 Dice 的数学定义

对于类别 `k`：

\[
Dice_k = \frac{2TP_k}{2TP_k + FP_k + FN_k}
\]

也可以写成预测区域和真实区域的形式：

\[
Dice_k = \frac{2|P_k \cap G_k|}{|P_k| + |G_k|}
\]

其中：

- `P_k`：预测为类别 `k` 的像素集合；
- `G_k`：真实属于类别 `k` 的像素集合。

如果：

```text
TP = 80
FP = 10
FN = 20
```

则：

\[
Dice = \frac{2\times80}{2\times80+10+20}=0.8421
\]

## 4.2 Dice 与 IoU 的关系

二者可以互相转换：

\[
Dice = \frac{2IoU}{1+IoU}
\]

\[
IoU = \frac{Dice}{2-Dice}
\]

Dice 通常数值比 IoU 高，但二者反映的都是区域重叠质量。

## 4.3 为什么 Dice 重要

Dice 对前景区域的重叠较敏感，常用于：

- 建筑物分割；
- 道路提取；
- 水体提取；
- 小目标分割；
- 类别不平衡场景。

当某个类别占比很小时，Dice 可以帮助观察模型是否真正覆盖了该类别。

## 4.4 手写 Dice

```python
def mean_dice(
    pred,
    target,
    num_classes,
    ignore_index=255,
    eps=1e-7,
):
    valid = target != ignore_index
    dices = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id) & valid
        target_class = (target == class_id) & valid

        tp = (pred_class & target_class).sum().float()
        fp = (pred_class & ~target_class).sum().float()
        fn = (~pred_class & target_class).sum().float()

        denominator = 2 * tp + fp + fn

        if denominator == 0:
            continue

        dice = (2 * tp + eps) / (denominator + eps)
        dices.append(dice)

    if len(dices) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(dices).mean()
```

调用：

```python
pred = torch.argmax(logits, dim=1)

dice = mean_dice(
    pred,
    masks,
    num_classes=7,
    ignore_index=255,
)
```

## 4.5 训练中的 Dice 指标

Dice 指标使用离散预测：

```python
pred = torch.argmax(logits, dim=1)
```

它适合验证和评价。

如果要把 Dice 作为训练损失，不能直接对 `argmax` 结果计算，因为 `argmax` 不可导。训练时需要使用 Softmax 概率：

```python
probs = torch.softmax(logits, dim=1)
```

这对应的是 Dice Loss，而不是离散 Dice 指标。后续可以单独学习：

```text
Dice coefficient：评价指标
Dice Loss：训练损失
```

---

# 5. 指标三：Pixel Accuracy / Overall Accuracy

## 5.1 数学定义

Pixel Accuracy 表示预测正确的像素数量占有效像素总数的比例：

\[
PixelAcc = \frac{\sum_k TP_k}{\sum_k(TP_k+FN_k)}
\]

也可以直接写成：

\[
PixelAcc = \frac{\text{预测正确的像素数}}{\text{有效像素总数}}
\]

如果有效像素中有：

```text
预测正确：9000
有效像素：10000
```

则：

```text
Pixel Accuracy = 0.9
```

## 5.2 为什么 Pixel Accuracy 重要

Pixel Accuracy 能够反映模型的总体像素分类情况，适合：

- 观察模型整体是否正常学习；
- 快速发现模型是否完全失效；
- 作为训练过程中的辅助指标；
- 与 mIoU、Dice 一起形成完整评价。

但它容易受到大类别影响。例如背景占据 90% 的图像中，模型只预测背景，也可能得到较高 Pixel Accuracy。因此不能单独使用 Pixel Accuracy 判断分割质量。

## 5.3 手写 Pixel Accuracy

```python
def pixel_accuracy(
    pred,
    target,
    ignore_index=255,
):
    valid = target != ignore_index

    correct = ((pred == target) & valid).sum().float()
    total = valid.sum().float()

    if total == 0:
        return torch.tensor(0.0, device=pred.device)

    return correct / total
```

调用：

```python
pred = torch.argmax(logits, dim=1)

acc = pixel_accuracy(
    pred,
    masks,
    ignore_index=255,
)

print("Pixel Accuracy:", acc.item())
```

## 5.4 使用混淆矩阵计算 Pixel Accuracy

```python
pixel_acc = torch.diag(confmat).sum() / confmat.sum()
```

其中：

```text
混淆矩阵对角线：预测正确的像素
混淆矩阵全部元素：有效像素总数
```

---

# 6. 三个指标的比较

| 指标 | 数学核心 | 优点 | 局限 |
|---|---|---|---|
| mIoU | 交集 / 并集 | 反映各类别区域重叠，适合类别不平衡 | 对边界小误差较敏感 |
| Dice/F1 | 2TP / (2TP+FP+FN) | 对前景和小目标较敏感 | 数值通常高于 IoU，跨论文比较要确认定义 |
| Pixel Accuracy | 正确像素 / 有效像素 | 直观，容易理解 | 容易被背景大类主导 |

推荐报告形式：

```text
mIoU：主要指标
Mean Dice：辅助重叠指标
Pixel Accuracy：总体像素正确率
Per-class IoU：分析各类别表现
```

---

# 7. 在验证函数中统一计算三个指标

```python
@torch.no_grad()
def evaluate_metrics(
    model,
    loader,
    device,
    num_classes,
    ignore_index=255,
):
    model.eval()

    confmat = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        pred = torch.argmax(logits, dim=1)

        update_confusion_matrix(
            confmat=confmat,
            pred=pred,
            target=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    tp = torch.diag(confmat).float()
    fp = confmat.sum(dim=0).float() - tp
    fn = confmat.sum(dim=1).float() - tp

    union = tp + fp + fn
    valid_classes = union > 0

    iou = torch.zeros_like(tp)
    dice = torch.zeros_like(tp)

    iou[valid_classes] = (
        tp[valid_classes] / union[valid_classes]
    )

    dice_denominator = 2 * tp + fp + fn
    valid_dice = dice_denominator > 0

    dice[valid_dice] = (
        2 * tp[valid_dice]
        / dice_denominator[valid_dice]
    )

    miou = iou[valid_classes].mean()
    mean_dice = dice[valid_dice].mean()
    pixel_acc = tp.sum() / confmat.sum().clamp_min(1)

    return {
        "miou": miou.item(),
        "mean_dice": mean_dice.item(),
        "pixel_acc": pixel_acc.item(),
        "per_class_iou": iou.cpu().tolist(),
        "confusion_matrix": confmat.cpu(),
    }
```

使用：

```python
metrics = evaluate_metrics(
    model=model,
    loader=val_loader,
    device=device,
    num_classes=7,
    ignore_index=255,
)

print("mIoU:", metrics["miou"])
print("Dice:", metrics["mean_dice"])
print("Pixel Accuracy:", metrics["pixel_acc"])
print("Per-class IoU:", metrics["per_class_iou"])
```

---

# 8. 为什么训练损失和评价指标要同时记录

训练损失回答：

```text
模型是否在优化目标函数？
```

mIoU、Dice 和 Pixel Accuracy 回答：

```text
模型的分割结果是否真正变好？
```

可能出现：

```text
train loss 下降，但 val mIoU 不再上升
```

这通常提示：

- 模型开始过拟合；
- 类别不平衡影响了训练；
- 标签或评价代码存在问题；
- 训练损失与实际评价目标不完全一致。

建议每个 epoch 记录：

```text
train_loss
val_loss
val_mIoU
val_Dice
val_PixelAcc
```

保存最佳模型时，通常以验证集 mIoU 为依据：

```python
if val_miou > best_miou:
    best_miou = val_miou
    torch.save(model.state_dict(), "best_miou.pth")
```

---

# 9. 当前 LoveDA 项目的指标检查清单

- [ ] 预测结果形状为 `[N, H, W]`；
- [ ] 标签形状为 `[N, H, W]`；
- [ ] 预测类别范围为 `0～6`；
- [ ] 标签有效类别范围为 `0～6`；
- [ ] `255` 区域不参与指标计算；
- [ ] mIoU 按类别分别计算后取平均；
- [ ] Pixel Accuracy 只统计有效像素；
- [ ] 空类别不会导致除零错误；
- [ ] 同时记录总体指标和各类别 IoU；
- [ ] Urban 和 Rural 可以分别统计；
- [ ] 保存最佳模型时依据验证集 mIoU；
- [ ] 指标计算不参与反向传播；
- [ ] 指标使用 `model.eval()` 和 `torch.no_grad()`。

---

# 10. 后续追加内容

## 追加记录 1

> 日期：
>
> 学习指标：
>
> 数学定义：
>
> 代码实现：
>
> 实验结果：
>
> 遇到的问题：
>
> 解决方法：

## 追加记录 2

> 日期：
>
> 学习指标：
>
> 数学定义：
>
> 代码实现：
>
> 实验结果：
>
> 遇到的问题：
>
> 解决方法：
