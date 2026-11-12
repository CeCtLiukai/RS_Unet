# RS-Unet 遥感分割训练流程与工程化教学

> 核心认知：一次完整的语义分割训练，先由模型根据输入图像产生 `logits`，再用标签计算损失，随后通过反向传播更新参数。`train_one_epoch` 管理一轮训练，`validate_one_epoch` 负责验证和指标统计，`fit` 负责组织多个 epoch，checkpoint 函数负责保存和恢复训练状态。
>
> 学习路线：训练基础 → 单个 batch → 单轮训练 → 单轮验证 → 多轮训练 → 权重保存与加载 → `train.py` 组装。

## 一、先理解一次训练到底做了什么

### 1. 一次 batch 的数据

假设 DataLoader 返回：

```python
images, masks, meta = next(iter(train_loader))
```

在遥感语义分割中，常见形状是：

```text
images: [B, 3, H, W]
masks:  [B, H, W]
```

其中 `B` 是 batch size，`3` 是 RGB 通道数，`H` 和 `W` 是图像高宽。mask 的每个像素保存一个类别编号，例如 `0` 到 `6`。忽略区域可以使用 `255`。

### 2. 模型输出不是最终类别图

```python
logits = model(images)
```

对于 7 类分割任务，输出形状通常是：

```text
logits: [B, 7, H, W]
```

每个像素位置有 7 个类别分数。分数还没有经过 `argmax`，也不应该在计算交叉熵前手动转成类别图。

```python
pred = torch.argmax(logits, dim=1)
```

这一步才把 `[B, C, H, W]` 转成 `[B, H, W]` 的预测类别图。训练计算损失使用 `logits`，验证指标使用 `pred`。

### 3. 交叉熵如何参与训练

```python
loss = criterion(logits, masks)
```

交叉熵会比较每个像素的预测分布和真实类别。`criterion` 常见写法是：

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
```

因此标签必须是整数类别编号，类型通常为：

```python
masks = masks.long()
```

### 4. 反向传播和参数更新

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

三步分别表示：

1. 清除上一批数据留下的梯度
2. 根据当前损失计算参数梯度
3. 使用优化器更新模型参数

完整的单 batch 流程是：

```python
images = images.to(device)
masks = masks.to(device)

logits = model(images)
loss = criterion(logits, masks)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 5. batch、epoch 和 iteration

- `batch` 表示一次送入模型的一小批样本
- `iteration` 表示完成一次 batch 的前向和反向传播
- `epoch` 表示完整遍历一次 DataLoader

如果训练集有 800 张图像，`batch_size=8`，那么一轮大约有 100 次 iteration。

## 二、训练阶段和验证阶段的区别

### 1. 训练阶段

训练阶段需要更新模型参数：

```python
model.train()
loss.backward()
optimizer.step()
```

### 2. 验证阶段

验证阶段只评估模型，不更新参数：

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

`model.train()` 和 `model.eval()` 会影响 BatchNorm、Dropout 等模块。验证时使用 `torch.no_grad()` 可以减少显存占用。

### 3. 验证阶段为什么需要混淆矩阵

分割指标不能只计算每张图的预测类别数量。需要累计整个验证集的：

- TP，真实类别和预测类别都为该类
- FP，预测为该类但真实不是该类
- FN，真实为该类但预测成其他类

然后计算：

```text
IoU_k = TP_k / (TP_k + FP_k + FN_k)
Dice_k = 2TP_k / (2TP_k + FP_k + FN_k)
```

最后对有效类别取平均得到 mIoU 和 mean Dice。

## 三、先手写一个最小训练循环

在工程化之前，建议先理解下面这段最小代码。

```python
for images, masks, _ in train_loader:
    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)
    loss = criterion(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

它只解决一件事，也就是让模型完成一次参数更新。后面的 `train_one_epoch` 只是把这段逻辑封装起来，并增加 loss 统计和进度条。

## 四、从最小循环过渡到工程化

训练代码可以按照四个层次逐步扩展：

```text
单 batch 训练
    ↓
train_one_epoch
    ↓
validate_one_epoch
    ↓
fit + checkpoint
```

本文将你的 `train_one_epoch` 和 `validate_one_epoch` 扩展成完整训练框架，并适配以下三个接口。

```python
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_dir,
):
    ...

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
):
    ...

def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    ...
```

---

## 1. `engine.py` 的整体职责

`engine.py` 负责训练过程本身，主要完成以下工作。

- 单轮训练
- 单轮验证
- loss 与 mIoU、Dice、Pixel Accuracy 的统计
- 学习率更新
- 保存最新权重
- 保存验证集 mIoU 最优权重
- 断点续训

`train.py` 负责组装对象，`engine.py` 负责执行过程。

```text
train.py
    │
    ├── 构建 Dataset 和 DataLoader
    ├── 构建 UNet
    ├── 构建 criterion
    ├── 构建 optimizer 和 scheduler
    │
    └── fit()
          ├── train_one_epoch()
          ├── validate_one_epoch()
          ├── scheduler.step()
          ├── save last.pth
          └── save best_miou.pth
```

---

## 2. 你的单轮训练函数需要理解什么

一轮 epoch 指模型完整遍历一次训练集。每个 batch 的计算顺序是：

```text
取出 images 和 masks
        ↓
转移到 device
        ↓
model.train()
        ↓
logits = model(images)
        ↓
loss = criterion(logits, masks)
        ↓
optimizer.zero_grad()
        ↓
loss.backward()
        ↓
optimizer.step()
```

你的核心代码是正确的。

```python
images = images.to(device)
masks = masks.to(device)

logits = model(images)
loss = criterion(logits, masks)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

需要注意 `images.to(device)` 必须重新赋值。`to()` 通常返回转移后的新张量。

```python
images = images.to(device)
```

下面这种写法不会改变原变量指向的张量。

```python
images.to(device)
```

---

## 3. 推荐的 `train_one_epoch`

```python
from tqdm import tqdm
import torch


def train_one_epoch(
    model,
    train_dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(
        train_dataloader,
        desc="Training",
        leave=False,
    )

    for images, masks, _ in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 前向传播
        logits = model(images)

        # logits 形状通常为 [B, C, H, W]
        # masks 形状通常为 [B, H, W]
        loss = criterion(logits, masks)

        # 清空上一轮梯度
        optimizer.zero_grad(set_to_none=True)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if total_samples == 0:
        raise RuntimeError("train_dataloader 为空，无法计算训练损失")

    average_loss = total_loss / total_samples
    return average_loss
```

### 为什么使用 `loss.item() * batch_size`

`CrossEntropyLoss` 默认返回一个 batch 内有效像素损失的平均值。训练过程中需要得到整个数据集的平均 loss，因此可以先乘以 batch 大小，再除以样本总数。

```python
average_loss = total_loss / total_samples
```

如果不同图像的尺寸不同，严格来说还可以按有效像素数统计。你的 LoveDA 训练通常采用统一裁剪尺寸，因此按 batch 样本数统计已经足够用于训练曲线。

---

## 4. 混淆矩阵更新函数

验证阶段不能直接对 `pred` 和 `masks` 求平均，需要先累计混淆矩阵。

设真实类别为 `target`，预测类别为 `pred`，混淆矩阵第 `i` 行第 `j` 列表示：

```text
真实类别为 i，预测类别为 j 的像素数量
```

```python
def update_confusion_matrix(
    confmat,
    pred,
    target,
    num_classes,
    ignore_index=255,
):
    """累计一个 batch 的语义分割混淆矩阵。"""

    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    # 去除 ignore 像素
    valid = target != ignore_index

    # 去除越界标签，防止 bincount 产生非法类别
    valid &= target >= 0
    valid &= target < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes

    target = target[valid]
    pred = pred[valid]

    if target.numel() == 0:
        return

    # 将二维索引 target * C + pred 映射为一维索引
    index = target * num_classes + pred
    hist = torch.bincount(
        index,
        minlength=num_classes * num_classes,
    )

    confmat += hist.reshape(num_classes, num_classes)
```

`ignore_index=255` 的像素不会参与 mIoU、Dice 和 Pixel Accuracy 的计算。

---

## 5. 推荐的 `validate_one_epoch`

```python
@torch.no_grad()
def validate_one_epoch(
    model,
    val_dataloader,
    criterion,
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

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(
        val_dataloader,
        desc="Validate",
        leave=False,
    )

    for images, masks, _ in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        pred = torch.argmax(logits, dim=1)

        update_confusion_matrix(
            confmat=confmat,
            pred=pred,
            target=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    if total_samples == 0:
        raise RuntimeError("val_dataloader 为空，无法计算验证损失")

    tp = torch.diag(confmat).float()
    fp = confmat.sum(dim=0).float() - tp
    fn = confmat.sum(dim=1).float() - tp

    union = tp + fp + fn

    # 只对验证集中出现过的类别计算 IoU
    valid_iou = union > 0
    iou = torch.zeros_like(tp)
    iou[valid_iou] = tp[valid_iou] / union[valid_iou]

    # Dice = 2TP / (2TP + FP + FN)
    dice_denominator = 2 * tp + fp + fn
    valid_dice = dice_denominator > 0
    dice = torch.zeros_like(tp)
    dice[valid_dice] = (
        2 * tp[valid_dice] / dice_denominator[valid_dice]
    )

    # 计算指标时分别使用自己的有效类别掩码
    miou = iou[valid_iou].mean() if valid_iou.any() else torch.tensor(0.0, device=device)
    mean_dice = dice[valid_dice].mean() if valid_dice.any() else torch.tensor(0.0, device=device)

    valid_pixels = confmat.sum().clamp_min(1)
    pixel_acc = tp.sum() / valid_pixels

    average_loss = total_loss / total_samples

    metrics = {
        "loss": average_loss,
        "miou": miou.item(),
        "dice": mean_dice.item(),
        "pixel_acc": pixel_acc.item(),
        "per_class_iou": iou.detach().cpu().tolist(),
    }

    return metrics
```

### 这里需要特别注意

你的原代码中写的是：

```python
mean_dice = dice[valid_iou].mean()
```

更严谨的写法是：

```python
mean_dice = dice[valid_dice].mean()
```

因为 IoU 和 Dice 的有效分母虽然通常同时有效，但它们代表不同的统计条件。分别使用对应的掩码更清晰。

---

## 6. 保存 checkpoint

一个 checkpoint 至少保存以下内容。

- 当前 epoch
- 模型参数
- 优化器状态
- 当前指标

```python
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    scheduler=None,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)
```

### 为什么保存优化器状态

模型参数决定网络学到了什么。优化器状态还保存了动量、学习率调整等信息。如果只保存模型参数，重新加载后可以继续推理，但续训时训练轨迹可能发生变化。

---

## 7. 加载 checkpoint

```python
def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    start_epoch = checkpoint.get("epoch", 0) + 1
    metrics = checkpoint.get("metrics", {})

    return start_epoch, metrics
```

### 两种加载方式

只加载模型用于测试。

```python
load_checkpoint(
    path="runs/unet/weights/best_miou.pth",
    model=model,
    device=device,
)
```

加载模型和优化器用于继续训练。

```python
start_epoch, old_metrics = load_checkpoint(
    path="runs/unet/weights/last.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)
```

---

## 8. `fit` 的作用

`fit` 是 epoch 级别的外层循环。它不负责具体的 batch 计算，而是管理训练过程。

每个 epoch 的顺序如下。

```text
train_one_epoch
        ↓
validate_one_epoch
        ↓
scheduler.step()
        ↓
保存 last.pth
        ↓
如果当前 mIoU 更高，保存 best_miou.pth
```

```python
from pathlib import Path


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_dir,
    num_classes,
    ignore_index=255,
    start_epoch=1,
):
    save_dir = Path(save_dir)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    history = []

    model.to(device)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\\nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model=model,
            train_dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            val_dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        # ReduceLROnPlateau 需要传入验证指标
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "miou": val_metrics["miou"],
            "dice": val_metrics["dice"],
            "pixel_acc": val_metrics["pixel_acc"],
            "lr": current_lr,
        }
        history.append(epoch_metrics)

        print(
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"mIoU={val_metrics['miou']:.4f}, "
            f"Dice={val_metrics['dice']:.4f}, "
            f"PixelAcc={val_metrics['pixel_acc']:.4f}, "
            f"lr={current_lr:.6g}"
        )

        save_checkpoint(
            path=weights_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_metrics,
            scheduler=scheduler,
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]

            save_checkpoint(
                path=weights_dir / "best_miou.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                scheduler=scheduler,
            )

            print(f"保存最佳权重，best_miou={best_miou:.4f}")

    return history
```

### `last.pth` 和 `best_miou.pth` 的区别

`last.pth` 保存最后一个 epoch 的状态，适合中断后继续训练。

`best_miou.pth` 保存验证集 mIoU 最高的状态，适合最终测试和推理。

---

## 9. 一个需要注意的参数问题

你最初要求的 `fit` 接口中没有 `num_classes` 和 `ignore_index`，但验证函数需要知道类别数，并且需要处理 `255` 标签。

因此工程中建议把接口扩展为：

```python
fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_dir,
    num_classes,
    ignore_index=255,
)
```

也可以把这两个值封装到一个 `metric` 对象中，但初学阶段直接传入更容易理解。

---

## 10. `train.py` 中如何调用

```python
from pathlib import Path
import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_model(config)
criterion = build_loss(config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["train"]["lr"],
    weight_decay=config["train"].get("weight_decay", 1e-4),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["train"]["epochs"],
)

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=config["train"]["epochs"],
    save_dir=Path("runs/unet_512_bs8"),
    num_classes=config["data"]["num_classes"],
    ignore_index=config["data"].get("ignore_index", 255),
)
```

---

## 11. 断点续训的完整写法

```python
start_epoch = 1

resume_path = "runs/unet_512_bs8/weights/last.pth"

if Path(resume_path).exists():
    start_epoch, old_metrics = load_checkpoint(
        path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print(f"从第 {start_epoch} 轮继续训练")

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=100,
    save_dir="runs/unet_512_bs8",
    num_classes=7,
    ignore_index=255,
    start_epoch=start_epoch,
)
```

如果 checkpoint 保存的是第 20 轮，`load_checkpoint` 会返回 `21`，这样不会重复训练第 20 轮。

---

## 12. 推荐的 `engine.py` 文件结构

```text
rsunet/engine.py
├── import torch
├── import tqdm
├── train_one_epoch()
├── update_confusion_matrix()
├── validate_one_epoch()
├── save_checkpoint()
├── load_checkpoint()
└── fit()
```

完整调用关系是：

```text
fit()
│
├── train_one_epoch()
│     ├── model.train()
│     ├── forward
│     ├── criterion
│     ├── backward
│     └── optimizer.step
│
├── validate_one_epoch()
│     ├── model.eval()
│     ├── torch.no_grad()
│     ├── forward
│     ├── update_confusion_matrix
│     └── 计算 mIoU、Dice、Pixel Accuracy
│
├── scheduler.step()
├── save_checkpoint(last.pth)
└── save_checkpoint(best_miou.pth)
```

---

## 13. 你的当前代码需要记住的四个重点

1. `images` 和 `masks` 必须接收 `.to(device)` 的返回值。
2. 交叉熵的输入是 `criterion(logits, masks)`，不能把原图传入损失函数。
3. `validate_one_epoch` 必须使用 `model.eval()` 和 `@torch.no_grad()`。
4. `best_miou.pth` 用于最终测试，`last.pth` 用于中断后续训。

当这一套代码跑通后，训练入口只需要负责构建组件并调用 `fit`，训练循环就不会继续堆积在 `train.py` 中。
