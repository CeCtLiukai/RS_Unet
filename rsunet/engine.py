from tqdm import tqdm
import torch
from .metrics import mean_iou, update_confusion_matrix
from rsunet import metrics
from pathlib import Path


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, device, ignore_index=255
):
    model.train()
    train_total_loss = 0.0
    train_total_samples = 0

    progress_bar = tqdm(train_dataloader, desc="Training")
    for images, masks, _ in progress_bar:
        batch_size = images.size(0)
        # 转移到gpu进行训练
        images = images.to(device)
        masks = masks.to(device)
        # 前向传播
        logits = model(images)
        # 计算损失
        loss = criterion(logits, masks, ignore_index)
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        train_total_loss += loss.item() * batch_size
        train_total_samples += batch_size

        progress_bar.set_postfix(loss=f"{(train_total_loss/train_total_samples):.4f}")
    average_loss = train_total_loss / train_total_samples
    return average_loss


@torch.no_grad()
def validate_one_epoch(
    model, val_dataloader, criterion, device, num_classes, ignore_index=255
):
    model.eval()

    # 初始化混淆矩阵
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    # 初始化损失
    val_total_loss = 0.0
    val_total_samples = 0
    progress_bar = tqdm(val_dataloader, desc="Validate")

    for images, masks, _ in progress_bar:
        batch_size = images.size(0)
        # 转移到gpu进行训练
        images = images.to(device)
        masks = masks.to(device)
        # 前向传播
        logits = model(images)
        # 计算损失
        loss = criterion(logits, masks, ignore_index)

        # 先取出预测的类别,沿着预测的类别取
        pred = torch.argmax(logits, dim=1)
        # 传入混淆矩阵的更新
        confmat = update_confusion_matrix(
            confmat,
            pred,
            masks,
            num_classes,
        )

        val_total_loss += loss.item() * batch_size
        val_total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{(val_total_loss/val_total_samples):.4f}",
        )

    # 计算三个指标
    tp = torch.diag(confmat).float()
    fp = confmat.sum(dim=0).float() - tp
    fn = confmat.sum(dim=1).float() - tp

    # 初始化dice
    iou = torch.zeros_like(tp)
    # 先计算iou
    union = tp + fp + fn
    # 校验计算指标时	只对出现过的类别算 IoU = tp/(tp + fp + fn)，防止除0
    valid_iou = union > 0
    # 计算单类iou
    iou[valid_iou] = tp[valid_iou] / union[valid_iou]

    # 初始化dice
    dice = torch.zeros_like(tp)
    # 再计算dice分母  2*tp + fp + fn = union + tp
    dice_denominator = union + tp
    # 校验分母
    valid_dice = dice_denominator > 0
    # 计算dice
    dice[valid_dice] = 2 * tp[valid_dice] / dice_denominator[valid_dice]

    # 最终指标
    miou = iou[valid_iou].mean()
    mean_dice = dice[valid_dice].mean()
    # 平均准确度,clamp_min(1)防止分母为 0
    pixel_acc = tp.sum() / confmat.sum().clamp_min(1)
    average_loss = val_total_loss / val_total_samples

    # 定义指标字典
    metrics = {
        "val_avg_loss": float(average_loss),
        "miou": miou.item(),
        "mean_dice": mean_dice.item(),
        "pixel_acc": pixel_acc.item(),
        "per_class_iou": iou.detach().cpu().tolist(),
        "confmat": confmat.detach().cpu().numpy(),
    }
    return metrics


def save_checkpoint(save_path, epoch, model, optimizer, metrics, scheduler=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "metrics": metrics,
    }

    # 填入要保存文件与保存路径
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载model权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器权重
    if optimizer is not None:
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

    # 加载调度器权重
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

    # 获取断点待续起始的epoch
    start_epoch = checkpoint.get("epoch", 0) + 1

    # 获取断点处的指标
    metrics = checkpoint.get("metrics")

    return start_epoch, metrics


def fit(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    num_class,
    ignore_index,
    device,
    save_dir,
    start_epoch,
    epochs,
):
    """
    :param model: 模型 unet
    :param train_dataloader: 训练集dataloader
    :param val_dataloader:  验证集dataloader
    :param criterion:   损失函数
    :param optimizer:  优化器
    :param scheduler:  学习率调度器
    :param num_class:  目标种类
    :param ignore_index:  背景/未标注 标签映射
    :param device:  设备
    :param save_dir: 权重保存文件夹
    :param start_epoch:  开始轮次
    :param epochs:  总轮次
    :return: 训练历史指标
    """

    # 设定权重保存路径
    save_dir = Path(save_dir)
    save_weights_dir = save_dir / "weights"
    save_weights_dir.mkdir(parents=True, exist_ok=True)

    #前置准备
    model.to(device)

    #指标初始化
    best_iou = -1.0
    history = []

    # epoch循环
    for epoch in range(start_epoch, epochs + 1):
        print(f'\nEpoch[{epoch}/{epochs}]')

        train_avg_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, ignore_index
        )

        val_epoch_metrics = validate_one_epoch(
            model, val_dataloader, criterion, device, num_class, ignore_index
        )

        #学习率单轮调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_metrics['val_avg_loss'])
            else:
                scheduler.step()

        #查看当前的学习率
        current_lr = optimizer.param_groups[0]['lr']

        #保存当前指标
        epoch_metrics = {
            'epoch': epoch,
            'lr': current_lr,
            'train_avg_loss': train_avg_loss,
            'val_avg_loss':val_epoch_metrics['val_avg_loss'],
            'miou': val_epoch_metrics['miou'],
            'mean_dice': val_epoch_metrics['mean_dice'],
            'pixel_acc': val_epoch_metrics['pixel_acc'],
            'per_class_iou': val_epoch_metrics['per_class_iou'],
            'confmat': val_epoch_metrics['confmat']
        }
        history.append(epoch_metrics)

        #打印
        print(
            f'train_avg_loss={train_avg_loss:.4f},'
            f'val_avg_loss={val_epoch_metrics["val_avg_loss"]:.4f},'
            f'mIoU={val_epoch_metrics["miou"]:.4f},'
            f'Dice={val_epoch_metrics["mean_dice"]:.4f},'
            f'PixelAcc={val_epoch_metrics["pixel_acc"]:.4f},'
            f'lr={current_lr:.6g}'
        )

        #保存最后一轮指标
        save_checkpoint(
            save_weights_dir/'last.pth',
            epoch,
            model,
            optimizer,
            epoch_metrics,
            scheduler
        )

        #保存最优指标
        if epoch_metrics['miou'] > best_iou:
            best_iou = epoch_metrics['miou']

            save_checkpoint(
                save_weights_dir/'best_iou.pth',
                epoch,
                model,
                optimizer,
                epoch_metrics,
                scheduler
            )
            print(f'保存最佳权重至{str(save_weights_dir/"best_iou.pth")}\nbest_iou={best_iou:.4f}')

    return history