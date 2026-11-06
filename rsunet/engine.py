from curses import noecho
from tkinter import NO

from tqdm import tqdm
import torch
from .metrics import mean_iou, update_confusion_matrix
from rsunet import metrics

def train_one_epoch(
        model,
        train_dataloader,
        criterion,
        optimizer,
        device
):
    model.train()
    train_total_loss = 0.0
    train_total_samples = 0

    progress_bar = tqdm(
        train_dataloader,
        desc='Training'
    )
    print(f'使用设备:{device}')
    for images, masks, _ in progress_bar:
        batch_size = images.size(0)
        #转移到gpu进行训练
        images = images.to(device)
        masks = masks.to(device)
        #前向传播
        logits  = model(images)
        #计算损失
        loss = criterion(logits , masks)
        #清空梯度
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()

        train_total_loss += loss.item() * batch_size
        train_total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    average_loss = train_total_loss/train_total_samples
    return average_loss


@torch.no_grad()
def validate_one_epoch(
        model,
        val_dataloader,
        criterion,
        device,
        num_classes,
):
    model.eval()

    #初始化混淆矩阵
    confmat = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device
    )
    #初始化损失
    val_total_loss = 0.0
    val_total_samples = 0
    progress_bar = tqdm(
        val_dataloader,
        desc='Validate'
    )
    print(f'使用设备:{device}')

    for images, masks, _ in progress_bar:
        batch_size = images.size(0)
        #转移到gpu进行训练
        images = images.to(device)
        masks = masks.to(device)
        #前向传播
        logits = model(images)
        #计算损失
        loss = criterion(logits, masks)

        #先取出预测的类别,沿着预测的类别取
        pred = torch.argmax(logits, dim=1)
        #传入混淆矩阵的更新
        update_confusion_matrix(
            confmat,
            pred,
            masks,
            num_classes,
        )

        val_total_loss += loss.item() * batch_size
        val_total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    #计算三个指标
    tp = torch.diag(confmat).float()
    fp = confmat.sum(dim=0).float() - tp
    fn = confmat.sum(dim=1).float() - tp

    #初始化dice
    iou = torch.zeros_like(tp)
    #先计算iou
    union = tp + fp + fn
    #校验计算指标时	只对出现过的类别算 IoU = tp/(tp + fp + fn)，防止除0
    valid_iou = (union>0)
    #计算单类iou
    iou[valid_iou] = (
        tp[valid_iou] / union[valid_iou]
    )

    #初始化dice
    dice = torch.zeros_like(tp)
    #再计算dice分母  2*tp + fp + fn = union + tp
    dice_denominator = union + tp
    #校验分母       
    valid_dice = (dice_denominator > 0)
    #计算dice
    dice[valid_dice] = (
        2*tp[valid_dice]
        /dice_denominator[valid_dice]
    )

    #最终指标
    miou = iou[valid_iou].mean()
    mean_dice = dice[valid_dice].mean()
    #平均准确度,clamp_min(1)防止分母为 0
    pixel_acc = tp.sum() / confmat.sum().clamp_min(1)
    average_loss = val_total_loss/val_total_samples

    #定义指标字典
    metrics = {
        'loss': average_loss,
        'miou': miou,
        'mean_dice': mean_dice,
        'pixel_acc': pixel_acc,
        #转为原生列表需要在cpu中进行
        'per_class_iou': iou.detach().cpu().tolist()
    }
    return metrics


def save_checkpoint(
        save_path,
        epoch,
        model,
        optimizer,
        metrics,
        scheduler=None
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics
    }

    #填入要保存文件与保存路径
    torch.save(checkpoint, save_path)

def load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        device='cpu'
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    #加载model权重
    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    #加载优化器权重
    if optimizer is not None:
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

    #加载调度器权重
    if scheduler is not None:
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

    #获取断点待续起始的epoch
    start_epoch = checkpoint.get('epoch', 0) + 1

    #获取断点处的指标
    metrics = checkpoint.get('metrics')

    return start_epoch, metrics