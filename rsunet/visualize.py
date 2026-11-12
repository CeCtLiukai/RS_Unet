from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#传入的数据很可能还在gpu上，先检查分离然后转为浮点数
def _to_float(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().item()  # 修复：detch -> detach，cup -> cpu
    return float(value)


def plot_training_curves(history, save_path):
    #创建文件保存路径
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)


    #获取各项数值
    epochs = [h['epoch'] for h in history]
    train_loss = [_to_float(h['train_avg_loss']) for h in history]
    val_loss = [_to_float(h['val_avg_loss']) for h in history]
    miou = [_to_float(h['miou']) for h in history]
    mean_dice = [_to_float(h['mean_dice']) for h in history]
    pixel_acc = [_to_float(h['pixel_acc']) for h in history]
    lr = [_to_float(h['lr']) for h in history]

    #创建画布，构建四个图（损失曲线，miou，（dice，pixel_acc），lr）
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    #创建损失函数曲线
    axes[0, 0].plot(
        epochs, 
        train_loss, 
        color="#7F17C4",
        marker='o',
        markersize=7,
        linewidth=1.5,
        label='Train_Loss'
    )
    axes[0, 0].plot(
        epochs, 
        val_loss, 
        color="#4496C9",
        marker='^',
        markersize=7,
        linewidth=1.5,
        label='Val_Loss'
    )
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')  # 修复：Epcoh -> Epoch
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    #创建miou曲线
    axes[0, 1].plot(
        epochs,
        miou,
        color="#37C41B",
        label='mIoU',
        marker='o',
        markersize=7,
        linewidth=1.5
    )
    axes[0, 1].set_title('mIoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    #控制y轴的范围
    axes[0, 1].set_ylim(0,1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    #绘制dice和pixel_acc
    axes[1,0].plot(
        epochs,
        mean_dice,
        color="#D2128F",
        marker='*',
        markersize=7,
        label='Dice',
        linewidth=1.5
    )
    axes[1,0].plot(
        epochs,
        pixel_acc,
        color="#12D2D2",
        marker='D',
        markersize=7,
        label='Pixel Accuracy',
        linewidth=1.5
    )
    axes[1,0].set_title('Segmentation Metrics')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim(0,1)
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)  # 修复：gird -> grid

    #绘制lr
    axes[1,1].plot(
        epochs,
        lr,
        color="#D89117D5",
        marker='*',
        markersize=7,
        label='Learning Rate',
        linewidth=1.5
    )
    axes[1,1].set_title('Learning Rate')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('LR')
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)  # 修复：gird -> grid

    #整体添加
    fig.suptitle("RS-Unet Training Curves")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(history, save_path, class_name=None, normalize=False):
    if not history:  # 修复：加上空列表保护
        return

    #创建文件保存路径
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confmat = np.asarray(
        history[-1]['confmat'],
        dtype=np.float64
    )

    if normalize:
        #对每一行数值进行归一化
        row_sum = confmat.sum(axis=1, keepdims=True)
        display_mat = np.divide(
            confmat,
            row_sum,
            out = np.zeros_like(confmat),
            where=(row_sum!=0)
        )

        value_format = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        display_mat = confmat  # 修复：去掉了末尾的逗号
        value_format = ".0f"
        title = "Confusion Matrix"

    num_class = confmat.shape[0]
    #如果没有传类别名称
    if class_name is None:
        class_name = [str(i) for i in range(confmat.shape[0])]

    #创建画布
    fig, axes = plt.subplots(figsize=(8, 7))
    image = axes.imshow(display_mat, cmap="Blues")
    #创建颜色棒,对哪张照片，哪个子图创建
    fig.colorbar(image, ax=axes)  # 修复：axes -> ax=axes

    #创建标注
    current_epoch = history[-1]['epoch'] # 提取出来防 f-string 嵌套报错
    axes.set_title(f'{title} - Epochs {current_epoch}')

    axes.set_xlabel('Predicted Class')
    axes.set_ylabel('True Class')

    #设置刻度与名称
    axes.set_xticks(np.arange(num_class))
    axes.set_xticklabels(class_name)
    axes.set_yticks(np.arange(num_class))
    axes.set_yticklabels(class_name)


    # 如果当前格子的数值大于这条线（说明颜色偏深），就用白字。
    # 如果当前格子的数值小于这条线（说明颜色偏浅），就用黑字。
    threshold = display_mat.max() / 2

    #填入数值
    for row in range(num_class):
        for col in range(num_class):
            #设置字体颜色
            color = (
                "white"
                if display_mat[row,col] > threshold
                else "black"
            ) 
            #填入数值
            axes.text(
                col,
                row,
                format(display_mat[row,col], value_format),
                ha="center",
                va="center",
                color=color,
            )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

def visualize_history(save_path, history, class_name = None):
    #创建文件保存路径
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        history=history,
        save_path=save_path
    )
    plot_confusion_matrix(
        history,
        save_path / "confusion_matrix.png",
        class_name=class_name,
        normalize=False,
    )

    plot_confusion_matrix(
        history,
        save_path / "confusion_matrix_normalized.png",
        class_name=class_name,
        normalize=True,
    )