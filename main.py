from rsunet.datasets.loveda_dataset import get_dataloaders, LoveDADataset, build_dataset
from torch import optim
from rsunet.models import unet
from rsunet.losses import seg_ce
from pathlib import Path
import yaml
from tqdm import tqdm
import torch
from rsunet.losses import seg_ce
from rsunet.models.unet import UNet, build_model
from rsunet.engine import train_one_epoch, validate_one_epoch
from rsunet.metrics import mean_iou
PROJECT_ROOT = Path(__file__).resolve().parent
config_path = PROJECT_ROOT/'configs'/'loveda_unet.yaml'

with open(config_path, mode='r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

#数据生成器
train_dataloader, val_dataloader = get_dataloaders(
    config=config,
    train_dataset=build_dataset(config, True),
    val_dataset=build_dataset(config, False)
)

#设备选择
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#构造model
model = build_model(config).to(device)
#构造优化器
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=config['optim']['lr'],
    weight_decay=config['optim']['weight_decay'],
)
#损失函数
criterion = seg_ce
 
avg_loss,miou,mean_dice,pixel_acc = validate_one_epoch(
    model,
    val_dataloader,
    criterion,
    device,
    config['data']['num_classes']
)
print(f'平均损失:{avg_loss:.4f}\n平均iou:{miou:.4f}\n平均dice:{mean_dice:.4f}\n平均准确度:{pixel_acc:.4f}\n')