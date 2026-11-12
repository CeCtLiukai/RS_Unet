import numpy as np
import torch
import yaml
from pathlib import Path
from rsunet.datasets.loveda_dataset import get_dataloaders, build_dataset
from rsunet.engine import train_one_epoch, validate_one_epoch, fit
from rsunet.losses import seg_ce
from rsunet.models.unet import build_model
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#设置随机种子
set_seed(42)

#路径设置
PROJECT_ROOT = Path(__file__).resolve().parent
config_path = PROJECT_ROOT/'configs'/'loveda_unet.yaml'

with open(config_path, mode='r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
#设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#模型设置
model = build_model(config)
model = model.to(device)
#数据加载
train_dataset = build_dataset(config, True)
val_dataset = build_dataset(config, False)

train_dataloader, val_dataloader = get_dataloaders(
    config,
    train_dataset,
    val_dataset,
)
#损失函数
criterion = seg_ce
#优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['optim']['lr'],
    weight_decay=config['optim']['weight_decay']
)
#调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['scheduler']['T_max'],
    eta_min=config['scheduler']['eta_min'],
)
#保存路径
save_dir = PROJECT_ROOT/'runs'/'unet_r50_512_bs8'

#训练论数
start_epoch = 1
epochs = 100

#数据配置
num_class = config['data']['num_classes']
ignore_index = config['data']['ignore_index']

#训练
print(f'使用设备:{device}')
history = fit(
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
    epochs
)