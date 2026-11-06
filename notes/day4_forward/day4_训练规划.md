
# RS-Unet 遥感分割项目工程结构规范
## 1. 数据集划分建议
LoveDA 数据集包含 Urban、Rural 两个域，验证集建议**同时包含两个域**，方便评估模型在不同场景下的泛化能力，目录结构：
```text
data/LoveDA/
├── Train/
│   ├── Urban/
│   └── Rural/
├── Val/
│   ├── Urban/
│   └── Rural/
└── Test/
    ├── Urban/
    └── Rural/
```
如果需要单独分析两个域性能，可以分开构建 DataLoader：
- `val_urban_loader`
- `val_rural_loader`
分别统计 Urban、Rural 的 mIoU。

> 注意：如果仅使用 `Val/Urban/` 做验证，指标只能代表城市场景，无法评估乡村场景效果，适合专门做跨域对比实验时使用。

## 2. 损失文件组织方案
### 初期（单损失，快速调试）
```text
rsunet/losses.py
```
### 后期（多损失扩展，Dice / Focal / 组合损失）
升级为包目录：
```text
rsunet/
├── losses/
│   ├── __init__.py
│   ├── cross_entropy.py
│   ├── dice.py
│   ├── focal.py
│   └── combined.py
```

## 3. engine.py 接口设计与职责划分
`engine.py` 专门封装训练、验证、断点保存加载逻辑，不和模型、数据集、主启动脚本耦合。
推荐函数清单：
```python
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    ...

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    metric,
    device,
):
    ...

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

### 各函数职责
- `train_one_epoch`：遍历一轮训练集，前向、算损失、反向、更新参数
- `validate_one_epoch`：遍历一轮验证集，**不更新参数**，计算 loss + 分割指标
- `fit`：epoch 外层循环，依次调用训练、验证，负责保存权重、记录指标
- `save_checkpoint`：序列化保存模型权重、优化器状态、当前轮次、指标（用于续训 & 保存最优权重）
- `load_checkpoint`：加载断点，支持继续训练或仅加载权重做推理测试

### 调用关系
```text
train.py
    ↓ fit()
        ├── train_one_epoch()
        ├── validate_one_epoch()
        ├── 计算 mIoU
        ├── 保存 best_miou.pth
        └── 保存 last.pth
```

## 4. train.py 职责：仅做组装，不写训练循环
`train.py` 只负责读取配置、构建数据集、dataloader、模型、损失、优化器，然后调用 `fit`，示例骨架：
```python
config = load_config(...)

train_dataset = build_dataset(...)
val_dataset = build_dataset(...)

train_loader = build_dataloader(...)
val_loader = build_dataloader(...)

model = build_model(config)
criterion = build_loss(config)
optimizer = build_optimizer(config)

fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=config["train"]["epochs"],
    save_dir=run_dir,
)
```

## 5. 完整项目目录树
```text
RS_Unet/
├── configs/
│   ├── loveda_unet.yaml
│   └── loveda_unet_r50.yaml
├── rsunet/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── loveda.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   └── backbones/
│   ├── losses.py
│   ├── metrics.py
│   └── engine.py
├── scripts/
│   ├── prepare_loveda.py
│   └── eval_official.py
├── train.py
├── test.py
├── data/
├── runs/
├── .gitignore
└── README.md
```

### 模块分工说明
- `datasets`：数据集扫描、文件配对、标签映射、样本读取
- `models`：UNet 网络、骨干网络定义
- `losses.py`：损失函数实现（CE / Dice / Focal）
- `metrics.py`：分割评价指标（mIoU、aAcc、逐类别 IoU）
- `engine.py`：训练验证循环、权重读写
- `train.py`：训练实验入口
- `test.py`：加载权重，推理、测试评估入口
- `scripts`：数据预处理脚本、对接官方评测脚本
- `configs`：yaml 配置文件，统一管理超参
- `runs`：日志、TensorBoard、权重输出目录

## 6. 最小可运行链路（优先实现顺序）
先打通基础链路，保证前向、损失、反向可以正常跑通，再逐步叠加功能：
```text
loveda.py
    ↓ transforms.py
    ↓ unet.py
    ↓ losses.py 手写交叉熵损失 seg_ce
    ↓ engine.py train_one_epoch
    ↓ engine.py validate_one_epoch
    ↓ train.py
```
先保证下面核心代码正常执行无报错：
```python
logits = model(images)
loss = criterion(logits, masks)
loss.backward()
optimizer.step()
```

基础链路跑通之后，再依次增加：
- mIoU 指标计算
- 学习率调度器 scheduler
- 最优权重 `best_miou.pth` 保存逻辑
- TensorBoard 日志记录
- 断点续训
- 测试集大图推理、结果保存

## 7. 开发节奏建议
1. 完成 Dataset + Transform，通过 data_check 脚本核验数据合法性
2. 模型前向测试，核对 logits shape
3. 损失函数调试，解决标签越界、ignore_index 过滤问题
4. 实现 engine 内单 epoch 训练、验证
5. 写 train.py 组装全部组件，跑通第一轮训练
6. 接入指标、日志、权重保存
7. 最后开发 test.py、scripts 评测脚本

> 该结构可以直接作为遥感分割论文实验工程的基础，后续更换模型、损失、数据集时改动边界清晰，易于维护和复现。