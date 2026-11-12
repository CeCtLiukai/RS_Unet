# Checkpoint 保存与断点续训：核心专题笔记

- **学习主题**：模型训练时保存什么 / 读取什么 / 断点续训的完整流程
- **定位**：训练工程专题笔记，承接 day2《从数据到张量》——day2 解决“数据能进模型”，这份解决“训练中断了怎么接着跑”
- **前置**：day2 笔记（Dataset 返回 `img (3,H,W) float32`、`mask (H,W) int64`）；评价指标笔记（`metrics` 字典里有 `val_mIoU`）
- **使用方式**：先讲清“为什么”，再给“可以抄的代码”

> 文档格式说明：本文按 A4 排版优化，行宽控制在 80 字符以内，表格不超过 4 列，不依赖 Mermaid 等需要插件渲染的语法。

## 目录

- 专题 1：训练时到底要保存什么
- 专题 2：`state_dict` 基础——模型和优化器里有什么
- 专题 3：`torch.save` 保存的写法与文件本质
- 专题 4：`torch.load` 加载与断点续训完整流程
- 专题 5：常见报错与根本原因
- 附录：Checkpoint 速查卡

---

> 这份笔记针对“保存了但不知道存了什么、加载了但不知道顺序”的薄弱点集中补强。

## 专题 1：训练时到底要保存什么

一句话定位：**保存的是“训练现场”，不是只保存模型权重。**

### 1.1 训练现场包括什么

一次训练中断后，要想“接着上次继续跑”，需要恢复四样东西：

| 要恢复的东西 | 存在哪 | 不恢复会怎样 |
|---|---|---|
| 模型权重 | `model.state_dict()` | 模型回到随机初始化 |
| 优化器状态 | `optimizer.state_dict()` | 动量/二阶矩清零，训练轨迹变 |
| 学习率调度 | `scheduler.state_dict()` | 学习率节奏错位 |
| 训练进度 | `epoch`、`metrics` | 不知道从第几轮继续、best 是多少 |

### 1.2 两种保存目的，存的东西不同

| 目的 | 存什么 | 文件习惯命名 |
|---|---|---|
| 只做推理/发布 | 只存 `model.state_dict()` | `model.pth` |
| 断点续训 | 存 model + optimizer + scheduler + epoch + metrics | `last.pth`、`best.pth` |

**推理不需要优化器**，因为不再更新参数；**续训必须存优化器**，因为动量、二阶矩、当前学习率都在里面。

### 1.3 为什么要存 `epoch` 和 `metrics`

- `epoch` 是权重的“时间戳”，加载后知道从第几轮继续（`start_epoch = epoch + 1`）；
- `metrics` 是评估依据，事后能知道 `best.pth` 对应哪一轮、mIoU 多少；
- 保存 `best` 时要先比较指标，不能无脑覆盖。

---

## 专题 2：`state_dict` 基础——模型和优化器里有什么

一句话定位：**`state_dict` 是 PyTorch 规定的“状态字典”，模型和优化器各有一套，结构不同、用途不同。**

### 2.1 模型里有什么

一个模型内部有两类东西：

- **参数（parameters）**：`weight`、`bias`，需要训练的张量；
- **缓冲区（buffers）**：如 BatchNorm 的 `running_mean`，不训练但随训练更新。

`model.state_dict()` 把它们按“名字 → 张量”列出来：

```python
model = nn.Linear(3, 2)
model.state_dict()
# OrderedDict([
#   ('weight', tensor([[...], [...]])),   # 形状 (2,3)
#   ('bias',   tensor([...]))             # 形状 (2,)
# ])
```

关键点：

- 键是**参数名**，值是**参数张量本身**；
- 是扁平的字典，没有嵌套；
- 只存权重和 buffer，不存模型结构、优化器、epoch。

### 2.2 优化器里有什么

优化器在训练时要维护额外状态：

- SGD + momentum：每个参数的**动量** `momentum_buffer`；
- Adam：每个参数的 `step`、一阶矩 `exp_avg`、二阶矩 `exp_avg_sq`；
- 还有超参数：学习率 `lr`、`momentum`、`weight_decay` 等。

`optimizer.state_dict()` 把这些列出来：

```python
{
    "state": {
        0: {"momentum_buffer": tensor(...)},   # 参数索引 0 的状态
        1: {"momentum_buffer": tensor(...)},   # 参数索引 1 的状态
    },
    "param_groups": [
        {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "params": [0, 1],   # 这个组包含哪些参数（索引）
        }
    ]
}
```

关键点：

- `state` 的键是**参数索引**，不是参数名；
- `param_groups["params"]` 是参数索引列表，说明这个组包含哪些参数；
- `param_groups["lr"]` 是**保存那一刻**的学习率，不是历史；
- 历史学习率由 `scheduler.state_dict()` 和训练日志记录。

### 2.3 为什么断点续训要存优化器

只存模型权重，断点续训会“失忆”：

| 只存 model | 同时存 model + optimizer |
|---|---|
| 权重恢复，但动量清零 | 动量、二阶矩恢复，训练轨迹连续 |
| 学习率从初始值开始 | `param_groups` 里的 `lr` 恢复 |
| Adam 的 `exp_avg` 丢失，前期更新偏差大 | Adam 状态连续，收敛更稳 |

### 2.4 `scheduler.state_dict()`

保存调度器内部状态，如 `last_epoch`、`step_size`、`gamma`，用于恢复后继续按原计划衰减。

只恢复 optimizer，不恢复 scheduler，学习率会从当前值开始，但衰减节奏可能错位。

### 2.5 一个可运行的打印例子

```python
import torch
import torch.nn as nn

model = nn.Linear(3, 2)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)

x = torch.randn(4, 3)
y = torch.randn(4, 2)
loss = nn.MSELoss()(model(x), y)
loss.backward()
optimizer.step()

print("model.state_dict():")
for k, v in model.state_dict().items():
    print(f"  {k}: {tuple(v.shape)}")

print("\noptimizer.state_dict():")
sd = optimizer.state_dict()
for idx, st in sd["state"].items():
    print(f"  参数索引 {idx}: {list(st.keys())}")
for g in sd["param_groups"]:
    print(f"  lr={g['lr']}, params={g['params']}")
```

运行后会看到：

- `model.state_dict()` 的键是 `weight`、`bias`；
- `optimizer.state_dict()["state"]` 的键是 `0`、`1`；
- `param_groups[0]["lr"] = 0.1`，`params = [0, 1]`。

---

## 专题 3：`torch.save` 保存的写法与文件本质

一句话定位：**`torch.save` 把整个 `checkpoint` 字典序列化成二进制文件，不是文本，必须用 `torch.load` 读回。**

### 3.1 保存的写法

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    "metrics": metrics,
}
torch.save(checkpoint, path)
```

### 3.2 保存出来是什么文件

- **二进制文件**，底层是 Python `pickle` + PyTorch 张量专用序列化；
- 不是文本，不能用 `json.load` 读；
- 用文本编辑器打开是乱码；
- 必须用 `torch.load` 读回。

### 3.3 后缀约定

| 后缀 | 习惯用途 | 说明 |
|---|---|---|
| `.pth` | 模型权重或 checkpoint | 最常见 |
| `.pt` | 模型权重或整个模型 | 和 `.pth` 等价 |
| `.ckpt` | 训练检查点 | 强调含 optimizer、epoch |
| `.bin` | 通用二进制 | 也有人用 |

后缀**不影响内容**，`.pth` 改成 `.ckpt` 还是同一个二进制格式。

### 3.4 注意点

- `torch.save` **不会自动创建父目录**，先 `mkdir(parents=True, exist_ok=True)`；
- 文件已存在会**直接覆盖**；
- `metrics` 里最好放原生类型（`.item()` / `.tolist()`），避免跨设备加载报错。

```python
from pathlib import Path

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, save_dir / "best.pth")
```

### 3.5 `save_checkpoint` 的封装

```python
def save_checkpoint(path, epoch, model, optimizer,
                    scheduler, metrics):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "metrics": {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items()
        },
    }
    torch.save(checkpoint, path)
```

---

## 专题 4：`torch.load` 加载与断点续训完整流程

一句话定位：**先重建同结构模型和优化器，再按序 `load_state_dict`，最后从 `ckpt["epoch"] + 1` 继续训练。**

### 4.1 `torch.load` 能加载回来吗

能，原样加载回来：

```python
ckpt = torch.load(path, map_location=device)

epoch           = ckpt["epoch"]
model_state     = ckpt["model_state_dict"]
optim_state     = ckpt["optimizer_state_dict"]
scheduler_state = ckpt["scheduler_state_dict"]
metrics         = ckpt["metrics"]
```

注意：

- **`map_location`**：GPU 保存、CPU 加载时要加，否则报设备不匹配；
- **`weights_only=True`**：新版本 PyTorch 推荐加，只加载张量和简单对象，更安全；
- **`None` 要判断**：保存时是 `None`，加载回来还是 `None`，不能直接 `load_state_dict(None)`。

### 4.2 加载：先重建，再恢复

**顺序不能反**：必须先创建同结构的模型和优化器，再 `load_state_dict`。

```python
# 1. 先建模型和优化器（结构必须和保存时一致）
model = UNet(num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# 2. 加载 checkpoint
ckpt = torch.load("checkpoints/last.pth", map_location=device)

# 3. 逐项恢复
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
if ckpt.get("scheduler_state_dict") is not None:
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

# 4. 从下一轮开始
start_epoch = ckpt["epoch"] + 1
best_metric = ckpt.get("metrics", {}).get("val_mIoU", 0.0)
```

### 4.3 训练循环从 `start_epoch` 接上

```python
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
    metrics = validate_one_epoch(...)
    scheduler.step()

    save_checkpoint("last.pth", epoch, model, optimizer, scheduler, metrics)
    if metrics["val_mIoU"] > best_metric:
        best_metric = metrics["val_mIoU"]
        save_checkpoint("best.pth", epoch, model, optimizer, scheduler, metrics)
```

### 4.4 断点续训的注意点

| 注意点 | 说明 |
|---|---|
| `map_location` | GPU 存、CPU 或另一张卡加载时必加 |
| `weights_only=True` | 新版本推荐，更安全 |
| scheduler 为 `None` | 保存存 `None`，加载要判断 `is not None` |
| `metrics` 转原生类型 | 保存前 `.item()` / `.tolist()` |
| 先建对象再加载 | `load_state_dict` 前必须先实例化同结构模型 |
| 学习率可能错位 | 某些 scheduler 恢复时 `last_epoch` 有已知问题 |
| `start_epoch = ckpt["epoch"] + 1` | 别从保存那一轮重复跑 |

### 4.5 一个最小可运行例子

```python
import torch
import torch.nn as nn
from pathlib import Path

# 1. 建模型和优化器
model = nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 2. 模拟训练一步
x = torch.randn(4, 3)
y = torch.randn(4, 2)
loss = nn.MSELoss()(model(x), y)
loss.backward()
optimizer.step()

# 3. 保存
Path("checkpoints").mkdir(exist_ok=True)
torch.save({
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": {"val_mIoU": 0.5},
}, "checkpoints/last.pth")

# 4. 重新建对象（模拟新进程）
model2 = nn.Linear(3, 2)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)

# 5. 加载
ckpt = torch.load("checkpoints/last.pth", map_location="cpu")
model2.load_state_dict(ckpt["model_state_dict"])
optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

start_epoch = ckpt["epoch"] + 1
best_metric = ckpt["metrics"]["val_mIoU"]
print(f"从第 {start_epoch} 轮继续，best mIoU = {best_metric}")
```

---

## 专题 5：常见报错与根本原因

### 5.1 保存时找不到目录

```text
FileNotFoundError: [Errno 2] No such file or directory
```

原因：`torch.save` 不建父目录。解决：

```python
from pathlib import Path
Path("checkpoints").mkdir(parents=True, exist_ok=True)
```

### 5.2 加载设备不匹配

```text
RuntimeError: Expected all tensors to be on the same device
```

原因：GPU 保存，CPU 加载。解决：

```python
ckpt = torch.load(path, map_location="cpu")
```

### 5.3 `load_state_dict` 键不匹配

```text
RuntimeError: Error(s) in loading state_dict for UNet:
Missing key(s) in state_dict: ...
Unexpected key(s) in state_dict: ...
```

原因：模型结构或参数名和保存时不一致。解决：确认模型定义、类别数、模块名一致。

### 5.4 `optimizer.load_state_dict(None)`

```text
TypeError: 'NoneType' object is not subscriptable
```

原因：保存时 `optimizer_state_dict` 是 `None`，加载直接传了 `None`。解决：先判断 `is not None`。

### 5.5 SGD 报“缺少 lr”

```text
ValueError: parameter group didn't specify a value of required optimization parameter lr
```

原因：传参数组列表时每组没写 `"lr"`，构造也没给默认值。解决：

```python
optimizer = torch.optim.SGD([
    {"params": [model.weight], "lr": 1e-4},
    {"params": [model.bias],   "lr": 1e-2},
], momentum=0.9)
```

### 5.6 常见坑速查

| 坑 | 原因 | 解法 |
|---|---|---|
| 保存文件找不到目录 | `torch.save` 不建目录 | 先 `mkdir(parents=True)` |
| 加载设备不匹配 | GPU 存 CPU 加载 | `map_location=device` |
| 键不匹配 | 模型结构不一致 | 确认模型定义、类别数一致 |
| `None` 报错 | 保存时是 `None` | 先判断 `is not None` |
| SGD 缺少 lr | 参数组没写 lr | 每组写 `"lr"` 或构造给默认值 |
| metrics 加载报错 | 存了 tensor | 保存前 `.item()` / `.tolist()` |

---

## 附录：Checkpoint 速查卡

### 键的作用

| 键 | 内容 | 加载方式 |
|---|---|---|
| `epoch` | 当前 epoch | `start_epoch = ckpt["epoch"] + 1` |
| `model_state_dict` | 模型权重 | `model.load_state_dict(...)` |
| `optimizer_state_dict` | 动量/二阶矩 + lr | `optimizer.load_state_dict(...)` |
| `scheduler_state_dict` | 调度器状态 | `scheduler.load_state_dict(...)` |
| `metrics` | 指标字典 | 读取、比较 best |

### `optimizer.state_dict()` 结构

```text
{
    "state": {参数索引: {优化器状态}},
    "param_groups": [
        {"lr": 当前学习率, "params": [参数索引列表], ...}
    ]
}
```

### 保存与加载口诀

```text
torch.save 二进制，后缀习惯不影响
checkpoint 存五样：epoch、model、optimizer、scheduler、metrics
model.state_dict 存权重，optimizer.state_dict 存动量和 lr
torch.load 原样读回，map_location 防设备错
断点续训先建对象，再 load_state_dict
start_epoch 记得 +1，别从保存那轮重跑
None 要先判断，metrics 先转原生类型
```

### 三个“形同意不同”的陷阱

| 词汇 | 含义 | 注意 |
|---|---|---|
| `model.state_dict()` | 参数名 → 权重 | 扁平字典，不含优化器 |
| `optimizer.state_dict()` | 参数索引 → 动量/二阶矩 | 嵌套字典，含当前 lr |
| `torch.save(model)` | 整个模型对象 | 依赖代码结构，不推荐 |