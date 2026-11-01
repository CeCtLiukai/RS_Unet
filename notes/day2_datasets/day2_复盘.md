# Day 2 复盘：Dataset 与 Transform 错题本

检查对象：

- `notes/day2_datasets/AugDataset_practice.py`
- `notes/day2_datasets/MiniTransform_practice.py`

结论：**10 个问题，其中 4 个会直接报错，3 个会静默产生错误结果。** 已全部修复，修复后脚本可运行。

---

## 一、问题总览

| 编号 | 位置 | 严重度 | 一句话 |
|---|---|---|---|
| A1 | `AugDataset.__init__` | 崩溃 | `str` 当 `Path` 用，`is_dir()` 直接报错 |
| A2 | `_get_samples` | 崩溃 | 目录不存在就 `raise`，`Val/Rural` 必炸 |
| A3 | `get_dataloaders` | 崩溃 | `persistent_workers=True` 与 `num_workers=0` 冲突 |
| A4 | `get_dataloaders` | 崩溃 | 验证集传了 `('Urban','Rural')`，触发 A2 |
| B1 | `CLASS_NUM` | 静默错误 | 8 个类别名，整体后移一位 |
| B2 | `get_dataloaders` | 静默错误 | `remap_zero_ignore=True` 硬编码，与配置相反 |
| B3 | `_get_samples` | 静默错误 | 孤儿文件只警告，随后被交集静默丢弃 |
| B4 | `get_dataloaders` | 指标偏差 | 验证集复用训练的 `batch_size` |
| C1 | `_get_samples` | 噪音 | `print(samples[:2])` 调试残留 |
| C2 | 两个文件 | 噪音 | 未使用的 `import cv2 / Optional` |

---

## 二、逐个分析

### A1：`str` 当 `Path` 用

**现象**

```python
data_root: Path = config['data']['root']    # 实际是 str
...
if not self.data_root.is_dir():             # AttributeError
```

```
AttributeError: 'str' object has no attribute 'is_dir'
```

**为什么错**

类型注解 `data_root: Path` 只是**写给人和 IDE 看的**，运行时不校验。`config['data']['root']` 从 yaml 读出来是 `str`，`str` 没有 `is_dir()`。

更深一层：`str` 和 `Path` 混用会「传染」——`self.data_root / "Train"` 如果 `self.data_root` 是 str，`/` 会变成字符串除法（`str.__truediv__` 不存在，直接 `TypeError`）。要么全 str（用 `os.path.join`），要么全 Path。**选 Path。**

**修改**

```python
self.data_root = Path(data_root)      # 在入口处立刻统一类型
```

**如何避免**

> 凡是「外部输入」（配置、命令行、函数参数），在进入类/函数的第一行就转成内部统一的类型。

写注解的时候同时写转换，两者绑定：

```python
self.data_root = Path(data_root)   # noqa: 注解要 Path，这里就保证是 Path
```

---

### A2：目录不存在不该 `raise`

**现象**

```python
if not img_dir.is_dir() or not mask_dir.is_dir():
    raise FileNotFoundError(...)
```

LoveDA 的 `Val` 下**只有 `Urban`**，没有 `Rural`。所以 `domains=('Urban','Rural')` 构建验证集时必定在这里抛异常。

**为什么错**

把两种完全不同的失败混为一谈：

| 情况 | 含义 | 正确反应 |
|---|---|---|
| `Val/Rural/` 不存在 | 这个域本来就没有 | 跳过，继续 |
| `12345.png` 没有对应掩码 | 数据坏了 | 立刻中断 |

前者是**正常的**，后者是**异常的**。代码要能分辨。

**修改**

```python
if not img_dir.is_dir() or not mask_dir.is_dir():
    print(f"[AugDataset] 跳过 {self.split}/{domain}: 目录不存在 {img_dir}")
    continue
```

**如何避免**

写 `raise` 之前先问一句：**"这种情况在正常数据里会出现吗？"**

会 → 跳过并打日志。
不会 → 抛异常。

一个实用的判断：目录/文件「结构」层面的缺失可以跳过（不同划分本来就不一样），「内容」层面的缺失必须报错（说明拷贝/解压出问题了）。

---

### A3：`persistent_workers` 的隐式依赖

**现象**

```python
persistent_workers=True     # 硬编码
num_workers=config['data']['num_workers']   # 可能是 0
```

配置里改成 `num_workers: 0`（调试常用）时：

```
ValueError: persistent_workers option needs num_workers > 0
```

**为什么错**

`persistent_workers` 的语义是「让子进程跨 epoch 存活」。没有子进程，这个选项无从谈起，所以 PyTorch 直接拒绝。

**修改**

```python
num_workers = int(data["num_workers"])
persist = num_workers > 0     # 由 num_workers 推导，而不是写死
```

**如何避免**

参数之间有依赖时，**永远由源头推导，不要各自写死**。同类陷阱还有：

- `pin_memory=True` 只有在用 GPU 时才有意义
- `batch_size` 和 `drop_last` 必须一起考虑（`drop_last=True` + `batch_size` 大于数据集长度 → 一个 batch 都没有）
- `worker_init_fn` 也只在 `num_workers > 0` 时才被调用

---

### A4：验证集的域不能照抄训练集

**现象**

```python
val_datasets = AugDataset(
    split='Val',
    domains=('Urban', 'Rural'),     # 照抄了训练集
    ...
)
```

**为什么错**

直接触发 A2。即使 A2 修好了，写死 domains 也违背了「配置驱动」的原则——`config['data']['val_domains']` 里明明写着 `[Urban]`，代码却不看。

**修改**

```python
domains=data["train_domains"] if training else data["val_domains"],
```

配套把训练集/验证集的构建抽成一个函数，避免两处写两遍：

```python
def build_dataset(config: dict, training: bool):
    data = config["data"]
    return AugDataset(
        data_root=data["root"],
        split="Train" if training else "Val",
        domains=data["train_domains"] if training else data["val_domains"],
        ...
    )
```

**如何避免**

> 训练集和验证集的差异如果不止一个参数，就该抽成同一个工厂函数，用 `training: bool` 分流。

写两遍的结果一定是：改了一处忘了另一处。这次的 A4 就是这么来的。

---

### B1：`CLASS_NUM` 数量的错位（最隐蔽）

**现象**

```python
CLASS_NUM = {
    0: "Nodata", 1: "background", 2: "building", 3: "road",
    4: "water", 5: "barren", 6: "forest", 7: "agriculture",
}
```

**为什么错**

LoveDA 只有 **7 个类别，像素值 0~6**：

```
0 no-data / background
1 building
2 road
3 water
4 barren
5 forest
6 agriculture
```

你写的表有 8 项（0~7），而且名称整体后移了一位：

| 像素值 | 你写的 | 实际应为 |
|---|---|---|
| 1 | background | **building** |
| 2 | building | **road** |
| 6 | forest | **agriculture** |
| 7 | agriculture | **不存在** |

后果：

- 可视化时 `building` 显示的位置其实是 `road`，图看起来「怪但说不上哪怪」
- 按类统计 IoU 时，每一类的名字都是错的，但数值本身还能算出来
- **完全不报错**，这是最危险的一类 bug

**修改**

```python
CLASS_NUM = {
    0: "no-data",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}
```

**如何避免**

在导入时就校验一次：

```python
assert len(CLASS_NUM) == config["data"]["num_classes"], \
    f"类别表 {len(CLASS_NUM)} 项，配置写的是 {config['data']['num_classes']}"
assert sorted(CLASS_NUM) == list(range(len(CLASS_NUM))), "类别索引必须从 0 连续排列"
```

再加一条实证校验（跑一次就够，别放进训练流程）：

```python
vals = set()
for m in mask_dir.iterdir():
    vals |= set(np.unique(np.array(Image.open(m))).tolist())
assert vals <= set(CLASS_NUM), f"掩码里出现了表外的值: {vals - set(CLASS_NUM)}"
```

> 类别索引有三重身份：**文件里的像素值 → tensor 里的整数 → 人类可读的名字**。任何一层错位都不会报错，只会让结果「说不通」。用 `len()` 和实测值域两头夹住它。

---

### B2：`remap_zero_ignore=True` 写死

**现象**

```python
remap_zero_ignore: bool = True          # 函数默认值
...
remap_zero_ignore=True                  # get_dataloaders 里又硬编码一次
```

而 `configs/loveda_unet.yaml` 里明确写着 `remap_zero_to_ignore: false`。

**为什么错**

一个参数有**三个真相**：配置里的 `false`、函数默认的 `True`、调用处硬编码的 `True`。其中两个覆盖了配置，配置形同虚设。

更实际的问题：0 到底是不是 NoData **还没确认**（要等直方图结论）。此时默认 `True` 等于先扔掉一类再验证，结论会被自己的预处理污染。

**修改**

```python
remap_zero_ignore=data.get("remap_zero_to_ignore", False),
```

**如何避免**

> 凡是配置里已有的项，代码里就不要再给一个「更有主见」的默认值。

默认值只应该出现在**配置缺失时**（`.get("key", 默认值)`），而不是在调用处再写一遍。

---

### B3：孤儿文件只警告，随后被静默丢弃

**现象**

```python
if only_img or only_mask:
    print(f"⚠️警告: 数据存在不匹配 ...")     # 只是打印

...
for stem in sorted(img_path.keys() & mask_path.keys()):   # 交集
    samples.append(...)
```

**为什么错**

打印完之后，循环用的是**交集**，那几张对不上的样本被**悄悄地排除掉了**。程序继续跑，数据集比磁盘上少几张图，没有任何人能发现——直到你发现训练集的样本数和别人论文里的对不上。

**修改**

```python
if only_img or only_mask:
    raise RuntimeError(
        f"图文不配对 {self.split}/{domain}："
        f"{len(only_img)} 张原图无标签（如 {only_img[:3]}）；"
        f"{len(only_mask)} 张标签无原图（如 {only_mask[:3]}）"
    )
```

**如何避免**

> 数据层不允许「静默减量」。

如果你确实想容忍少量坏样本（比如大规模数据集里偶有损坏图），那就**把丢弃的数量累计下来并打印总数**，让它可见：

```python
self.n_skipped += len(only_img) + len(only_mask)
...
print(f"[AugDataset] 共跳过 {self.n_skipped} 个不完整样本")
```

「静默」不等于「宽容」，前者是 bug，后者是策略。区分标准是有没有人知道。

---

### B4：验证集复用训练的 `batch_size`

**现象**

```python
val_dataloader = DataLoader(val_datasets, batch_size=config['data']['batch_size'], ...)
```

**为什么错**

训练要大批量是为了梯度稳定，验证不需要——验证只是前向，batch 越小越省显存，而且显存紧张时训练可能 OOM。

**修改**

配置里加 `val_batch_size`，代码用回退：

```python
batch_size=int(data.get("val_batch_size") or data["batch_size"]),
```

**如何避免**

> 训练和推理的资源需求不一样，别默认「同一套参数」。

---

### C1 / C2：调试残留

```python
print(samples[:2])       # 调试时想看配对结果，忘了删
import cv2               # 从 data_check.py 抄过来就没用上
from typing import Optional   # 同上
```

**如何避免**

- 调试输出用 `logging.debug()` 而不是 `print`，上线时改日志级别就行
- 提交前跑一次 `python -m pyflakes <file>`（或看 IDE 的灰字提示），未使用的导入会直接列出来

---

### C3：`from_config` 直接下标

```python
crop_size=config["augment"]["crop_size"]     # 键不存在 → KeyError
hflip=config["augment"]["hflip"]             # 值留空 → 传 None 进去
```

你的 yaml 现在填好了，所以能跑。但空值仍会漏进构造函数：`random_scale=None` 侥幸能处理，`crop_size=None` 会在 `int(None)` 处炸。

**修改**

```python
aug = config.get("augment") or {}
norm = aug.get("normalize") or {}
return cls(
    crop_size=aug.get("crop_size") or 512,
    random_scale=aug.get("random_scale"),
    hflip=aug.get("hflip") or 0.0,
    ...
)
```

注意用 `or` 而不是 `.get(key, default)`：yaml 里写成 `crop_size:`（后面空着）解析出来是 `None` 而**不是缺失**，`.get` 的默认值不生效，只有 `or` 能兜住。

---

### C4：`max()` 夹宽高会拉变形

```python
out_w = max(self.crop_size, int(round(scale_ratio*w)))
out_h = max(self.crop_size, int(round(scale_ratio*h)))
```

方形图（LoveDA 全是 1024×1024）看不出来。但换成 1024×2048 的图、比例为 0.2 时：

```
out_w = max(512, 205) = 512
out_h = max(512, 410) = 512      ← 1:2 被压成 1:1
```

**修改**：夹**比例**而不是夹宽高

```python
scale_ratio = max(scale_ratio, self.crop_size / min(w, h))
out_w = max(self.crop_size, int(round(scale_ratio * w)))
out_h = max(self.crop_size, int(round(scale_ratio * h)))
```

这样长短边同比例放大，形状不变。

---

### C5：`astype(np.uint8)` 静默截断

```python
img = img.astype(np.uint8)
```

如果调用方传的是已经归一化的 `float 0~1` 数组，`astype` 会把 0.7 变成 **0**（截断，不是四舍五入），整张图几乎全黑——而且不报错。

**修改**：显式声明契约，错了就拦住

```python
if img.dtype != np.uint8:
    raise TypeError(f"img 需要 uint8，收到 {img.dtype}")
```

---

### C6：`if self.transform:` 的隐藏陷阱

```python
if self.transform:          # 依赖对象的「真值」
```

现在没问题（`MiniTransform` 没有 `__len__`，任何实例都是真）。但如果哪天你写一个组合式子类：

```python
class Compose:
    def __init__(self, ops): self.ops = ops
    def __len__(self): return len(self.ops)      # ← 有了 __len__
```

那么 `Compose([])` 或任何 `len() == 0` 的对象会被判为 **False**，transform 被静默跳过——图像不再归一化，训练照样能跑，只是效果很差。

**修改**

```python
if self.transform is not None:
```

> 判「有没有这个对象」用 `is None`，判「这个对象是不是空的」才用真值。`0`、`""`、`[]`、`len()==0` 都会被当成 False。

---

### C7：`self.hflip > random.random()` 的写法

```python
if self.hflip > random.random():
```

数学上等价于 `random.random() < self.hflip`，是正确的。但读的人要在大脑里做一次不等号翻转，而且 `hflip` 在左边容易让人误以为是「阈值」。

**修改**

```python
if random.random() < self.hflip:
```

「概率 < 阈值」的语序是全世界的共识，别创新。

---

### C8：变量名复用导致类型漂移

原来的 `__call__`：

```python
img = Image.fromarray(img)      # ndarray → Image
...
img = img.resize(...)           # Image
...
img = np.array(img, ...)        # Image → ndarray
```

同一个名字先后装了三种类型。IDE 会报一堆类型错，人读代码也要不断回忆「这里 img 是什么」。

**修改**：按类型换名

```python
pil_img = Image.fromarray(...)
arr = np.array(pil_img, ...)
img_t = torch.from_numpy(...)
```

> 变量名带上类型的影子（`pil_`、`arr`、`tensor`），后面调试时能省很多猜测。

---

## 三、知识扩展

### 1. `Path` 与 `str` 的边界

```python
Path("D:/data") / "Train" / "Urban"     # 拼路径
Path("D:/data").is_dir()                 # 判断类型
Path("a/b.png").stem                     # 'b'
Path("a/b.png").suffix                   # '.png'
```

**原则**：外部输入进来时转一次，内部全程用 `Path`，只在需要字符串的地方（`cv2.imread` 除外，它连 str 都要 Unicode 处理）转出去。

一个容易忽略的点：`Path` 在 Windows 上默认是 `WindowsPath`，比较时大小写不敏感、分隔符自动处理，比手写字符串拼接可靠得多。

### 2. 两类数据错误

| | 结构缺失 | 内容损坏 |
|---|---|---|
| 例子 | `Val/Rural/` 不存在 | 掩码少一张、尺寸对不上 |
| 是否正常 | 正常 | 异常 |
| 处理 | 跳过 + 日志 | `raise` |
| 判据 | 换个 split/domain 就没有 | 同一批数据里不该出现 |

这个区分不只用在路径上，也用在**类别缺失**上：某个 batch 里没有第 6 类像素是正常的，但掩码里出现第 7 类就是数据坏了。

### 3. 类别索引的三重身份

```
PNG 文件  ──np.array──▶  uint8 索引  ──to_tensor──▶  int64 类别号  ──CLASS_NAMES──▶  "building"
```

三个环节都可能错位：

- 读的时候 `convert('L')` 走了亮度公式（P 模式陷阱）
- 转 tensor 的时候 dtype 没转成 int64
- 名字表多一项或少一项

**前两个会报错（CrossEntropyLoss 拒绝越界 target），第三个永远不会。**

所以名字表一定要用数量断言 + 实测值域断言夹住。

### 4. `np.where` 的 dtype 提升

```python
mask = np.array([0, 1, 2], dtype=np.uint8)
np.where(mask == 0, 255, mask)          # dtype 还是 uint8，255 能放下
np.where(mask == 0, 1000, mask)         # dtype 变成 int64
```

如果目标值放不下（比如 uint8 里塞 300），NumPy 会**不做检查地**按类型转换，得到 300 % 256 = 44。所以掩码的重映射之后一定要显式 `astype`：

```python
mask = np.where(mask == 0, self.ignore_index, mask).astype(np.uint8)
```

（当前 `ignore_index=255` 恰好放得下，但这属于「碰巧对」。）

### 5. 插值：为什么掩码必须 NEAREST

| 操作 | 原图 | 掩码 |
|---|---|---|
| resize | BILINEAR | **NEAREST** |
| crop | 任意 | 任意（不插值） |
| flip | 任意 | 任意（不插值） |

只有 resize 会「计算」新像素。BILINEAR 在类别 1 和 3 之间会算出 2——一个根本不存在的类。NEAREST 只做复制，值域不变。

验证手段：resize 之后 `set(np.unique(mask))` 必须是原来的子集。

### 6. Python 真值陷阱

| 写法 | 结果 |
|---|---|
| `if self.transform:` | `None`、`False`、`0`、`[]`、`len()==0` 都为假 |
| `if self.transform is not None:` | 只有 `None` 为假 |

自定义类一旦实现 `__len__` 或 `__bool__`，就会参与真值判断。对「可选对象」一律用 `is None`。

### 7. DataLoader 的参数依赖

```python
persistent_workers  →  需要 num_workers > 0
pin_memory          →  只有 device 是 cuda 才有收益
drop_last=True      →  batch_size 必须 ≤ 数据集长度，否则 0 个 batch
worker_init_fn      →  只在 num_workers > 0 时被调用
```

这类「A 依赖 B」的参数，写成由 B 推导（`persist = num_workers > 0`），而不是各自从配置里读。

### 8. 类型注解不是校验

```python
def f(x: int):
    return x + 1

f("a")      # 运行时照样跑到 "a" + 1 才报错
```

注解的价值在于：IDE 补全、静态检查（mypy / pyright）、以及**给人看的契约**。想让它变成真校验，要么 `assert`，要么显式 `isinstance`，要么用 pydantic 这类库。

这次的 A1 就是「注解写了 Path，运行时给了 str」——注解没有帮你挡住，因为挡不住。

---

## 四、提交前自查清单

针对 Dataset / Transform 这一类代码：

- [ ] 外部输入（str vs Path、tuple vs list）是否在入口统一
- [ ] 目录缺失 → 跳过；文件不配对 → `raise`
- [ ] `CLASS_NAMES` 数量 `== num_classes`，索引从 0 连续
- [ ] 掩码 resize 用 `NEAREST`，且 `torch.unique` 是原值域的子集
- [ ] 掩码 dtype 是 `int64`
- [ ] 归一化后反算回 `[0,1]` 无偏
- [ ] 训练集 / 验证集的 split 与 domains 分别取配置，不互相照抄
- [ ] 配置里已有的开关，代码里不要另给默认值
- [ ] `if transform is not None`，不是 `if transform`
- [ ] `persistent_workers` 由 `num_workers` 推导
- [ ] 无调试 `print` 残留、无未使用 import
- [ ] 真跑一次：`len(ds)`、`ds[0]` 的形状/dtype/值域

---

## 五、修改后的验证结果

```
samples: 2522
img  (3, 512, 512) torch.float32
mask (512, 512)    torch.int64 [1, 5]
meta {'stem': '1366', 'domain': 'Urban'}
```

与 `rsunet/datasets/loveda_dataset.py` 的正式实现结果一致（2522 个样本）。

下一步可以把两个文件对比着看：正式版多了 `file_list`、`domain_counts`、`__repr__` 这些工程化的部分，但核心逻辑现在已经和练习版等价了。
