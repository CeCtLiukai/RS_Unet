# LoveDA 数据集校验学习笔记

- **学习主题**：遥感图像语义分割（LoveDA 数据集）的数据检查脚本开发
- **学习产出**：`scripts/data_check.py`
  文件校验 → 配对检查 → 标签统计 → 可视化抽检 → 输入模拟
- **核心收获**：用一次真实任务把 pathlib / PIL / OpenCV / NumPy /
  Matplotlib / PyTorch 的工程套路串起来

> 文档格式说明：本文按 A4 排版优化，行宽控制在 80 字符以内，表格不超过 4 列，
> 不依赖 Mermaid 等需要插件渲染的语法，可直接用 pandoc 或 Markdown PDF 插件导出。

## 目录

- 一、今日学习主线
- 二、知识模块总结
  - 1. pathlib.Path —— 路径操作
  - 2. PIL.Image —— 图片完整性校验
  - 3. OpenCV —— 图像与掩码读取
  - 4. NumPy + Counter —— 标签像素统计
  - 5. Python 语言细节
  - 6. Matplotlib —— 可视化抽检
  - 7. LoveDA 数据集业务认知
  - 8. PyTorch 端到端输入检查
  - 9. 今日踩坑清单
- 三、学习建议
- 附录 A：三种图像库速查对比
- 附录 B：导出 PDF 的方法

---

## 一、今日学习主线

```
configs/loveda_unet.yaml
        ↓
1. check_files        文件完整性 / 尺寸 / 模式
        ↓
2. check_pairing      原图与标签配对 / 尺寸一致
        ↓
3. label_histogram    标签像素值分布统计
        ↓
4. 可视化抽检         原图 / 掩码 / mask==0 二值图
        ↓
5. PyTorch 输入模拟   dtype / 值域 assert
        ↓
Dataset → DataLoader → 训练
```

**关键认知**：写模型之前先花时间把数据查干净，能省掉训练时大量
"为什么 loss 变 NaN / 为什么会崩"的排查时间。

---

## 二、知识模块总结

### 1. `pathlib.Path` —— 路径操作

`Path` 是对字符串路径的面向对象封装，所有 `os.path` 的功能都能用它完成。
核心优势：跨平台（自动处理 `/` 和 `\`）、可链式调用、永远返回 `Path` 对象。

#### 1.1 创建 Path

| 写法 | 含义 |
|---|---|
| `Path("data/LoveDA")` | 相对路径，不解析、不检查 |
| `Path(r"D:\data\LoveDA")` | 绝对路径 |
| `Path.cwd()` | 当前工作目录 |
| `Path.home()` | 用户主目录 |
| `Path(__file__).resolve().parent` | 当前脚本所在目录 |

`Path(__file__).resolve().parent` 是最常用的一行，它让"找配置文件"
不依赖从哪里启动脚本：

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "configs" / "loveda_unet.yaml"
```

#### 1.2 拼接与拆解

拼接只用 `/` 运算符，不要用字符串相加：

```python
root = Path("D:/Postgraduate_Study/Data/loveda")
img_dir = root / "Train" / "Urban" / "images_png"
```

拆解相关属性（以 `p = Path("D:/data/train/a.PNG")` 为例）：

| 属性 | 结果 | 说明 |
|---|---|---|
| `p.name` | `a.PNG` | 只取文件名 |
| `p.stem` | `a` | 去掉最后一个后缀 |
| `p.suffix` | `.PNG` | 最后一个后缀 |
| `p.parent` | `D:\data\train` | 父目录，仍是 Path |
| `p.parents[1]` | `D:\data` | 上两级，越界会 IndexError |
| `p.parts` | `('D:\\', 'data', 'train', 'a.PNG')` | 路径各段元组 |
| `p.anchor` | `D:\` | 路径根 |
| `p.drive` | `D:` | 盘符 |
| `p.as_posix()` | `D:/data/train/a.PNG` | 统一用正斜杠 |

多后缀文件：`Path("a.tar.gz").suffixes` 得到 `['.tar', '.gz']`。

#### 1.3 判断与查询

| 方法 | 说明 |
|---|---|
| `p.exists()` | 存在（文件或目录） |
| `p.is_file()` / `p.is_dir()` | 区分类型 |
| `p.is_absolute()` | 是否绝对路径 |
| `p.is_symlink()` | 是否符号链接 |
| `p.is_relative_to(other)` | 是否是另一个路径的子路径（3.9+） |
| `p.stat().st_size` | 字节数 |

**注意**：`exists()` 会触发一次磁盘访问，循环里不要反复调用它做重复判断。

#### 1.4 遍历目录

| 方法 | 说明 |
|---|---|
| `p.iterdir()` | 直接子项（含文件和目录），生成器 |
| `p.glob("*.png")` | 当前层按模式匹配，生成器 |
| `p.rglob("*.png")` | 递归匹配，等价 `glob("**/*.png")` |

三者都返回**生成器**，需要长度、下标、多次遍历时先 `list()`：

```python
files = list(img_dir.glob("*.png"))
print(len(files), files[:3])
```

#### 1.5 读写文件

```python
text = p.read_text(encoding="utf-8")
p.write_text("hello", encoding="utf-8")

data = p.read_bytes()
p.write_bytes(b"\x00\x01")

with p.open("r", encoding="utf-8") as f:
    for line in f:
        ...
```

`read_text` / `write_text` 一定要显式写 `encoding="utf-8"`，
否则 Windows 默认用 GBK，读写中文会出错。

#### 1.6 修改与删除

| 方法 | 说明 |
|---|---|
| `p.mkdir(parents=True, exist_ok=True)` | 递归建目录，可反复执行 |
| `p.rename(new)` | 重命名 |
| `p.replace(new)` | 重命名，跨盘或已存在时覆盖 |
| `p.unlink(missing_ok=True)` | 删除文件，不存在也不报错 |
| `p.rmdir()` | 删除空目录 |
| `p.touch()` | 创建空文件或更新时间戳 |
| `p.resolve()` | 解析成绝对路径，展开 `..` 和软链接 |

#### 1.7 三个完整示例

#### 示例 1：扫描并配对原图与掩码

```python
from pathlib import Path

def scan_pairs(img_dir: Path, mask_dir: Path):
    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}

    only_img = set(imgs) - set(masks)
    only_mask = set(masks) - set(imgs)

    pairs = [(imgs[s], masks[s]) for s in sorted(imgs.keys() & masks.keys())]
    return pairs, only_img, only_mask
```

用 `stem` 而不是完整路径做键，是因为原图和掩码目录不同，只有文件名相同。

#### 示例 2：安全地建目录再保存

```python
out = PROJECT_ROOT / "reports" / "preview_urban.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
```

`parents=True` 表示中间层级不存在时一并创建；
`exist_ok=True` 表示目录已存在时不抛 `FileExistsError`。

#### 示例 3：配置里的相对路径转绝对路径

```python
root_cfg = cfg["data"]["root"]          # "data/LoveDA" 或 "D:/datasets/LoveDA"
root = Path(root_cfg).expanduser()
if not root.is_absolute():
    root = (PROJECT_ROOT / root).resolve()
```

这样配置写相对路径时按项目根解析，不受"从哪个目录运行脚本"影响。

#### 1.8 常见坑

| 坑 | 原因 | 解法 |
|---|---|---|
| `'WindowsPath' object is not subscriptable` | Path 不支持切片 | 先 `list()` 物化 |
| `'list' object has no attribute 'glob'` | 对列表调用了 Path 方法 | 保持传 Path，或函数内分别处理 |
| `类型错误: str + Path` | Path 不能和字符串相加 | 用 `/` 或 `str(p)` |
| 生成器被消费完 | `glob()` 返回值只能遍历一次 | `list()` 后复用 |
| `.PNG` 匹配不到 `.png` | 换个平台（Linux）就大小写敏感 | 用 `suffix.lower() == ".png"` 判断 |

**版本差异（本机 Python 3.8）**：`with_stem()`（3.9+）、
`is_relative_to()`（3.9+）、`hardlink_to()`（3.10+）都不可用。

---

### 2. `PIL.Image` —— 图片完整性校验

Pillow 的定位是"通用图像文件操作"，适合做校验、格式转换、元数据读取。
它**不会**帮你做数值运算（那是 NumPy 的活）。

#### 2.1 打开、关闭与懒加载

```python
from PIL import Image

with Image.open("a.png") as im:
    im.verify()
```

`Image.open()` 是**惰性**的：只读文件头，不立即解码像素。
像素在 `im.load()` 或第一次访问像素时才解码。
这意味着：

- 打开很快，适合只取 `size` / `mode`
- 不 `close()` 会一直占用文件句柄，Windows 下文件被锁定，无法删除或覆盖

所以一律用 `with` 语句。

#### 2.2 核心属性

| 属性 | 示例值 | 说明 |
|---|---|---|
| `im.size` | `(1024, 1024)` | **宽 × 高**（注意顺序） |
| `im.width` / `im.height` | `1024` | 单独取宽高 |
| `im.mode` | `"RGB"` | 像素模式 |
| `im.format` | `"PNG"` | 文件格式 |
| `im.info` | `{...}` | 元数据字典 |
| `im.n_frames` | `1` | 多帧图（GIF/TIFF）帧数 |

常见 `mode`：

| mode | 含义 | 分割任务中的意义 |
|---|---|---|
| `L` | 8 位单通道灰度 | 掩码的理想格式 |
| `RGB` | 三通道彩色 | 原图 |
| `RGBA` | 三通道 + alpha | 带透明通道，注意丢弃 alpha |
| `P` | 调色板模式 | 需 `convert("L")` 或 `convert("RGB")` |
| `I;16` | 16 位灰度 | 遥感原始影像常见，转 uint8 前要缩放 |

#### 2.3 常用方法

| 方法 | 说明 |
|---|---|
| `im.verify()` | 校验文件完整性，调用后对象失效 |
| `im.load()` | 真正解码像素 |
| `im.convert("RGB")` | 模式转换 |
| `im.resize((w, h), Image.NEAREST)` | 缩放，掩码必须用 NEAREST |
| `im.crop((left, top, right, bottom))` | 裁剪 |
| `im.transpose(Image.FLIP_LEFT_RIGHT)` | 水平翻转 |
| `im.rotate(90, expand=True)` | 旋转，`expand` 防裁切 |
| `im.thumbnail((w, h))` | 原地生成缩略图（保持比例） |
| `im.save(path, format="PNG")` | 保存 |
| `im.copy()` | 复制，断开与原文件的关系 |
| `im.split()` | 拆通道，返回元组 |

`resize` 的插值参数很重要：

- 图像缩放：`Image.BILINEAR` 或 `Image.BICUBIC`
- **掩码/类别图缩放：必须 `Image.NEAREST`**，否则插值会造出
  2.5 这样的"不存在的类别"

#### 2.4 `verify()` 的两个关键特性

1. **调用后对象立即失效**，必须重新 `Image.open` 才能读 `size` / `mode`，
   否则报 `I/O operation on closed file`。
2. **查不出"结构完整但像素无法解码"的图**，需要再 `im.load()` 真正解码一次。

典型写法（两遍打开）：

```python
try:
    with Image.open(p) as im:
        im.verify()          # 第一遍：结构完整性
except Exception as e:
    bad.append((p, repr(e)))
    continue

with Image.open(p) as im:
    size, mode = im.size, im.mode   # 第二遍：取信息
```

#### 2.5 与 NumPy 互转

```python
import numpy as np
from PIL import Image

arr = np.array(im)              # Image -> ndarray，形状 (H, W, C)
im2 = Image.fromarray(arr)      # ndarray -> Image
```

转换后 `arr` 仍可能是只读视图，需要修改时先 `arr.copy()`。

---

### 3. OpenCV —— 图像与掩码读取

OpenCV 的定位是"图像处理算法库"，读进来就是 NumPy 数组，适合做数值统计和
预处理。**读分割掩码首选 OpenCV**，因为能精确控制通道和位深。

#### 3.1 `cv2.imread` 的读取模式

| 参数 | 数值 | 结果 |
|---|---|---|
| `cv2.IMREAD_COLOR` | 1（默认） | 三通道 BGR，丢弃 alpha |
| `cv2.IMREAD_GRAYSCALE` | 0 | 单通道灰度 |
| `cv2.IMREAD_UNCHANGED` | -1 | 原样读取，含 alpha |
| `cv2.IMREAD_ANYDEPTH` | 2 | 保留 16 位等原始位深 |

掩码必须这样读：

```python
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
```

不加参数会得到 `(H, W, 3)`，类别统计彻底失真。

#### 3.2 常用函数

| 函数 | 说明 |
|---|---|
| `cv2.imread(path, flag)` | 读图，失败返回 `None` |
| `cv2.imwrite(path, arr)` | 写图，注意路径也要 `str` |
| `cv2.cvtColor(img, code)` | 颜色空间转换 |
| `cv2.resize(src, dsize, interpolation=)` | 缩放，`dsize` 是 `(宽, 高)` |
| `cv2.imencode(ext, arr)` | 编码到内存，配合 `.tofile()` 写盘 |
| `cv2.imdecode(buf, flag)` | 从内存解码，中文路径的解法 |
| `cv2.split(img)` / `cv2.merge` | 拆/合通道 |
| `cv2.threshold` | 二值化 |
| `cv2.copyMakeBorder` | 边缘填充 |
| `cv2.imshow` / `cv2.waitKey` | 可视化（脚本里一般不用） |

#### 3.3 为什么掩码必须用 `IMREAD_GRAYSCALE`

对单通道类别图，`IMREAD_GRAYSCALE` 是"原样读取"，像素值就是类别 ID。

但对**彩色**标签图，OpenCV 会按 `0.299R + 0.587G + 0.114B`
加权平均，把颜色的组合压成一个灰度值——类别 ID 会被彻底破坏。
所以使用前必须确认掩码本身是单通道 `L` / uint8 类别索引图。

#### 3.4 缩放插值的选择

| 插值 | 用途 |
|---|---|
| `cv2.INTER_NEAREST` | 掩码、类别图、标签 |
| `cv2.INTER_LINEAR` | 一般图像放大（默认） |
| `cv2.INTER_AREA` | 图像缩小，效果更平滑 |
| `cv2.INTER_CUBIC` | 高质量放大，较慢 |

```python
mask_small = cv2.resize(mask, (512, 512),
                        interpolation=cv2.INTER_NEAREST)
```

**高频错误**：`dsize=(width, height)`，而 NumPy 的 `shape` 是 `(height, width)`。

#### 3.5 中文路径

Windows 下 `cv2.imread` 对非 ASCII 路径会返回 `None`。解决方案是
"读字节 → 内存解码"：

```python
import numpy as np, cv2

def imread_unicode(path, flag=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flag)
```

写入同理：

```python
def imwrite_unicode(path, img):
    ext = str(path).rsplit(".", 1)[-1]
    ok, buf = cv2.imencode("." + ext, img)
    if ok:
        buf.tofile(str(path))
    return ok
```

#### 3.6 常见坑

| 坑 | 原因 | 解法 |
|---|---|---|
| `cv2.imread` 报参数类型错误 | 不接受 Path | 传 `str(p)` |
| 读中文路径得到 `None` | 编码问题 | `np.fromfile` + `imdecode` |
| 读失败静默返回 `None` | OpenCV 不抛异常 | 每次读后判空 |
| 颜色红蓝颠倒 | OpenCV 是 BGR | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| 缩放出错 | `dsize` 是 `(宽, 高)` | 与 shape 顺序区分 |

#### 3.7 示例：读掩码并统计类别

```python
import cv2, numpy as np

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"读取失败: {mask_path}")

vals, cnts = np.unique(mask, return_counts=True)
for v, c in zip(vals.tolist(), cnts.tolist()):
    print(f"类别 {v}: {c} 像素")
```

---

### 4. NumPy + `collections.Counter` —— 标签像素统计

`label_histogram` 的核心数据流（6 步）：

```
准备(Counter+抽样)
  → 读图(cv2)
  → 单图统计(np.unique)
  → 全局累加(Counter.update)
  → 求总量(sum)
  → 输出占比(字典推导)
```

- **`np.unique(m, return_counts=True)`**：单张图去重 + 升序 + 频次，
  返回 `(vals, cnts)`。比 `np.bincount` 通用（不需要预知最大值）。
- **为什么还要 `Counter`**：`np.unique` 只管单张图；多张图的 `vals`
  长度和内容不同，无法直接相加。`Counter.update()` 的语义是
  "键存在则值相加、不存在则新建"，自动完成跨图对齐累加。
- **`counter.update(zip(vals.tolist(), cnts.tolist()))`**：
  - `zip` 把两个平行列表配成 `(值, 次数)` 序列；
  - `.tolist()` 必须：NumPy 标量（`np.uint8`）与原生 `int` 的哈希
    不保证一致，不转会导致统计分裂；
  - `Counter.update` 接受键值对迭代器，`dict()` 是多余的中间层。
- **`sorted(counter.items())` 为什么必要**：`np.unique` 只排单张图内部；
  `Counter` 跨图累加后插入顺序取决于首次出现时机，`sorted` 保证输出
  按类别 ID 升序、跨运行可预测。
- **`{v: (c, c/total) for v, c in sorted(...)}`**：输出
  `{像素值: (次数, 占比)}`，占比是判断类别不平衡的核心指标。
- **`total = sum(counter.values())`**：空 counter 会
  `ZeroDivisionError`，需要 `if total == 0: return {}`。

---

### 5. Python 语言细节

| 知识点 | 要点 |
|---|---|
| 三元表达式 | `a if cond else b`；`None/0/[]/""` 均为假，传 `sample=0` 会走全量，严谨写法是 `if sample is not None` |
| `zip` 是迭代器 | 惰性生成、只能消费一次；`dict()` 可直接消费二元组序列 |
| `dict` 的 key | 必须可哈希：`int/str/tuple/frozenset/None`；`list/dict/set` 不行 |
| 集合运算 | `set(a) - set(b)` 找单边缺失；`set(a) & set(b)` 找共有键 |
| 计数器两种写法 | `d[k] = d.get(k, 0) + 1`；`Counter.update(...)` 自动累加 |
| dict 保序 | Python 3.7+ 保序，`sorted` 后推导即得有序结果 |

---

### 6. Matplotlib —— 可视化抽检

- **网格画布**：`fig, axes = plt.subplots(3, n, figsize=(4*n, 12))`
  - `fig` 是整张画布（保存用），`axes` 是子图数组（定位用）
  - `figsize` 单位是**英寸**，最终像素 ≈ `figsize × dpi`
- **掩码上色**：`imshow(msk, cmap="tab10", vmin=0, vmax=6)`
  - `tab10` 让 7 个类别颜色分明
  - `vmin/vmax` 必须固定，否则不同图同一类别颜色不一致
- **二值图核心公式**：
  `zero_bin = (msk == 0).astype(np.uint8) * 255`
  - 把"像素值为 0 的位置"变白、其余变黑，配合 `cmap="gray"`
  - 这是肉眼判断 `0` 是背景还是未标注的唯一手段
- **布局与保存**：
  - `plt.tight_layout()` 防标题重叠，**必须在 `savefig` 之前**
  - `plt.savefig(path, dpi=150, bbox_inches="tight")`
  - 事先 `Path(path).parent.mkdir(parents=True, exist_ok=True)`
- **无界面环境**：脚本开头 `matplotlib.use("Agg")`，
  必须在 `import pyplot` 之前，可消掉 Qt 字体警告

---

### 7. LoveDA 数据集业务认知

- 掩码是 `L` 模式、单通道 uint8、`1024×1024`、7 个类别（0~6）。
- **`0` 的含义（今天的重要结论）**：
  - Urban 抽样：第三行 `mask==0` 全黑，这 8 张图里没有像素值 0；
  - Rural 抽样：Image 4 下半部纯黑边缘（遥感图 NoData）在第三行变白，
    说明黑边像素值就是 0。
  - 结论：LoveDA 的 `0` 是**背景 / NoData**，不是未标注。
    训练时建议把黑边区域映射为 `ignore_index=255` 跳过 Loss，
    防止模型学着预测黑边。
- **类别不平衡（长尾）**：Urban 抽样统计——标签 1 占 43.3%、
  5 占 33.9%、3 占 13.9%、2 占 6.8%、6 占 2.2%（0 和 4 未出现）。
  Rural 场景更严重，需考虑 class weights / Dice Loss / Focal Loss，
  并做随机 Crop / Flip / Rotate 增强。
- **可视化验证了 img/mask 配对正确**（原图蓝色屋顶与掩码方块位置一致），
  说明前面 `stem` 配对逻辑没问题。

---

### 8. PyTorch 端到端输入检查（Sanity Check）

```python
imgs, msks = next(iter(DataLoader(ds, batch_size=2)))
assert imgs.dtype == torch.float32   # 网络权重默认 float32
assert msks.dtype == torch.int64     # CrossEntropyLoss 要求 Long
assert msks.min() >= 0 and \
    set(msks.unique().tolist()) <= set(range(num_classes)) | {255}
```

- 第三句含义：标签里只能出现合法类别 `0~6` 和 `255`（ignore index），
  出现 8、100 等脏值直接拦在训练门外。
- **顺序依赖**：这一步依赖 `Dataset` / `DataLoader`，目前还没写，
  先搁置，等写完再回来补。
- Dataset 模板要点：`__getitem__` 里读图后
  `img = torch.from_numpy(img).float()`、
  `mask = torch.from_numpy(mask).long()`；
  如需忽略 `0`，先 `mask[mask == 0] = 255`。

---

### 9. 今日踩坑清单（重点复习）

| 坑 | 原因 | 解法 |
|---|---|---|
| `'list' object has no attribute 'glob'` | 函数内部对传入的 list 调用 `.glob()` | 函数内直接遍历 list，或统一传 Path |
| `'WindowsPath' object is not subscriptable` | Path 不支持切片 | 先 `list(glob(...))` 物化 |
| `FileNotFoundError: ...smaple_png` | 拼写错误 `smaple` vs `sample` | 路径常量定义一次到处引用 |
| `sample: None` 报 missing argument | 把类型注解写成了默认值 | 正确写法 `sample=None` |
| `savefig` 收到目录 | 传了目录而非带文件名的路径 | 拼 `f"{split}_{domain}.png"` |
| `tqdm` 进度条不动 | 处理太快，刷新间隔没到 | 加 `mininterval=0.01` |
| 类型检查器报"重载不匹配" | 存根 `.pyi` 重载复杂 | 代码没错可忽略，或 `# type: ignore` |
| `im.verify()` 后读 `im.size` 报错 | verify 后对象已关闭 | 重新 `Image.open` |

---

## 三、学习建议

> 原话："我发现这些代码我理解意思但是我自己写却不会，
> 我总是记不清楚每一个对象或类有什么方法和属性。"

### 1. 问题定位

看得懂代码是**被动理解**（上下文帮你把 API 用对了）；
自己写不出是**主动产出**（需要独立完成"目标 → 工具 → 参数"的检索）。

两者之间差的是：**把 API 变成自己的检索结构**。
资深工程师也记不住成千上万的 API，差别只在检索得快。
所以问题不是"记忆量不够"，而是"记忆的组织方式"不对。

### 2. 五个立刻能用的建议

1. **放弃背 API，改记"套路"（数据流）**
   - OpenCV 读图：`cv2.imread(路径, 模式)` → NumPy 数组（失败返回 None）
   - PIL 读图：`Image.open(路径)` → Image 对象 → `np.array()` 转数组
   - 转张量：`torch.from_numpy(arr).float() / .long()`

   今天整份脚本就是"读图 → 校验 → 统计 → 可视化"四个套路的拼接。
   先记套路骨架，参数靠 IDE 补全。

2. **给每个对象建"身份卡"（三问法）**
   - 从哪来：`Image.open()`、`cv2.imread()`、`Path(...)`、`np.array()`
   - 能做什么：`Path.glob/.stem/.parent`、
     `Image.mode/.verify`、`ndarray.shape/.tolist()`
   - 能去哪：Path → str、Image → ndarray → Tensor

3. **善用三个"记忆外挂"**
   - IDE 自动补全：输入 `p.` 弹出方法列表
   - `dir(obj)` / `help(obj)`：陌生对象直接打印
   - 搜索引擎 / 大模型：忘函数名就搜，大牛的代码不是背出来的

4. **"抄 → 改 → 独立写"三级跳**
   - 第一遍照着抄，理解结构
   - 第二遍只留骨架自己填
   - 第三遍隔天不看参考独立重写

   改写是学习编程最高效的方式。

5. **建自己的错题本 / 片段库**
   只记今天踩过的坑和好用的片段：
   `p.parent` 拿父目录、`IMREAD_GRAYSCALE` 读掩码、
   `(msk == 0).astype(np.uint8) * 255` 画二值图、
   `assert msks.dtype == torch.int64`。

### 3. 分阶段学习路线

**本周（巩固）**

- 不打开原文件，把 `data_check.py` 重写一遍，写完对照 diff，
  看漏了哪些坑（尤其 `m is None`、`total == 0`、`mkdir`）
- 把今天的坑整理成错题卡，每类配一句"为什么错 + 怎么写对"
- 用费曼技巧讲清"0 是背景还是未标注、为什么用二值图判断"

**下周（进阶）**

- 按模板填出 `LoveDADataset`（`__init__` / `__len__` / `__getitem__`），
  跑通那 4 行 assert
- 自己实现 `__getitem__` 的数据流：
  读图 → 预处理 → `torch.from_numpy().float()/.long()` → 返回
- 写训练循环时先做小 batch 冒烟测试（2~4 张图跑 1 个 epoch）

**持续（内化）**

- 写代码前先画数据流：输入什么 → 用什么工具 → 输出什么
- 学完一个模块做一次"独立写 + 对照"，用间隔重复（1/3/7 天）对抗遗忘
- 把常用函数沉淀进自己的 `utils.py`

### 4. 一句话总结

> 你今天的进度其实非常扎实——没有盲目跑模型，
> 而是踏踏实实把数据检查做完了，这在很多研究生里都是稀缺的好习惯。
> 现在缺的不是聪明，而是把"看得懂"转化为"写得对"的刻意练习。
> 记套路、用外挂、勤重写、建错题本，
> 两周后你会明显感觉到"写不出来"的障碍在缩小。

---

## 附录 A：三种图像库速查对比

| 维度 | pathlib | PIL.Image | OpenCV |
|---|---|---|---|
| 定位 | 路径操作 | 文件级图像操作 | 图像处理算法 |
| 读入结果 | Path 对象 | Image 对象 | NumPy 数组 |
| 适合 | 扫目录、建路径 | 校验、查属性、格式转换 | 像素统计、预处理 |
| 懒加载 | 不适用 | 是（只读头） | 否（立即解码） |
| 失败行为 | 抛异常 | 抛异常 | 返回 `None` |
| 接受 Path | 是 | 是 | 否，需 `str()` |
| 彩色顺序 | 不适用 | RGB | BGR |
| 典型用法 | `p.glob("*.png")` | `Image.open(p).size` | `cv2.imread(str(p), 0)` |

**本项目的分工**：

- `pathlib` 负责找文件、配对、建输出目录
- `PIL` 负责校验完整性与读取 `size` / `mode`
- `OpenCV` 负责真正读像素做统计
- `NumPy` 负责数值计算
- `PyTorch` 负责转张量进模型

---

## 附录 B：导出 PDF 的方法

### 方法一：pandoc（推荐，排版最稳）

需要先安装 pandoc 与 TeX Live / MiKTeX。命令行：

```text
pandoc notes/data_check.md -o notes/data_check.pdf ^
  --pdf-engine=xelatex ^
  -V CJKmainfont="Microsoft YaHei" ^
  -V geometry:margin=2.2cm ^
  -V fontsize=10pt ^
  --toc --toc-depth=2 ^
  -V toc-title="目录" ^
  --highlight-style=tango
```

说明：

- `xelatex` 才能正确处理中文，`pdflatex` 会报缺少字形
- `CJKmainfont` 换成系统里已安装的中文字体即可
- `--toc` 自动生成目录，把开头的手写目录删掉也可以

### 方法二：VS Code 插件 Markdown PDF

安装 "Markdown PDF" 插件，打开 md 文件后
`Ctrl+Shift+P` → `Markdown PDF: Export (pdf)`。

优点是零配置，缺点是对中文表格和代码高亮支持一般。

### 方法三：Typora / Obsidian 直接导出

- Typora：`文件 → 导出 → PDF`，排版最好看，需要正版授权
- Obsidian：装 "Better Export PDF" 插件

### 排版约定（本文已遵循）

1. 正文行宽 ≤ 80 字符，避免代码块被强制折行
2. 表格 ≤ 4 列，避免 A4 竖版溢出
3. 不使用 Mermaid（多数 PDF 工具不渲染）
4. 代码块标注语言，便于语法高亮
5. 章节标题最多三级；示例小标题降为四级，
   `--toc-depth=2` 时目录只保留章节，不显示例
