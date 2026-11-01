# Matplotlib (plt) 核心用法教程
> 核心认知：plt (Pyplot) 的核心是**状态机（State Machine）**。你可以把它想象成“画笔”和“全局画布”。你每调用一次 `plt.plot()` / `plt.imshow()`，都是在**当前激活的画布**上绘制；fig/ax 面向对象写法是推荐工程范式，能避开状态机混乱的坑。
> ✅ 两条路线：
> - plt 状态机写法：适合快速调试、一行行画图
> - fig, ax 面向对象（OO）写法：深度学习、批量可视化、子图多的场景优先用，**强烈推荐CV项目统一用ax**

## 一、基础骨架：画布与子图
在神经网络/遥感分割任务里，经常要把原图、标签、预测图、特征图并排对比，子图是最高频功能。
### 1. 单张画布两种写法
#### 写法1：plt状态机（简单脚本）
```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()  # 弹出窗口显示；脚本模式阻塞程序，循环可视化不推荐频繁show
```
#### 写法2：面向对象（推荐，无状态混乱）
```python
fig = plt.figure(figsize=(6,4)) # 先创建画布对象
ax = fig.add_subplot()          # 在画布上创建单个绘图轴
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

### 2. 创建子图（最常用）
`plt.subplots(nrows, ncols)` 一次性生成画布fig + 绘图轴数组axes，axes的索引和网格一一对应。
```python
# 1行2列，画布尺寸宽10英寸、高5英寸
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0] 代表第一个子图，axes[1] 代表第二个
axes[0].plot([1, 2, 3])
axes[0].set_title("图1")
axes[1].plot([3, 2, 1])
axes[1].set_title("图2")

plt.tight_layout()
plt.show()
```
> 补充：多行多列子图索引
> ```python
> fig, axes = plt.subplots(2,2,figsize=(8,8))
> axes[0,0].plot([1,2]) # 第0行第0列
> axes[1,1].plot([2,1]) # 第1行第1列
> ```

### 3. 子图进阶：自定义布局、跨子图（gridspec）
当需要子图大小不一样、图片跨格显示（比如大图占一整行，下方放两个小对比图），用GridSpec：
```python
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10,6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :]) # 第一行，横跨两列
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
```

## 二、显示图像（重点：imshow，CV/遥感核心）
`plt.imshow()` 接收二维(H,W)灰度数组 / 三维(H,W,3)RGB数组，**不接受(C,H,W)**，这是和tensor最容易冲突的点。
### 1. 显示RGB原图（OpenCV坑）
OpenCV `cv2.imread` 默认读取为BGR通道顺序，plt识别RGB，不转换颜色完全错位。
```python
import cv2
img = cv2.imread("a.png")
if img is None:
    raise FileNotFoundError("图片路径错误！")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

### 2. 灰度图、单通道特征图、离散标签Mask
核心参数：
- `cmap`：颜色映射表
- `vmin, vmax`：像素映射上下限（分割任务重中之重）
- `interpolation`：插值方式（遥感/特征图常用 `nearest`，避免标签边缘模糊！）

```python
# 灰度图
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

# 分割标签mask，类别0~6，必须固定vmin/vmax，防止颜色自动缩放错乱
plt.imshow(mask, cmap='tab10', vmin=0, vmax=6, interpolation='nearest')
```
⚠️ **必坑指南**：不设置vmin/vmax，matplotlib会根据当前图片像素最大最小值自动归一映射。如果一张图缺少某一类，同类别的颜色和其他图对不上，对比实验直接失效。
> 常用cmap对照表
> - `gray`：灰度图
> - `tab10 / tab20`：离散分类，类别少首选
> - `jet / viridis`：连续热力图、特征图（viridis更友好，色盲兼容）

### 3. 叠加显示（原图+半透明Mask，分割可视化刚需）
```python
plt.imshow(img_rgb)
# alpha 透明度：0完全透明，1不透明
plt.imshow(mask, cmap='tab10', alpha=0.5, vmin=0, vmax=6, interpolation='nearest')
plt.axis('off')
plt.show()
```

### 4. 张量可视化补充（PyTorch Tensor）
模型输出tensor默认shape `[B,C,H,W]`、数据在GPU、归一化到[-1,1]，一套标准反归一流程：
```python
import numpy as np
img_tensor = img_tensor[0] # 取batch里第一张图
img_show = img_tensor.cpu().detach().numpy() # 脱离计算图+转到cpu
img_show = img_show.transpose(1, 2, 0) # C,H,W → H,W,C
# 反归一化 mean=0.5 std=0.5
img_show = img_show * 0.5 + 0.5
img_show = np.clip(img_show, 0, 1) # 截断到合法区间，防止异常值
plt.imshow(img_show)
```

## 三、坐标轴、刻度、标题、图例、字体美化（论文图必备）
很多人画图只满足“能看”，论文图需要精细控制样式
```python
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, y, label='curve')

ax.set_title("标题", fontsize=12, fontfamily='SimHei') # SimHei解决中文乱码
ax.set_xlabel("X轴", fontsize=10)
ax.set_ylabel("Y轴", fontsize=10)

ax.set_xlim(0,10)  # x轴显示范围
ax.set_ylim(0,1)   # y轴显示范围
ax.set_xticks([0,2,4,6,8,10]) # 自定义刻度位置
ax.set_xticklabels(["0","2","4","6","8","10"]) # 自定义刻度文字

ax.legend(loc='upper right') # 图例位置
ax.grid(True, linestyle='--', alpha=0.6) # 网格线，虚线+半透明
```
> 中文乱码全局设置（放在文件最顶部）
> ```python
> plt.rcParams['font.sans-serif'] = ['SimHei']
> plt.rcParams['axes.unicode_minus'] = False # 负号正常显示
> ```

## 四、画布清理与保存（循环可视化防内存泄漏，训练脚本重中之重）
### 1. 坐标轴开关
```python
ax.axis('off')        # 推荐：面向对象写法，单独关闭一个子图坐标轴
plt.axis('off')       # 状态机写法：关闭当前激活画布坐标轴
ax.axis('on')         # 开启坐标轴
```
### 2. 布局调整
```python
plt.tight_layout()    # 自动适配子图，防止标题、标签重叠，必须在savefig之前
# 精细版：手动控制子图间距
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.02, hspace=0.1)
```
### 3. 保存图片
```python
# dpi分辨率；bbox_inches='tight'裁掉多余白边；facecolor背景色
plt.savefig("output.png", dpi=300, bbox_inches='tight', facecolor='white')
```
> 论文建议dpi≥300；支持png/jpg/pdf/svg矢量图（svg适合LaTeX论文）
### 4. 画布清理（循环画图必写！）
循环遍历数据集批量可视化时，不释放画布会持续占用内存，程序越来越慢、图片错乱叠加
```python
plt.clf()   # clear figure：清空当前画布内容，画布对象还在
plt.close() # 关闭figure对象，释放内存，循环结束优先用这个
plt.close('all') # 一次性关闭所有画布，兜底
```

## 五、神经网络高频场景模板（直接复制进训练代码）
### 1. Loss曲线（train loss + val loss，可扩展mIoU、acc）
```python
epochs = range(1, len(train_loss)+1)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=1.5)
ax.plot(epochs, val_loss, label='Val Loss', color='#d62728', linestyle='--', linewidth=1.5)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()
```

### 2. 分割对比模板：原图 / Label / Pred 三栏子图（遥感/分割最常用）
```python
fig, axes = plt.subplots(1,3,figsize=(12,4))
# 原图
axes[0].imshow(img_rgb)
axes[0].set_title("Image")
axes[0].axis('off')
# 标签
axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=6, interpolation='nearest')
axes[1].set_title("Ground Truth")
axes[1].axis('off')
# 预测结果
axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=6, interpolation='nearest')
axes[2].set_title("Prediction")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("compare.png", dpi=200, bbox_inches='tight')
plt.close()
```

## 六、后端Backend & 坑点汇总（对应你PyCharm预览失败的问题）
matplotlib后端决定图片怎么渲染、怎么显示：
1. `Agg`：无GUI后端，只能保存图片，不能弹窗预览（深度学习库经常强制设置这个）
2. `TkAgg`：Windows弹窗后端；`macosx`：Mac；`module://backend_interagg`：PyCharm专业版SciView面板
> 放在所有import最前面修改后端：
> ```python
> import matplotlib
> matplotlib.use('TkAgg')
> import matplotlib.pyplot as plt
> ```
⚠️ 高频坑清单
1. Tensor是`C,H,W`，直接imshow报错/颜色错乱，必须transpose转`H,W,C`
2. 标签图interpolation默认是bilinear，会模糊类别边界，分割任务强制`interpolation='nearest'`
3. 循环可视化只plt.show不close，内存泄漏
4. OpenCV图片BGR顺序忘记转RGB
5. 不设置vmin/vmax，不同样本mask颜色不一致
6. 中文无字体设置，方框乱码
7. 远程服务器、Colab环境没有GUI，不能show，只能save

## 七、状态机 vs 面向对象写法选择建议
- 快速调试、单行画图：plt状态机写法，代码短
- 深度学习批量可视化、多子图、论文图、循环绘图：**统一用fig, ax面向对象写法**，不受全局状态干扰，不容易出现“画到上一张图里”的诡异bug
