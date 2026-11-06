# forward

# 遥感语义分割 模型前向、标签映射与损失函数学习路线

## 1. 学习目标

本教程目标不是直接复制一份 UNet 和 loss 代码，而是打通**标签预处理 → 网络前向传播 → 损失计算 → 反向更新参数**整条链路，理解分割训练最核心的数据流。
最终希望能够独立完成以下工作：

- 读懂标准 UNet Encoder-Decoder + skip 连接前向流程；
- 理解类别标签重映射、ignore_index 的设计目的；
- 理解手写交叉熵损失如何摊平 batch、过滤忽略像素；
- 定位标签越界、ignore 不生效这类常见报错；
- 区分“标签改数字”和“损失里过滤像素”两件独立工作；
- 把 Dataset、Model、Loss、Optimizer 串成完整训练循环；
- 读懂分割论文里 ignore label、像素级平均 loss 的含义。

配套材料：你手写的标准 UNet 代码、`seg_ce` 手写交叉熵损失、LoveDA 标签映射逻辑。

---

## 2. UNet 模型基础结构与前向概念

### 2.1 UNet 整体结构

标准 UNet 分为编码器（下采样）、瓶颈层、解码器（上采样+skip拼接）、输出头：

```
输入图像 [N, 3, H, W]
→ Inc入口卷积（升通道）
→ Encoder：多次 MaxPool + DoubleConv，保存每一层浅层特征（skip）
→ Bottleneck（最深特征图，分辨率最小，通道最多）
→ Decoder：逐层上采样 + 和对应skip拼接 + DoubleConv融合
→ Outc 1×1卷积 → logits [N, num_classes, H, W]
```

语义分割输出：**每个像素每个类别的预测分值（logits，未经过softmax）**，不是概率。

### 2.2 DoubleConv、Down、Up 模块职责

- `DoubleConv`：(3×3 Conv + BN + ReLU) ×2，基础特征提取块；
- `Down`：MaxPool2d 分辨率减半 + DoubleConv，编码器下采样，通道翻倍；
- `Up`：上采样（转置卷积 / bilinear upsample）→ 尺寸对齐padding → 通道拼接skip → DoubleConv融合特征。

> 
> 关键点：遥感图边长不一定是2的幂，Up里增加padding居中补零，让上采样后的特征图和skip特征图高宽完全一致，保证cat操作合法。
> 隐患：padding补0会引入无信息边界像素，遥感任务边界指标容易下降；原始论文UNet方案是裁剪skip特征而不是pad上采样结果。

### 2.3 前向主流程代码逐段理解

```
def forward(self, x):
    skips = []
    x = self.inc(x)
    for down in self.downs:
        skips.append(x)
        x = down(x)

    for i, up in enumerate(self.ups):
        x = up(x, skips[-(i + 1)])

    return self.outc(x)
```

1. `skips = []`：列表缓存编码器每一层的浅层细节特征；
2. `x = self.inc(x)`：入口卷积，把输入3通道升到base_channels（默认64）；
3. 编码器循环：**先存当前特征，再做下采样**；存入的就是后续要拼接的skip；循环结束得到瓶颈深层语义特征；
4. 解码器循环：`skips[-(i+1)]` 倒序取skip，先拼最深的skip，逐步还原分辨率；
5. `self.outc`：1×1卷积，把64通道映射到`num_classes`通道，输出logits。

举个示例（depth=5，输入 [2,3,256,256]，num_classes=7）

```
inc: [2,64,256,256] → skips[0]
down1: [2,128,128,128] → skips[1]
down2: [2,256,64,64] → skips[2]
down3: [2,512,32,32] → skips[3]
down4: [2,1024,16,16] 瓶颈
up0: 瓶颈 + skips[3] → [2,512,32,32]
up1: + skips[2] → [2,256,64,64]
up2: + skips[1] → [2,128,128,128]
up3: + skips[0] → [2,64,256,256]
outc: [2,7,256,256] logits
```

---

## 3. 标签重映射（Dataset阶段）

### 3.1 场景说明

以 LoveDA 为例：原始标注类别编号 `1~7`，原始mask中的`0`代表无效/背景，不参与loss计算；模型设置 `num_classes=7`，合法标签范围 `0~6`。
映射代码：

```
if self.remap_zero_to_ignore:
    mask = np.where(
        (mask >= 1) & (mask <= 7),
        mask - 1,        # 原值1→0，7→6，转为模型合法标签
        self.ignore_index # 其余像素（原值0、8及以上）赋值ignore_index，常用255
    ).astype(np.uint8)
```

> 
> 重点区分：
> 这一步**仅仅修改mask里数字**，它本身不会控制loss是否计算这个像素；只是给“不参与训练的像素打上一个约定编号255”。

### 3.2 常见坑

1. dtype 坑：`ignore_index=-1` 时不能用 `uint8`，uint8无法存负数，会溢出变成255；想用-1要改成`int64`；
2. 非法标签坑：如果原始mask存在值=7，映射失败，送入loss就会触发索引越界报错 `index 7 out of bounds for dimension 1 with size7`；
3. 分支开关：`remap_zero_to_ignore` 是bool开关，如果为False，标签不会转换，原始0会被当成有效类别0计算loss。

---

## 4. 手写交叉熵损失 seg_ce 原理

### 4.1 函数完整代码

```
def seg_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index=255):
    n, c, h, w = logits.shape
    logits = logits.permute(0, 2, 3, 1)
    logits = logits.reshape(-1, c)

    target = target.reshape(-1).long()

    valid = target != ignore_index
    logits = logits[valid]
    target = target[valid]

    if target.numel() == 0:
        return logits.sum() * 0.0

    log_prob = torch.log_softmax(logits, dim=1)
    selected_log_probs = log_prob[
        torch.arange(target.numel(), device=target.device),
        target
    ]
    loss = -selected_log_probs.mean()
    return loss
```

### 4.2 逐行流程（batch级数据流）

输入：

- `logits`：`[N, C, H, W]` 网络输出，C=num_classes；
- `target`：`[N, H, W]`，已经在Dataset完成标签映射，包含合法标签`0~6`和忽略标记`255`。

1. **全部摊平**：把整个batch内所有图片的所有像素打散，不再区分图片、行列坐标
   - logits：`[N,C,H,W] → [N*H*W, C]`，每一行代表**一个像素的全部类别分值**
   - target：`[N,H,W] → [N*H*W]`，一维，每个元素是该像素标签
2. **valid过滤**：`valid = target != ignore_index`
   - 不是删除某一个通道，而是**整行像素剔除**：只要这个像素标签等于255，该像素对应的全部C个logits和标签一起扔掉；
   - 过滤之后，只剩下有效像素参与后续计算；
3. **全batch无有效像素兜底**：如果整张batch全部是ignore像素，返回0 loss，防止后续计算报错；
4. `log_softmax`：在类别维度做softmax转概率再取对数；
5. 索引取出真值类别的log概率；
6. 取负、求均值，得到当前batch损失标量。

> 
> 核心特点：loss是**整个batch全部有效像素的算术平均**，不是单张图loss求平均，等价于PyTorch原生 `CrossEntropyLoss(reduction='mean', ignore_index=255)`。

### 4.3 数字示例（batch_size=2，H=2,W=2，C=7，ignore_index=255）

映射后target展平：`[255, 0, 3, 255, 1, 255, 2, 5]`

```
valid = [False, True, True, False, True, False, True, True]
过滤后target = [0, 3, 1, 2, 5]
logits过滤后shape：[5,7]
```

只有这5个像素参与交叉熵计算，3个255像素完全不参与梯度回传。

### 4.4 经典报错：IndexError: index 7 is out of bounds

原因：target里面出现了7；合法标签只能是0~6。
排查方式：在loss开头打印 `torch.unique(target)`，定位非法标签来源，根源一般在Dataset标签映射环节。
兜底方案（治标）：过滤行增加 `valid = valid & (target < c)`。

---

## 5. 反向传播与参数更新

损失得到标量loss之后，进入标准训练三步：

```
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

1. `optimizer.zero_grad()`：清空上一轮迭代残留梯度，梯度会累加，不清零会计算错误；
2. `loss.backward()`：沿着计算图反向传播，对所有可训练参数计算梯度 `.grad`；**只有valid保留下来的有效像素会产生梯度，被过滤的255像素无梯度贡献**；
3. `optimizer.step()`：优化器（Adam/SGD等）根据梯度、学习率更新网络权重。

返回 `tensor(1.9973, grad_fn=<NegBackward0>)` 含义：

- 1.9973：当前batch的loss数值；
- `NegBackward0`：计算图完整、支持反向传播，来源于代码 `loss = -selected_log_probs.mean()` 的取负操作。

---

## 6. 整条链路串联（完整数据流）

```
Dataset __getitem__
    ↓
标签映射：原值1~7→0~6，原值0→255（仅修改数字）
    ↓
DataLoader打包成batch [imgs, masks]
    ↓
UNet forward，输出logits [N,7,H,W]
    ↓
seg_ce loss：
    摊平所有像素 → valid筛掉target=255像素 → 有效像素算CE
    ↓
loss标量
    ↓
zero_grad → backward → step（更新权重）
```

一句话区分两件事：

- Dataset：给不需要训练的像素打上编号255；
- seg_ce：看到编号255就剔除，不计算loss、不回传梯度。
**缺一不可：只改标签不写valid过滤，会索引越界报错；只写valid过滤、Dataset不把无效像素改成255，则无效像素会被当成正常类别参与训练。**

---

## 7. 常见错误及原因

### 错误1：Up模块拼接时报尺寸不匹配

原因：上下采样后特征高宽不一致；解决方案：Up里F.pad居中对齐；原始UNet方案裁剪skip。

### 错误2：IndexError 索引越界（index=7）

原因：标签映射逻辑漏洞，target中混入7；排查Dataset np.where映射条件。

### 错误3：ignore_index设置255，但无效像素依然参与loss

原因：Dataset没有把无效mask值转为255，valid条件没有命中任何像素。

### 错误4：loss一直是0

原因：当前batch所有像素都是255，触发兜底分支 `target.numel()==0`。

### 错误5：Windows下DataLoader num_workers>0直接报multiprocessing spawn报错

原因：Windows multiprocessing默认spawn，主训练循环、DataLoader实例化代码必须放在 `if __name__ == "__main__":` 内部。

### 错误6：以为255是网络的一个类别通道

误区：255只是忽略标记；网络通道数=num_classes，只有0~6共7类通道，不存在255通道。

---

## 8. 编程训练题

建议按顺序完成，巩固整条链路：

### 题目1：跑通UNet前向

输入 `[2,3,256,256]`，实例化UNet，打印输出shape，确认输出 `[2,7,256,256]`。

### 题目2：手写标签映射函数

输入原始mask数组（值0~~7），实现映射：1~~7→0~6，0→255，返回映射后mask。

### 题目3：测试seg_ce单batch

构造模拟logits、模拟target（包含255），调用seg_ce，观察loss输出，验证过滤逻辑。

### 题目4：复现越界报错

手动构造target里面包含7，送入seg_ce，观察IndexError。

### 题目5：最小训练循环

组装 model + seg_ce + Adam优化器，写极简训练循环，包含 zero_grad / backward / step，包在 `if __name__ == "__main__"` 内，使用 `num_workers=0` 调试。

### 题目6：增加loss打印

在循环中打印loss.item()，同时打印每一轮target唯一值，监控标签是否异常。

---

## 9. 学习延伸方向

掌握这条链路之后，可以继续拓展：

- DiceLoss、FocalLoss、LovaszLoss 实现，同样要兼容ignore_index过滤逻辑；
- 指标计算（mIoU、PA），指标计算同样需要过滤ignore标签；
- 滑动窗口预测（sliding window），大图分块推理、结果拼接；
- 梯度裁剪、学习率调度；
- DDP分布式训练（Windows下DDP同样严格要求`if __name__ == "__main__"`）；
- 对比 bilinear upsample 和 ConvTranspose2d 两种Up的效果差异；
- 边界padding带来精度下降的优化方案。

---

## 10. 最终检查清单

在开始训练前核对：

- UNet输出通道数 = num_classes；
- Dataset标签映射逻辑正确，原始标注1~~7映射到0~~6，无效像素转为ignore_index；
- mask dtype和ignore_index匹配（255可用uint8；-1必须int64）；
- Up模块尺寸对齐逻辑正常，拼接不报错；
- loss函数内valid过滤开启；
- 非法标签（>=num_classes）不会流入loss；
- 训练循环顺序：zero_grad → forward → loss → backward → step；
- Windows训练脚本主逻辑包裹 `if __name__ == "__main__":`；
- 能单独取一个batch测试前向、loss计算，无报错；
- 理解loss是batch内有效像素平均，不是逐图平均。

当以上全部检查通过，模型前向、标签映射、损失、反向更新整套流程就达到论文实验可用标准。