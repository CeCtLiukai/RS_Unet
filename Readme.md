# RS_Unet

> 🛰️ 一个面向初学者的简明 U-Net 遥感语义分割学习仓库

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

---

## 📖 简介

**RS_Unet** 是我在学习遥感图像语义分割过程中整理出的一个**入门级**代码仓库。

这里记录了我从零复现 U-Net 分割模型的全过程 —— 包括遇到的坑、踩过的雷、以及最终写出来的每一行代码。仓库结构力求清晰简洁，**适合刚接触遥感分割的同学对照学习**，也欢迎各位大佬批评指正、交流讨论。

> 💡 如果你也在学习遥感分割，希望这个仓库能帮你少走一点弯路。

---

## ✨ 特点

- 🧱 **结构清晰**：数据 / 模型 / 损失 / 指标 / 训练引擎分层解耦，代码可读性优先
- 🚀 **开箱即用**：`configs/` + `train.py` 一行命令开始训练
- 📊 **完整评测**：支持 mIoU、aAcc、per-class IoU，并附带官方评测口径脚本
- 🗂️ **实验管理**：每次实验自动保存配置快照、日志、TensorBoard、权重、可视化结果
- 🎓 **学习导向**：面向初学者，注释详尽，不追求花哨 trick

---

## 🛠️ 技术栈

| 分类 | 技术 |
| :--- | :--- |
| 语言 | Python 3.8+ |
| 深度学习框架 | PyTorch |
| 数据增强 | torchvision.transforms / 自定义 transforms |
| 可视化 | TensorBoard |
| 配置管理 | YAML |
| 评价指标 | mIoU / aAcc / per-class IoU |
| 数据集 | LoveDA（遥感语义分割数据集） |
| 模型 | U-Net（backbone 可切换，如 ResNet-50） |

---

## 📂 目录结构

```text
RS_Unet/
├── configs/                      # 实验配置
│   ├── loveda_unet.yaml          # 数据路径、类别数、超参
│   └── loveda_unet_r50.yaml
├── rsunet/                       # 核心代码包（可 import）
│   ├── datasets/
│   │   ├── loveda.py             # LoveDA Dataset
│   │   └── transforms.py         # 裁剪 / 翻转 / 归一化
│   ├── models/
│   │   ├── unet.py
│   │   └── backbones/
│   ├── losses.py                 # CE / Dice / Focal
│   ├── metrics.py                # mIoU、aAcc、per-class IoU
│   └── engine.py                 # train / val 循环
├── train.py                      # 训练入口，读取 config
├── test.py                       # 测试入口，读取 config + 权重
├── scripts/
│   ├── prepare_loveda.py         # 拆分配置 / 统计类别 / 生成清单
│   └── eval_official.py          # 官方评测口径
├── data/                         # 只放数据（.gitignore）
│   └── LoveDA/
│       ├── Train/{Urban,Rural}/
│       ├── Val/Urban/
│       └── Test/{Urban,Rural}/
├── runs/                         # 实验输出（.gitignore）
│   └── unet_r50_512_bs8/
│       ├── config.yaml           # 本次实验配置快照
│       ├── train.log
│       ├── tb/                   # TensorBoard 日志
│       ├── weights/
│       │   ├── best_miou.pth
│       │   └── last.pth
│       └── preds/                # 测试可视化结果
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/<your-name>/RS_Unet.git
cd RS_Unet
pip install -r requirements.txt
```

### 2. 准备数据

下载 [LoveDA 数据集](https://github.com/Junjue-Wang/LoveDA)，放置到 `data/LoveDA/` 下，并运行：

```bash
python scripts/prepare_loveda.py
```

### 3. 开始训练

```bash
python train.py --config configs/loveda_unet.yaml
```

### 4. 测试与评测

```bash
python test.py --config configs/loveda_unet.yaml --weight runs/<exp>/weights/best_miou.pth
```

---

## 📊 实验结果

| 模型 | Backbone | 输入尺寸 | Batch | mIoU | aAcc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| U-Net | ResNet-50 | 512×512 | 8 | _待补充_ | _待补充_ |

> 结果持续更新中，欢迎提交 PR 分享你的实验数据 🙌

---

## 🐛 踩坑记录

学习过程中遇到的问题会同步整理到仓库 Issue 或笔记中，例如：

- LoveDA 数据集目录结构与官方 `test` 标签缺失的处理
- 遥感图像大尺寸切图与重叠裁剪
- 类别不均衡下的损失函数选择
- 多卡 / 混合精度训练的简单实践

---

## 🤝 交流与贡献

- 有任何问题欢迎提 [Issue](../../issues) 交流
- 发现 bug 或有改进建议，欢迎 PR
- 如果这个仓库对你的学习有帮助，欢迎点个 ⭐ Star

---

## 👤 作者

**燕山大学 · 自动化专业**

一名正在学习遥感语义分割的初学者，热衷于把踩过的坑整理成可复用的代码与笔记。

---

## 📄 License

本项目基于 [MIT License](./LICENSE) 开源，仅供学习交流使用。