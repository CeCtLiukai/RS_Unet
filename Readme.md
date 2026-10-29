<!-- RS_Unet/
├── configs/
│   ├── loveda_unet.yaml          # 数据路径、类别数、超参
│   └── loveda_unet_r50.yaml
├── rsunet/                       # 代码包（可 import）
│   ├── datasets/
│   │   ├── loveda.py             # LoveDA Dataset
│   │   └── transforms.py         # 裁剪/翻转/归一化
│   ├── models/
│   │   ├── unet.py
│   │   └── backbones/
│   ├── losses.py                 # CE / Dice / Focal
│   ├── metrics.py                # mIoU、aAcc、per-class IoU
│   └── engine.py                 # train/val 循环
├── train.py                      # 入口，读 config
├── test.py                       # 入口，读 config + 权重
├── scripts/
│   ├── prepare_loveda.py         # 拆分配置/统计类别/生成清单
│   └── eval_official.py          # 官方评测口径
├── data/                         # 只放数据，.gitignore
│   └── LoveDA/
│       ├── Train/{Urban,Rural}/
│       ├── Val/Urban/
│       └── Test/{Urban,Rural}/
├── runs/                         # .gitignore
│   └── unet_r50_512_bs8/
│       ├── config.yaml           # 本次实验的配置快照
│       ├── train.log
│       ├── tb/                   # tensorboard
│       ├── weights/
│       │   ├── best_miou.pth
│       │   └── last.pth
│       └── preds/                # 测试可视化
├── .gitignore
└── Readme.md -->
