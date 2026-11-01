"""标准 U-Net（Ronneberger et al., 2015）。

    x = inc(x)                       # 64
    down × (depth-1)                 # 64 → 128 → 256 → 512 → 1024
    up   × (depth-1)，每次拼接 skip   # 1024 → 512 → 256 → 128 → 64
    outc: 1×1 卷积 → num_classes

输入 [N, 3, H, W]，输出 [N, num_classes, H, W]（与输入同分辨率）。
H、W 不要求是 2 的幂：上采样时若尺寸与 skip 差一两个像素，会自动 pad 对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3×3 Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """2×2 最大池化 + DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """上采样 + 拼接 skip + DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            cat_ch = in_ch + out_ch
        else:
            # 转置卷积：同时完成上采样和通道减半
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            cat_ch = out_ch * 2
        self.conv = DoubleConv(cat_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 输入边长不是 2^(depth-1) 的整数倍时，上采样后可能比 skip 小一圈
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        base_channels: int = 64,
        bilinear: bool = False,
        depth: int = 5,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth 至少为 2，收到 {depth}")

        chs = [base_channels * 2 ** i for i in range(depth)]  # 64,128,256,512,1024

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [Down(chs[i], chs[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [Up(chs[i + 1], chs[i], bilinear) for i in reversed(range(depth - 1))]
        )
        self.outc = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        for down in self.downs:
            skips.append(x)      # 保存下采样前的特征用于拼接
            x = down(x)

        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 1)])   # 由深到浅取回 skip

        return self.outc(x)


def build_model(cfg: dict) -> nn.Module:
    """配置 -> 模型。配置键名的映射只出现在这一个函数里。"""
    m = cfg["model"]
    model_type = str(m.get("type", "unet")).lower()
    if model_type != "unet":
        raise ValueError(f"暂不支持的 model.type={model_type!r}")

    return UNet(
        in_channels=int(m.get("in_channels", 3)),
        num_classes=int(m.get("num_classes") or cfg["data"]["num_classes"]),
        base_channels=int(m.get("base_channels", 64)),
        bilinear=bool(m.get("bilinear", False)),
        depth=int(m.get("depth", 5)),
    )


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=7)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"in  {tuple(x.shape)}")
    print(f"out {tuple(y.shape)}")
    print(f"params {n / 1e6:.2f} M")
