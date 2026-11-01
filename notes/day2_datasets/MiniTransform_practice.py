"""MiniTransform：手写的成对变换（练习版）。

与 AugDataset 之间的契约：
    输入  img : np.ndarray, uint8, HWC(RGB)
          mask: np.ndarray, uint8, HW，值为类别索引
    输出  img : torch.float32, [3, S, S]，已归一化
          mask: torch.int64,   [S, S]

关键点：几何变换必须让 img 和 mask 共用同一次随机结果，
所以写成一个 __call__(img, mask)，而不是两个独立的 transform。
"""

import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class MiniTransform:
    def __init__(
        self,
        crop_size: int = 512,
        random_scale: Optional[Tuple[float, float]] = (0.5, 2.0),
        hflip: float = 0.5,
        vflip: float = 0.5,
        color_jitter: float = 0.0,
        mean: Sequence[float] = MEAN,
        std: Sequence[float] = STD,
        training: bool = True,
    ):
        self.crop_size = int(crop_size)
        self.random_scale = tuple(random_scale) if random_scale else None
        self.hflip = float(hflip)
        self.vflip = float(vflip)
        self.color_jitter = float(color_jitter)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.training = bool(training)

    @classmethod
    def from_config(cls, config: dict, training: bool) -> "MiniTransform":
        """配置 -> 对象。用 .get + or 兜住 yaml 里留空（解析成 None）的键。"""
        aug = config.get("augment") or {}
        norm = aug.get("normalize") or {}
        return cls(
            crop_size=aug.get("crop_size") or 512,
            random_scale=aug.get("random_scale"),
            hflip=aug.get("hflip") or 0.0,
            vflip=aug.get("vflip") or 0.0,
            color_jitter=aug.get("color_jitter") or 0.0,
            mean=norm.get("mean") or MEAN,
            std=norm.get("std") or STD,
            training=training,
        )

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        # 调用方保证是 uint8；如果传进来 float 0~1，这里会静默截断成 0/1
        if img.dtype != np.uint8:
            raise TypeError(f"img 需要 uint8，收到 {img.dtype}")

        pil_img = Image.fromarray(img.astype(np.uint8, copy=False), mode="RGB")
        pil_mask = Image.fromarray(mask.astype(np.uint8, copy=False))

        if self.training:
            pil_img, pil_mask = self._scale(pil_img, pil_mask)
            pil_img, pil_mask = self._crop(pil_img, pil_mask)
            pil_img, pil_mask = self._flip(pil_img, pil_mask)
            if self.color_jitter > 0.0:
                pil_img = self._jitter(pil_img)
        else:
            # 测试和训练的尺寸必须保持一样
            size = (self.crop_size, self.crop_size)
            pil_img = pil_img.resize(size, Image.Resampling.BILINEAR)
            pil_mask = pil_mask.resize(size, Image.Resampling.NEAREST)
        return self._to_tensor(pil_img, pil_mask)

    def _scale(self, img: Image.Image, mask: Image.Image):
        if self.random_scale is None:
            return img, mask

        w, h = img.size
        scale_ratio = random.uniform(*self.random_scale)
        # 夹住缩放系数本身，而不是分别夹宽高——否则非方形图会被拉变形
        scale_ratio = max(scale_ratio, self.crop_size / min(w, h))
        out_w = max(self.crop_size, int(round(scale_ratio * w)))
        out_h = max(self.crop_size, int(round(scale_ratio * h)))

        img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
        mask = mask.resize((out_w, out_h), Image.Resampling.NEAREST)
        return img, mask

    def _crop(self, img: Image.Image, mask: Image.Image):
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            raise ValueError(f"图像 {w}x{h} 小于 crop_size={self.crop_size}")

        # 防止超出边界
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        box = (x1, y1, x1 + self.crop_size, y1 + self.crop_size)

        return img.crop(box), mask.crop(box)

    def _flip(self, img: Image.Image, mask: Image.Image):
        # hflip = 水平翻转 = 左右镜像；vflip = 垂直翻转 = 上下镜像
        if random.random() < self.hflip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.vflip:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return img, mask

    def _jitter(self, img: Image.Image):
        jitter_ratio = random.uniform(-self.color_jitter, self.color_jitter)
        img = ImageEnhance.Brightness(img).enhance(1 + jitter_ratio)
        jitter_ratio = random.uniform(-self.color_jitter, self.color_jitter)
        img = ImageEnhance.Contrast(img).enhance(1 + jitter_ratio)
        return img

    def _to_tensor(self, img: Image.Image, mask: Image.Image):
        arr = np.array(img, dtype=np.float32) / 255.0          # HWC, 0~1
        arr = (arr - self.mean) / self.std                      # (H,W,3) - (3,) 广播
        img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_t, mask_t
