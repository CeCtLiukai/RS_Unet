import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PairedTransform:
    def __init__(
        self,
        crop_size: int = 512,
        scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        hflip: float = 0.5,
        vflip: float = 0.5,
        color_jitter: float = 0.0,
        mean: Sequence[float] = MEAN,
        std: Sequence[float] = STD,
        training: bool = True,
    ):
        self.crop_size = int(crop_size)
        self.scale_range = tuple(scale_range) if scale_range else None
        self.hflip = float(hflip)
        self.vflip = float(vflip)
        self.color_jitter = float(color_jitter)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.training = training

    @classmethod
    def from_config(cls, cfg: dict, training: bool = True) -> 'PairedTransform':

        aug = cfg.get('augment') or {}
        norm = aug.get('normalize') or {}
        return cls(
            crop_size=aug.get('crop_size') or 512,
            scale_range=aug.get('random_scale'),
            hflip=aug.get('hflip') or 0.0,
            vflip=aug.get('vflip') or 0.0,
            color_jitter=aug.get('color_jitter') or 0.0,
            mean=norm.get('mean') or MEAN,
            std=norm.get('std') or STD,
            training=training,
        )

    def __call__(self, img: Image.Image, mask: np.ndarray):
        mask = Image.fromarray(mask)

        if self.training:
            img, mask = self._random_scale(img, mask)
            img, mask = self._random_crop(img, mask)
            img, mask = self._random_flip(img, mask)  # ← 修复 1 后这里才有效
            if self.color_jitter > 0:
                img = self._jitter(img)
        else:
            side = (self.crop_size, self.crop_size)
            img = img.resize(side, Image.Resampling.BILINEAR)
            mask = mask.resize(side, Image.Resampling.NEAREST)

        return self._to_tensor(img, mask)

    def _random_scale(self, img: Image.Image, mask: Image.Image):
        if not self.scale_range:
            return img, mask

        #选取一个缩放的大小
        ratio = random.uniform(*self.scale_range)
        #统计当前的大小
        w, h = img.size
        #最后应当被裁减的大小
        out_w = max(self.crop_size, int(round(ratio*w)))
        out_h = max(self.crop_size, int(round(ratio*h)))

        return (
            img.resize((out_w, out_h), Image.Resampling.BILINEAR),
            mask.resize((out_w, out_h), Image.Resampling.NEAREST)
        )

    def _random_crop(self, img: Image.Image, mask: Image.Image):
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            raise ValueError(f'图像 {w}x{h} 小于 crop_size={self.crop_size}')

        x1 = random.randint(0, w-self.crop_size)
        y1 = random.randint(0, h-self.crop_size)
        x2 = x1+self.crop_size
        y2 = y1+self.crop_size

        box = (x1, y1, x2, y2)
        return img.crop(box), mask.crop(box)

    def _random_flip(self, img, mask):
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
        img = ImageEnhance.Brightness(img).enhance(1+jitter_ratio)
        jitter_ratio = random.uniform(-self.color_jitter, self.color_jitter)
        img = ImageEnhance.Contrast(img).enhance(1+jitter_ratio)
        return img

    def _to_tensor(self, img, mask):
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC，0~1
        arr = (arr - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_t, mask_t
