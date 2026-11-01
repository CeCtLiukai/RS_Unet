import numpy as np
import torch
import yaml
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Sequence

from MiniTransform_practice import MiniTransform

PRACTICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PRACTICE_DIR.parents[1]

config_path = PROJECT_ROOT / "configs" / "loveda_unet.yaml"
with open(config_path, mode="r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# LoveDA 官方 7 类，索引与掩码像素值一一对应（0~6，共 7 项）
CLASS_NUM = {
    0: "no-data",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class AugDataset(Dataset):
    """目标是编写一个能够同时识别 Urban 和 Rural 场景的通用 Dataset。"""

    def __init__(
        self,
        data_root,
        split: str = "Train",
        domains: Sequence[str] = ("Urban", "Rural"),
        transform=None,
        ignore_index: int = 255,
        remap_zero_ignore: bool = False,
        mean: Sequence[float] = MEAN,
        std: Sequence[float] = STD,
    ):
        """
        :param data_root: 数据根路径，str 或 Path 都可以：'D:/Postgraduate_Study/Data/loveda'
        :param split: 数据集划分：Train / Val
        :param domains: 区域划分 ('Urban', 'Rural')；同一个 split 下两个域会被合并成一张清单
        """
        # 统一转成 Path：调用方很可能直接传 config 里的字符串
        self.data_root = Path(data_root)
        self.split = split
        self.domains = list(domains)
        self.ignore_index = int(ignore_index)
        self.remap_zero_ignore = bool(remap_zero_ignore)
        self.transform = transform
        # 仅在没有传 transform 时作为兜底归一化使用
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"错误: 传入的数据根路径不存在 {self.data_root}")

        self.samples = self._get_samples()
        if not self.samples:
            raise ValueError("错误: 没有生成任何图像-标签对")

    def _get_samples(self):
        samples = []
        for domain in self.domains:
            img_dir = self.data_root / self.split / domain / "images_png"
            mask_dir = self.data_root / self.split / domain / "masks_png"

            # 目录缺失属于"这个域本来就没有"（例如 Val 下没有 Rural），跳过而不是报错
            if not img_dir.is_dir() or not mask_dir.is_dir():
                print(f"[AugDataset] 跳过 {self.split}/{domain}: 目录不存在 {img_dir}")
                continue

            img_path = {
                p.stem: p
                for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            }
            mask_path = {
                p.stem: p
                for p in mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            }

            only_img = sorted(set(img_path) - set(mask_path))
            only_mask = sorted(set(mask_path) - set(img_path))
            if only_img or only_mask:
                # 文件缺失属于数据坏了，必须中断：静默丢样本在训练里查不出来
                raise RuntimeError(
                    f"图文不配对 {self.split}/{domain}："
                    f"{len(only_img)} 张原图无标签（如 {sorted(only_img)[:3]}）；"
                    f"{len(only_mask)} 张标签无原图（如 {sorted(only_mask)[:3]}）"
                )

            for stem in sorted(img_path):
                samples.append((img_path[stem], mask_path[stem], domain))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path, domain = self.samples[index]

        with Image.open(img_path) as img:
            img_arr = np.array(img.convert("RGB"))
        with Image.open(mask_path) as mask:
            if mask.mode == "P":
                mask_arr = np.array(mask)            # 调色板图直接取索引，不能 convert('L')
            else:
                mask_arr = np.array(mask.convert("L"))
        mask_arr = mask_arr.astype(np.uint8)

        if self.remap_zero_ignore:
            mask_arr = np.where(mask_arr == 0, self.ignore_index, mask_arr).astype(np.uint8)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_arr, mask_arr)
        else:
            arr = img_arr.astype(np.float32) / 255.0
            arr = (arr - self.mean) / self.std
            img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
            mask_t = torch.from_numpy(mask_arr.astype(np.int64))

        meta = {"stem": img_path.stem, "domain": domain}
        return img_t, mask_t, meta


def build_dataset(config: dict, training: bool):
    data = config["data"]
    return AugDataset(
        data_root=data["root"],
        split="Train" if training else "Val",
        domains=data["train_domains"] if training else data["val_domains"],
        transform=MiniTransform.from_config(config, training=training),
        ignore_index=data["ignore_index"],
        remap_zero_ignore=data.get("remap_zero_to_ignore", False),
    )


def get_dataloaders(config: dict):
    data = config["data"]
    train_datasets = build_dataset(config, training=True)
    val_datasets = build_dataset(config, training=False)

    num_workers = int(data["num_workers"])
    # persistent_workers 只在 num_workers > 0 时合法，为 0 会直接 ValueError
    persist = num_workers > 0

    train_dataloader = DataLoader(
        train_datasets,
        batch_size=int(data["batch_size"]),
        pin_memory=bool(data["pin_memory"]),
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=persist,
    )
    val_dataloader = DataLoader(
        val_datasets,
        batch_size=int(data.get("val_batch_size") or data["batch_size"]),
        pin_memory=bool(data["pin_memory"]),
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,          # 验证集不能丢尾巴
        persistent_workers=persist,
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    ds = build_dataset(config, training=True)
    print(ds.samples[0])
    print(f"samples: {len(ds)}")
    img_t, mask_t, meta = ds[0]
    print("img ", tuple(img_t.shape), img_t.dtype)
    print("mask", tuple(mask_t.shape), mask_t.dtype, torch.unique(mask_t).tolist())
    print("meta", meta)
