import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rsunet.datasets.transform import PairedTransform


class LoveDADataset(Dataset):
    IMG_DIR = "images_png"
    MASK_DIR = "masks_png"

    def __init__(
        self,
        root: str,
        split: str,
        domains: Sequence[str] = ("Urban", "Rural"),
        transform=None,
        ignore_index: int = 255,
        remap_zero_to_ignore: bool = False,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在: {root}")

        self.split = split
        self.domains = list(domains)
        self.transform = transform
        self.ignore_index = int(ignore_index)
        self.remap_zero_to_ignore = bool(remap_zero_to_ignore)

        self.samples: List[Tuple[Path, Path, str]] = self._scan()
        if not self.samples:  # ← 修复 2
            raise RuntimeError(
                f"{self.root / self.split} 下没有找到任何图文对，"
                f"请确认 split={split!r} 与 domains={self.domains}"
            )

    def _scan(self) -> List[Tuple[Path, Path, str]]:
        samples: List[Tuple[Path, Path, str]] = []
        for domain in self.domains:
            img_dir = self.root / self.split / domain / self.IMG_DIR
            mask_dir = self.root / self.split / domain / self.MASK_DIR

            if not img_dir.is_dir() or not mask_dir.is_dir():  # ← 修复 6
                print(f"[LoveDADataset] 跳过 {self.split}/{domain}：目录不存在")
                continue

            imgs = {
                p.stem: p
                for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"  # ← 修复 5
            }
            masks = {
                p.stem: p
                for p in mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            }

            only_img = sorted(imgs.keys() - masks.keys())
            only_mask = sorted(masks.keys() - imgs.keys())
            if only_img or only_mask:  # ← 修复 3
                raise RuntimeError(  # ← 修复 4
                    f"图文不配对 {self.split}/{domain}："
                    f"{len(only_img)} 图无掩码（如 {only_img[:3]}）；"
                    f"{len(only_mask)} 掩码无原图（如 {only_mask[:3]}）"
                )

            for stem in sorted(imgs):
                samples.append((imgs[stem], masks[stem], domain))

        return samples  # ← 修复 1：移到循环外

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        with Image.open(path) as mask:
            if mask.mode == "P":
                return np.array(mask, dtype=np.uint8)
            return np.array(mask.convert("L"), dtype=np.uint8)

    def __getitem__(self, idx: int):
        img_path, mask_path, domain = self.samples[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
        mask = self._load_mask(mask_path)

        if self.remap_zero_to_ignore:
            mask = np.where(mask == 0, self.ignore_index, mask).astype(np.uint8)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
            mask = torch.from_numpy(mask.astype(np.int64))

        meta = {"stem": img_path.stem, "domain": domain}
        return img, mask, meta

    @property
    def domain_counts(self) -> Dict[str, int]:  # ← 修复 7
        return dict(Counter(d for _, _, d in self.samples))

    def __repr__(self) -> str:
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(self.domain_counts.items()))
        return (


            f"LoveDADataset(split={self.split!r}, n={len(self.samples)}, "
            f"{detail}, ignore_index={self.ignore_index}, "
            f"remap_zero={self.remap_zero_to_ignore})"
        )


def build_dataset(config: dict, training: bool = True) -> LoveDADataset:
    """配置 -> 数据集。配置键名的映射只出现在这一个函数里。"""
    data = config["data"]
    return LoveDADataset(
        root=data["root"],
        split="Train" if training else "Val",
        domains=data["train_domains"] if training else data["val_domains"],
        transform=PairedTransform.from_config(config, training=training),
        ignore_index=data["ignore_index"],
        remap_zero_to_ignore=data.get("remap_zero_to_ignore", False),
    )


def _worker_init(worker_id: int):
    """让每个 worker 的随机序列互相独立。Windows 上是 spawn，本来就独立；
    Linux 的 fork 模式会继承父进程状态，不加这个所有 worker 的增广会一模一样。"""
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def get_dataloaders(config: dict, train_dataset: LoveDADataset, val_dataset: LoveDADataset):
    data = config["data"]
    num_workers = int(data["num_workers"])
    pin_memory = bool(data["pin_memory"])
    # 这两个只在 num_workers > 0 时合法
    persist = num_workers > 0
    init_fn = _worker_init if num_workers > 0 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(data["batch_size"]),
        shuffle=True,
        drop_last=bool(data["drop_last"]),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist,
        worker_init_fn=init_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(data.get("val_batch_size") or data["batch_size"]),
        shuffle=False,
        drop_last=False,  # 验证集不能丢尾巴，否则指标有偏
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist,
        worker_init_fn=init_fn,
    )
    return train_dataloader, val_dataloader