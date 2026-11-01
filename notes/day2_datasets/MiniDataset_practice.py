import torch
import yaml
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

current_path = Path(__file__).resolve().parent
PROJECT_ROOT = current_path.parents[1]

config_path = PROJECT_ROOT/'configs'/'loveda_unet.yaml'
with open(config_path, mode='r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

data_root = Path(config['data']['root'])
img_dir = data_root/'Train'/'Rural'/'images_png'
mask_dir = data_root/'Train'/'Rural'/'masks_png'

class MiniSegDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path):
        #准备文件路径
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        #检查文件路径
        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f'错误❌️:原图文件夹{str(self.img_dir)}或标签文件夹{str(self.mask_dir)}[不存在]')

        self.img_path = [p for p in self.img_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png']
        self.mask_path = [p for p in self.mask_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png']

        if not self.img_path:
            raise ValueError(f'错误❌️:原图文件夹{str(self.img_dir)}下面未发现图片')
        if not self.mask_path:
            raise ValueError(f'错误❌️:标签文件夹{str(self.mask_dir)}下面未发现图片')

    def __len__(self):
        #统计文件长度
        return len(self.img_path)

    def __getitem__(self, idx):
        #根据文件加载img与mask
        img = self.img_path[idx]
        with Image.open(img) as img:
            img = img.convert('RGB')
            img_arr = np.array(img, dtype=np.float32)

        mask = self.mask_path[idx]
        with Image.open(mask) as mask:
            if str(mask.mode).lower() == 'p':
                mask_arr = np.array(mask, dtype=np.int64)
            else:
                mask.convert('L')
                mask_arr = np.array(mask, dtype=np.int64)

        #转为tensor前需要对image中的数值先做归一化
        img_arr = img_arr/255.0

        #转为tensor
        img_tensor = torch.from_numpy(img_arr)
        img_tensor.permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_arr)

        return img_tensor, mask_tensor


minisegdataset = MiniSegDataset(img_dir, mask_dir)
print("dataset length:", len(minisegdataset))
image, mask = minisegdataset[0]
print("image:", image.shape, image.dtype)
print("mask:", mask.shape, mask.dtype)