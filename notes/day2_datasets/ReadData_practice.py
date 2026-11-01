import yaml
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

current_path = Path(__file__).resolve().parent
PROJECT_ROOT = current_path.parents[1]
# region 数据扫描与配对
config_path = PROJECT_ROOT/'configs'/'loveda_unet.yaml'
with open(config_path, mode='r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

data_root = Path(config['data']['root'])
#读取图片
img_dir = data_root/'Train'/'Rural'/'images_png'
mask_dir = data_root/'Train'/'Rural'/'masks_png'

img_path = [p for p in img_dir.iterdir() if p.suffix.lower() == '.png']
mask_path = [p for p in mask_dir.iterdir() if p.suffix.lower() == '.png']

img = img_path[0]
mask = mask_path[0]

with Image.open(img) as img:
    img.convert('RGB')
    '统一转为numpy数组,方便查看尺寸,查看数据类型,转为Tensor等操作'
    img_array = np.array(img)
    '转完都是uint8'
    print(type(img_array), type(img_array[0][0][0]))

with Image.open(mask) as mask:
    if str(mask.mode).lower() == 'p':
        mask_array = np.array(mask)
    else:
        mask.convert('L')
        mask_array = np.array(mask)
    '转完都是uint8'
    print(type(mask_array), type(mask_array[0][0]))

'转完之后我们就可以查看img和mask的形状'
#检查形状
print(f'img_shape:{img_array.shape}')
print(f'mask_shape:{mask_array.shape}')
#检查类型
print(f'img_dtype:{img_array.dtype}')
print(f'mask_dtype:{mask_array.dtype}')
#检查数值
print(f'mask_val_cnts:{np.unique(mask_array, return_counts=True)}')
#endregion

