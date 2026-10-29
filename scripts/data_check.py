from turtle import st
import matplotlib.pyplot as plt
import cv2, numpy as np
from collections import Counter
from PIL import Image
from pathlib import Path
import os
import ply
import yaml
from tqdm import tqdm

#需要检查的路径：
# D:\Postgraduate_Study\Data\loveda\Train\Rural\images_png
# D:\Postgraduate_Study\Data\loveda\Train\Rural\masks_png
# D:\Postgraduate_Study\Data\loveda\Train\Urban\images_png
# D:\Postgraduate_Study\Data\loveda\Train\Urban\masks_png

# D:\Postgraduate_Study\Data\loveda\Val\Rural\images_png
# D:\Postgraduate_Study\Data\loveda\Val\Urban\masks_png
# D:\Postgraduate_Study\Data\loveda\Val\Rural\images_png
# D:\Postgraduate_Study\Data\loveda\Val\Urban\masks_png


with open('./configs/loveda_unet.yaml','r',encoding='utf-8') as f:
    configs = yaml.safe_load(f)
root_path = Path(configs['data']['root'])

#路径构造
Domains = ['Urban','Rural']
Splits = ['Train','Val'] 

#抽样数量
n=8
sample_save_pths = './logs/sample_png'

'检查 1：文件可读性 + 尺寸 + 通道'
#要求接收图片**路径(path,str)**的列表
def check_files(path,expect_size=None):
    #[("出错的图片路径", "具体的报错信息")]
    #{'具体size':数量}
    #{'颜色模式名称': 数量}
    bad,sizes,mode = [],{},{}
    #查找出错的文件及原因
    for p in tqdm(path,desc="校验遥感图像"):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((str(p),repr(e)))
            continue

        with Image.open(p) as im:
            sizes[im.size] = sizes.get(im.size,0)+1
            mode[im.mode] = mode.get(im.mode,0)+1
    return bad,sizes,mode


'检查 2：图像/掩码成对 + 尺寸一致'
def check_pairing(img_path, mask_path):
    imgs = {p.stem: p for p in img_path.glob('*.PNG')}
    masks = {p.stem: p for p in mask_path.glob('*.PNG')}

    only_imgs = set(imgs.keys()) - set(masks.keys())
    only_masks =set(masks.keys()) - set(imgs.keys())

    mismatched = []
    for stem in imgs.keys() & masks.keys():
        with Image.open(imgs[stem]) as a, Image.open(masks[stem]) as b:
            if a.size != b.size:
                mismatched.append((stem,a.size,b.size))
    return only_imgs, only_masks, mismatched

    
'# 检查 3：标签像素值统计（最关键）'
def label_histogram(mask_dir,sample:None):
    counter = Counter()
    if isinstance(mask_dir,Path):
        mask_dir = list(mask_dir.glob('*.PNG'))
    file = mask_dir[0:sample] if sample else mask_dir
    print(f'🔍掩码像素校验路径：{str(mask_dir[0].parent)} | 掩码像素校验数量{len(file)}')
        #读取图片信息
    for p in tqdm(file,mininterval=0.01):
        m = cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        #统计图片信息
        #返回此次图片的内容,两个list,单图排序
        vals, cnts = np.unique(m, return_counts=True) # type: ignore
        #键已存在则值相加，键不存在则新建，对dict进行操作
        counter.update(dict(zip(vals.tolist(),cnts.tolist())))
        #统计所有的像素数
    total = sum(counter.values())
        #返回结果，整体排序
    return {v:(c, c/total)for v, c in sorted(counter.items())}


'检查 4：可视化抽样'
def visualize_mask_check(img_paths,mask_paths,save_paths,n=8):
    n = min(n,len(img_paths))
    #绘制画板
    fig, axes = plt.subplots(3,n,figsize=(4 * n,12))

    #展示图片
    for i in range(n):
        img = cv2.cvtColor(cv2.imread(str(img_paths[i])),cv2.COLOR_BGR2RGB)#type:ignore
        msk = cv2.imread(str(mask_paths[i]),cv2.IMREAD_GRAYSCALE)

        #展示原图
        axes[0,i].imshow(img)
        axes[0,i].set_title(f"Image {i}", fontsize=8)
        axes[0,i].axis('off')
        # 第 2 行：掩码（用 gray 或指定 colormap，让类别颜色可辨）
        axes[1, i].imshow(msk, cmap="tab10", vmin=0, vmax=6)
        axes[1, i].set_title(f"Mask {i}", fontsize=8)
        axes[1, i].axis("off")
        # 第 3 行：mask == 0 的二值图（白 = 0，黑 = 非0）
        zero_bin = (msk == 0).astype(np.uint8) * 255
        axes[2, i].imshow(zero_bin, cmap="gray")
        axes[2, i].set_title(f"mask==0 {i}", fontsize=8)
        axes[2, i].axis("off")

    #调整子图间距
    plt.tight_layout()
    plt.savefig(save_paths, dpi=150, bbox_inches="tight")
    plt.close()



#pipeline处理
for split in Splits:
    for domain in Domains:
        img_dir = root_path/split/domain/'images_png'
        mask_dir = root_path/split/domain/'masks_png'
        #对配对后的进行检查
        if not all([img_dir.exists(),mask_dir.exists()]):
            print(f"\n ⚠️ 跳过 {split} / {domain}：目录不存在\n    {img_dir}")
            continue

        print(f"\n{'=' * 62}")
        print(f" 检查: {split} / {domain}")
        print('=' * 62)
        img_path = list(img_dir.glob('*.PNG'))
        mask_path = list(mask_dir.glob('*.PNG'))

        #检查原图
        if img_path:
            b,s,m = check_files(img_path)
            print(f" 🖼️ 原图 | {len(img_path):>6,} 张 | 损坏 {len(b)} | 尺寸 {s} | 模式 {m}")
        #检查标签
        if mask_path:
            b,s,m = check_files(mask_path)
            print(f" 🏷️ 标签 | {len(mask_path):>6,} 张 | 损坏 {len(b)} | 尺寸 {s} | 模式 {m}")
        
        if img_path and mask_path:
            oi,om,mismatched =check_pairing(img_dir,mask_dir)
            if mismatched:
                print(f"❌ 警告: 有{len(mismatched)}张原图与标签不匹配！例如: {mismatched[:2]}")
            if oi:
                print(f"❌ 警告: 有 {len(oi)} 张原图找不到对应的标签！例如: {list(oi)[:2]}")
            if om:
                print(f"❌ 警告: 有 {len(om)} 张标签找不到对应的原图！例如: {list(om)[:2]}")
            if not om and not oi and not mismatched:
                print("✅ 原图与标签完美匹配，数量与尺寸一一对应！")

        label_check_dict = label_histogram(mask_dir,500)#type:ignore
        sampled = min(500, len(mask_path))
        print(f"\n 标签像素分布（抽样 {sampled:,} / {len(mask_path):,} 张）")
        print(f" {'value':>5}  {'ratio':>9}  {'count':>16}")
        print(f" {'-' * 5}  {'-' * 9}  {'-' * 16}")
        for key in label_check_dict:#type:ignore
            cnt, ratio = label_check_dict[key]#type:ignore
            print(f" {key:>5}  {ratio:>9.3%}  {cnt:>16,}")

        print(f'👀可视化抽样 | 抽样数量：{n} | 保存目录: {sample_save_pths}')
        save_file = Path(sample_save_pths) / f"{split}_{domain}.png"
        visualize_mask_check(img_path,mask_path,save_file)
