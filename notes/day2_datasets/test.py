from pathlib import Path
import numpy as np
from PIL import Image
from collections import Counter


def confirm_loveda_zero(mask_dir, sample_size=1000):
    """
    验证 LoveDA 掩码中 0 是否为 NoData（官方标准）。
    结论基于：官方文档 + 本地数据分布。
    """
    paths = list(Path(mask_dir).glob("*.png"))
    if not paths:
        print("目录下没有找到 png 文件，请检查路径。")
        return

    # 抽样统计（全量跑会慢，抽样已足够确认）
    if len(paths) > sample_size:
        import random
        paths = random.sample(paths, sample_size)

    zero_pixel_total = 0
    total_pixel = 0
    unique_values = Counter()

    for p in paths:
        with Image.open(p) as img:
            m = np.array(img)
        # 统计所有出现过的像素值（官方合法值应为 0~7）
        for v in np.unique(m):
            unique_values[v] += 1

        zero_mask = (m == 0)
        zero_pixel_total += zero_mask.sum()
        total_pixel += m.size

    print(f"抽样文件数: {len(paths)}")
    print(f"出现过的像素值集合: {sorted(unique_values.keys())}")
    print(f"0 像素总占比: {zero_pixel_total / total_pixel:.2%}")
    print("-" * 40)

    # 官方结论：0 是 NoData，真实类别是 1~7
    if 0 in unique_values:
        print("✅ 确认：数据中存在像素值 0。")
        print("   根据 LoveDA 官方定义，0 是 NoData（无效区域），必须忽略。")
        print("   处理方式：在 Dataset 中设置 remap_zero_to_ignore=True")
        print("   模型输出通道数应为 7（对应类别 1~7）。")
    else:
        print("⚠️ 抽样中未发现像素值 0，但官方标准仍是：若出现 0，按 NoData 忽略。")

    # 检查是否有超出 0~7 的异常值
    invalid = [v for v in unique_values if v > 7]
    if invalid:
        print(f"⚠️ 发现异常像素值（>7）：{invalid}，请检查数据完整性。")


# 替换成你的实际路径
confirm_loveda_zero(r"D:/Postgraduate_Study/Data/loveda/Train/Rural/masks_png")