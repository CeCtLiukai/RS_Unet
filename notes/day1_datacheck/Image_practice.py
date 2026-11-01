from pathlib import Path
from PIL import Image
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve().parent
PROJECT_ROOT = current_file.parents[1]
"""
题目 1：防弹级数据加载（PIL 校验 + CV 读取 + 尺寸同步）
场景：在 Dataset 中，你必须确保读入的原图和标签是完美配对的。但有时候原图是完好的，标签却损坏了。
任务：编写一个函数 load_and_check_pair(img_path, mask_path)。
先使用 PIL 的 Image.open 和 verify() 检查 img_path 和 mask_path 的结构完整性。
如果校验通过，使用 OpenCV 读取原图（默认彩色）和标签（灰度）。
核心考点：如何确保 CV 读取成功了？如果返回 None 怎么办？
对比 PIL 读取的尺寸和 CV 读取的数组形状（shape），如果尺寸不一致或读取失败，抛出明确的错误。
cv读取的是uint8
"""
config_path = PROJECT_ROOT / "configs" / "loveda_unet.yaml"
with open(config_path, mode="r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_ROOT = Path(config["data"]["root"])
# 加载一对pair
img_dir = DATA_ROOT / "Train" / "Rural" / "images_png"
mask_dir = DATA_ROOT / "Train" / "Rural" / "masks_png"

if not img_dir:
    raise FileNotFoundError(f"❌错误:当前路径{img_dir}下没有原图文件")
if not mask_dir:
    raise FileNotFoundError(f"❌错误:当前路径{mask_dir}下没有标签文件")

img_path = [p for p in img_dir.glob("*.PNG")][0]
mask_path = [p for p in mask_dir.glob("*.PNG")][0]


def load_and_check_pair(img, mask):
    target = [img, mask]
    for p in target:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            raise ValueError(
                f"❌️文件信息损坏，无法打开路径{str(p)}下的照片\n错误信息:{e}"
            )

    print(f"校验通过，使用cv读取原图与标签信息")

    cv_img = cv2.imread(str(img), cv2.IMREAD_COLOR)
    cv_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    # 校验cv的读图结果
    if cv_img is None:
        raise FileNotFoundError(f"❌️未能找到原始图片信息,请检查图片路径:{img}")
    if cv_mask is None:
        raise FileNotFoundError(f"❌️未能找到标签图片信息,请检查图片路径:{mask}")

    # 校验图片尺寸
    # 先对cv读取后的图片转换通道顺序
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    with Image.open(img) as a, Image.open(mask) as b:
        if a.size != b.size:
            raise ValueError(f"❌️错误:原图尺寸{a.size}与标签尺寸不匹配{b.size}")

        im = np.array(im)
        if im.shape != cv_img.shape:
            raise ValueError(f"错误PIL与CV尺寸不匹配,请检查图片路径:{img}")
        return cv_img, cv_mask


"""
题目 2：把 OpenCV 变成 PIL（保存图片）
场景：你用 OpenCV 读取了一张图片，做了一堆处理，现在你想用 PIL 自带的方法把它优雅地保存下来（因为 PIL 保存图片时对格式的控制更好）。
任务：用 cv2.imread() 读取一张彩色图片。把它转换成 PIL 的 Image 对象（注意：Image.fromarray() 之前，你需要确保颜色通道顺序是对的，否则存出来的图颜色会变成“阿凡达”）。
用 PIL 的 save() 方法把它保存为 。
"""
save_path = PROJECT_ROOT / "reports" / "output.png"

cv_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
p1_img = Image.fromarray(cv_img)
# 创建文件夹是对目录使用的
save_path.parent.mkdir(parents=True, exist_ok=True)
p1_img.save(save_path)


"""# 题目 3：双剑合璧的“防弹”读取函数
场景：你要在 Dataset 的 __getitem__ 里读取数据，要求既安全又标准。安全性交给 PIL 校验，标准化交给 OpenCV。
任务：编写一个函数 safe_load_image(img_path)：
第一步（安全）：用 PIL 打开图片，调用 verify()。如果这一步报错，直接返回 None（或者抛出异常）。
第二步（重新打开）：因为 verify() 后对象失效，用 PIL 重新打开，获取它的 size 和 mode（用于后续判断）。
第三步（读取）：用 OpenCV 读取这张图片（读取为 RGB 格式的数组）。
第四步（验证）：如果 cv2.imread() 返回了 None，说明 OpenCV 解码失败，打印警告。如果成功，打印出 PIL 获取的尺寸和 OpenCV 数组的 shape。
(这道题把你笔记里所有的坑都踩了一遍，非常考验基本功)
"""


def safe_load_image(img_path):
    try:
        with Image.open(img_path) as im:
            im.verify()
    except Exception as e:
        raise ValueError(f"错误,图片{img_path}无法正常加载\n错误信息{e}")

    with Image.open(img_path) as im:
        img_size = im.size
        img_mode = im.mode

        cv_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if cv_img is None:
            print(f"❌错误，当前图片{img_path}解码错误")
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return img_size, img_mode, cv_img.shape


"""
题目 4：给标签上色并保存（业务逻辑）
场景：LoveDA 的掩码（Mask）是单通道灰度的（里面只有 0, 1, 2...6）。你想把它画成彩色的（比如 0 是黑色，1 是红色），存成一张图。
任务：编写函数 colorize_and_save(mask_path, save_path)：用 cv2.imread() 以灰度模式（IMREAD_GRAYSCALE）读取标签。
使用 NumPy 的布尔索引或者 np.where，把里面的像素值映射成 RGB 颜色（比如：标签为 1 的地方变成 [255, 0, 0]，标签为 0 的地方变黑）。
难点：映射后的数据，形状会从 (1024, 1024) 变成 (1024, 1024, 3)，并且数据类型需要是 uint8。
把转换好的 RGB 数组，用 PIL 的 Image.fromarray() 存成图片。
"""
save_color_path = PROJECT_ROOT / "reports" / "mask.png"
save_color_path.parent.mkdir(parents=True, exist_ok=True)


def colorize_and_save(mask_path, save_color_path):
    # 读取灰度图
    cv_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if cv_mask is None:
        raise ValueError(f"❌错误：图片对象不存在{mask_path}")
    print(cv_mask.shape)

    # 新建画布
    rgb = np.zeros((*cv_mask.shape, 3), dtype=np.uint8)

    rgb[cv_mask == 0] = [0, 0, 0]  # 黑
    rgb[cv_mask == 1] = [255, 0, 0]  # 红
    rgb[cv_mask == 2] = [0, 255, 0]  # 绿
    rgb[cv_mask == 3] = [0, 0, 255]  # 蓝
    rgb[cv_mask == 4] = [125, 125, 0]
    rgb[cv_mask == 5] = [0, 125, 125]
    rgb[cv_mask == 6] = [125, 0, 125]

    # 转为pil
    pil_mask = Image.fromarray(rgb, mode="RGB")
    pil_mask.save(save_color_path)

    # 读图
    pil2np = cv2.imread(save_color_path)
    pil2np = cv2.cvtColor(pil2np, cv2.COLOR_BGR2RGB)

    plt.imshow(pil2np)
    plt.axis("off")
    plt.show()


colorize_and_save(mask_path, save_color_path)
