from pathlib import Path
import torch
import cv2
import yaml
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parents[1]
"""
题目 1：基础拼接与后缀提取
场景：你正在准备 LoveDA 数据集，配置文件里写了一个根目录 root_dir = "D:/Postgraduate_Study/Data/loveda"。
任务：请使用 pathlib，写出代码来获取 Train/Urban/images_png 文件夹下所有 .png 图像的纯文件名（不包含后缀，如 1234），并将它们存入一个列表中。
"""

# step1:my root directory is stored in the YAML file,I have to extract it from the YAML file.
config_path = PROJECT_ROOT / "configs" / "loveda_unet.yaml"
with open(config_path, mode="r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# step2:
# root_dir = D:/Postgraduate_Study/Data/loveda'
root_dir = Path(config["data"]["root"])
train_urban_images_dir = root_dir / "Train" / "Urban" / "images_png"

# step3:
img_paths = []
for p in train_urban_images_dir.glob("*.PNG"):
    img_paths.append(p)
img_name = [p.stem for p in img_paths]


"""
# 题目 2：路径拆解与安全建目录（结合你的踩坑）
# 场景：你想把可视化抽检的结果保存到 PROJECT_ROOT / "reports" / "preview_urban.png"。
# 任务：请写出代码，获取 preview_urban.png 所在的父目录路径。了防止你的脚本因为“reports 文件夹不存在”而崩溃，请写出创建该目录的代码path.mkdir（要求：如果存在不报错，且支持多级目录创建）
"""
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
axes.plot([1, 2, 3], [4, 5, 6])
axes.set_title("My Plot")
# 目标是保存在这个的上一级，如果没有则递归创建目录，存在则不报错
parents_path = (PROJECT_ROOT / "reports" / "preview_urban.png").parent
# 创建目录
parents_path.mkdir(parents=True, exist_ok=True)
# fig保存需要给出要保存的文件名称
fig.savefig(
    PROJECT_ROOT / "reports" / "preview_urban.png", dpi=150, bbox_inches="tight"
)

"""
题目 3：配对检查中的“生成器陷阱”
场景：你想检查原图（images）和掩码（masks）是否一一对应。
任务：请看下面这段代码，它试图找出只在原图存在、掩码缺失的图片：
"""
# from pathlib import Path
# 这里拿到的是Path对象
# img_dir = Path("data/loveda/Train/Urban/images_png")
# mask_dir = Path("data/loveda/Train/Urban/masks_png")
#
# 试图获取所有图片
# 这里不能glob因为这个返回的是一个生成器，必须转化为list或for遍历，取出来这些数据
# imgs = img_dir.glob("*.png")
# masks = mask_dir.glob("*.png")
#
# # 试图找出只存在于 images 中，而 masks 中没有的文件
# only_in_imgs = set(imgs.stems) - set(masks.stems) # 这里生成器没有stems属性
"answer:"
# glob() / rglob() / iterdir() 返回的都是生成器，不是列表；
# .stem 是单个 Path 对象的属性（单数），要配合推导式用，别写成 .stems。


"""
题目 4：大小写敏感与文件名清洗
场景：你的数据集是从不同渠道汇总的，后缀名既有 .png 也有 .PNG。你写了一个 iterdir() 循环来筛选所有图片。
任务：请写一段代码，遍历一个文件夹中的所有文件，只筛选出后缀名为 .png（不区分大小写）的文件，并将它们保存到一个列表中。
"""
# 这个是依次遍历然后把符合条件的放入内存,全部条目
img_only_png = [
    p for p in train_urban_images_dir.iterdir() if p.suffix.lower() == ".png"
]

"""
题目 5：PyTorch 数据加载前的路径转换
场景：在写 LoveDADataset 的 __getitem__ 函数时，你需要读取一张图片，并将其转化为 PyTorch 张量。
任务：假设你有一个 Path 对象 img_path，你知道 OpenCV 的 cv2.imread() 不接受 Path 对象，
请写出将它转换为 cv2.imread() 能够接受的格式的代码。读取成功后，你需要将 NumPy 数组转为 PyTorch 的 float32 张量，请写出这行转换代码。
"""
# 一个图片的路径
img_path = img_paths[0]
# cv2.imread读出来的是一个numpy张量

img = cv2.imread(str(img_path))
# cv2读出来的是bgr，通道转换
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 使用numpy转tensor不复制
tensor1 = torch.from_numpy(img).float()
tensor1 = tensor1 / 255.0
# 归一化
tensor1 = tensor1.permute(2, 0, 1)
print(tensor1, type(tensor1))
