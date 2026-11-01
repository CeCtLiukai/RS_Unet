from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import yaml

file_dir = Path(__file__).resolve().parent
PROJECT_ROOT = file_dir.parents[1]
config_path = PROJECT_ROOT/'configs'/'loveda_unet.yaml'

with open(config_path, mode='r', encoding='utf-8')as f:
    configs = yaml.safe_load(f)

DATA_ROOT = Path(configs['data']['root'])
img_dir = DATA_ROOT/'Train'/'Rural'/'images_png'
mask_dir = DATA_ROOT/'Train'/'Rural'/'masks_png'

img_path = [p for p in img_dir.glob('*.PNG')]
mask_path = [p for p in mask_dir.glob('*.PNG')]

'绘制单个图'
'1.创建画布2.加载图片3.坐标轴关闭4.设置title5.展示图片'
img = cv2.imread(img_path[0], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(img)
plt.axis('off')
plt.title('Urban_image')
plt.show()

'绘制两排图'
'1.创建多个画布'
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()
for i in range(5):
    img = img_path[i]
    img = cv2.imread(str(img), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = mask_path[i]
    mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    axes[i].imshow(img)
    axes[i].imshow(mask, alpha=0.2, cmap='tab10', vmin=0, vmax=6)
    axes[i].set_title(f'img_{i}', fontsize=10, fontweight='bold')
    axes[i].axis('off')

    axes[i+5].imshow(mask, cmap='tab10', vmin=0, vmax=6)
    axes[i+5].set_title(f'mask{i}', fontsize=10, fontweight='bold')
    axes[i+5].axis('off')

plt.tight_layout()
plt.show()

'绘制loss曲线'
save_path = PROJECT_ROOT/'reports'/'loss_curve.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
train_loss = [0.85, 0.62, 0.41, 0.30, 0.24]
val_loss = [0.88, 0.65, 0.45, 0.33, 0.27]
epochs = range(0, len(train_loss))

fig, axes = plt.subplots(figsize=(7, 4))
#设置总标签
axes.set_title('Training Loss Curve')
#设置横纵坐标轴
axes.set_xlabel('Epochs')
axes.set_ylabel('Loss')
#设置曲线
axes.plot(epochs, train_loss, label='train_loss', color='#1f77b4', linewidth=1.5, marker='o', markersize=3)
axes.plot(epochs, val_loss, label='val_loss',color='#d62728',linewidth=1.5,linestyle='--',marker='s',markersize=3)
#绘制图例
axes.legend()
#打开网格
axes.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()