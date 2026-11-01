import numpy as np
import torch
import yaml
from PIL import Image

from rsunet.datasets.loveda_dataset import build_dataset, get_dataloaders
from rsunet.datasets.transform import PairedTransform

FAILED = []

def check(name, fn):
    try:
        fn()
        print(f"  [OK]   {name}")
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        FAILED.append(name)

def main():
    cfg = yaml.safe_load(open("configs/loveda_unet.yaml", encoding="utf-8"))

    # --- 1. 数据集构建 -------------------------------------------------
    train_ds = build_dataset(cfg, training=True)
    val_ds = build_dataset(cfg, training=False)
    print(f"train={train_ds}\nval={val_ds}")

    check("训练集非空", lambda: (_ for _ in ()).throw(AssertionError())
          if len(train_ds) == 0 else None)
    check("验证集非空", lambda: (_ for _ in ()).throw(AssertionError())
          if len(val_ds) == 0 else None)

    check("验证集只含 Urban", lambda: (
        None if list(val_ds.domain_counts) == ["Urban"]
        else (_ for _ in ()).throw(AssertionError(val_ds.domain_counts))))

    # --- 2. 单样本形状 / dtype -----------------------------------------
    img, mask, meta = train_ds[0]
    s = cfg["augment"]["crop_size"]
    check("img 形状", lambda: None if tuple(img.shape) == (3, s, s)
          else (_ for _ in ()).throw(AssertionError(img.shape)))
    check("img dtype", lambda: None if img.dtype == torch.float32
          else (_ for _ in ()).throw(AssertionError(img.dtype)))

    check("mask 形状", lambda: None if tuple(mask.shape) == (s, s)
          else (_ for _ in ()).throw(AssertionError(mask.shape)))
    check("mask dtype", lambda: None if mask.dtype == torch.int64
          else (_ for _ in ()).throw(AssertionError(mask.dtype)))

    check("meta 有 domain", lambda: None if meta.get("domain") in ("Urban", "Rural")
          else (_ for _ in ()).throw(AssertionError(meta)))

    # --- 3. 掩码值域：只能出现真实类别或 ignore_index -------------------
    n_cls = cfg["data"]["num_classes"]
    ign = cfg["data"]["ignore_index"]

    def mask_range():
        vals = torch.unique(mask).tolist()
        bad = [v for v in vals if not (0 <= v < n_cls or v == ign)]
        assert not bad, f"越界值 {bad}，全部取值 {vals}"

    check("mask 值域合法", mask_range)

    # --- 4. 图像归一化：反归一化后应落在 [0,1] -------------------------
    def denorm_ok():
        mean = torch.tensor(cfg["augment"]["normalize"]["mean"]).view(3, 1, 1)
        std = torch.tensor(cfg["augment"]["normalize"]["std"]).view(3, 1, 1)
        x = img * std + mean
        assert float(x.min()) > -0.01 and float(x.max()) < 1.01, \
            f"反归一化范围 [{x.min():.3f}, {x.max():.3f}]"

    check("归一化可逆", denorm_ok)

    # --- 5. 训练集与验证集不重叠 --------------------------------------
    def no_overlap():
        tr = {p.stem for p, _, _ in train_ds.samples}
        va = {p.stem for p, _, _ in val_ds.samples}
        assert not (tr & va), f"{len(tr & va)} 个重复样本"

    check("train/val 无重叠", no_overlap)

    # --- 6. 成对变换：图和掩码必须同步 ---------------------------------
    def flip_sync():
        t = PairedTransform(crop_size=1024, scale_range=None,
                            hflip=1.0, vflip=0.0, training=True)
        # 左黑右白，掩码左 1 右 2
        arr = np.zeros((1024, 1024, 3), np.uint8)
        arr[:, 512:] = 255
        img_ = Image.fromarray(arr)
        mask_ = np.zeros((1024, 1024), np.uint8)
        mask_[:, 512:] = 2
        mask_[:, :512] = 1

        x, y = t(img_, mask_)
        # 水平翻转后：左侧应为原来的右侧
        assert int(y[0, 0]) == 2, f"掩码没翻转，左上角={int(y[0,0])}"
        assert float(x[0, 0, 0]) > 0, "图像没翻转（左侧仍是黑色）"

    check("翻转时图与掩码同步", flip_sync)

    def nearest_mask():
        t = PairedTransform(crop_size=300, scale_range=(0.5, 0.5), training=True)
        mask_ = np.zeros((1024, 1024), np.uint8)
        mask_[512:] = 6
        x, y = t(Image.new("RGB", (1024, 1024)), mask_)
        vals = set(torch.unique(y).tolist())
        assert vals <= {0, 6}, f"掩码被插值污染，出现 {sorted(vals)}"

    check("掩码 resize 不插值", nearest_mask)

    # --- 7. DataLoader 能组 batch -------------------------------------
    def batch_ok():
        cfg["data"]["num_workers"] = 0        # 冒烟测试不开子进程
        trl, _ = get_dataloaders(cfg, train_ds, val_ds)
        x, y, m = next(iter(trl))
        assert tuple(x.shape) == (cfg["data"]["batch_size"], 3, s, s), x.shape
        assert tuple(y.shape) == (cfg["data"]["batch_size"], s, s), y.shape
        assert len(m["domain"]) == cfg["data"]["batch_size"], m

    check("batch 组装", batch_ok)

    print()
    if FAILED:
        print(f"{len(FAILED)} 项失败: {FAILED}")
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
