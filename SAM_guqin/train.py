"""
古琴弦 UNet 训练脚本（针对细线分割优化版）
==========================================

相比原版的核心改动：
1. 增强策略：移除 RandomRotate90 / VerticalFlip / MotionBlur，保留符合琴弦先验的变换
2. 分辨率：提高到 768，细线能保留至少 1 像素宽度
3. Loss：Focal + Tversky（比 Dice 对稀疏目标更稳），并且 alpha 与 IoU 阈值对齐
4. Mask 缩放：强制 nearest 插值，避免二值化后出现断线
5. Encoder：可选 efficientnet-b0（比 resnet34 参数更少但小样本上更稳）
6. 记录 loss 曲线，支持断点续训
7. 修正路径冲突

用法：
    python3 train.py --image_dir ./GuQin --mask_dir ./masks_strings --save_dir ./checkpoints
"""

import argparse
import os
import random
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)  # 避免和 DataLoader 多进程冲突


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the Guqin string segmentation model.")
    parser.add_argument("--data_root", type=Path, default=script_dir,
                        help="Dataset root. Defaults to the SAM_guqin directory.")
    parser.add_argument("--image_dir", type=Path, default=None,
                        help="Image directory. Defaults to <data_root>/GuQin.")
    parser.add_argument("--mask_dir", type=Path, default=None,
                        help="Mask directory. Defaults to <data_root>/masks_strings.")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Checkpoint directory. Defaults to <data_root>/checkpoints.")
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Optional checkpoint used to initialize/fine-tune the model.")
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


args = parse_args()

# ════════════════════════════════════════════════════════
#  配置区
# ════════════════════════════════════════════════════════
DATA_ROOT   = args.data_root.expanduser().resolve()
IMAGE_DIR   = (args.image_dir or (DATA_ROOT / "GuQin")).expanduser().resolve()
MASK_DIR    = (args.mask_dir or (DATA_ROOT / "masks_strings")).expanduser().resolve()
SAVE_DIR    = (args.save_dir or (DATA_ROOT / "checkpoints")).expanduser().resolve()

INPUT_SIZE  = args.input_size
BATCH_SIZE  = args.batch_size
ACCUM_STEPS = max(1, args.accum_steps)
NUM_EPOCHS  = args.epochs
LR          = args.lr
VAL_RATIO   = args.val_ratio
SEED        = args.seed
ENCODER     = args.encoder
NUM_WORKERS = args.num_workers
VAL_WORKERS = args.val_workers
PATIENCE    = args.patience
# ════════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")
print(f"[INFO] 图像目录: {IMAGE_DIR}")
print(f"[INFO] 掩码目录: {MASK_DIR}")
print(f"[INFO] 保存目录: {SAVE_DIR}")


# ── Dataset ──────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class GuqinDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.transform = transform
        mask_files = sorted(Path(mask_dir).glob("*_mask.png"))
        self.pairs = []
        for mf in mask_files:
            stem = mf.stem.replace("_mask", "")
            img_file = None
            for ext in IMG_EXTS:
                candidate = Path(img_dir) / (stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                print(f"  [WARN] 找不到 {stem} 的图像，跳过")
                continue
            self.pairs.append((img_file, mf))

        if not self.pairs:
            raise FileNotFoundError(f"没有找到配对，请检查 {img_dir} 和 {mask_dir}")
        print(f"[INFO] 有效样本对: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)   # 【改】先用 uint8，交给 albumentations 用 nearest 插值

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].float().unsqueeze(0)  # 转 float
        return img, mask


# ── 数据增强（针对细线优化）──────────────────────────────
# 关键原则：
#   1. 不做 90 度旋转、垂直翻转（琴弦有强方向先验）
#   2. 不做 MotionBlur（会直接糊掉细线）
#   3. 几何变换用 nearest 插值，避免 mask 抗锯齿断线
#   4. 小角度旋转 OK（±5 度模拟相机轻微抖动）
#   5. Resize 不要用 RandomResizedCrop 的大范围缩放
train_tf = A.Compose([
    # 【改】用固定 Resize 替代 RandomResizedCrop，细线不能承受大幅缩放
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),                   # 水平翻转 OK（弦左右对称）
    # 【删】 A.VerticalFlip - 现实中不会上下颠倒拍摄
    # 【删】 A.RandomRotate90 - 琴弦绝不会变成垂直
    A.Affine(                                  # 【新】小幅旋转+平移，模拟实际相机抖动
        rotate=(-5, 5),
        translate_percent=(-0.05, 0.05),
        scale=(0.9, 1.1),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # 【关键】mask 必须用 nearest，否则二值化后断线
        p=0.5,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.4),
    A.GaussNoise(p=0.3),                       # 【改】去掉 var_limit 参数（新版 API 已改名）
    # 【删】 A.MotionBlur - 对 1-2 像素宽的弦是毁灭性的
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ── 损失函数 ──────────────────────────────────────────────
class StringLoss(nn.Module):
    """
    Focal + Tversky 组合：
      - Focal 处理像素级极端不平衡
      - Tversky 是 Dice 的广义形式，beta > alpha 时更重视 recall（宁多勿漏）
        对"找出所有琴弦"这种任务更合适
    """
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        # 【改】focal_alpha 从 0.85 降到 0.75，避免过度膨胀预测
        self.fa, self.fg = focal_alpha, focal_gamma
        self.ta, self.tb = tversky_alpha, tversky_beta

    def focal(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return (self.fa * (1 - pt) ** self.fg * bce).mean()

    def tversky(self, pred, target, smooth=1.0):
        """Tversky = TP / (TP + α·FP + β·FN)，α < β 惩罚漏检更重"""
        p = torch.sigmoid(pred)
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()
        return 1 - (tp + smooth) / (tp + self.ta * fp + self.tb * fn + smooth)

    def forward(self, pred, target):
        return 0.5 * self.focal(pred, target) + 0.5 * self.tversky(pred, target)


# ── 评估指标 ──────────────────────────────────────────────
def calc_metrics(pred_logits, target, threshold=0.5):
    """【改】threshold 从 0.35 提到 0.5，和标准二分类对齐，避免虚高"""
    pred  = (torch.sigmoid(pred_logits) > threshold)
    target = target.bool()
    inter = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou   = (inter / (union + 1e-6)).item()

    # 【新】额外返回 precision / recall，对细线任务更有诊断价值
    tp = (pred & target).float().sum()
    fp = (pred & ~target).float().sum()
    fn = (~pred & target).float().sum()
    prec = (tp / (tp + fp + 1e-6)).item()
    rec  = (tp / (tp + fn + 1e-6)).item()
    return iou, prec, rec


# ── 模型 ──────────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = ENCODER,
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = 1,
    activation      = None,
).to(device)

if args.init_checkpoint:
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    ckpt = torch.load(init_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] 从已有权重初始化: {init_checkpoint}")


# ── 数据集拆分 ────────────────────────────────────────────
full_dataset = GuqinDataset(IMAGE_DIR, MASK_DIR)
n_val   = max(1, int(len(full_dataset) * VAL_RATIO))
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED))

class TransformWrapper(Dataset):
    """直接持有 pairs 列表，不依赖 Subset 的内部结构"""
    def __init__(self, subset, transform):
        self.pairs = [subset.dataset.pairs[i] for i in subset.indices]
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        out  = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)

train_loader = DataLoader(TransformWrapper(train_ds, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(TransformWrapper(val_ds, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=VAL_WORKERS, pin_memory=True)

print(f"[INFO] 训练集: {n_train} 张  验证集: {n_val} 张")


# ── 优化器 & 调度器 ───────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# 【改】用 warmup + cosine，小数据集前期稳定性更好
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=max(1, NUM_EPOCHS * len(train_loader) // ACCUM_STEPS),
    pct_start=0.1, anneal_strategy='cos', final_div_factor=100)
criterion = StringLoss()

# 【新】AMP 混合精度，显存 + 速度双收益
scaler = torch.cuda.amp.GradScaler()

# ── 训练循环 ──────────────────────────────────────────────
best_iou = 0.0
patience, no_improve = PATIENCE, 0
history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_prec": [], "val_rec": []}

print("\n开始训练...\n")
for epoch in range(1, NUM_EPOCHS + 1):
    # ---- train ----
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for step, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss  = criterion(preds, masks) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_loss += loss.item() * ACCUM_STEPS
    train_loss /= len(train_loader)

    # ---- val ----
    model.eval()
    val_loss, val_iou, val_prec, val_rec = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
            iou, prec, rec = calc_metrics(preds, masks)
            val_iou += iou; val_prec += prec; val_rec += rec
    n = len(val_loader)
    val_loss /= n; val_iou /= n; val_prec /= n; val_rec /= n

    lr_now = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
          f"Loss {train_loss:.4f} → {val_loss:.4f} | "
          f"IoU {val_iou:.4f}  P {val_prec:.3f}  R {val_rec:.3f} | "
          f"lr {lr_now:.2e}")

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)
    history["val_prec"].append(val_prec)
    history["val_rec"].append(val_rec)

    # ---- 保存最优模型 ----
    if val_iou > best_iou:
        best_iou   = val_iou
        no_improve = 0
        ckpt = os.path.join(SAVE_DIR, "guqin_best.pth")
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),   # 【新】可断点续训
                    "scheduler": scheduler.state_dict(),
                    "iou":   best_iou}, ckpt)
        print(f"  ✓ 保存最优模型 IoU={best_iou:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\n[早停] {patience} 轮未提升，停止训练")
            break

    # 每轮都保存 history，防止训练中断丢失
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

print(f"\n训练结束，最优 IoU: {best_iou:.4f}")
print(f"模型保存在: {SAVE_DIR}/guqin_best.pth")
