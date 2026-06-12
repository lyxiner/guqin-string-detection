"""Train a UNet model for Guqin string segmentation."""

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

cv2.setNumThreads(0)


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

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")
print(f"[INFO] 图像目录: {IMAGE_DIR}")
print(f"[INFO] 掩码目录: {MASK_DIR}")
print(f"[INFO] 保存目录: {SAVE_DIR}")

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
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].float().unsqueeze(0)
        return img, mask


train_tf = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        rotate=(-5, 5),
        translate_percent=(-0.05, 0.05),
        scale=(0.9, 1.1),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.5,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.4),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class StringLoss(nn.Module):
    """Focal + Tversky loss for sparse string masks."""
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.fa, self.fg = focal_alpha, focal_gamma
        self.ta, self.tb = tversky_alpha, tversky_beta

    def focal(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return (self.fa * (1 - pt) ** self.fg * bce).mean()

    def tversky(self, pred, target, smooth=1.0):
        p = torch.sigmoid(pred)
        tp = (p * target).sum()
        fp = (p * (1 - target)).sum()
        fn = ((1 - p) * target).sum()
        return 1 - (tp + smooth) / (tp + self.ta * fp + self.tb * fn + smooth)

    def forward(self, pred, target):
        return 0.5 * self.focal(pred, target) + 0.5 * self.tversky(pred, target)

def calc_metrics(pred_logits, target, threshold=0.5):
    pred  = (torch.sigmoid(pred_logits) > threshold)
    target = target.bool()
    inter = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou   = (inter / (union + 1e-6)).item()

    tp = (pred & target).float().sum()
    fp = (pred & ~target).float().sum()
    fn = (~pred & target).float().sum()
    prec = (tp / (tp + fp + 1e-6)).item()
    rec  = (tp / (tp + fn + 1e-6)).item()
    return iou, prec, rec

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

full_dataset = GuqinDataset(IMAGE_DIR, MASK_DIR)
n_val   = max(1, int(len(full_dataset) * VAL_RATIO))
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED))

class TransformWrapper(Dataset):
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=max(1, NUM_EPOCHS * len(train_loader) // ACCUM_STEPS),
    pct_start=0.1, anneal_strategy='cos', final_div_factor=100)
criterion = StringLoss()

scaler = torch.cuda.amp.GradScaler()

best_iou = 0.0
patience, no_improve = PATIENCE, 0
history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_prec": [], "val_rec": []}

print("\n开始训练...\n")
for epoch in range(1, NUM_EPOCHS + 1):
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

    if val_iou > best_iou:
        best_iou   = val_iou
        no_improve = 0
        ckpt = os.path.join(SAVE_DIR, "guqin_best.pth")
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iou":   best_iou}, ckpt)
        print(f"  ✓ 保存最优模型 IoU={best_iou:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\n[早停] {patience} 轮未提升，停止训练")
            break

    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

print(f"\n训练结束，最优 IoU: {best_iou:.4f}")
print(f"模型保存在: {SAVE_DIR}/guqin_best.pth")
