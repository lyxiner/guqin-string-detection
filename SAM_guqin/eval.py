"""
Run inference and evaluation for the Guqin string segmentation model.

用法：
  python eval.py --ckpt /path/to/guqin_best.pth --image /path/to/test.jpg
  python eval.py --ckpt ... --image_dir /path/to/images --out_dir /path/to/results
  python eval.py --ckpt ... --image_dir /path/to/images --mask_dir /path/to/masks_strings --eval
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import segmentation_models_pytorch as smp

INPUT_SIZE = 768
ENCODER    = "resnet34"
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    model = smp.Unet(
        encoder_name=ENCODER, encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] 加载模型: {ckpt_path}")
    print(f"[INFO] 训练时最优 IoU: {ckpt.get('iou', 'N/A'):.4f} "
          f"@ epoch {ckpt.get('epoch', 'N/A')}")
    return model

def preprocess(img_rgb):
    """Convert an RGB image to a normalized model input tensor."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0).to(device)


@torch.no_grad()
def predict_resize(model, img_rgb):
    """Resize the full image to model input size, then upsample the probability map."""
    H, W = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = preprocess(img_small)
    logits = model(x)
    prob   = torch.sigmoid(logits)
    prob   = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    return prob.squeeze().cpu().numpy()


@torch.no_grad()
def predict_sliding(model, img_rgb, tile=INPUT_SIZE, overlap=0.25):
    """Run tiled inference and blend overlapping areas with a Gaussian weight."""
    H, W = img_rgb.shape[:2]
    stride = int(tile * (1 - overlap))

    pad_h = max(0, tile - H)
    pad_w = max(0, tile - W)
    if pad_h > 0 or pad_w > 0:
        img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT_101)
        H, W = img_rgb.shape[:2]

    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    cy, cx = tile / 2, tile / 2
    sigma  = tile / 4
    weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    prob_sum   = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] != H - tile: ys.append(H - tile)
    if xs[-1] != W - tile: xs.append(W - tile)

    for y in ys:
        for x in xs:
            patch = img_rgb[y:y+tile, x:x+tile]
            prob  = torch.sigmoid(model(preprocess(patch))).squeeze().cpu().numpy()
            prob_sum  [y:y+tile, x:x+tile] += prob * weight
            weight_sum[y:y+tile, x:x+tile] += weight

    prob_map = prob_sum / (weight_sum + 1e-8)
    return prob_map[:H-pad_h if pad_h else H, :W-pad_w if pad_w else W]


def overlay_mask(img_rgb, prob_map, threshold=0.5, color=(255, 0, 0), alpha=0.6):
    """Overlay a binary prediction mask on the RGB image."""
    mask = (prob_map > threshold).astype(np.uint8)
    overlay = img_rgb.copy()
    overlay[mask > 0] = (np.array(color) * alpha +
                         overlay[mask > 0] * (1 - alpha)).astype(np.uint8)
    return overlay, mask


def save_multi_threshold(img_rgb, prob_map, save_path, thresholds=(0.3, 0.5, 0.7)):
    """Save a threshold comparison image."""
    panels = [img_rgb]
    for th in thresholds:
        ov, _ = overlay_mask(img_rgb, prob_map, threshold=th)
        cv2.putText(ov, f"thr={th}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        panels.append(ov)
    heat = (prob_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    panels.append(heat)

    h = panels[0].shape[0]
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    combo = np.concatenate(panels, axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou  = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    return iou, prec, rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      required=True, help="训练好的 .pth 路径")
    parser.add_argument("--image",     default=None,  help="单张图像路径")
    parser.add_argument("--image_dir", default=None,  help="批量图像目录")
    parser.add_argument("--mask_dir",  default=None,  help="真值 mask 目录（算指标用）")
    parser.add_argument("--out_dir",   default="./infer_results", help="结果保存目录")
    parser.add_argument("--mode",      default="sliding", choices=["resize", "sliding"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval",      action="store_true", help="算数值指标")
    args = parser.parse_args()

    model = load_model(args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)
    predict_fn = predict_sliding if args.mode == "sliding" else predict_resize

    if args.image:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        print(f"[INFO] 原图尺寸: {img.shape}")
        prob = predict_fn(model, img)
        save_path = os.path.join(args.out_dir, Path(args.image).stem + "_multi.png")
        save_multi_threshold(img, prob, save_path)

        _, mask = overlay_mask(img, prob, threshold=args.threshold)
        mask_path = os.path.join(args.out_dir, Path(args.image).stem + "_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"[INFO] 多阈值对比图: {save_path}")
        print(f"[INFO] 最终 mask:   {mask_path}")
        return

    if not args.image_dir:
        print("[ERROR] 需要 --image 或 --image_dir"); return

    img_paths = sorted([p for p in Path(args.image_dir).iterdir()
                        if p.suffix.lower() in IMG_EXTS])
    print(f"[INFO] 批量处理 {len(img_paths)} 张图像")

    ious, precs, recs = [], [], []
    for ip in img_paths:
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        prob = predict_fn(model, img)
        pred_mask = (prob > args.threshold).astype(np.uint8)

        cv2.imwrite(os.path.join(args.out_dir, ip.stem + "_pred_mask.png"),
                    (pred_mask * 255).astype(np.uint8))
        overlay, _ = overlay_mask(img, prob, threshold=args.threshold)
        cv2.imwrite(os.path.join(args.out_dir, ip.stem + "_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if args.eval and args.mask_dir:
            gt_path = Path(args.mask_dir) / (ip.stem + "_mask.png")
            if not gt_path.exists():
                print(f"  [WARN] 无真值: {gt_path}"); continue
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = (gt > 127).astype(np.uint8)
            iou, prec, rec = compute_metrics(pred_mask, gt)
            ious.append(iou); precs.append(prec); recs.append(rec)
            print(f"  {ip.name:<40} IoU={iou:.4f} P={prec:.3f} R={rec:.3f}")

    if args.eval and ious:
        print("\n========== 整体指标 ==========")
        print(f"mean IoU       : {np.mean(ious):.4f}")
        print(f"mean Precision : {np.mean(precs):.4f}")
        print(f"mean Recall    : {np.mean(recs):.4f}")
        print(f"样本数          : {len(ious)}")


if __name__ == "__main__":
    main()
