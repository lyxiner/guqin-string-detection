"""
LabelMe line 标注 → 二值掩码 PNG
---------------------------------
输入：172 个 JSON 文件（shape_type="line"）
输出：同名的 _mask.png（白色=弦，黑色=背景）

用法：
    python json2mask.py \
        --json_dir /home/wz/Git/SAM_guqin/annotations \
        --img_dir  /home/wz/Git/SAM_guqin \
        --out_dir  /home/wz/Git/SAM_guqin/masks \
        --line_width 7
"""

import json
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def json_to_mask(json_path, img_h, img_w, line_width=7):
    """把一个 LabelMe JSON 里所有 line 形状画成二值掩码"""
    with open(json_path, "r") as f:
        data = json.load(f)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["shape_type"] != "line":
            continue
        pts = shape["points"]
        x1, y1 = int(round(pts[0][0])), int(round(pts[0][1]))
        x2, y2 = int(round(pts[1][0])), int(round(pts[1][1]))
        # 画线，thickness 对应弦宽（像素）
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=line_width)

    # 轻微膨胀，弥补标注偏差
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir",    required=True, help="JSON 文件目录")
    parser.add_argument("--img_dir",     required=True, help="原始图像目录")
    parser.add_argument("--out_dir",     required=True, help="掩码输出目录")
    parser.add_argument("--line_width",  type=int, default=7,
                        help="弦线宽度（像素），默认 7")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {args.json_dir} 下没有找到 JSON 文件")

    print(f"共找到 {len(json_files)} 个 JSON 文件，开始转换...")

    ok, skip = 0, 0
    for jf in json_files:
        # 读取图像尺寸（优先从 JSON 里取，避免再读图）
        with open(jf) as f:
            meta = json.load(f)

        img_h = meta.get("imageHeight")
        img_w = meta.get("imageWidth")

        # JSON 里没有尺寸则从图像文件读
        if not img_h or not img_w:
            img_name = meta.get("imagePath", jf.stem + ".jpg")
            img_path = Path(args.img_dir) / img_name
            if not img_path.exists():
                print(f"  [SKIP] 找不到图像: {img_path}")
                skip += 1
                continue
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

        mask = json_to_mask(jf, img_h, img_w, args.line_width)

        out_name = jf.stem + "_mask.png"
        out_path = Path(args.out_dir) / out_name
        cv2.imwrite(str(out_path), mask)
        ok += 1
        print(f"  [{ok}/{len(json_files)}] {jf.name} → {out_name}")

    print(f"\n完成！成功 {ok} 张，跳过 {skip} 张")
    print(f"掩码保存在: {args.out_dir}")


if __name__ == "__main__":
    main()