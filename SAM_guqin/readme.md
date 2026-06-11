 使用labelme进行标注，需要先conda激活labelme虚拟环境，然后在终端输入labelme即可打开该软件

GuQin/ 原图 + masks_strings/ 弦掩码 -> 训练 UNet -> checkpoints/guqin_best.pth -> 用 eval.py 做推理/评估。

  关键入口在 train.py:36。它明确把原图目录设成 GuQin/，把掩码目录设成 masks_strings/，并按同名规则配对：xxx.jpg 对 xxx_mask.png，见 train.py:60。训练时只看“有 mask 的样本”，所以 GuQin/ 里图片比 mask 多是允
  许的。

  masks_strings 怎么来：

  1. 这个目录现在仓库里已经有了：/home/wz/Git/SAM_guqin/masks_strings
     我检查到当前里面有 172 个 mask。
  2. 最直接的生成方式是用 json2mask.py:45。
     这个脚本会读取 LabelMe 的 line 标注 JSON，把每根弦画成二值 mask。你现在 GuQin/*.json 里就是这种格式，像 GuQin/IMG_20260423_175329.json:1。
  3. 按当前仓库结构，建议你这样生成或重建 masks_strings：

  cd /home/wz/Git/SAM_guqin

  python json2mask.py \
    --json_dir /home/wz/Git/SAM_guqin/GuQin \
    --img_dir /home/wz/Git/SAM_guqin/GuQin \
    --out_dir /home/wz/Git/SAM_guqin/masks_strings \
    --line_width 7

  这会输出 *_mask.png 到 masks_strings/。--line_width 7 是弦宽，太粗太细都可以自己调。
  
cd /home/wz/Git/SAM_guqin
  python train.py

  训练完成后模型会保存在 checkpoints/guqin_best.pth，见 train.py:294。

  推理/评估用 eval.py:177：

  python eval.py --ckpt  /home/wz/Git/SAM_guqin/checkpoints/guqin_best.pth   --image /home/wz/Git/SAM_guqin/eval01.jpg     --mode resize --threshold 0.5
  
输出拟合结果：
 python /home/wz/Git/SAM_guqin/mask_to_strings.py