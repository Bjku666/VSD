"""将 VisDrone DET 标注转换为 YOLO txt，并同步复制图像。

该工具保留原始 train/val 划分，输出 Ultralytics 训练所需的
归一化 xywh 标签格式。
"""

from pathlib import Path
from PIL import Image
import shutil

SRC_ROOTS = {
    "train": Path("/mnt/disk2/lhr/VSD/data/VisDrone/raw/VisDrone2019-DET-train"),
    "val": Path("/mnt/disk2/lhr/VSD/data/VisDrone/raw/VisDrone2019-DET-val"),
}

DST_ROOT = Path("/mnt/disk2/lhr/VSD/prepared/visdrone_yolo")
VALID_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def convert_one_split(split_name, src_root):
    img_dir = src_root / "images"
    ann_dir = src_root / "annotations"

    out_img_dir = DST_ROOT / "images" / split_name
    out_lab_dir = DST_ROOT / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"[{split_name}] found {len(img_paths)} images")

    for img_path in img_paths:
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"

        target_img = out_img_dir / img_path.name
        if not target_img.exists():
            shutil.copy2(img_path, target_img)

        with Image.open(img_path) as im:
            w_img, h_img = im.size

        yolo_lines = []

        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) < 6:
                        continue

                    x, y, w, h = map(float, parts[:4])
                    score = int(float(parts[4]))
                    category = int(float(parts[5]))

                    if category not in VALID_CLASSES:
                        continue
                    if score == 0:
                        continue
                    if w <= 0 or h <= 0:
                        continue

                    cls = category - 1

                    x_center = (x + w / 2) / w_img
                    y_center = (y + h / 2) / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img

                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    w_norm = min(max(w_norm, 0.0), 1.0)
                    h_norm = min(max(h_norm, 0.0), 1.0)

                    yolo_lines.append(
                        f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )

        with open(out_lab_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

def main():
    for split_name, src_root in SRC_ROOTS.items():
        convert_one_split(split_name, src_root)
    print("Done.")

if __name__ == "__main__":
    main()