"""VisDrone YOLO 标签可视化抽检脚本。

脚本会随机抽取训练图像，绘制 YOLO 边框并保存调试图，
用于快速核对标注质量。
"""

from pathlib import Path
import random
import cv2

root = Path("/mnt/disk2/lhr/VSD/prepared/visdrone_yolo")
img_dir = root / "images" / "train"
lab_dir = root / "labels" / "train"
save_dir = root / "debug_vis"
save_dir.mkdir(exist_ok=True)

class_names = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
sample_paths = random.sample(img_paths, min(20, len(img_paths)))

for img_path in sample_paths:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = lab_dir / f"{img_path.stem}.txt"

    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = parts
                cls = int(cls)
                xc, yc, bw, bh = map(float, [xc, yc, bw, bh])

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    class_names[cls],
                    (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imwrite(str(save_dir / img_path.name), img)

print(f"saved debug images to {save_dir}")