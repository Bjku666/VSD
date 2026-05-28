#!/usr/bin/env python3
"""根据 results 中的指标文件重建关键结果图。

只生成最终结果图，不生成训练过程图：
- required_metrics_overview.png
- per_class_ap50_95.png
- confusion_matrix.png
- confusion_matrix_normalized.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/disk2/lhr/VSD")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"无效 JSON 文件：{path}")
    return data


def _save_required_overview(metrics: dict[str, Any], out_path: Path) -> None:
    keys = [
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "AP_small",
        "Recall_small",
        "AP_dark",
        "Recall_dark",
        "AP_dark-small",
    ]
    labels = [k for k in keys if k in metrics]
    values = [float(metrics[k]) for k in labels]
    if not values:
        return

    plt.figure(figsize=(10, 4.8))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    bars = plt.bar(labels, values, color=colors[: len(values)])
    plt.ylim(0, max(1.0, max(values) * 1.15))
    plt.ylabel("score")
    plt.xticks(rotation=30, ha="right")
    plt.title("Required metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_per_class_ap(metrics: dict[str, Any], out_path: Path) -> None:
    per_class = metrics.get("per_class_AP")
    if not isinstance(per_class, dict) or not per_class:
        return
    classes = list(per_class.keys())
    ap50 = [float(per_class[c].get("AP50", 0.0)) for c in classes if isinstance(per_class[c], dict)]
    ap95 = [float(per_class[c].get("AP50-95", 0.0)) for c in classes if isinstance(per_class[c], dict)]
    if not ap50 and not ap95:
        return

    x = np.arange(len(classes))
    width = 0.36
    plt.figure(figsize=(8, 4.6))
    plt.bar(x - width / 2, ap50, width, label="AP50", color="#4C78A8")
    plt.bar(x + width / 2, ap95, width, label="AP50-95", color="#F58518")
    plt.ylim(0, max(1.0, max(ap50 + ap95) * 1.15))
    plt.ylabel("AP")
    plt.xticks(x, classes, rotation=25, ha="right")
    plt.title("Per-class AP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_confusion_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"无效混淆矩阵 CSV：{path}")
    labels = rows[0][1:]
    values = []
    for row in rows[1:]:
        values.append([float(x) for x in row[1:]])
    return labels, np.asarray(values, dtype=float)


def _save_confusion(labels: list[str], matrix: np.ndarray, out_path: Path, normalize: bool) -> None:
    values = matrix.copy()
    title = "Confusion Matrix"
    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum != 0)
        title = "Confusion Matrix Normalized"

    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(values, cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    threshold = values.max() * 0.55 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            color = "white" if values[i, j] > threshold else "black"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def render_run(run_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    metrics_path = run_dir / "required_metrics.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        overview = run_dir / "required_metrics_overview.png"
        per_class = run_dir / "per_class_ap50_95.png"
        _save_required_overview(metrics, overview)
        _save_per_class_ap(metrics, per_class)
        if overview.exists():
            outputs.append(overview)
        if per_class.exists():
            outputs.append(per_class)

    confusion_csv = run_dir / "confusion_matrix.csv"
    if confusion_csv.exists():
        labels, matrix = _load_confusion_csv(confusion_csv)
        cm = run_dir / "confusion_matrix.png"
        cmn = run_dir / "confusion_matrix_normalized.png"
        _save_confusion(labels, matrix, cm, normalize=False)
        _save_confusion(labels, matrix, cmn, normalize=True)
        outputs.extend([cm, cmn])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-val", default=str(ROOT / "results" / "val"))
    args = parser.parse_args()

    results_val = Path(args.results_val)
    generated: list[Path] = []
    for run_dir in sorted(p for p in results_val.iterdir() if p.is_dir()):
        generated.extend(render_run(run_dir))

    if not generated:
        print("没有生成任何关键图。")
        return
    print("已生成关键图：")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
