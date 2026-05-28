#!/usr/bin/env python3
"""精简 experiments/results 中的重复实验产物。

默认只打印将删除的内容；加 `--apply` 才会真正删除。
清理策略：
1. 保留训练权重、args.yaml、results.csv、metrics_summary、required_metrics。
2. 删除 results 下 Ultralytics 自动生成的重复图片目录。
3. 删除 experiments/e3/e4 中已经汇总到 results 的阶段性验证图片目录。
4. 删除 experiments 中训练过程自动生成的图片。
5. 删除 e3/e4 的推理缓存 pred_cache.pkl。
6. 保留 results 中的最终关键图片。
7. 删除空的 results/test 占位目录。
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/disk2/lhr/VSD")


@dataclass(frozen=True)
class DeleteTarget:
    path: Path
    reason: str


def _existing(paths: list[DeleteTarget]) -> list[DeleteTarget]:
    return [item for item in paths if item.path.exists()]


def collect_targets(root: Path) -> list[DeleteTarget]:
    targets: list[DeleteTarget] = []

    # results 中的 ultralytics_val 是验证脚本的中间图和原生统计图；
    # 核心指标已经落在同级 metrics_summary/required_metrics 文件中。
    for path in sorted((root / "results" / "val").glob("*/ultralytics_val")):
        targets.append(DeleteTarget(path, "results 下的 Ultralytics 重复验证图目录"))

    # E5/E6 验证临时 YAML 已经可以由脚本再生成，不需要作为最终结果保存。
    for path in sorted((root / "results" / "val").glob("*/_generated_eval_data")):
        targets.append(DeleteTarget(path, "results 下的临时验证 YAML 目录"))

    # test 目录当前只有占位小文件，未形成有效测试结果。
    test_dir = root / "results" / "test"
    if test_dir.exists():
        targets.append(DeleteTarget(test_dir, "未使用的 results/test 占位目录"))

    # experiments/e3/e4 中这些目录是阶段复跑的验证产物；
    # canonical 指标已经保存在 results/e3/e4 对应目录。
    fusion_patterns = [
        root / "experiments" / "e3_late_fusion_nms" / "e3_stage1_rerun",
        root / "experiments" / "e4_late_fusion_wbf" / "e4_wbf_grid_search_20260318_174347",
    ]
    for base in fusion_patterns:
        for name in ("rgb_only", "ir_only", "late_fusion_nms", "late_fusion_wbf_best", "figures"):
            path = base / name
            if path.exists():
                targets.append(DeleteTarget(path, "experiments 下已汇总到 results 的阶段验证产物"))
        cache_path = base / "pred_cache.pkl"
        if cache_path.exists():
            targets.append(DeleteTarget(cache_path, "experiments 下可重新生成的推理缓存"))

    # e4 早期 equal_weight 验证目录也已有 results/S1_baselines/e4_late_fusion_wbf_val 作为统一入口。
    equal_weight_val = (
        root
        / "experiments"
        / "e4_late_fusion_wbf"
        / "e4_yolo11n_rgb_ir_wbf_equal_weight"
        / "val"
    )
    if equal_weight_val.exists():
        targets.append(DeleteTarget(equal_weight_val, "experiments 下重复的 equal-weight 验证目录"))

    for path in sorted((root / "experiments").glob("*/*/*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            targets.append(DeleteTarget(path, "experiments 训练 run 中的自动生成图片"))

    return _existing(targets)


def format_size(path: Path) -> str:
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--apply", action="store_true", help="真正删除；默认只 dry-run")
    args = parser.parse_args()

    root = Path(args.root)
    targets = collect_targets(root)
    if not targets:
        print("没有发现需要清理的重复产物。")
        return

    total_bytes = 0
    print("将清理以下重复产物：" if args.apply else "Dry-run，将清理以下重复产物：")
    for item in targets:
        size_text = format_size(item.path)
        if item.path.is_file():
            total_bytes += item.path.stat().st_size
        else:
            total_bytes += sum(p.stat().st_size for p in item.path.rglob("*") if p.is_file())
        print(f"- {item.path}  [{size_text}]  {item.reason}")

    print(f"合计约 {total_bytes / 1024 / 1024:.2f} MB")
    if not args.apply:
        print("未执行删除。确认无误后加 --apply。")
        return

    for item in targets:
        if item.path.is_dir():
            shutil.rmtree(item.path)
        else:
            item.path.unlink()
    print("清理完成。")


if __name__ == "__main__":
    main()
