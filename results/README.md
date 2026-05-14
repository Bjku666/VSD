# VSD 评测结果目录

评测规范（重构后）：

- `experiments/` 只保留训练过程必要产物：权重、`args.yaml`、`results.csv`、少量阶段报告。
- `results/` 保留最终评测结果：`metrics_summary`、`required_metrics`、混淆矩阵 CSV、关键 PNG 图。
- 不保留重复的 Ultralytics 中间验证目录，例如 `ultralytics_val/`。
- 不保留训练过程图片，例如 `train_batch*.jpg`、`val_batch*.jpg`、训练曲线 PNG。

当前标准目录：

- `results/val/e1_rgb_only_val`
- `results/val/e2_ir_only_val`
- `results/val/e3_late_fusion_nms_val`
- `results/val/e4_late_fusion_wbf_val`
- `results/val/e5_feature_fusion_single_val`
- `results/val/e6_feature_fusion_multiscale_val`
- `results/dark_small_next`

脚本入口：

- E1: `scripts/e1_val_rgb_only.sh`
- E2: `scripts/e2_val_ir_only.sh`
- E3: `scripts/e3_val_late_fusion_nms.sh`
- E4: `scripts/e4_val_late_fusion_wbf.sh`
- E5: `scripts/e5_val_feature_fusion_single.py`
- E6: `scripts/e6_val_feature_fusion_multiscale.py`

辅助脚本：

- 关键图重建：`scripts/render_key_result_figures.py`
- 重复产物清理：`scripts/cleanup_result_artifacts.py`
