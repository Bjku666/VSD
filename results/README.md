# VSD Evaluation Outputs

评测规范（重构后）：

- 训练产物放在 `experiments/`
- 评测产物统一放在 `results/`
- 不再使用 stage1 benchmark 聚合脚本

当前标准目录：

- `results/val/e1_rgb_only_val`
- `results/val/e2_ir_only_val`
- `results/val/e3_late_fusion_nms_val`
- `results/val/e4_late_fusion_wbf_val`
- `results/val/e5_feature_fusion_single_val`

- `results/test/e1_rgb_only_test`
- `results/test/e2_ir_only_test`
- `results/test/e3_late_fusion_nms_test`
- `results/test/e4_late_fusion_wbf_test`
- `results/test/e5_feature_fusion_single_test`

脚本入口：

- E1: `scripts/e1_val_rgb_only.sh`
- E2: `scripts/e2_val_ir_only.sh`
- E3: `scripts/e3_val_late_fusion_nms.sh`
- E4: `scripts/e4_val_late_fusion_wbf.sh`
- E5: `scripts/e5_val_feature_fusion_single.py`
