# VSD Agent Runbook

## 执行原则

1. 每次只执行一个阶段。
2. 每个实验开始前检查依赖，依赖不满足则标记 blocked。
3. 每个实验完成后检查必须产物，产物不完整则标记 failed。
4. 每个实验完成后更新 [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md)。
5. 每个实验完成后更新 [results/val/dark_small_experiment_leaderboard.md](results/val/dark_small_experiment_leaderboard.md)。
6. 所有日志写入 [results/val/logs/](results/val/logs/)。
7. 所有关键结果必须汇总到结果总表。
8. 不允许使用 test set 调参。
9. 不允许引用旧 weights/trained/。
10. 不允许删除原始数据或预训练权重。

## 第一阶段执行顺序

1. E0
2. E1
3. E2
4. E3
5. E4

## 后续阶段执行顺序

### 第二阶段

1. E5
2. E6

### 第三阶段

1. E7_1
2. E7_2
3. E7_3
4. E7_4
5. E7_5
6. E7_6

### 第四阶段

1. E8_1
2. E8_2
3. E8_3
4. E8_4

### 第五阶段

1. E5/E6 补 FP/image、FPPI_dark、FPPI_low-contrast、AP_tiny、Recall_tiny、AP_low-contrast、object-level AP 和效率指标
2. E5/E6 补 seed=1,2，输出 mean ± std
3. E10_2：E6 768
4. 暂停 E10_1、E10_3、E10_4，除非 E10_2 明确优于 E6 640

### 第六阶段

1. E11-1：E6 + P2 detection head，P2 只做检测头
2. E12-1：E6 + residual gated fusion
3. 若 FP/image 高于 E4 WBF 明显，优先 E22 hard negative mining
4. 否则进入 E13 scale-aware / center-aware loss

## 当前路线约束

- E6 multi-scale fusion 是当前主线基线。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 不继续搜索 WBF 权重。
- 不继续 IR-only 960 + resampling。
- 不启动 RT-DETR / YOLOv10 / YOLO11s 强模型对照。
- 不运行 test set。

## 通过条件

### E0
- [ ] pair_audit.json 存在
- [ ] train_thresholds.json 存在
- [ ] subset_counts.csv 存在
- [ ] protocol_summary.md 存在
- [ ] RGB / IR / RGB-IR yaml 已生成
- [ ] train-only 阈值确认

### E1 / E2
- [ ] best.pt 存在
- [ ] last.pt 存在
- [ ] metrics_summary.md 存在
- [ ] required_metrics.json 存在
- [ ] required_metrics.csv 存在
- [ ] full / dark / small / dark-small / tiny / low-contrast 均完成评估

### E3 / E4
- [ ] 使用 E1 / E2 新权重
- [ ] 融合结果保存到新目录
- [ ] FPPI_dark 和 FPPI_low-contrast 已统计

## 失败处理

- 失败后先保留错误日志，不允许直接跳过。
- 失败实验必须标记 failed，并在状态表里写明原因。
- 只在当前阶段通过后再进入下一阶段。
