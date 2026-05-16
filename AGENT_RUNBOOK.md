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
