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

## 总阶段划分

| 大阶段 | 阶段名称 | 实验范围 | 当前状态 | 结论 |
| --- | --- | --- | --- | --- |
| S0 | 数据协议阶段 | E0 | done | 协议、子集和指标口径已固定 |
| S1 | 基础基线阶段 | E1-E4 | done | E2 是暗弱支撑基线，E4 是低误报 WBF 参考线 |
| S2 | 融合主线确认阶段 | E5-E6 | done | E6 是当前主线基线 |
| S3 | 高分辨率/后融合/重采样初筛 | E7、E8、E10_2 | done / paused | 高分辨率、WBF 权重、IR 重采样暂不作为主线 |
| S4 | 小目标头/门控/loss 初筛 | E11_1、E12_1、E13_2、E13_3 | done / partial | 都能降 FP，但 AP_dark-small 低于 E6 |
| S5 | 当前阶段：诊断优化阶段 | E20_0、E18_1/E18_2、E12_1b、E13_loss_check、E22_0 | next | 分析 E6 与 E12/E13 差异，寻找保 AP 降 FP 的改法 |
| S6 | 方法二次优化阶段 | E12_1b 正式版、E13_2b/E13_3b、E22_2、E14 | not started | 形成最终方法候选 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 | not started | 多 seed、效率、强模型、test set、复现冻结 |

当前定位：S4 已完成，S5 刚开始。上一轮 E7/E10/E11/E12/E13 的作用是初筛；当前只做诊断优化，不继续普通扩展。

## 当前阶段执行清单

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| E6/E12/E13 差异诊断 | E20_0 | 是 | 找出 E12/E13 压掉了哪些 TP，消除了哪些 FP |
| E6 多 seed | E18_1 / E18_2 | 是 | 验证 E6 是否稳定最优 |
| 弱残差门控 | E12_1b | 是 | 继承 E12 降 FP，但保留 E6 AP_dark-small |
| loss 实现检查 | E13_loss_check | 是 | 检查 E13_2/E13_3 是否真正不同 |
| hard negative taxonomy | E22_0 | 是 | 分类 E6/E12/E13 的误报类型 |
| scale+center loss | E13_4 | 暂缓 | 不建议 GPU 空闲就立刻作为主线跑 |
| P2 普通扩展 | E11_2/E11_3 | 暂停 | E11_1 已低于 E6 |
| 普通门控扩展 | E12_2/E12_3/E12_4 | 暂停 | 先做 E12_1b 弱门控 |
| E6 768 重采样扩展 | E10_3/E10_4 | 暂停 | E10_2 没超过 E6 640 |
| 强模型/test | E15/E21 | 禁止 | 还没到论文最终验证阶段 |

## 当前阶段暂停项

- 暂停 E10_3 / E10_4：E10_2 没有明确优于 E6 640。
- 暂停 E11_2 / E11_3：E11_1 AP_dark-small、AP_tiny 和总体 mAP 均低于 E6。
- 暂停普通 E12_2 / E12_3 / E12_4：E12_1 低误报有效，但 AP_dark-small 低于 E6，改做 E12_1b/1c/1d。
- E13_4 只保留 dry-run 记录，当前暂缓；先完成 E13_loss_check。
- 继续暂停 E15 强模型对照和 E21 test set。

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

## Demo 脚本

- `scripts/train/run_dark_small_experiment_demos.sh list` 列出 E0_1-E24_3 的全部编号。
- 默认 dry-run；只有设置 `RUN_MODE=run` 才会执行已实现实验。
- E7-E10 已接入 `dark_small_experiment_runner.py`；E11-E24 当前是编号一致的命令模板/占位入口，等待对应模型、loss 或分析脚本落地。
- 当前路线下不要用 demo 启动 E15 强模型对照或 E21 test set，除非已经进入最终论文阶段。

## 失败处理

- 失败后先保留错误日志，不允许直接跳过。
- 失败实验必须标记 failed，并在状态表里写明原因。
- 只在当前阶段通过后再进入下一阶段。
