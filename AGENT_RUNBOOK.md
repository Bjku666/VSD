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
| S5 | 诊断优化阶段 | E20_0、E18_1/E18_2、E12_1b、E13_loss_check、E22_0 | done | 找到主要 FP 类型，确认 E13_3b 是当前最佳低误报候选但 AP_dark-small 三 seed 稳定性仍不足 |
| S6 | 诊断驱动的目标保持型背景抑制阶段 | E23、E18_check、E22_1、E13_3b-light、E22_2a/b、E14、E24_0 | running | 围绕 object-level 口径、seed 核查、hard negative 定向优化、target-scoped light loss 和 CEBS 候选推进 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 full | not started | 暂不进入 test / 强模型 |

当前定位：S5 诊断完成，进入 S6。上一轮 E7/E10/E11/E12/E13 的作用是初筛和诊断；当前不继续普通 YOLO 扩展，只围绕 object-level evaluator、误报 taxonomy、target-scoped light loss、background_far HN 和 CEBS 候选推进。

## 当前阶段执行清单

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| object-level evaluator | E23 | 已完成 | 输出 dark-small/tiny/low-contrast object-level AP/Recall，并对比 image-level |
| E13_3b seed 独立性核查 | E18_check | 已完成 | seed=1/2 args 和 weight hash 不同，但关键指标完全一致；multi-seed invalid，需重跑 seed=2 |
| hard negative list 构建与去重 | E22_1 | 已完成 | 拆分 background_far、class_confusion、localization_error、near_object_background、duplicate_or_conf_threshold；仅 background_far train-allowed |
| E13_3b-light | E13_3b-light | 已完成，不作为候选 | FP/image 与 FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 1.5x | E22_2a | blocked | 必须先有 train-split hard negative source；不能直接用 val FP 训练 |
| background_far hard negative 2x | E22_2b | blocked | 同上；不允许 3x/5x，不允许 class_confusion |
| CEBS alpha=0.05/0.10 | E14_1/E14_2 | 等待 GPU/部分完成 | E14_1 已完成训练并等待 GPU1 补验证；E14_2 已完成 image-level 并等待 GPU0 object-level |
| CEBS 组合候选 | E14_3/E14_4 | 待 E13_3b-light 后执行 | 若 CEBS 不超过 light/HN 候选，不作为主方法硬用 |
| candidate freeze | E24_0 | 待最佳候选有效后执行 | 冻结配置、权重路径、metrics、protocol、commit、训练和验证参数 |
| scale+center loss 后续扩展 | E13_4b/E13_4c | 暂停 | E13_4b/E13_4c 误报明显升高，不作为当前最佳方向 |
| P2 普通扩展 | E11_2/E11_3 | 暂停 | E11_1 已低于 E6 |
| 普通门控扩展 | E12_1c/E12_1d/E12_2/E12_3/E12_4 | 暂停 | gate 路线降 FP 但伤 AP_dark-small，S6 不继续 |
| E6 768 重采样扩展 | E10_3/E10_4 | 暂停 | E10_2 没超过 E6 640 |
| 强模型/test | E15/E21 | 禁止 | 不运行 test set，不启动 RT-DETR / YOLOv10 / YOLO11s |
| hard negative 3x/5x | E22_2 3x/5x variants | 禁止 | class_confusion 占比高，盲目强采样可能压掉真实车辆目标 |

## 当前阶段暂停项

- 暂停 E10_3 / E10_4：E10_2 没有明确优于 E6 640。
- 暂停 E11_2 / E11_3：E11_1 AP_dark-small、AP_tiny 和总体 mAP 均低于 E6。
- 暂停 E12_1c / E12_1d / E12_2 / E12_3 / E12_4：E12_1b 低误报有效，但 AP_dark-small 明显低于 E6 和 E13_3b。
- 暂停 E13_4b / E13_4c 后续扩展：scale+center 组合未超过 E13_3b，且误报明显升高。
- 暂停 hard negative 3x / 5x：S6 只允许 background_far 1.5x / 2x。
- 继续暂停 E15 强模型对照和 E21 test set。

## 当前路线约束

- E6 multi-scale fusion 是当前主线基线。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 不继续搜索 WBF 权重。
- 不继续 IR-only 960 + resampling。
- 不启动 RT-DETR / YOLOv10 / YOLO11s 强模型对照。
- 不运行 test set。
- S6 只允许执行 E23、E18_check、E22_1、E13_3b-light、E22_2a、E22_2b、E14_1、E14_2、E14_3、E14_4、E24_0。
- E22_1 只生成 hard negative list，不直接训练。
- E22_2a/E22_2b 只使用 background_far hard negative 轻量采样，不使用 class_confusion/all 3x/5x；训练源必须来自 train split，不允许直接用 val FP 泄漏到训练。
- 运行 E13 相关脚本时若出现 `CXXABI_1.3.15` / `cv2` 导入错误，先设置 `LD_LIBRARY_PATH=/mnt/disk2/lhr/conda_envs/vsd/lib`。
- 当前实时日志入口：E14_1 原训练日志为 `tail -f results/val/logs/e14_1_e6_cebs_a005_gpu1_20260524_2126.log`，补验证等待日志为 `tail -f results/val/logs/e14_1_e6_cebs_a005_manual_val_gpu1_20260525_1231.log`；E14_2 原训练/验证日志为 `tail -f results/val/logs/e14_2_e6_cebs_a010_gpu0_20260524_2228.log`，object-level 等待日志为 `tail -f results/val/logs/e14_2_e6_cebs_a010_object_eval_gpu0_20260525_1231.log`；E13_3b-light 日志为 `results/val/logs/e13_3b_light_target_center_loss_gpu0_20260524_143843.log`，已到 `VALIDATE_EXIT 0`。
- E23 object-level evaluator 已支持 `--validator e14`；E14 object-level 评估需传入 CEBS 参数，当前 E14_2 已排队：`scripts/train/e14_2_object_wait_gpu0.sh`。

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
