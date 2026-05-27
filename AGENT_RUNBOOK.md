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
11. 不得删除、终止或清理用户未明确授权的进程，只能操作当前任务自己启动的进程。
12. 可并行任务优先分配到空闲 GPU，不与当前正在训练的任务抢占同一张卡。

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
| S6 | 诊断驱动的目标保持型背景抑制阶段 | E23、E18_check、E22_1、E13_3b-light、E22_2a/b、E14、E24_0 | awaiting_review | object-level 口径已落地；target-scoped light loss、train-split HN 和 CEBS 均未形成新候选 |
| S6.5 | 诊断驱动的分类混淆修正与候选可靠性校准阶段 | corrected multi-seed rerun；E25_1_full、E26_1_full；第二批为 E26_2a/b、E27、E28、E9_1、E16、E19 | seedfix_multiseed_running | 旧 E18/E25_0 多 seed 结果 invalid 且旧产物/日志已删除；GPU0 正在 batch48 重跑 corrected multi-seed；E25_1_full/E26_1_full 已完成但不通过 object-level gate |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 full | not started | 暂不进入 test / 强模型 |

当前定位：旧 E18/E25_0 多 seed 结果已判定 invalid，旧输出目录和日志已按用户要求删除。2026-05-27 已修复 E6/E13 trainer 的 dataloader seed 逻辑，并在 GPU0 启动 batch48 corrected multi-seed 队列；日志为 `results/val/logs/seedfix_multiseed_gpu0_b48_20260527.log`。E26_2b 仍在 GPU1 batch48 训练。

当前设计符合度：方向符合 S6.5，但原状态表把缓存版 E25_1/E26_1/E27_1 和第二批 E26_2a 写得过于靠前。按新 gate 修正后，缓存结果只能证明“校准/可靠性有信号”，不能写成完整实验结论。

## 当前阶段执行清单

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| corrected multi-seed 重跑 | seedfix/E18/E25_0 | running_gpu0_b48 | 已修复 dataloader seed 逻辑；GPU0 batch48 顺序重跑 E6 seed0/1/2、E13_3b seed0/1/2、E25_0 seed42/43/44。每个 seed 补齐 validation、object-level、prediction export；旧多 seed 产物/日志已删除，不得再用于均值/方差/候选判断 |
| E6 完整重推理 calibration / threshold / NMS sweep | E25_1_full | done_not_candidate | 完整重推理完成；selected=illumination_wise dark0.35_other0.40 / NMS 0.50，FP/image=1.279101、FPPI_dark=2.196023，但 AP_dark-small_object=0.078896，未通过 gate |
| class-wise threshold 完整复核 | E26_1_full | done_not_candidate | 完整复核完成；FP/image=1.147720、FPPI_dark=1.781250、class_confusion FP=661，但 AP_dark-small_object=0.066720，未通过 gate |
| E6 calibration 缓存版 | E25_1 | 已完成，preliminary | 基于 E20 train/val 缓存预测完成；NMS 仅记录缓存 IoU=0.70，object AP 图是 cached dark-small object AP50 proxy，不是完整 AP50-95 |
| class-wise threshold 缓存版 | E26_1 | 已完成，preliminary | 阈值 car=0.40、truck=0.50、bus=0.50、van=0.45、freight_car=0.45；val FP/image=1.147720、FPPI_dark=1.781250、class_confusion FP=661，只作为 full 版复核依据 |
| class_confusion classification-only loss 1.25x | E26_2a | 第二批暂停 | 只改分类 loss，不动 bbox/dfl/assignment；不得把 class_confusion 当背景负样本；仅在第一批完成并复核后决定是否重跑 |
| class_confusion classification-only loss 1.50x | E26_2b | 第二批暂停 | 仅在 E26_2a 有效且 object-level AP 未明显下降后执行 |
| metadata verifier 完整复核 | E27_1_full | 第二批暂停 | 重新生成 E6 train/val predictions 后做 verifier；缓存版 E27_1 只作为信号，不作为完整结论 |
| metadata verifier 缓存版 | E27_1 | 已完成，preliminary | train TP vs background_far 负样本，排除 class_confusion/localization_error 负训练；best score_final 阈值 0.03，FP/image=1.144997、FPPI_dark=1.781250 |
| object-level evaluator | E23 | 已完成 | 输出 dark-small/tiny/low-contrast object-level AP/Recall，并对比 image-level |
| E13_3b seed 独立性核查 | E18_check | 已完成 | seed=1/2 args 和 weight hash 不同，但关键指标完全一致；multi-seed invalid，需重跑 seed=2 |
| hard negative list 构建与去重 | E22_1 | 已完成 | 拆分 background_far、class_confusion、localization_error、near_object_background、duplicate_or_conf_threshold；仅 background_far train-allowed |
| E13_3b-light | E13_3b-light | 已完成，不作为候选 | FP/image 与 FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 1.5x | E22_2a | 已完成，不作为候选 | 使用 train-split background_far HN；image-level AP_dark-small 提升但 FP/FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 2x | E22_2b | 已完成，不作为候选 | 使用 train-split background_far HN；image-level AP_dark-small 提升但 FP/FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| CEBS alpha=0.05/0.10 | E14_1/E14_2 | 已完成，不作为候选 | E14_1 image-level 有提升但 object-level AP_dark-small 低于 E6；E14_2 降 FP 但 image/object dark-small AP 均低于 E6 |
| CEBS 组合候选 | E14_3/E14_4 | E14_3 已完成，不作为候选；E14_4 跳过 | E14_3 未超过 B 候选，且 HN 路线未提供组合依据 |
| candidate freeze | E24_0 | 阻塞，无有效候选 | 当前没有满足 image/object 条件且 seed 状态有效的候选；E24 CLI 仍是 demo placeholder |
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
- 暂停 E26_2b：必须等 E26_2a 有效训练和验证结果后再判断是否执行。
- 暂停 E26_2a/E26_2b/E27_1_full/E27_2/E27_3/E28_1/E28_2/E9_1/E16/E19：第一批 E25_0、E25_1_full、E26_1_full 完成并复核前不进入第二批。

## 当前路线约束

- E6 multi-scale fusion 是当前主线基线。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 不继续搜索 WBF 权重。
- 不继续 IR-only 960 + resampling。
- 不启动 RT-DETR / YOLOv10 / YOLO11s 强模型对照。
- 不运行 test set。
- S6.5 当前优先完成 corrected multi-seed 重跑；旧 E18/E25_0 多 seed 结果不得用于结论。第二批继续训练中的 E26_2b 保留，不主动终止。
- S6.5 硬通过条件：AP_dark-small_object >= 0.098，AP_tiny_object / AP_low-contrast_object 不明显下降，FP/image < 1.469027，FPPI_dark < 2.536932，FPPI_low-contrast < E6。
- E22_1 只生成 hard negative list，不直接训练。
- E22_2a/E22_2b 只使用 background_far hard negative 轻量采样，不使用 class_confusion/all 3x/5x；训练源必须来自 train split，不允许直接用 val FP 泄漏到训练。
- E25_1/E26_1/E27_1 当前使用缓存预测 `conf>=0.25, NMS IoU=0.70`，因此低于 0.25 的阈值和真实多 NMS IoU 重新推理仍未完成；不能把缓存限制下的结果写成完整结论。
- 运行 E13 相关脚本时若出现 `CXXABI_1.3.15` / `cv2` 导入错误，先设置 `LD_LIBRARY_PATH=/mnt/disk2/lhr/conda_envs/vsd/lib`。
- 当前 S6 关键日志入口：E22_2a object-level 日志为 `results/val/logs/e22_2a_hn15_object_eval_gpu0_20260526_0012.log`；E22_2b object-level 日志为 `results/val/logs/e22_2b_hn2_object_eval_gpu1_20260526.log`；E14_3 object-level 日志为 `results/val/logs/e14_3_cebs_a005_object_eval_gpu0_20260526.log`。
- E23 object-level evaluator 已支持 `--validator e6` 和 `--validator e14`；E14 object-level 评估需传入 CEBS 参数。
- S6 可复现审计入口：`python scripts/s6_repro_audit.py`，报告目录为 `results/val/s6_repro_audit/`。当前审计 status=pass、failures=0；警告为 E18 multi-seed invalid、E22_2a 训练日志含早期中断 traceback 但最终完成、git worktree dirty。

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
- E7-E10 和 S6 的 E13_3b_light / E14 / E22_2a / E22_2b 已接入 `dark_small_experiment_runner.py`；E18_check、E22_1、E23 有真实 CLI demo；E24 仍是 demo placeholder，且当前无有效候选可冻结。
- 当前阶段 demo 可用 `scripts/train/run_dark_small_experiment_demos.sh S6` 一次性 dry-run。
- 当前阶段可复现审计可用 `python scripts/s6_repro_audit.py`；如需让任何 warning 也失败，追加 `--strict`。
- S6.5 离线校准入口：`python scripts/e25_e26_offline_calibration.py e25_0|e25_1|e26_1`。
- 当前路线下不要用 demo 启动 E15 强模型对照或 E21 test set，除非已经进入最终论文阶段。

## 失败处理

- 失败后先保留错误日志，不允许直接跳过。
- 失败实验必须标记 failed，并在状态表里写明原因。
- 只在当前阶段通过后再进入下一阶段。
