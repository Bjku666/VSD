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
| S6.5 | 诊断驱动的分类混淆修正与候选可靠性校准阶段 | corrected seedfix audit；E25_0 aggregate；E25_1_full、E26_1_full、E27_1_full；E26_2a/b failed audit samples | audit_only_completed_no_candidate | S6.5 audit-only 收口完成，暂无有效新候选；E6 仍是当前主线基线；E13_3b 只有单 seed / 低误报信号，但 corrected multi-seed 与 object-level gate 下尚未形成稳定候选 |
| S7-A | val-only 架构候选孵化阶段 | S7_0、S7_1、S7_3、S7_4、S7_2、S7_5、S7_6、S7_7 | s7_1a_reviewed_next_s7_3a | S7_0 已完成；S7_1a 已完成并判定 not_candidate；下一步转入 S7_3a Evidential reliability fusion lite |
| S7-B | 论文完整验证阶段 | E24_full、E15/E16/E19、E18_full、E20_full、E21 | blocked_no_valid_candidate | 只有 S7-A 至少产生一个通过 gate 的候选后才允许启动 |

当前定位：S7 不再直接表示论文最终验证，而是先进入 `S7-A：val-only 架构候选孵化阶段`。从 E6 corrected baseline 出发，优先验证 `S7_1 UTAH-lite quality-aligned head` 和 `S7_3 Evidential reliability fusion lite`，不再继续普通 YOLO 模块堆叠。`S7-B` 在 S7-A 产出有效候选前保持 blocked。

当前设计符合度：S6.5 已确认暂无有效新候选，E6 仍是当前主线基线。E12/E14/E25/E26/E27 说明只调阈值、普通 gate 或背景抑制会伤 object-level dark-small；因此 S7 首批转向“置信度-定位质量错位”和“模态可靠性波动”的可验证模块化落地。

## 当前阶段执行清单

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| S6.5 freeze & audit refresh | S7_0 | done | 冻结产物已生成到 `results/S7_architecture_incubation/s7_0_freeze_audit_refresh/` |
| UTAH-lite quality-aligned head | S7_1a | done_not_candidate | 2026-05-29 gate review：FP 与 object-level 同时劣化，`FP metrics down = 0/3`，不进入 `S7_1b` |
| Evidential reliability fusion lite | S7_3a | next_to_implement_and_launch | 在 E6 多尺度融合处加入 RGB/IR evidence、uncertainty 和 conflict 建模；当前推荐先做 `P4/P5-only` 版本 |
| Offset alignment lite | S7_4 | after_S7_3_or_if_localization_error_high | 在 P3/P4/P5 融合前加入 zero-init offset / deformable alignment，优先解决弱错位与 localization_error |
| RS/aLRP-lite ranking loss | S7_2 | after_S7_1_stable | 仅对 dark-small / tiny / low-contrast top-k 预测加小权重 ranking loss，训练风险更高，放在 UTAH-lite 稳住后 |
| Frequency/Retinex shallow branch | S7_5 | later_branch | RGB 反射残差 + IR 高频边缘残差，先做浅层辅助，不替换主干 |
| 组合候选 | S7_6 | blocked_until_single_modules_pass | 优先 `S7_1 + S7_3`，其次 `S7_1 + S7_4`；禁止把所有模块一次性堆叠 |
| corrected multi-seed validation | S7_7 | blocked_until_candidate | 对通过者做 seed0/1/2 full re-inference、统一验证、object-level 和 prediction export；通过后才允许进入 S7-B |

## 当前阶段暂停项

- 暂停 E10_3 / E10_4：E10_2 没有明确优于 E6 640。
- 暂停 E11_2 / E11_3：E11_1 AP_dark-small、AP_tiny 和总体 mAP 均低于 E6。
- 暂停 E12_1c / E12_1d / E12_2 / E12_3 / E12_4：E12_1b 低误报有效，但 AP_dark-small 明显低于 E6 和 E13_3b。
- 暂停 E13_4b / E13_4c 后续扩展：scale+center 组合未超过 E13_3b，且误报明显升高。
- 暂停 hard negative 3x / 5x：S6 只允许 background_far 1.5x / 2x。
- 继续暂停 E15 强模型对照和 E21 test set。
- 暂停 E27_2/E27_3/E28_1/E28_2/E9_1/E16/E19：当前阶段 audit-only 收口，暂无有效候选，不进入第二批设计外执行。

## 当前路线约束

- E6 multi-scale fusion 是当前主线基线。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 不继续搜索 WBF 权重。
- 不继续 IR-only 960 + resampling。
- 不启动 RT-DETR / YOLOv10 / YOLO11s 强模型对照。
- 不运行 test set。
- 旧 E18/E25_0 多 seed 结果保持 invalid，不得用于结论。E26_2a/E26_2b 已完成但 failed_train_split_source_unverified，不再视为有效候选任务。
- S7-A 有效候选 gate：AP_dark-small_object 不低于 E6 基线，或不低于 E6 corrected multi-seed 的统计置信下界；FP/image、FPPI_dark、FPPI_low-contrast 均低于 E6；AP_tiny_object、AP_low-contrast_object 不发生实质下降；image-level AP 不能靠牺牲 object-level AP 换来。
- E22_1 只生成 hard negative list，不直接训练。
- E22_2a/E22_2b 只使用 background_far hard negative 轻量采样，不使用 class_confusion/all 3x/5x；训练源必须来自 train split，不允许直接用 val FP 泄漏到训练。
- 后续所有 hard negative / class confusion 训练任务，必须在训练启动前检查来源 split == train 且非空；不满足则直接 blocked。
- E25_1/E26_1/E27_1 当前使用缓存预测 `conf>=0.25, NMS IoU=0.70`，因此低于 0.25 的阈值和真实多 NMS IoU 重新推理仍未完成；不能把缓存限制下的结果写成完整结论。
- E25_1/E26_1/E27_1 缓存版只能写 preliminary，只能说明“校准/可靠性有信号”；真正候选判断必须以 E25_1_full、E26_1_full、E27_1_full 这类完整重推理结果为准。
- cached prediction 只能写 preliminary；S7-A 正式候选必须 full re-inference + unified validation + object-level 复核。
- 运行 E13 相关脚本时若出现 `CXXABI_1.3.15` / `cv2` 导入错误，先设置 `LD_LIBRARY_PATH=/mnt/disk2/lhr/conda_envs/vsd/lib`。
- 当前 S6 关键日志入口：E22_2a object-level 日志为 `results/val/logs/e22_2a_hn15_object_eval_gpu0_20260526_0012.log`；E22_2b object-level 日志为 `results/val/logs/e22_2b_hn2_object_eval_gpu1_20260526.log`；E14_3 object-level 日志为 `results/val/logs/e14_3_cebs_a005_object_eval_gpu0_20260526.log`。
- E23 object-level evaluator 已支持 `--validator e6` 和 `--validator e14`；E14 object-level 评估需传入 CEBS 参数。
- S6 可复现审计入口：`python scripts/s6_repro_audit.py`，报告目录为 `results/val/s6_repro_audit/`。当前审计 status=pass、failures=0；警告为 E18 multi-seed invalid、E22_2a 训练日志含早期中断 traceback 但最终完成、git worktree dirty。
- S7-A 当前已落盘 `S7_0` 与 `S7_1a`：`S7_0` 冻结产物在 `results/S7_architecture_incubation/s7_0_freeze_audit_refresh/`；`S7_1a` 入口为 `scripts/train/run_dark_small_experiment_demos.sh S7_1a`，默认 `device=0,1`、`batch=96`、`workers=16`。
- `S7_1a` 已完成 dry-run、Python 编译与 CPU 建图 smoke；真实训练需在可访问 GPU 的宿主环境启动。当前 Codex 沙箱内无法访问 NVIDIA driver，且后台进程会被沙箱回收。
- 用户将自行执行最终启动命令；推荐后台启动脚本：`scripts/train/launch_s7_1a_utah_lite_dual4090_background.sh`。日志写入 `results/S7_architecture_incubation/logs/`，PID 文件为 `results/S7_architecture_incubation/logs/s7_1a_utah_lite_a06_b04.pid`。
- `S7_1a` 的 val/object/gate 结果已完成，结论为 `not_candidate`；当前不执行 `S7_1b`。下一步应先补齐 `S7_3a` 的实现、manifest 和 demo，再启动新的单项训练。

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

- `scripts/train/run_dark_small_experiment_demos.sh list` 列出 E0_1-E24_3 的全部编号；执行时会打印 `Sx / Ex` 阶段前缀。
- `scripts/train/run_dark_small_experiment_demos.sh S7` 列出当前 S7-A 可执行 demo；`scripts/train/launch_s7_1a_utah_lite_dual4090_background.sh` 后台启动双卡高吞吐训练。
- 默认 dry-run；只有设置 `RUN_MODE=run` 才会执行已实现实验。
- E7-E10 和 S6 的 E13_3b_light / E14 / E22_2a / E22_2b 已接入 `dark_small_experiment_runner.py`；E18_check、E22_1、E23 有真实 CLI demo；E24 仍是 demo placeholder，且当前无有效候选可冻结。
- 当前阶段 demo 可用 `scripts/train/run_dark_small_experiment_demos.sh S6` 一次性 dry-run。
- `scripts/train/run_e6_mainline_demos.sh` 也会打印 `Sx / Ex` 阶段前缀，便于区分当前属于哪一阶段。
- 当前阶段可复现审计可用 `python scripts/s6_repro_audit.py`；如需让任何 warning 也失败，追加 `--strict`。
- S6.5 离线校准入口：`python scripts/e25_e26_offline_calibration.py e25_0|e25_1|e26_1`。
- S6.5 正式完整复核入口：`python scripts/e25_e26_full_calibration.py e25_1_full|e26_1_full` 与 `python scripts/e27_metadata_verifier.py` 的 full re-inference 产物；候选判断必须看 full 版，不看 cached 版。
- 当前路线下不要用 demo 启动 E15 强模型对照或 E21 test set，除非已经进入最终论文阶段。

## 失败处理

- 失败后先保留错误日志，不允许直接跳过。
- 失败实验必须标记 failed，并在状态表里写明原因。
- 只在当前阶段通过后再进入下一阶段。
