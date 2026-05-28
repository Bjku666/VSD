# S7 实验落盘方案：从 S6.5 audit-only 到专家方案可验证候选

## 0. 当前结论

S6.5 已经完成 audit-only 收口，但没有有效新候选。E6 multi-scale fusion 继续作为唯一稳定主线基线。E13_3b 只保留为“单 seed / 低误报信号”，不得作为候选冻结。E25_1_full、E26_1_full、E27_1_full 能降低 FP/image 或 FPPI_dark，但 dark-small/tiny/low-contrast 的 object-level AP 明显下降，因此不能继续沿阈值/后处理路线强推。

S7 不建议直接进入 test set、强模型对照或候选冻结。S7 应先重定义为“专家方案拆解后的可验证架构候选孵化阶段”，只有产生满足 gate 的 val 候选后，再进入论文完整验证。

## 1. S7 总目标

在不使用 test set 调参、不引用旧 weights/trained、不使用 split 未验证的 hard negative/class confusion 来源的前提下，从 E6 基线出发，解决三个已经被 S5-S6.5 暴露出的核心矛盾：

1. 置信度与定位质量不单调：高分背景/弱定位框导致 FP 与 localization_error。
2. 模态融合不可靠：RGB/IR 在暗夜、强热背景、反光路面下发生模态冲突。
3. 后处理过度剪枝：阈值、class-wise threshold、metadata verifier 都能降 FP，但会杀掉 dark-small/tiny 真目标。

## 2. S7 阶段拆分

### S7-A：val-only 架构候选孵化，不跑 test

| ID | 名称 | 专家方案来源 | 实验目的 | 主要改动 | 通过条件 |
| --- | --- | --- | --- | --- | --- |
| S7_0 | S6.5 freeze & audit refresh | 规范化前置 | 冻结 S6.5 证据和 E6 参考线 | 重建 E6 corrected seed0/1/2 汇总、确认 E25/E26/E27 只作 not_candidate/preliminary | audit pass；leaderboard/state/report 一致 |
| S7_1 | UTAH-lite quality-aligned head | 统一任务对齐头 | 解决 cls score 与 IoU/定位质量错位 | 在 E6 head 加轻量 IoU-quality 分支；推理 score = cls^alpha * q^beta；只用 train/val | FPPI_dark 下降且 AP_dark-small_object 不低于 E6 |
| S7_2 | RS/aLRP-lite ranking loss | 全局排序损失 | 直接优化正负样本排序，抑制 hard FP | 仅对 dark-small/tiny/low-contrast top-k 预测加 ranking loss；先小权重 warmup | AP_dark-small 与 object AP 不降，FP 不升 |
| S7_3 | Evidential reliability fusion lite | Dirichlet/EDL 动态模态融合 | 从网络内部做模态可靠性，不再靠离线 verifier | 每尺度生成 RGB/IR evidence、uncertainty、conflict map；残差式调制 E6 fusion | 比 E12 gate 保 AP；比 E6 降 FP/FPPI |
| S7_4 | Offset alignment lite | 偏移量引导跨模态对齐 | 针对 RGB/IR 弱错位与 localization_error | P3/P4/P5 融合前加入 zero-init offset/deformable alignment；先不开对比损失 | localization_error 下降，AP_tiny_object 不降 |
| S7_5 | Frequency/Retinex shallow branch | 反射域/频域解耦 | 增强暗弱边缘与低对比目标边界 | RGB 反射/亮度残差、IR 高频边缘残差作为浅层辅助，不替换主干 | AP_low-contrast_object 与 AP_dark-small_object 有提升 |
| S7_6 | 组合候选 | 专家方案组合落地 | 只组合单项通过 gate 的模块 | 优先 UTAH-lite + EDL；其次 UTAH-lite + alignment；禁止把所有模块一次堆叠 | 组合项必须优于任一单项，且 FP/FPPI 不反弹 |
| S7_7 | multi-seed candidate validation | 论文候选可靠性 | 对 S7_1-S7_6 通过者做 seed0/1/2 | corrected dataloader seed；统一验证；object-level；prediction export | 均值或统计置信下界通过有效候选 gate |

### S7-B：论文完整验证，只在 S7-A 产生候选后执行

| ID | 名称 | 执行条件 | 输出 |
| --- | --- | --- | --- |
| E24_full | candidate freeze full | S7_7 至少一个候选通过 gate | 冻结候选权重、配置、阈值、metrics、hash |
| E15/E16/E19 | 强模型/横向对照 | E24_full 后 | YOLO11s/RT-DETR/其他强基线只作论文对照，不参与调参 |
| E18_full | 多 seed 完整统计 | E24_full 后 | E6 与候选 3 seed/5 seed 均值方差 |
| E20_full | error analysis full | E24_full 后 | taxonomy、TIDE-like、FP source、case study |
| E21 | test set final | 仅最终冻结后一次 | test 只报告，不再回调参数 |

## 3. S7 gate

沿用并强化当前 gate：

- AP_dark-small_object 不低于 E6，或不低于 E6 corrected multi-seed 的统计置信下界；
- FP/image、FPPI_dark、FPPI_low-contrast 均低于 E6；
- AP_tiny_object、AP_low-contrast_object 不发生实质下降；
- image-level AP_dark-small、AP_tiny、AP_low-contrast 不能只靠牺牲 object-level AP 换来；
- 所有 hard negative / class confusion 训练源必须在训练前通过 split == train 且非空校验；
- cached prediction 只能 preliminary，正式候选必须 full re-inference + unified validation + object-level 复核。

## 4. 推荐立即执行顺序

1. S7_0：冻结 S6.5 证据、修正 runbook/state 中“不要进入原 S7 final”的表述。
2. S7_1：先做 UTAH-lite，因为它直接对应当前 localization_error/class_confusion 与高分 FP，工程风险最低。
3. S7_3：再做 EDL reliability fusion lite，因为 E25/E26/E27 已证明“可靠性有信号”，但必须内生到网络。
4. S7_4：若 localization_error 仍高，再做 offset alignment lite。
5. S7_2：ranking loss 放在 UTAH/EDL 后做小权重实验，避免训练不稳定。
6. S7_5：Retinex/FFT 作为低对比增强支线，不抢主线。
7. S7_6/S7_7：只组合通过单项 gate 的模块，并做 corrected multi-seed。

## 5. 可以写入 EXPERIMENT_STATE 的落盘语句

当前建议将“下一步”改为：

> S7 暂不按论文最终验证阶段直接启动。由于 S6.5 audit-only 收口后暂无有效候选，E24_0 仍 blocked_no_valid_candidate，E15/E16/E19 强模型对照和 E21 test set 继续禁止。下一阶段先执行 S7-A val-only 架构候选孵化：从 E6 corrected baseline 出发，按 UTAH-lite、Evidential reliability fusion、offset alignment、RS/aLRP-lite、Frequency/Retinex shallow branch 的顺序做单项 ablation；仅当至少一个单项或组合通过 AP_dark-small_object、AP_tiny_object、AP_low-contrast_object 与 FP/FPPI gate 后，才进入 E24_full 候选冻结和 S7-B 论文完整验证。
