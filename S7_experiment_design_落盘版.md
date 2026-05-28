# S7 实验设计落盘版：从 S6.5 诊断结果到专家方案可执行化

生成时间：2026-05-28  
输入依据：`专家方案.md`、`AGENT_RUNBOOK.md`、`EXPERIMENT_STATE.md`、`dark_small_experiment_leaderboard.csv/md`

## 1. 当前阶段结论

当前结果不支持继续做普通 YOLO 模块堆叠。E6 仍是主基线；E12/E13/E14/E22/E25/E26/E27 的主要价值是暴露问题，而不是形成可冻结候选。

| 实验 | 方向 | 现在代表什么 | 关键证据 |
| --- | --- | --- | --- |
| E6 | 多尺度 RGB-IR 融合主线 | 当前唯一稳健主基线 | mAP50-95=0.635715, AP_dark-small=0.512464, AP_dark-small_obj=0.100028, FP/image=1.469027 |
| E13_3b | small-scoped center-aware loss | 有低误报信号，但不是新主线 | AP_dark-small=0.513947, FP/image=1.285909；但 corrected multi-seed 稳定性不足 |
| E12_1/E12_1b | 残差门控融合 | 证明“门控能降误报”，但会压掉暗弱目标 | FP/image 降到 1.278/1.257，但 AP_dark-small 降到 0.506/0.504 |
| E22_2a/b | background_far hard negative | image-level AP 有假信号，object-level 不通过 | E22_2b AP_dark-small=0.521550 最高，但 FP/image=1.562968、AP_dark-small_obj=0.095848 |
| E14_1/2/3 | CEBS / 背景抑制 | 可降 FP 或微升 image AP，但 object-level 掉 | E14_1 AP_dark-small=0.515630，但 AP_dark-small_obj=0.087590；E14_2 FP/image=1.238257，但 AP_dark-small=0.505991 |
| E25/E26/E27 | 阈值、NMS、metadata verifier 后处理 | 不适合作为论文主贡献 | FP 能降，但 AP_dark-small_object 从 0.100028 降到 0.078896/0.066720/0.066380 |
| E26_2a/b | class_confusion 分类重加权 | 训练源校验失败，不可用 | split 字段为空，failed_train_split_source_unverified；且 FP/image 升高到 1.68/1.75 |

### 1.1 现在的实验说明了四件事

1. **E6 是当前结构天花板，不是普通 baseline。**  
   它同时保持了 AP_dark-small、object-level AP、FP/image 和速度的综合平衡，因此后续 S7 所有实验必须以 E6 为第一对照。

2. **“降误报”和“保目标”之间存在硬冲突。**  
   E12、E14、E25、E26、E27 都能降低 FP，但 object-level AP_dark-small 明显下降。这说明后处理阈值、普通门控和背景抑制并没有真正学到目标，而是在删候选。

3. **image-level AP_dark-small 不能单独作为候选标准。**  
   E22_2b 的 AP_dark-small 最高，但 object-level AP_dark-small 低于 E6，FP 也升高，因此它不是候选，只能说明 hard negative 对排序有扰动信号。

4. **专家方案的落地点应该优先放在“质量排序 + 动态可信融合”，而不是先做大而全的 Retinex/Frequency Transformer。**  
   当前最直接的问题是置信度、定位质量和真实目标保持不一致；所以 S7 第一刀应落在 UTAH/task-aligned score 与 RS/aLRP proxy，再做 EDL/Dirichlet 动态融合。

## 2. S7 阶段重新定义

原 runbook 中 S7 写的是“论文完整验证阶段”，但当前没有有效候选，因此不建议直接进入旧 S7 的 test / 强模型 / 候选冻结。

建议把 S7 改成：

> **S7：专家方案最小可验证落地阶段**  
> 目标不是堆更多 YOLO 组件，而是把专家方案拆成 3 个可证伪模块：  
> **任务对齐重评分 UTAH-lite → Dirichlet 不确定性融合 EDL-gate → 频域/Retinex 轻量解耦 probe**。  
> 每个模块必须单独通过 gate 后才允许组合。

## 3. S7 实验矩阵

| 阶段/编号 | 实验名 | 具体做法 | 目的 | 建议输出目录 |
| --- | --- | --- | --- | --- |
| S7-0 / E29_0 | 协议重锁定与诊断基准 | 不训练；重跑/汇总 E6、E13_3b corrected seed、object-level、TIDE/taxonomy、score-IoU 相关性 | 给后续模块提供唯一有效比较口径；修复 S6.5 旧结果混杂问题 | results/val/s7_0_protocol_lock/ |
| S7-1 / E29_1 | UTAH-lite 任务对齐重评分头 | 在 E6 检测头后增加 quality/IoU 分支；推理分数改为 score = cls^α × quality^β；正样本 quality 监督为 IoU 或 task-aligned target | 优先解决当前最明显的“高置信 FP / object-level 掉点” | results/val/s7_e29_1_utah_lite/ |
| S7-1 / E29_2 | UTAH-lite + RS/aLRP proxy | 只在 small/dark-small 正负样本上加轻量排序损失；先用 pairwise ranking proxy，不直接全量替换 YOLO loss | 把专家方案中的 AP/Rank&Sort 思路最小化落地 | results/val/s7_e29_2_utah_rank_proxy/ |
| S7-2 / E30_1 | EDL/Dirichlet 不确定性融合门控 | 在 E6 多尺度融合处加入 RGB/IR evidence head：e=softplus(x)，u=K/S，用 u 生成动态模态权重；先只作用 P3/P4/P5 | 解决静态融合在夜间/热背景下把坏模态噪声注入的问题 | results/val/s7_e30_1_edl_gate/ |
| S7-2 / E30_2 | EDL gate + modality dropout | 训练时随机压低 RGB 或 IR 信噪比，强制 gate 学会单模态接管；不改变检测头 | 验证专家方案中“知道自己不知道”的动态断流价值 | results/val/s7_e30_2_edl_dropout/ |
| S7-3 / E31_0 | 频域/Retinex 离线可行性诊断 | 不先上大模块；导出 RGB Retinex 反射代理、IR 高频边缘图，统计它们与 dark-small TP/FP 的相关性 | 避免一开始堆 FreDFT/Retinex 大模块导致贡献失焦 | results/val/s7_e31_0_freq_retinex_probe/ |
| S7-3 / E31_1 | IR 高频残差轻量注入 | 浅层 IR 分支加入固定/可学习高通残差或 Laplacian residual，参数量受控；只做单 seed 初筛 | 解决热扩散边界模糊；比完整 Transformer 频域模块更稳 | results/val/s7_e31_1_ir_hf_residual/ |
| S7-4 / E32_1 | 组合候选 A | E29 通过后 + E30 通过后组合；不加入 E31，先验证“质量对齐 + 不确定性融合” | 形成真正区别于 YOLO 魔改的主方法雏形 | results/val/s7_e32_1_utah_edl/ |
| S7-4 / E32_2 | 组合候选 B | 仅当 E31_1 单独通过，再叠加到 E32_1 | 验证频域/反射域是否是增益来源，而不是噪声 | results/val/s7_e32_2_ear_yolo_lite/ |
| S7-5 / E33 | 候选冻结与 multi-seed | 对 E32 最优候选跑 seed 0/1/2；导出 full/object-level/taxonomy/predictions；严禁 test 调参 | 只有 multi-seed 通过才进入论文完整验证/测试 | results/val/s7_e33_candidate_freeze_multiseed/ |

## 4. S7 推荐执行顺序

### 第一优先级：E29，先解决排序与重评分

**先做 E29_1/E29_2，而不是先做 Retinex/Frequency。**

原因：当前所有失败结果共同指向 score ranking 问题。  
模型不是完全找不到目标，而是常出现：
- 高置信背景 FP；
- 阈值提高后真实 dark-small 目标被删；
- image-level AP 有时升，但 object-level AP 下降；
- 后处理能降 FP，却无法提升 object-level AP。

E29 的最小实现：

```text
E6 neck/head output
  ├─ cls branch
  ├─ box/dfl branch
  └─ quality branch / task-aligned score branch
final_score = cls_score^alpha * quality_score^beta
```

训练监督：
- positive 的 `quality_target = IoU(pred_box, gt_box)` 或 task-aligned target；
- negative 的 quality target = 0；
- small/dark-small 样本可以有轻量权重，但不要复用 E26_2 的未校验 class_confusion 源。

建议先设：
- `alpha=0.5, beta=0.5` 或 `alpha=0.75, beta=0.25` 做 train-split sweep；
- sweep 只允许来自 train split；
- val 只用于最终报告，不反向调参。

### 第二优先级：E30，做不确定性动态融合

E30 是对专家方案中 Dirichlet/EDL 的可执行化，不要写成普通 attention gate。

最小实现：
```text
e_rgb = softplus(h_rgb)
e_ir  = softplus(h_ir)
S_rgb = sum(e_rgb + 1)
S_ir  = sum(e_ir + 1)
u_rgb = K / S_rgb
u_ir  = K / S_ir

w_rgb = normalize(1 - u_rgb)
w_ir  = normalize(1 - u_ir)

f_fused = w_rgb * f_rgb + w_ir * f_ir
```

关键是输出并记录 `u_rgb/u_ir` 的统计：
- dark subset 上 RGB 是否更高不确定；
- 白天热背景上 IR 是否更高不确定；
- FP 样本是否对应高冲突/高不确定；
- TP 样本是否由低不确定模态接管。

### 第三优先级：E31，只做轻量 probe，不先上大模块

专家方案中的 Retinex、频域 Transformer、对比学习和 deformable alignment 都很大。S7 不宜一次性全上，否则结果不可解释。

E31 应先做两件低风险事情：
1. 离线 probe：RGB Retinex 反射代理、IR 高频残差和 TP/FP 的相关性；
2. 轻量注入：只给 IR 浅层加高频残差，或只给 RGB 加反射代理辅助，不与 E29/E30 同时改。

若 E31 单独不通过，不进入组合。

## 5. S7 通过条件

| gate | 标准 |
| --- | --- |
| 单 seed 初筛 gate | AP_dark-small ≥ E6 + 0.004，或 AP_dark-small_obj ≥ E6 + 0.003；同时 FP/image 不高于 E6 + 0.05，mAP50-95 不低于 E6 - 0.003 |
| object-level 强 gate | AP_dark-small_object 必须 ≥ 0.100028；若 image-level AP 升但 object-level 低于 E6，标记 image-level trap，不进入组合 |
| 低误报 gate | FP/image ≤ 1.469027，FPPI_dark ≤ 2.536932；若 AP 升但 FP 明显升高，只能作为诊断，不作为候选 |
| 效率 gate | GFLOPs 增量 ≤ 20%，FPS 尽量 ≥ 400；若 EDL/UTAH 后低于 350 FPS，需要在论文中明确代价或做轻量化 |
| multi-seed gate | E32/E33 候选至少 3 seed；均值优于 E6 corrected mean，且 std 不得覆盖全部增益；旧 invalid seed 结果不得进入统计 |
| test gate | 只有 E33 freeze 后才允许一次性跑 test；禁止在 test set 上选阈值、调 NMS 或调 α/β |

## 6. runbook/state 建议改写片段

### 6.1 AGENT_RUNBOOK.md 中 S7 行建议替换为

```markdown
| S7 | 专家方案最小可验证落地阶段 | E29_0、E29_1、E29_2、E30_1、E30_2、E31_0、E31_1、E32_1、E32_2、E33 | not_started | 不进入 test / 强模型；先验证 UTAH-lite、EDL-gate、频域/Retinex probe，只有 E33 freeze 后才进入论文完整验证 |
```

### 6.2 EXPERIMENT_STATE.md 当前阶段建议替换为

```markdown
## 当前阶段

S7：专家方案最小可验证落地阶段；从 E6 主线出发，优先验证 task-aligned 重评分和不确定性动态融合，不再继续普通 YOLO 模块堆叠。

## 当前任务

第一批只允许执行：
1. E29_0 protocol lock / score-IoU diagnostic；
2. E29_1 UTAH-lite；
3. E29_2 UTAH-lite + ranking proxy。

E29 未通过前，不启动 E30/E31 组合实验；E33 freeze 前不运行 test set。
```

### 6.3 leaderboard 新增字段建议

为了支撑专家方案，不只记录 AP/FP，还应新增：

```text
score_iou_corr
high_conf_fp_count
quality_branch_auc
rgb_uncertainty_dark_mean
ir_uncertainty_thermal_bg_mean
modality_conflict_score
rank_violation_rate
```

这些字段能把论文贡献从“模块堆叠”转成“诊断—理论机制—指标闭环”。

## 7. 最终建议

S7 不应写成“我又改了 YOLO 的某个模块”。  
更好的论文主线是：

> **现有 RGB-IR YOLO 多尺度融合已经达到强工程基线，但暗弱小目标的主要瓶颈不是感受野不足，而是跨模态可靠性波动和分类置信度—定位质量错位。本文提出一种证据感知的任务对齐融合检测框架，通过不确定性驱动的动态模态融合与质量排序一致性监督，在保留小目标召回的同时降低高置信背景误报。**

按这个主线，S7 第一阶段只需要证明两件事：
1. **UTAH-lite/Ranking 能把高置信 FP 压下去，同时不伤 object-level AP；**
2. **EDL-gate 能在暗/热背景/模态冲突样本中给出可解释的动态权重。**

只要这两点成立，后面再加 Retinex/频域解耦才有论文结构价值。
