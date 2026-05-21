# 暗弱小目标检测实验流程记录

本文档记录 VSD 项目的实验协议、结果存放规范和后续实验路线。旧训练结果、旧验证结果和旧历史权重已经清空，后续从数据协议审计和 E1/E2 基线开始重新训练。当前总体判断：实验设计合理，已覆盖 RGB/IR 单模态、后融合、特征级融合、高分辨率、小目标增强、暗弱增强和强模型对照等主要方向；但要达到论文级闭环，还必须补充低对比度协议、多 seed 稳定性、效率分析、per-class/tiny-size 分析、配准扰动鲁棒性和最终 test set 锁定评测。

## 1. 项目目录与存放规范

- 原始数据：`data/`
- 预处理数据：`prepared/`
- 数据与实验配置：`configs/`
- Python 工具脚本：`scripts/`
- shell 启动脚本：`scripts/train/`
- 预训练与固定权重：`weights/`
- 实验过程、最终指标、权重、关键图和日志：`results/`
- 验证集实验主目录：`results/val/`
- 统一日志目录：`results/val/logs/`
- 实验总表：`results/val/dark_small_experiment_leaderboard.md`
- 批量实验矩阵：`configs/experiments/dark_small_next.yaml`
- 执行状态表：`EXPERIMENT_STATE.md`
- Agent 执行手册：`AGENT_RUNBOOK.md`

新实验统一写入 `results/val/<experiment_id>.../`。训练过程权重保留在对应实验目录的 `weights/` 下，至少保留 `best.pt`；需要断点续训的实验还必须保留 `last.pt`。日志统一写入 `results/val/logs/`。

## 2. 研究目标与核心指标

研究问题：在无人机 RGB-IR 场景下，提升暗弱小目标，尤其是 `dark-small` 和 `low-contrast` 目标的检测能力。

核心指标：

- 常规指标：`mAP50`、`mAP50-95`、`Precision`、`Recall`
- 小目标指标：`AP_tiny`、`Recall_tiny`、`AP_small`、`Recall_small`
- 暗弱指标：`AP_dark`、`Recall_dark`、`AP_dark-small`
- 低对比度指标：`AP_low-contrast`、`Recall_low-contrast`、`FPPI_low-contrast`
- 误报指标：`False Positives/image`、`FPPI_dark`、背景混淆率
- 类别细分：`AP_dark-small_car`、`AP_dark-small_truck`、`AP_dark-small_bus`、`AP_dark-small_van`、`AP_dark-small_freight_car`
- 效率指标：`Params`、`FLOPs`、`FPS`、`GPU memory`、模型大小

排序优先级：

1. `AP_dark-small`
2. `Recall_small` / `Recall_tiny`
3. 更低的 `False Positives/image` 和 `FPPI_low-contrast`
4. `mAP50-95`
5. 效率指标

## 3. 数据与评测协议

数据主线使用 DroneVehicle RGB-IR。二次数据集定义为 `DroneVehicle-DarkSmall` 派生评测协议，不重新随机切分官方 train/val/test。它不是新造一个主数据集，而是在原生 DroneVehicle 上增加统一预处理、RGB/IR 配对审计、专项子集、固定评测脚本和统计报告。

预处理流程：

1. 原始数据只放在 `data/DroneVehicle/raw/`，不覆盖、不删除、不直接训练。
2. 保持官方 train、val、test 主划分不变，不重新随机打乱。
3. 检查 RGB 图像、IR 图像、RGB XML、IR XML 四者配对完整性。
4. 裁除原图四周白边，再调整标注坐标，避免白边污染亮度和对比度统计。
5. 保留原始 OBB/polygon 标注，另行生成当前 YOLO detect 使用的 AABB 标签。
6. 基于统一裁边后的图像统计 brightness、object area 和 local contrast。
7. 只用 train split 统计阈值，再应用到 val/test 子集，避免信息泄漏。
8. 生成 RGB、IR、RGB-IR 三类训练配置。
9. 生成 `dark`、`small`、`dark-small`、`low-contrast`、`tiny` 等 train/val/test 派生子集。

阈值定义：

- `dark`：裁边后 RGB 灰度均值低于 train brightness p25。
- `small`：目标面积低于 train object area p25。
- `tiny`：目标面积低于 train object area p10。
- `low-contrast`：目标区域与外扩背景环 Weber/local contrast 低于 train contrast p25。
- `dark-small`：dark 图像中包含 small 目标；同时保留 image-level 子集和 object-level 指标口径。

必须输出的审查与统计产物：

- RGB/IR 配对审计：`pair_audit.json`、`bad_pairs.csv`、配对框 IoU 分布。
- 阈值文件：`train_thresholds.json`，必须注明阈值只来自 train split。
- 主划分统计：images、pairs、objects、类别分布。
- 子集统计：dark、small、dark-small、low-contrast、tiny 的 images、objects、类别分布。
- 尺寸、亮度、对比度分布统计。
- label 转换检查和裁边检查。

数据审查规则：

- RGB/IR 文件 stem 和顺序必须一一对应。
- RGB/IR 图像尺寸必须一致。
- RGB/IR 标签数量和类别直方图必须一致。
- 配对框 IoU 过低样本记录为 possible registration error，不直接静默删除。
- train/val/test 泄漏检查必须基于真实路径，不能只看 stem。
- val/test 子集只用于评测，不能反向用于调训练权重。

Agent 执行约束：

1. 不允许删除 raw data。
2. 不允许覆盖已有 best.pt，除非输出目录是新实验目录。
3. 不允许引用旧 weights/trained/。
4. 不允许用 val/test 统计阈值。
5. 不允许在 test 上调 WBF 权重、重采样倍率或模块参数。
6. 不允许跳过 E0 直接训练 E1/E2。
7. 不允许实验失败后直接标记 done。
8. 不允许把临时结果写到未编号目录。

第一轮执行范围：

- 只允许执行 E0、E1、E2、E3、E4。
- E0 不通过，不允许进入 E1-E21。
- E7 及之后实验在 E1/E2 新权重就绪前一律 blocked。

正式评测规则：

- 所有模型选择、WBF 权重选择、重采样倍率选择和模块选择只能基于 val。
- test set 只在最终 3 到 5 个模型上锁定评测一次，不能反复调参。
- 每个实验至少保存 `metrics_summary.md`、`required_metrics.json`、`required_metrics.csv`、关键混淆矩阵和必要可视化。
- 关键模型需要 3 seed，报告 `mean ± std`。

## 4. 当前结果状态与基线重跑

当前已经删除旧训练结果、旧验证结果和旧历史权重。`results/val/` 只保留空的统一日志目录，`weights/` 只保留预训练权重。旧指标只作为实验设计背景，不再作为当前可复现实验结果引用。

重新开始后的第一轮必须按下面顺序重建可复现实验底座：

| 编号 | 实验 | 输入 | 方法 | 输出位置 |
| --- | --- | --- | --- | --- |
| E0 | DroneVehicle-DarkSmall 协议审计 | RGB+IR | 配对、裁边、阈值、子集统计 | `prepared/`、`results/dataset_audit/` |
| E1 | RGB-only baseline | RGB | YOLO11n 640 | `results/val/e1_yolo11n_rgb_only_640_ddp/` |
| E2 | IR-only baseline | IR | YOLO11n 640 | `results/val/e2_yolo11n_ir_only_640_ddp/` |
| E3 | Late Fusion + NMS | RGB+IR | 结果级 NMS 融合 | `results/val/e3_late_fusion_nms_val/` |
| E4 | Late Fusion + WBF | RGB+IR | 结果级 WBF 融合 | `results/val/e4_late_fusion_wbf_val/` |
| E5 | 单层特征融合 | RGB+IR | 双骨干 + 单层 concat + 1x1 conv | `results/val/e5_feature_fusion_single/` |
| E6 | 多尺度特征融合 | RGB+IR | 双骨干 + 多尺度融合 | `results/val/e6_feature_fusion_multiscale/` |

重跑后才能恢复 E7/E8/E10 等批量实验。E7 的 WBF 权重搜索依赖 E1/E2 新训练出的 `best.pt`，不能再引用旧 `weights/trained`。

## 5. 总路线

| 大阶段 | 阶段名称 | 实验范围 | 当前状态 | 结论 |
| --- | --- | --- | --- | --- |
| S0 | 数据协议阶段 | E0 | done | DroneVehicle-DarkSmall 协议作为基础 |
| S1 | 基础基线阶段 | E1-E4 | done | E2 是暗弱支撑基线，E4 是低误报参考线 |
| S2 | 融合主线确认阶段 | E5-E6 | done | E6 是当前主线基线 |
| S3 | 高分辨率/后融合/重采样初筛 | E7、E8、E10_2 | done / paused | 高分辨率、WBF 权重、IR 重采样暂不作为主线 |
| S4 | 小目标头/门控/loss 初筛 | E11_1、E12_1、E13_2、E13_3 | done / partial | 都能降 FP，但 AP_dark-small 低于 E6 |
| S5 | 当前阶段：诊断优化阶段 | E20_0、E18_1/E18_2、E12_1b、E13_loss_check、E22_0 | next | 分析 E6 与 E12/E13 差异，寻找保 AP 降 FP 的改法 |
| S6 | 方法二次优化阶段 | E12_1b 正式版、E13_2b/E13_3b、E22_2、E14 | not started | 形成最终方法候选 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 | not started | 多 seed、效率、强模型、test set、复现冻结 |

历史推荐优先级（当前执行以 S5 当前阶段表为准）：

1. `E0`：先固定 DroneVehicle-DarkSmall 协议，生成审计和子集统计。
2. `E1 / E2`：重新训练 RGB-only 和 IR-only 基线，生成后续 WBF 依赖的 `best.pt`。
3. `E3 / E4`：重新评估 NMS/WBF 后融合。
4. `E7-4 / E7-5 / E7-6`：基于新 E1/E2 权重做 WBF 权重搜索。
5. `E9`：6ch early fusion，作为 E6 双骨干路线的必要性对照。
6. `E10`：E5/E6 高分辨率与重采样扩展。
7. `E7-3`：补 RGB-only 768 单模态对照。
8. `E8-1`：close_mosaic 对照；E8-3/E8-4 暂不优先。
9. `E11-E14`：小目标、暗弱、低对比增强。
10. `E15-E21`：论文级完整验证。


## 5.1 当前阶段重排（2026-05-21）

当前定位：S4 已完成，S5 刚开始。基础协议、基线、E6 主线、高分辨率初筛、P2 初筛、门控初筛和 loss 初筛都已经完成；现在进入 E6 主线后的误报-召回诊断优化阶段。

S5 的核心问题不是继续证明 E6 强，而是解决：E6 的 AP_dark-small 最强但 FPPI_dark 偏高，E12/E13 降低 FPPI_dark 但 AP_dark-small 下降。因此当前目标是保住 E6 的 AP_dark-small，同时继承 E12/E13 的低误报能力。

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

## 6. 后续实验编号

### E0 协议固定与子集构建

- E0-1：RGB/IR 配对审计。
- E0-2：官方 train/val/test 泄漏检查。
- E0-3：`dark`、`small`、`dark-small` 子集确认。
- E0-4：`low-contrast` 子集构建。
- E0-5：`tiny` / `small` / `medium` / `large` 尺寸分桶。
- E0-6：统一指标协议，加入 `AP_low-contrast`、`Recall_low-contrast`、`FPPI_low-contrast`、`AP_tiny`。
- E0-7：`dark-small`、`tiny`、`low-contrast` 的 image-level 与 object-level evaluator 分离实现。

E0 验收标准：

1. train/val/test 样本无路径重复、无 stem 重复泄漏。
2. RGB/IR pair success rate = 100%；异常样本全部进入 `bad_pairs.csv`。
3. `dark`、`small`、`tiny`、`low-contrast` 阈值全部来自 `train_thresholds.json`。
4. val/test 不参与任何阈值统计。
5. 每个子集的 images、objects、class distribution 已输出。
6. `dark-small` 子集样本量不低于最小规模下限。
7. `dark-small` 的 image-level 和 object-level 口径写清楚。
8. 所有 yaml 能被训练脚本和验证脚本正常读取。

E0 不通过，不允许进入 E1-E21。

子集规模下限：

- `val_dark-small` 至少包含 100 images 或 300 boxes。
- `test_dark-small` 至少包含 300 images 或 800 boxes。
- `val_low-contrast` 至少包含 100 images 或 300 boxes。
- `test_low-contrast` 至少包含 300 images 或 800 boxes。

如果不满足下限，则允许把阈值从 p20 调到 p25 或 p30，但阈值仍然只能由 train 统计，不能用 val/test 调参。

### E7 强基线扩展实验

- E7-1：IR-only，`imgsz=768`
- E7-2：IR-only，`imgsz=960`
- E7-3：RGB-only，`imgsz=768`
- E7-4：WBF，RGB:IR = `0.4:0.6`
- E7-5：WBF，RGB:IR = `0.3:0.7`
- E7-6：WBF，RGB:IR = `0.5:0.5`

验收标准：`AP_dark-small` 高于 E2/E4，`Recall_small` 不低于 E2，WBF 误报不明显高于 E4。

### E8 Mosaic 与难样本采样实验

- E8-1：IR-only 768，`close_mosaic=20`
- E8-2：IR-only 768，dark-small 样本 3 倍重采样
- E8-3：IR-only 960，dark-small 样本 3 倍重采样
- E8-4：最佳 WBF 配置 + dark-small 重采样后的 IR 模型

验收标准：`AP_dark-small` 和 `Recall_small` 同时提升，`False Positives/image` 不大幅上升。

### E9 6 通道 Early Fusion 实验

- E9-1：YOLO11n 6ch early fusion，`imgsz=640`
- E9-2：YOLO11n 6ch early fusion，`imgsz=768`

判断规则：

- 若 6ch 接近 E4，说明双骨干复杂融合必要性降低。
- 若 6ch 低于 E4 但高于 E5/E6，说明当前双骨干实现仍需改进。
- 若 6ch 低于 E6，双骨干多尺度融合才有继续作为主线的理由。

### E10 特征融合扩展实验

- E10-1：E5，`imgsz=768`
- E10-2：E6，`imgsz=768`
- E10-3：E6，`imgsz=768` + small 重采样
- E10-4：E6，`imgsz=768` + dark-small 重采样

验收标准：E10 系列应在 `AP_dark-small` 或 `Recall_small` 上超过 E6 640，且 `mAP50-95` 不明显下降。

### E11 小目标检测头实验

- E11-1：IR-only + P2 小目标头
- E11-2：最佳融合结构 + P2 小目标头
- E11-3：P2 head + 高分辨率 768

重点观察 `AP_tiny`、`AP_small`、`Recall_tiny`、`Recall_small`、`AP_dark-small` 和误报。

### E12 暗弱感知门控融合实验

- E12-1：E6 + 简单通道门控
- E12-2：E6 + 空间门控
- E12-3：E6 + 暗区感知门控
- E12-4：E6 + dark-small 重采样 + 暗区感知门控

验收标准：`AP_dark-small` 优于 E4，`AP_dark` 和 `Recall_dark` 不下降，背景混淆率低于 NMS 融合。

### E13 Tiny-aware 损失与样本权重实验

- E13-1：baseline loss
- E13-2：small 样本权重
- E13-3：dark 样本权重
- E13-4：dark-small 样本权重
- E13-5：位置敏感 bbox loss
- E13-6：WIoU/CIoU 对比

验收标准：`AP_dark-small`、`AP_tiny` 或 `Recall_small` 明显提升，总体 mAP 不应大幅下降。

### E14 低对比度增强与背景抑制实验

- E14-1：RGB gamma/CLAHE 增强
- E14-2：IR 轻量归一化和弱噪声扰动
- E14-3：IR 背景抑制 attention
- E14-4：低对比度子集评估

注意：RGB 可做较强亮度/对比度增强；IR 不宜过强增强，避免破坏热辐射分布。

### E15 强模型横向对照

- E15-1：YOLOv10n
- E15-2：RT-DETR-R18
- E15-3：YOLO11s RGB-only / IR-only / fusion
- E15-4：最终方法与强模型对照

目的：确认最终方法收益不是单纯来自模型容量或新架构。

### E16 RGB-IR 配准扰动鲁棒性实验

- E16-1：RGB/IR 正常配准
- E16-2：IR 随机平移 ±2 px
- E16-3：IR 随机平移 ±4 px
- E16-4：IR 随机平移 ±8 px
- E16-5：门控融合 + 配准扰动

目的：证明融合方法不只提升精度，还能抑制错位模态带来的噪声。

### E17 Low-contrast 全模型评测

- E17-1：构建 low-contrast 子集
- E17-2：E2/E4/E9/E10-best/最终方法在 low-contrast 上评测
- E17-3：报告 `AP_low-contrast`、`Recall_low-contrast`、`FPPI_low-contrast`

### E18 多 seed 稳定性实验

对关键实验运行 `seed=0,1,2`，报告 `mean ± std`。

必须多 seed 的实验：

- E6
- E9 6ch
- 最终方法

建议多 seed 的实验：

- E1 RGB-only
- E2 IR-only
- E4 WBF
- 强模型对照

### E19 效率与部署指标

每个主模型报告：

- `Params`
- `FLOPs`
- `FPS`
- `GPU memory`
- 模型权重大小

重点比较 IR-only vs WBF、6ch vs E6、E6 vs E6+P2、E6 vs 最终方法。

### E20 Per-class、Tiny-size 与失败案例分析

补充：

- `AP_dark-small_car`
- `AP_dark-small_truck`
- `AP_dark-small_bus`
- `AP_dark-small_van`
- `AP_dark-small_freight_car`
- `AP_tiny`
- `Recall_tiny`

失败类型统计保存为 `failure_case_taxonomy.csv`，建议分类：

- `dark miss`
- `tiny miss`
- `low-contrast miss`
- `thermal hotspot FP`
- `lamp FP`
- `edge FP`
- `class confusion`
- `registration error`

### E21 Test set 最终锁定评测

test set 只用于最终锁定评估。建议最终只评估：

- E2 IR-only best
- E4 WBF best
- E9 6ch best
- E10 E6 best
- 最终方法
- 强模型 best

test set 锁定记录表必须单独保留，字段至少包括：模型名、val 选择理由、是否进入 test、test 权重路径、test 日期、对应协议版本。

### E22 Hard Negative Mining 实验

- E22_0：收集 E6 / E12_1 / E13_2 在 dark / low-contrast 上的 FP，构建 hard negative taxonomy 和 list，暂不训练。
- E22_1：hard negative list 构建与去重。
- E22_2：E6 + hard negative 2x 采样。
- E22_3：E12_1b + hard negative 2x 采样。
- E22_4：仅在 2x 不牺牲 recall 时考虑 3x 采样。
- E22_5：hard negative + CEBS。

重点观察 `FPPI_dark`、`FPPI_low-contrast`、`AP_dark-small`、`Recall_dark-small`。

### E23 Object-level Subset Evaluator

- E23-1：实现 object-level 的 `dark-small`、`tiny`、`low-contrast` 评测器。
- E23-2：输出 `AP_dark-small_object`、`Recall_dark-small_object`、`AP_tiny_object`、`AP_low-contrast_object`。
- E23-3：对比 image-level 与 object-level 口径差异。

### E24 配置冻结与复现审计

- E24-1：冻结 `configs/frozen/` 下的最终配置。
- E24-2：检查 config path、commit hash、protocol version、seed、weight path、result path。
- E24-3：确认 leaderboard 与实际结果目录一致。

## 7. 执行依赖与淘汰规则

### 7.1 实验依赖关系

| 实验 | 必须依赖 |
| --- | --- |
| E1 | E0 通过 |
| E2 | E0 通过 |
| E3 / E4 | E1 / E2 新权重 |
| E7 WBF | E1 / E2 新 `best.pt` |
| E8 重采样 | E0 的 `train_dark-small` 列表 |
| E9 | E0 RGB/IR 同步配置 |
| E10 | E5/E6 原生训练与验证通过 |
| E11 | E2 或 E10-best |
| E12 | E6 或 E10-best |
| E14 | E0 low-contrast 子集 |
| E16 | E10 / E12 融合模型 |
| E21 | E15 / E18 / E19 / E20 完成后 |

### 7.2 模型淘汰规则

1. 若 `AP_dark-small` 提升小于 0.5 且 `FPPI_dark` 上升，则不进入下一阶段。
2. 若 `Recall_small` 提升但 `FP/image` 大幅上升，则转入背景抑制分支，不作为主方法。
3. 若 E6 仍低于 E4 WBF 超过 3 个 `mAP50-95` 点，则不强行以 E6 为论文主线。
4. 若 6ch early fusion 接近 WBF，则双骨干创新必须证明效率或鲁棒性优势。
5. 若 P2 只提升 Recall 但显著增加误报，则必须和 CEBS 或 hard negative mining 绑定。

### 7.3 主线切换条件

主线 A：RGB-IR 双分支融合路线。

- 启用条件：E6 的 `AP_dark-small` ≥ E4 WBF，或 E6 的 `Recall_small` 明显高于 E4 且 `FP/image` 不高于 E4 的 10%。

主线 B：IR 强基线 + WBF 工程路线。

- 启用条件：E6 仍明显低于 E4 WBF；E9 6ch 也无法超过 E4；E2 / E7 IR 高分辨率在 dark-small 上最稳。

### 7.4 hard negative mining 独立实验

- E22-1：收集 E2 / E4 / E10-best 在 dark / low-contrast 上的 FP。
- E22-2：按 thermal hotspot、lamp、edge、background vehicle-like 分类。
- E22-3：hard negative 2x 采样。
- E22-4：hard negative 3x 采样。
- E22-5：hard negative + CEBS。

重点观察 `FPPI_dark`、`FPPI_low-contrast`、`AP_dark-small`、`Recall_dark-small`。

### 7.5 资源预算表

| 阶段 | 实验 | 数量 | 是否 3 seed | 优先级 |
| --- | --- | ---: | --- | --- |
| 基线 | E1-E4 | 4 | E1/E2 建议 | 必须 |
| 融合对照 | E9 | 2 | 必须 | 必须 |
| 融合扩展 | E10 | 4 | E6 关键项必须 | 必须 |
| 高分辨率 | E7 | 6 | 可单 seed 初筛 | 高 |
| 小目标 | E8 / E11 / E13 | 多项 | 关键模型 3 seed | 高 |
| 创新模块 | E12 / E14 | 多项 | 最终候选 3 seed | 高 |
| 完整验证 | E15-E21 | 多项 | 必须 | 论文阶段 |

原则：先单 seed 初筛，进入最终候选后再补 `seed=1,2`。

### 7.6 配置冻结机制

- 建议建立 `configs/frozen/`，保存每个阶段最终使用的配置。
- 每个实验结果表必须记录 `config path`、`commit hash`、`data protocol version`、`seed`、`weight path`、`result path`。
- 最终冻结配置至少包括 `e0_protocol_v1`、`e1_rgb_baseline`、`e2_ir_baseline`、`e4_wbf_best`、`final_method`。

### 7.7 图表产物清单

必须生成图：

1. brightness distribution
2. object area distribution
3. contrast distribution
4. dark / small / dark-small / low-contrast 子集样例图
5. PR curve
6. confusion matrix
7. FP / FN 可视化
8. per-class AP bar
9. AP_dark-small vs FPS trade-off
10. 模态权重热力图，如果做 E12 门控融合

如果做暗弱感知门控融合，还应保存 RGB weight map、IR weight map、fusion attention map、dark-small case visualization。

### 7.8 E23 / E24 建议补充实验

- E23：object-level subset evaluator，输出 `AP_dark-small_object`、`Recall_dark-small_object`、`AP_tiny_object`、`AP_low-contrast_object`。
- E24：configuration freezing and reproducibility audit，检查配置、权重、协议版本和结果目录是否一致。

优先级：E23 > E24 > E22。

## 8. 最终方法完整消融表

最终论文必须有组合消融：

| 设置 | P2 | Gate | Reweight | CEBS | AP_dark-small |
| --- | --- | --- | --- | --- | ---: |
| Baseline | × | × | × | × |  |
| +P2 | √ | × | × | × |  |
| +Gate | × | √ | × | × |  |
| +Reweight | × | × | √ | × |  |
| +CEBS | × | × | × | √ |  |
| +P2+Gate | √ | √ | × | × |  |
| +P2+Gate+Reweight | √ | √ | √ | × |  |
| Full | √ | √ | √ | √ |  |

## 9. 闭环验收与复现要求

### 9.1 最终输出清单

最终每个关键实验至少保存：

- `metrics_summary.md`
- `required_metrics.json`
- `required_metrics.csv`
- 关键混淆矩阵
- 必要可视化
- 协议版本号
- 配置路径
- 权重路径
- seed

### 9.2 test set 锁定记录表

| 模型 | val 选择理由 | 是否进入 test | test 权重路径 | test 日期 | 协议版本 |
| --- | --- | --- | --- | --- | --- |
| E2 IR best | dark recall 强 |  |  |  |  |
| E4 WBF best | 误报低 |  |  |  |  |
| E9 6ch best | 低成本融合 |  |  |  |  |
| E10 / E12 best | 融合主线 |  |  |  |  |
| Final | 最终方法 |  |  |  |  |

### 9.3 复现检查清单

- [ ] 原始 DroneVehicle 未被覆盖或删除
- [ ] 官方 train / val / test 主划分保持不变
- [ ] RGB / IR 文件 stem 一一对应
- [ ] RGB / IR 图像尺寸一致
- [ ] RGB / IR label 数量一致
- [ ] RGB / IR 类别直方图一致
- [ ] 完成四周白边裁剪
- [ ] 裁剪后 label 坐标正确
- [ ] 原始 OBB / polygon 标注已保留
- [ ] YOLO AABB label 已生成
- [ ] dark 阈值只由 train 统计
- [ ] small 阈值只由 train 统计
- [ ] low-contrast 阈值只由 train 统计
- [ ] val / test 只被筛选，不参与阈值学习
- [ ] 生成 RGB、IR、RGB-IR 三套 yaml
- [ ] 生成 dark / small / dark-small / low-contrast 子集
- [ ] 子集样本量足够，不是几十张小样本
- [ ] 每个子集有类别分布统计
- [ ] 每个子集有亮度、尺寸、对比度统计
- [ ] 支持 `AP_dark-small`、`Recall_small`、`FP/image` 评测
- [ ] 明确 `AP_dark-small` 是 image-level 还是 object-level
- [ ] 所有子集 list、阈值、脚本可复现

## 10. 最终论文主线建议

当前最稳论文主线：

> RGB-IR 双分支检测框架 + 暗弱感知跨模态融合 + 小目标检测优化。

如果 E9 后 E5/E6 仍不理想，则改为更稳的工程主线：

> IR 强基线 + WBF 后融合 + dark-small 样本重采样 + 小目标检测头。

论文主表必须突出 `AP_dark-small`、`Recall_small`、`AP_tiny`、`AP_low-contrast`、`FP/image` 和效率指标。不能只追求全量 `mAP50`。
