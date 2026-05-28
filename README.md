# VSD 项目目录说明

本项目围绕 DroneVehicle/VisDrone 无人机目标检测实验展开，当前重点是 RGB-IR 暗弱小目标检测。根目录保留唯一 `README.md`；主要一级目录使用 `perface.md` 说明用途。

## 顶层目录

- `configs/`：数据集、子集和批量实验配置。
- `data/`：原始公开数据和压缩包。
- `prepared/`：转换后可直接用于 YOLO/Ultralytics 的数据。
- `scripts/`：数据处理、训练、验证、融合评估、结果清理和关键图重建脚本。实验推进以编号实验脚本和 runner 为主，不再把结果重组、目录改写、报告生成类工具当作实验 demo 入口。
- `scripts/train/`：所有 shell 启动脚本。
- `results/`：实验过程、最终指标、权重、关键图和日志。
- `weights/`：预训练权重；旧历史训练权重已删除，后续按实验重新生成。

## 当前状态

- 旧训练结果、旧验证结果和旧历史权重已经清空，后续从 E0 数据协议和 E1/E2 基线开始重新训练。
- `weights/` 当前只保留预训练权重；新训练权重随实验保存在 `results/val/<experiment>/weights/`。
- 当前核心问题仍然是 `AP_small`、`AP_tiny`、`AP_dark-small` 和低对比度目标，而不是只追求总 mAP。
- 二次数据集协议采用 `DroneVehicle-DarkSmall` 思路：不重新打乱官方 train/val/test，只基于官方划分派生专项评测子集。

## 当前实验路线

当前执行以 `EXPERIMENT_STATE.md`、`AGENT_RUNBOOK.md` 和 `S7_experiment_plan_luopan.md` 为准；`shiyan.md` 主要保留历史规划。当前实验路线为：

1. S6.5 先按 audit-only 收口，冻结 E6 corrected baseline，以及 E25/E26/E27 的 not_candidate / failed audit 结论。
2. S7 不直接进入论文最终验证，先拆成 `S7-A` 和 `S7-B` 两段。
3. `S7-A` 只做 val-only 架构候选孵化，从 E6 出发做单项可解释 ablation。
4. 第一优先实验是 `S7_1: E6 + UTAH-lite quality-aligned head`。
5. 第二优先实验是 `S7_3: Evidential reliability fusion lite`，把可靠性信号从后处理移回融合层。
6. 若定位误差仍高，再做 `S7_4: offset alignment lite`；`S7_2 ranking loss` 放在其后做小权重验证。
7. `S7_5 Frequency/Retinex shallow branch` 作为低对比增强支线，不抢主线。
8. 只有单项通过 gate 后，才允许进入 `S7_6` 组合候选与 `S7_7` corrected multi-seed full re-inference。
9. `S7-B` 的 `E24_full`、`E15/E16/E19`、`E18_full`、`E20_full` 和 `E21` 只有在 S7-A 产生有效候选后才允许启动。

## 目录规范

- 新实验输出统一写入 `results/val/<experiment_id>.../`。
- 训练日志统一写入 `results/val/logs/`。
- shell 脚本统一放在 `scripts/train/`。
- 实验总表为 `results/val/dark_small_experiment_leaderboard.md`。
- 根目录不再保留 `logs/`、`old/`、`experiments/` 作为运行路径。
- 子目录不再放 `perface.md`，只在主要一级目录放说明文件。
