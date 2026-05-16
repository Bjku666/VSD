# VSD Experiment State

## 当前阶段
阶段 0：协议固定

## 当前任务
E1 RGB-only baseline

## 全局约束
- 不允许删除 data/DroneVehicle/raw/
- 不允许删除 weights/pretrained/
- 不允许引用旧 weights/trained/
- 所有阈值只允许来自 train split
- 不允许在 test 上调参
- 第一轮只允许执行 E0-E4

## 任务状态表

| 实验 | 状态 | 依赖 | 输出目录 | 是否通过 | 备注 |
| --- | --- | --- | --- | --- | --- |
| E0 | done | raw data | prepared/, results/dataset_audit/ |  | 协议审计、裁边、阈值、子集统计 |
| E1 | running | E0 done | results/val/e1_yolo11n_rgb_only_640_ddp/ |  | RGB-only baseline |
| E2 | running | E0 done | results/val/e2_yolo11n_ir_only_640_ddp/ |  | IR-only baseline |
| E3 | blocked | E1/E2 best.pt | results/val/e3_late_fusion_nms_val/ |  | Late Fusion + NMS |
| E4 | blocked | E1/E2 best.pt | results/val/e4_late_fusion_wbf_val/ |  | Late Fusion + WBF |

## 状态更新规则
- 每完成一个实验，先保存日志，再更新状态表。
- 实验失败必须标记为 failed，并保留错误日志。
- 只有验收通过后才能改为 done。
