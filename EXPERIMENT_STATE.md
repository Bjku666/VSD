# VSD Experiment State

## 当前阶段

阶段 4：E6 主线补评估与小目标增强

## 当前任务

E6 multi-scale fusion 主线补评估 / E10_2 E6 768 / E11-E12 小目标与可靠性融合

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
| E1 | done | E0 done | results/val/e1_yolo11n_rgb_only_640_ddp/ |  | RGB-only baseline |
| E2 | done | E0 done | results/val/e2_yolo11n_ir_only_640_ddp/ |  | IR-only baseline |
| E3 | done | E1/E2 best.pt | results/val/e3_late_fusion_nms_val/ |  | Late Fusion + NMS |
| E4 | done | E1/E2 best.pt | results/val/e4_late_fusion_wbf_val/ |  | Late Fusion + WBF |
| E5 | done | E3/E4 done | results/val/yolo11n_e5_rgb_ir_640_ddp/ | yes | Single-layer fusion；补评估完成：mAP50-95=0.624957，AP_dark-small=0.504808，AP_tiny=0.538322，AP_low-contrast=0.624837，AP_dark-small_object=0.097728，FP/image=1.584071，FPPI_dark=2.704545 |
| E6 | done | E5 best.pt | results/val/yolo11n_e6_rgb_ir_640_ddp/ | yes | Multi-scale fusion；补评估完成：mAP50-95=0.635715，AP_dark-small=0.512464，AP_tiny=0.555855，AP_low-contrast=0.637506，AP_dark-small_object=0.100028，FP/image=1.469027，FPPI_dark=2.536932 |
| E7_1 | done | E2 best.pt / E0 done | results/val/e7_1_ir_only_768_gpu0/ |  | IR-only 768；统一验证已完成，required_metrics.json 已生成并归档到 results/val/e7_1_ir_only_768_val/ |
| E7_2 | done | E7_1 validation reviewed | results/val/e7_2_ir_only_960/ |  | IR-only 960；统一验证已完成，AP_dark-small 未超过 E2，因此转入 E8_2 |
| E7_3 | pending | E1 done | results/val/e7_3_rgb_only_768/ |  | RGB-only 768；优先级低于 IR 高分辨率和 WBF 搜索 |
| E7_5 | done | E1/E2 best.pt | results/val/e7_5_wbf_rgb03_ir07/ | no | WBF RGB:IR=0.3:0.7；AP_dark-small=0.472421，低于 E7_4/E2 |
| E7_6 | skipped | E4 done | results/val/e7_6_wbf_rgb05_ir05/ |  | 与 E4 等权 WBF 口径重复，跳过 |
| E8_2 | done | E7_2 done / dark-small subset | results/val/e8_2_ir_only_768_darksmall_x3/ | no | IR-only 768 + dark-small 3x 重采样；统一验证已完成，AP_dark-small=0.499926，未超过 E2 |
| E10_1 | stopped_deprioritized | E5 done | results/val/e10_1_e5_768/ | no | 发现仍有 E5 768 训练进程后已停止；目录仅有 args.yaml、labels.jpg 和空 weights/，无 results/metrics，不计入有效实验 |
| E10_2 | running | E6 done | results/val/e10_2_e6_768_val/ |  | E6 768；后台运行中，PID=10410，log=results/val/logs/e10_2_e6_768_20260519_161657.log |
| E10_3 | paused | E10_2 clearly better than E6 640 | results/val/e10_3_e6_768_small_x3_val/ |  | 暂停；只有 E10_2 明确优于 E6 640 后再考虑 small resampling |
| E10_4 | paused | E10_2 clearly better than E6 640 | results/val/e10_4_e6_768_darksmall_x3_val/ |  | 暂停；只有 E10_2 明确优于 E6 640 后再考虑 dark-small resampling |

## 当前结论与下一步

- E6 multi-scale fusion 是当前主线基线：mAP50-95=0.635715，AP_small=0.586603，Recall_small=0.748135，AP_dark=0.585551，AP_dark-small=0.512464。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 暂停继续 WBF 权重搜索、IR-only 960/resampling 扩展、RT-DETR/YOLOv10/YOLO11s 强模型对照。
- A1/A2/A3 已完成：E5/E6 已补 FP/image、FPPI_dark、FPPI_low-contrast、AP_tiny、Recall_tiny、AP_low-contrast、Recall_low-contrast、Params、GFLOPs、FPS、GPU memory。E6 在 AP 与误报上均优于 E5。下一步按顺序补 seed=1,2；再跑 E10_2；再做 E11-1 P2 head、E12-1 residual gated fusion。object-level dark-small AP 已补：E5 AP_dark-small_object=0.097728、Recall=0.505895；E6 AP_dark-small_object=0.100028、Recall=0.657039。下一步仍待补 seed=1,2。

## 状态更新规则

- 每完成一个实验，先保存日志，再更新状态表。
- 实验失败必须标记为 failed，并保留错误日志。
- 只有验收通过后才能改为 done。
