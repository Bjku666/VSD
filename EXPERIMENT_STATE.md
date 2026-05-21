# VSD Experiment State

## 当前阶段

S5：当前阶段，E6 主线确立后的误报-召回诊断优化阶段

## 当前任务

S5 当前执行：E20_0 差异诊断 / E18_1-E18_2 E6 多 seed / E12_1b 弱残差门控 / E13_loss_check / E22_0 hard negative taxonomy

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
| E10_2 | done | E6 done | results/val/e10_2_e6_768_val/ | partial | E6 768；双卡完成，mAP50-95=0.640562，AP_tiny=0.557324，AP_low-contrast=0.641939，但 AP_dark-small=0.509453 低于 E6 640，FP/image=1.488087 高于 E6 640；不解锁 E10_3/E10_4 |
| E10_3 | paused | E10_2 clearly better than E6 640 | results/val/e10_3_e6_768_small_x3_val/ |  | 暂停；只有 E10_2 明确优于 E6 640 后再考虑 small resampling |
| E10_4 | paused | E10_2 clearly better than E6 640 | results/val/e10_4_e6_768_darksmall_x3_val/ |  | 暂停；只有 E10_2 明确优于 E6 640 后再考虑 dark-small resampling |
| E11_1 | done | E6 done | results/val/e11_1_e6_p2_head_val/ | no | E6 + P2 detection head；mAP50-95=0.623174，AP_dark-small=0.497853，AP_tiny=0.540172，FP/image=1.194690；误报下降但 AP 明显低于 E6，因此暂停 E11_2/E11_3 |
| E11_2 | paused | E11_1 beats E6 on AP_dark-small | results/val/e11_2_e6_p2_aux_head_val/ |  | 暂停；E11_1 未超过 E6，暂不继续普通 P2 扩展 |
| E11_3 | paused | E11_1 beats E6 on AP_dark-small | results/val/e11_3_e6_p2_weighted_val/ |  | 暂停；E11_1 未超过 E6，暂不继续普通 P2 扩展 |
| E12_1 | done | E6 done / E11_1 reviewed | results/val/e12_1_residual_gated_fusion_val/ | partial | E6 + residual gated fusion；mAP50-95=0.635959，AP_dark=0.589785，AP_dark-small=0.506156，AP_tiny=0.551706，FP/image=1.278421，FPPI_dark=2.159091；误报明显低于 E6，但 dark-small AP 低于 E6，因此暂停 E12_2/E12_3/E12_4，转入 E13 loss |
| E12_1b | pending_S5_3 | E20_0 done / E18_1-E18_2 reviewed | results/val/e12_1b_weak_residual_gated_fusion_val/ |  | E6 + weak residual gated fusion；identity-preserving residual gate，lambda=0.1；目标 AP_dark-small >= E6 且 FP/image < E6 |
| E12_1c | queued_later | E12_1b reviewed | results/val/e12_1c_identity_bias_residual_gate_val/ |  | E6 + residual gated fusion with identity bias；仅在 E12_1b 显示方向有效后启动 |
| E12_1d | queued_later | E12_1b/E20_0 reviewed | results/val/e12_1d_target_preserving_gate_val/ |  | E6 + target-preserving gate；仅在差异诊断定位被压掉的 dark-small TP 后启动 |
| E12_2 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_2_spatial_gate_val/ |  | 暂停普通 spatial gate 扩展；优先弱残差/保目标门控 |
| E12_3 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_3_dark_aware_reliability_gate_val/ |  | 暂停普通 dark-aware gate 扩展；优先弱残差/保目标门控 |
| E12_4 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_4_residual_gate_p2_val/ |  | 暂停普通 E12+P2 扩展；P2 路线需重新设计为低权重/辅助头 |
| E13_2 | done | E6 done / E12_1 reviewed | results/val/e13_2_e6_scale_aware_loss_val/ | partial | E6 + scale-aware loss；mAP50-95=0.631939，AP_dark-small=0.508839，AP_tiny=0.555102，AP_low-contrast=0.634011，FP/image=1.226685，FPPI_dark=2.068182；误报下降，但 AP_dark-small 和总体 mAP 低于 E6，因此不作为主线提升 |
| E13_3 | done | E13_2 reviewed / center-aware loss smoke passed | results/val/e13_3_e6_center_aware_loss_val/ | partial | E6 + center-aware loss；mAP50-95=0.631939，AP_dark-small=0.508839，AP_tiny=0.555102，AP_low-contrast=0.634011，FP/image=1.226685，FPPI_dark=2.068182；结果与 E13_2 一致，误报下降但 AP_dark-small 和总体 mAP 低于 E6 |
| E13_4 | paused_non_mainline | E13_loss_check done | results/val/e13_4_e6_scale_center_aware_loss_val/ |  | dry-run 已通过但不作为当前主线；先检查 E13_2/E13_3 loss 路径、权重和作用范围 |
| E13_loss_check | pending_S5_4 | E13_2/E13_3 done | results/val/e13_loss_check/ |  | 不训练；检查 E13_2/E13_3 是否真的不同、loss 权重是否过强、是否作用于所有 box、是否影响分类置信度分布 |
| E13_2b | queued_later | E13_loss_check done | results/val/e13_2b_tiny_dark_scale_loss_val/ |  | scale-aware loss 只作用 tiny + dark-small，轻量辅助项 |
| E13_3b | queued_later | E13_loss_check done | results/val/e13_3b_tiny_dark_center_loss_val/ |  | center-aware loss 只作用 tiny + dark-small，轻量辅助项 |
| E13_4b | queued_later | E13_loss_check done | results/val/e13_4b_tiny_dark_scale_center_w005_val/ |  | scale+center auxiliary loss，loss_weight=0.05 |
| E13_4c | queued_later | E13_loss_check done | results/val/e13_4c_tiny_dark_scale_center_w010_val/ |  | scale+center auxiliary loss，loss_weight=0.10 |
| E18_1 | pending_S5_2 | E6 done | results/val/e18_1_e6_seed1_val/ |  | E6 seed=1；用于判断 E6 是否稳定优于 E12_1/E13_2 |
| E18_2 | pending_S5_2 | E6 done | results/val/e18_2_e6_seed2_val/ |  | E6 seed=2；与 E6 seed=0 汇总 mean ± std |
| E18_3 | queued_later | E18_1-E18_2 reviewed | results/val/e18_3_e12_1_seed1_val/ |  | 可选：若 E6 std 较大，再补 E12_1 seed=1 |
| E18_4 | queued_later | E18_1-E18_2 reviewed | results/val/e18_4_e13_2_seed1_val/ |  | 可选：若 E6 std 较大，再补 E13_2 seed=1 |
| E20_0 | pending_S5_1 | E6/E12_1/E13_2 predictions available | results/val/e20_0_error_delta_analysis/ |  | E6 vs E12_1 vs E13_2 差异诊断；输出 E6 TP 但 E12_1/E13_2 FN、被 E12_1/E13_2 消除的 FP，以及 dark/tiny/low-contrast/class 统计 |
| E22_0 | pending_S5_5 | E20_0 done | results/val/e22_0_hard_negative_taxonomy/ |  | 基于 E6/E12_1/E13_2 FP 构建 hard negative taxonomy 和 list；暂不训练 3x/5x |

## 当前结论与下一步

- E6 multi-scale fusion 是当前主线基线：mAP50-95=0.635715，AP_small=0.586603，Recall_small=0.748135，AP_dark=0.585551，AP_dark-small=0.512464。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- 暂停继续 WBF 权重搜索、IR-only 960/resampling 扩展、RT-DETR/YOLOv10/YOLO11s 强模型对照。
- A1/A2/A3 已完成：E5/E6 已补 FP/image、FPPI_dark、FPPI_low-contrast、AP_tiny、Recall_tiny、AP_low-contrast、Recall_low-contrast、Params、GFLOPs、FPS、GPU memory。E10_2 已完成：总体 mAP、AP_tiny、AP_low-contrast 有小幅提升，但 dark-small 主指标和误报不优于 E6 640，不解锁 E10_3/E10_4。E11_1 已完成：误报下降但 AP_dark-small、AP_tiny 和总体 mAP 均低于 E6，因此暂停 E11_2/E11_3。E12_1 已完成：误报下降明显，但 AP_dark-small 低于 E6，因此暂停普通 E12_2/E12_3/E12_4，改做 E12_1b 弱残差保目标门控。E13_2/E13_3 已完成：均误报下降，但 AP_dark-small 和总体 mAP 低于 E6，且结果一致；先做 E13_loss_check，不把 E13_4 作为当前主线。仍待补 E6 seed=1,2。


## 实验阶段总览

| 大阶段 | 阶段名称 | 实验范围 | 当前状态 | 阶段结论 |
| --- | --- | --- | --- | --- |
| S0 | 数据协议阶段 | E0 | done | DroneVehicle-DarkSmall 协议、子集和指标口径已固定 |
| S1 | 基础基线阶段 | E1-E4 | done | E2 是暗弱支撑基线，E4 是低误报 WBF 参考线 |
| S2 | 融合主线确认阶段 | E5-E6 | done | E6 multi-scale fusion 确认为当前主线基线 |
| S3 | 高分辨率/后融合/重采样初筛 | E7、E8、E10_2 | done / paused | 高分辨率、WBF 权重搜索和 IR 重采样暂不作为主线；E10_3/E10_4 暂停 |
| S4 | 小目标头/门控/loss 初筛 | E11_1、E12_1、E13_2、E13_3 | done / partial | P2、gate、loss 都能降低 FP，但 AP_dark-small 低于 E6 |
| S5 | 当前阶段：诊断优化阶段 | E20_0、E18_1/E18_2、E12_1b、E13_loss_check、E22_0 | next | 分析 E6 与 E12/E13 差异，寻找保 AP 降 FP 的改法 |
| S6 | 方法二次优化阶段 | E12_1b 正式版、E13_2b/E13_3b、E22_2、E14 | not started | 形成最终方法候选 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 | not started | 多 seed、效率、强模型、test set 和复现冻结 |

当前进度：S4 已完成，S5 刚开始。现在不是继续普通模块扩展，而是先诊断 E6 的高 AP 与 E12/E13 的低误报之间差在哪里。

## 当前阶段执行顺序

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| E6/E12/E13 差异诊断 | E20_0 | 是 | 找出 E12/E13 压掉了哪些 TP，消除了哪些 FP |
| E6 多 seed | E18_1 / E18_2 | 是 | 验证 E6 是否稳定优于 E12_1/E13_2 |
| 弱残差门控 | E12_1b | 是 | 继承 E12 降 FP，但保留 E6 AP_dark-small |
| loss 实现检查 | E13_loss_check | 是 | 检查 E13_2/E13_3 是否真正不同，确认 loss 权重和作用范围 |
| hard negative taxonomy | E22_0 | 是 | 分类 E6/E12/E13 的误报类型，暂不训练 3x/5x |
| scale+center loss | E13_4 | 暂缓 | 不作为当前主线，先完成 E13_loss_check |
| P2 普通扩展 | E11_2 / E11_3 | 暂停 | E11_1 已低于 E6 |
| 普通门控扩展 | E12_2 / E12_3 / E12_4 | 暂停 | 先做 E12_1b 弱门控 |
| E6 768 重采样扩展 | E10_3 / E10_4 | 暂停 | E10_2 没超过 E6 640 |
| 强模型/test | E15 / E21 | 禁止 | 还没到论文最终验证阶段 |

当前阶段目标：保留 E6 的 AP_dark-small，同时继承 E12/E13 的低误报优势。

## Demo 脚本

- 全量编号 demo 入口：`scripts/train/run_dark_small_experiment_demos.sh`，覆盖 E0_1-E24_3，默认 dry-run。
- 当前 E6 主线快捷 demo：`scripts/train/run_e6_mainline_demos.sh`。
- 已实现的 manifest 实验可用 `RUN_MODE=run scripts/train/run_e6_mainline_demos.sh E13_2` 或 runner 直接启动；未实现的后续分支只输出编号一致的命令模板，不会误跑 test set。

## 状态更新规则

- 每完成一个实验，先保存日志，再更新状态表。
- 实验失败必须标记为 failed，并保留错误日志。
- 只有验收通过后才能改为 done。
