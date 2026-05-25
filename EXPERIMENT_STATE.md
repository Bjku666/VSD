# VSD Experiment State

## 当前阶段

S6：诊断驱动的目标保持型背景抑制阶段，围绕 object-level evaluator、seed 核查、hard negative 定向优化、E13_3b-light 和 CEBS 候选推进

## 当前任务

S6 当前只允许执行：E23 / E18_check / E22_1 / E13_3b-light / E22_2a / E22_2b / E14_1 / E14_2 / E14_3 / E14_4 / E24_0

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
| E12_1b | done | E20_0 done / E18_1-E18_2 reviewed | results/val/e12_1b_weak_residual_gated_fusion_val/ | no | E6 + weak residual gated fusion；mAP50-95=0.632290，AP_dark-small=0.503857，AP_tiny=0.549441，AP_low-contrast=0.633238，FP/image=1.257318；误报下降但 AP_dark-small 低于 E6 seed0 和 E18_1，因此不解锁 E12_1c/E12_1d |
| E12_1c | paused | E12_1b reviewed | results/val/e12_1c_identity_bias_residual_gate_val/ |  | 暂停；E12_1b 已显示弱残差门控降 FP 但 AP_dark-small 明显低于 E6 和 E13_3b，S6 不继续 gate 扩展 |
| E12_1d | paused | E12_1b/E20_0 reviewed | results/val/e12_1d_target_preserving_gate_val/ |  | 暂停；E12_1b 已显示弱残差门控降 FP 但 AP_dark-small 明显低于 E6 和 E13_3b，S6 不继续 gate 扩展 |
| E12_2 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_2_spatial_gate_val/ |  | 暂停普通 spatial gate 扩展；优先弱残差/保目标门控 |
| E12_3 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_3_dark_aware_reliability_gate_val/ |  | 暂停普通 dark-aware gate 扩展；优先弱残差/保目标门控 |
| E12_4 | paused | E12_1 beats E6 on AP_dark-small | results/val/e12_4_residual_gate_p2_val/ |  | 暂停普通 E12+P2 扩展；P2 路线需重新设计为低权重/辅助头 |
| E13_2 | done | E6 done / E12_1 reviewed | results/val/e13_2_e6_scale_aware_loss_val/ | partial | E6 + scale-aware loss；mAP50-95=0.631939，AP_dark-small=0.508839，AP_tiny=0.555102，AP_low-contrast=0.634011，FP/image=1.226685，FPPI_dark=2.068182；误报下降，但 AP_dark-small 和总体 mAP 低于 E6，因此不作为主线提升 |
| E13_3 | done | E13_2 reviewed / center-aware loss smoke passed | results/val/e13_3_e6_center_aware_loss_val/ | partial | E6 + center-aware loss；mAP50-95=0.631939，AP_dark-small=0.508839，AP_tiny=0.555102，AP_low-contrast=0.634011，FP/image=1.226685，FPPI_dark=2.068182；结果与 E13_2 一致，误报下降但 AP_dark-small 和总体 mAP 低于 E6 |
| E13_4 | paused_non_mainline | E13_loss_check done | results/val/e13_4_e6_scale_center_aware_loss_val/ |  | 暂停；E13_4b/E13_4c 已显示 scale+center 组合未保住低误报优势，S6 不继续该方向扩展 |
| E13_loss_check | done | E13_2/E13_3 done | results/val/e13_loss_check/ | yes | 不训练；已确认 E13_2/E13_3 关键指标完全一致，loss 代码分支存在但辅助项作用于所有 foreground boxes，后续应缩小到 tiny/dark-small 并降低权重 |
| E13_2b | done | E13_loss_check done / E12_1b reviewed | results/val/e13_2b_tiny_dark_scale_loss_val/ | partial | small-scoped scale-aware loss；mAP50-95=0.631207，AP_dark-small=0.512472，AP_tiny=0.551849，AP_low-contrast=0.631612，FP/image=1.322668，FPPI_dark=2.244318；AP_dark-small 基本回到 E6 seed0，误报低于 E6 |
| E13_3b | done | E13_loss_check done / E13_2b reviewed | results/val/e13_3b_tiny_dark_center_loss_val/ | yes | small-scoped center-aware loss；mAP50-95=0.635858，AP_dark-small=0.513947，AP_tiny=0.558752，AP_low-contrast=0.637488，FP/image=1.285909，FPPI_dark=2.090909；当前最优候选，AP 和 FP 均优于 E6 seed0 |
| E13_3b-light | done_not_candidate | E23 / E18_check / E22_1 | results/val/e13_3b_light_target_center_loss_val/ | no | 训练和验证完成，VALIDATE_EXIT 0；mAP50-95=0.636497，AP_dark-small=0.509436，AP_tiny=0.553970，AP_low-contrast=0.638198，但 FP/image=1.664398、FPPI_dark=2.923295 明显升高；object-level AP_dark-small=0.095211，低于 E6 的 0.100028 |
| E13_4b | done | E13_3b done | results/val/e13_4b_tiny_dark_scale_center_w005_val/ | partial | small-scoped scale+center loss，aux_weight=0.05；mAP50-95=0.630100，AP_dark-small=0.517772，AP_tiny=0.547149，AP_low-contrast=0.632103，FP/image=1.671886，FPPI_dark=3.068182；AP_dark-small 高于 E6 seed0，但误报明显升高，不作为低误报最优候选 |
| E13_4c | done | E13_4b finished | results/val/e13_4c_tiny_dark_scale_center_w010_val/ | no | small-scoped scale+center loss，aux_weight=0.10；mAP50-95=0.635766，AP_dark-small=0.511115，AP_tiny=0.552800，AP_low-contrast=0.636350，FP/image=1.674609，FPPI_dark=3.017045；未优于 E13_3b，且误报明显高于 E6 |
| E18_1 | done | E6 done | results/val/e18_1_e6_seed1_val/ | yes | E6 seed=1；mAP50-95=0.636696，AP_dark-small=0.519368，AP_tiny=0.553853，AP_low-contrast=0.636971，FP/image=1.413887 |
| E18_2 | done | E6 done | results/val/e18_2_e6_seed2_val/ | yes | E6 seed=2；mAP50-95=0.637496，AP_dark-small=0.508066，AP_tiny=0.550223，AP_low-contrast=0.638622，FP/image=1.440436 |
| E18_3 | queued_later | E18_1-E18_2 reviewed | results/val/e18_3_e12_1_seed1_val/ |  | 可选：若 E6 std 较大，再补 E12_1 seed=1 |
| E18_4 | queued_later | E18_1-E18_2 reviewed | results/val/e18_4_e13_2_seed1_val/ |  | 可选：若 E6 std 较大，再补 E13_2 seed=1 |
| E18_5 | done | E13_3b reviewed / GPU0 free | results/val/e18_5_e13_3b_seed1_val/ | partial | E13_3b seed=1；mAP50-95=0.634911，AP_dark-small=0.506625，AP_tiny=0.555215，AP_low-contrast=0.635295，FP/image=1.351259，FPPI_dark=2.352273 |
| E18_6 | done_rerun | E13_3b reviewed / GPU1 free | results/val/e18_6_e13_3b_seed2_rerun_val/ | partial | E13_3b seed=2 重跑已完成并验证，VALIDATE_EXIT 0；mAP50-95=0.634911，AP_dark-small=0.506625，AP_tiny=0.555215，AP_low-contrast=0.635295，FP/image=1.351259，FPPI_dark=2.352273 |
| E18_check | done_invalid | E18_5/E18_6 done | results/val/e18_check_e13_3b_seed_integrity/ | no | S6-2 完成：seed/目录/weight hash 不同，但关键 required_metrics 逐项完全一致；E13_3b multi-seed 标记 invalid_requires_seed2_rerun，需重跑 seed=2 后再使用均值 |
| E20_0 | done | E6/E12_1/E13_2 predictions available | results/val/e20_0_error_delta_analysis/ | yes | E6 vs E12_1 vs E13_2 差异诊断完成：conf=0.25、IoU=0.50；E6 TP=21011/FP=2753/FN=1451，E12_1 消除 E6 FP 660 个但漏掉 E6 TP 399 个，E13_2 消除 E6 FP 677 个但漏掉 E6 TP 407 个 |
| E22_0 | done | E20_0 done | results/val/e22_0_hard_negative_taxonomy/ | yes | hard negative taxonomy 完成：FP 共 7713 条，background_far=3393，class_confusion=3127，localization_error=918，duplicate_or_conf_threshold=223，near_object_background=52；暂不训练 3x/5x |
| E22_1 | done | E22_0 done | results/val/e22_1_hard_negative_lists/ | yes | S6-3 完成：输出 5 类 hard negative TSV 列表；background_far=3393 条且唯一 train-allowed，class_confusion/localization_error 等仅诊断导出 |
| E22_2a | blocked | E18_check valid / E22_1 done / train-split HN source | results/val/e22_2a_e13_3b_light_background_far_hn15_val/ |  | S6-5：E13_3b-light + background_far hard negative 1.5x；当前 E18_check invalid，且 E22_1 源于 val FP，不能直接训练，需重跑 seed=2 并准备 train-split HN |
| E22_2b | blocked | E18_check valid / E22_1 done / train-split HN source | results/val/e22_2b_e13_3b_light_background_far_hn2_val/ |  | S6-5：E13_3b-light + background_far hard negative 2x；不允许 3x/5x，不混入 class_confusion |
| E23 | done | E6 weights / train-only thresholds | results/val/e23_object_level_evaluator/ | yes | S6-1 完成：E6 object-level metrics 输出；AP_dark-small_object=0.100028，Recall_dark-small_object=0.657039，AP_tiny_object=0.054049，Recall_tiny_object=0.500678，AP_low-contrast_object=0.246427，Recall_low-contrast_object=0.725633 |
| E14_1 | train_done_val_wait_gpu1 | E23 done / CEBS implementation | results/val/e14_1_e6_cebs_a005_val/ |  | E6 + CEBS alpha=0.05；训练已完成，但原链式验证 shell 判断失败未写 required_metrics；已排队 GPU1 空闲后补跑验证，实时日志：results/val/logs/e14_1_e6_cebs_a005_manual_val_gpu1_20260525_1231.log |
| E14_2 | image_done_object_wait_gpu0 | E14_1 reviewed | results/val/e14_2_e6_cebs_a010_val/ | partial | E6 + CEBS alpha=0.10；image-level 验证完成：mAP50-95=0.633557，AP_dark-small=0.505991，AP_tiny=0.554227，AP_low-contrast=0.633506，FP/image=1.238257，FPPI_dark=2.059659；已排队 GPU0 空闲后补跑 object-level，日志：results/val/logs/e14_2_e6_cebs_a010_object_eval_gpu0_20260525_1231.log |
| E14_3 | planned | E13_3b-light done / E14_1 reviewed | results/val/e14_3_e13_3b_light_cebs_a005_val/ |  | E13_3b-light + CEBS alpha=0.05 |
| E14_4 | planned | E14_3 reviewed / train-split background_far HN | results/val/e14_4_e13_3b_light_cebs_hn15_val/ |  | E13_3b-light + CEBS + background_far HN 1.5x；仅在 B/C 候选需要时执行 |
| E24_0 | next | E13_3b reviewed | results/val/e24_0_e13_3b_candidate_freeze/ |  | S6-4：冻结 E13_3b candidate config；记录 configs/frozen/e13_3b_candidate.yaml、seed0/1/2 权重路径、metrics、protocol version、commit hash、training args 和 evaluation args |

## 当前结论与下一步

- E6 multi-scale fusion 是当前主线基线：mAP50-95=0.635715，AP_small=0.586603，Recall_small=0.748135，AP_dark=0.585551，AP_dark-small=0.512464。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- S5 done：E20_0、E22_0、E13_loss_check、E18 多 seed、E12_1b 均已完成。普通 gate / P2 / 全局 loss 不再继续；主要 FP 来源为 background_far、class_confusion、localization_error。
- E13_3b 是当前最有希望的低误报候选：单 seed 同时做到 AP_dark-small 略高于 E6、FP/image 明显低于 E6、FPPI_dark 明显低于 E6。但 E13_3b 三 seed 汇总为 mAP50-95=0.635227±0.000547，AP_dark-small=0.509066±0.004227，AP_tiny=0.556394±0.002042，FP/image=1.329476±0.037730，FPPI_dark=2.265152±0.150898；稳定降低误报，但 AP_dark-small 均值低于 E6 三 seed 均值。
- S6 running：E23 object-level evaluator 已补齐，E18_check 已判定旧 E13_3b seed=1/2 指标完全重复，E22_1 已输出 per-taxonomy hard negative list；E18_6 seed=2 已完成单卡重跑但指标仍与 seed=1 完全一致，需要继续标记为 suspicious；E13_3b-light 已完成但 FP 显著升高且 object-level AP_dark-small 低于 E6，不作为候选；E14_2 alpha=0.10 已完成 image-level，降低 FP 但 AP_dark-small 低于 E6/E13_3b；E14_1 验证和 E14_2 object-level 已排队等待 GPU 空闲。
- 暂停 E12_1c/E12_1d、E13_4b/E13_4c 后续扩展、E11_2/E11_3、E10_3/E10_4、E15 强模型对照和 E21 test set；不运行 test set，不启动 RT-DETR / YOLOv10 / YOLO11s，不训练 hard negative 3x/5x。


## 实验阶段总览

| 大阶段 | 阶段名称 | 实验范围 | 当前状态 | 阶段结论 |
| --- | --- | --- | --- | --- |
| S0 | 数据协议阶段 | E0 | done | DroneVehicle-DarkSmall 协议、子集和指标口径已固定 |
| S1 | 基础基线阶段 | E1-E4 | done | E2 是暗弱支撑基线，E4 是低误报 WBF 参考线 |
| S2 | 融合主线确认阶段 | E5-E6 | done | E6 multi-scale fusion 确认为当前主线基线 |
| S3 | 高分辨率/后融合/重采样初筛 | E7、E8、E10_2 | done / paused | 高分辨率、WBF 权重搜索和 IR 重采样暂不作为主线；E10_3/E10_4 暂停 |
| S4 | 小目标头/门控/loss 初筛 | E11_1、E12_1、E13_2、E13_3 | done / partial | P2、gate、loss 都能降低 FP，但 AP_dark-small 低于 E6 |
| S5 | 诊断优化阶段 | E20_0、E18_1/E18_2、E12_1b、E13_loss_check、E22_0 | done | 找到主要 FP 类型，确认 E13_3b 是当前最佳低误报候选但 AP_dark-small 三 seed 稳定性仍不足 |
| S6 | 诊断驱动的目标保持型背景抑制阶段 | E23、E18_check、E22_1、E13_3b-light、E22_2a/b、E14、E24_0 | running | object-level 口径已落地；E13_3b 多 seed 当前 invalid；下一步围绕 target-scoped light loss、train-split HN 和 CEBS 推进 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 full | not started | 暂不进入 test / 强模型 |

当前进度：S5 诊断完成，进入 S6。现在不是继续普通 YOLO 模块扩展，而是围绕 object-level 口径、误报 taxonomy、target-scoped light loss 和目标保持型背景抑制做闭环。

## 当前阶段执行顺序

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| object-level evaluator | E23 | 已完成 | 输出 E6 的 AP/Recall_dark-small_object、tiny_object、low-contrast_object，并生成 scope 对比 |
| E13_3b seed 独立性核查 | E18_check | 已完成 | 关键指标完全一致，multi-seed invalid，需要重跑 seed=2 |
| hard negative list 构建与去重 | E22_1 | 已完成 | 只构建列表，不训练；background_far 是唯一 train-allowed taxonomy |
| E13_3b-light | E13_3b-light | 已完成，不作为候选 | FP/image 与 FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 1.5x | E22_2a | blocked | 需 E18_check valid 且使用 train-split HN 来源，不能直接用 val FP 训练 |
| background_far hard negative 2x | E22_2b | blocked | 同上，只允许 background_far，不允许 3x/5x |
| CEBS alpha=0.05/0.10 | E14_1/E14_2 | 等待 GPU/部分完成 | E14_1 已完成训练并等待 GPU1 补验证；E14_2 已完成 image-level 并等待 GPU0 object-level |
| CEBS 组合候选 | E14_3/E14_4 | 待 E13_3b-light 后执行 | 不强行使用 CEBS；若不超过 B 候选则作为讨论实验 |
| candidate freeze | E24_0 | 待候选有效后执行 | 冻结配置、权重路径、metrics、protocol、commit、训练和验证参数 |
| scale+center loss 后续扩展 | E13_4b / E13_4c | 暂停 | 误报明显升高，不作为当前最佳方向 |
| P2 普通扩展 | E11_2 / E11_3 | 暂停 | E11_1 已低于 E6 |
| 普通门控扩展 | E12_1c / E12_1d / E12_2 / E12_3 / E12_4 | 暂停 | gate 路线降 FP 但伤 AP_dark-small，S6 不继续 |
| E6 768 重采样扩展 | E10_3 / E10_4 | 暂停 | E10_2 没超过 E6 640 |
| 强模型/test | E15 / E21 | 禁止 | 不运行 test set，不启动 RT-DETR / YOLOv10 / YOLO11s |
| hard negative 3x/5x | E22_2 3x/5x variants | 禁止 | class_confusion 占比高，盲目强采样可能压掉真实车辆目标 |

当前阶段目标：确认 E13_3b 复现独立性，在保住 AP_dark-small 的前提下定向降低 background_far 带来的 FP/image 和 FPPI_dark。

## Demo 脚本

- 全量编号 demo 入口：`scripts/train/run_dark_small_experiment_demos.sh`，覆盖 E0_1-E24_3，默认 dry-run。
- 当前 E6 主线快捷 demo：`scripts/train/run_e6_mainline_demos.sh`。
- 已实现的 manifest 实验可用 `RUN_MODE=run scripts/train/run_e6_mainline_demos.sh E13_2` 或 runner 直接启动；未实现的后续分支只输出编号一致的命令模板，不会误跑 test set。

## 状态更新规则

- 每完成一个实验，先保存日志，再更新状态表。
- 实验失败必须标记为 failed，并保留错误日志。
- 只有验收通过后才能改为 done。
