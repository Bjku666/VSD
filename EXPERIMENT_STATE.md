# VSD Experiment State

## 当前阶段

S6.5：诊断驱动的分类混淆修正与候选可靠性校准阶段；第一批仅执行 E25_0 / E25_1_full / E26_1_full，完成后停止

## 当前任务

S6.5 seed 独立性修复重跑已完成：旧 E18/E25_0 多 seed 产物已确认模型张量/指标存在异常，不再作为有效多 seed 结论。2026-05-27 已修复 E6/E13 trainer 的 dataloader seed 逻辑，并以 batch=48 完成 corrected multi-seed 队列。GPU0 的 E6 seed0/1/2 已完成训练、验证和导出；E13_3b seed0/1/2 已由另一台服务器接管并完成收尾，本机已加阻断保护避免重复训练；GPU1 的 E25_0 seed42/43/44 已完成训练、验证和导出。所有新产物写入 [results/val/seedfix_e6_seed0_seedfix_b48_20260527/](results/val/seedfix_e6_seed0_seedfix_b48_20260527/) 、[results/val/seedfix_e6_seed1_seedfix_b48_20260527/](results/val/seedfix_e6_seed1_seedfix_b48_20260527/) 、[results/val/seedfix_e6_seed2_seedfix_b48_20260527/](results/val/seedfix_e6_seed2_seedfix_b48_20260527/) 、[results/val/seedfix_e13_3b_seed0_seedfix_b48_20260527/](results/val/seedfix_e13_3b_seed0_seedfix_b48_20260527/) 、[results/val/seedfix_e13_3b_seed1_seedfix_b48_20260527/](results/val/seedfix_e13_3b_seed1_seedfix_b48_20260527/) 、[results/val/seedfix_e13_3b_seed2_seedfix_b48_20260527/](results/val/seedfix_e13_3b_seed2_seedfix_b48_20260527/) 、[results/val/seedfix_e25_0_e13_3b_seed42_seedfix_b48_20260527/](results/val/seedfix_e25_0_e13_3b_seed42_seedfix_b48_20260527/) 、[results/val/seedfix_e25_0_e13_3b_seed43_seedfix_b48_20260527/](results/val/seedfix_e25_0_e13_3b_seed43_seedfix_b48_20260527/) 、[results/val/seedfix_e25_0_e13_3b_seed44_seedfix_b48_20260527/](results/val/seedfix_e25_0_e13_3b_seed44_seedfix_b48_20260527/) 新目录；旧 invalid 输出目录和旧日志已按用户要求删除。
2026-05-27 19:58：GPU1 E25_0 seed42 队列曾停在 epoch 52/100，已改为显式 `--resume-path weights/last.pt` 续跑，避免从头覆盖；后续 seed43/44 仍由同一 GPU1 队列顺序执行。
E26_2a 曾被重启为高吞吐版本（batch=48、workers=8），日志入口为 [results/val/logs/e26_2a_class_confusion_cls125_b48_20260526.log](results/val/logs/e26_2a_class_confusion_cls125_b48_20260526.log)。在新 S6.5 gate 下，该任务属于第二批，第一批 full 版完成前不得作为当前阶段推进依据；不主动终止任何用户未授权进程。
2026-05-26 旧第一批流水线曾完成 E25_0 seed42/43/44，但后续审计发现三份 best/last checkpoint 的模型张量完全一致，训练曲线也逐项一致；同时 seed43/44 的早期 GPU1 导出与后续 GPU0 覆盖训练混杂。因此 E25_0 旧输出状态为 seed_pipeline_failed；旧输出和日志已删除，必须等待 seedfix 队列重新产出后再判断。

## 全局约束

- 不允许删除 data/DroneVehicle/raw/
- 不允许删除 weights/pretrained/
- 不允许引用旧 weights/trained/
- 所有阈值只允许来自 train split
- 不允许在 test 上调参
- 不得删除、终止或清理用户未明确授权的进程，只能操作当前任务自己启动的进程
- 可并行任务优先分配到空闲 GPU，不与当前正在训练的任务抢占同一张卡
- 阶段推进必须遵循 AGENT_RUNBOOK 的当前阶段 gate

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
| E7_3 | running | E1 done | results/val/e7_3_rgb_only_768/ |  | RGB-only 768；当前已启动，等待统一验证 |
| E8_1 | running | E7_2 done / E7_3 running | results/val/e8_1_ir_only_768_closemosaic20_val/ |  | IR-only 768 + close_mosaic=20；当前已启动，等待统一验证 |
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
| E18_1 | deleted_invalid_old_seed_logic | E6 done | deleted | no | 旧 E6 seed=1；因 dataloader seed 逻辑已修复，旧产物已删除；新 seedfix 队列重跑 E6 seed0/1/2 |
| E18_2 | deleted_invalid_old_seed_logic | E6 done | deleted | no | 旧 E6 seed=2；因 dataloader seed 逻辑已修复，旧产物已删除；新 seedfix 队列重跑 E6 seed0/1/2 |
| E18_3 | queued_later | E18_1-E18_2 reviewed | results/val/e18_3_e12_1_seed1_val/ |  | 可选：若 E6 std 较大，再补 E12_1 seed=1 |
| E18_4 | queued_later | E18_1-E18_2 reviewed | results/val/e18_4_e13_2_seed1_val/ |  | 可选：若 E6 std 较大，再补 E13_2 seed=1 |
| E18_5 | deleted_invalid_old_seed_logic | E13_3b reviewed / GPU0 free | deleted | no | 旧 E13_3b seed=1；与 seed=2 关键指标完全一致，旧产物已删除；新 seedfix 队列重跑 E13_3b seed0/1/2 |
| E18_6 | deleted_invalid_old_seed_logic | E13_3b reviewed / GPU1 free | deleted | no | 旧 E13_3b seed=2；与 seed=1 关键指标完全一致，旧产物已删除；新 seedfix 队列重跑 E13_3b seed0/1/2 |
| E18_check | deleted_invalid_old_seed_logic | E18_5/E18_6 done | deleted | no | 旧 seed integrity 报告已删除；E13_3b multi-seed 需等待 seedfix 队列重新产出 |
| E20_0 | done | E6/E12_1/E13_2 predictions available | results/val/e20_0_error_delta_analysis/ | yes | E6 vs E12_1 vs E13_2 差异诊断完成：conf=0.25、IoU=0.50；E6 TP=21011/FP=2753/FN=1451，E12_1 消除 E6 FP 660 个但漏掉 E6 TP 399 个，E13_2 消除 E6 FP 677 个但漏掉 E6 TP 407 个 |
| E22_0 | done | E20_0 done | results/val/e22_0_hard_negative_taxonomy/ | yes | hard negative taxonomy 完成：FP 共 7713 条，background_far=3393，class_confusion=3127，localization_error=918，duplicate_or_conf_threshold=223，near_object_background=52；暂不训练 3x/5x |
| E22_1 | done | E22_0 done | results/val/e22_1_hard_negative_lists/ | yes | S6-3 完成：输出 5 类 hard negative TSV 列表；background_far=3393 条且唯一 train-allowed，class_confusion/localization_error 等仅诊断导出 |
| E22_2a | done_not_candidate | E22_1 done / train-split HN source | results/val/e22_2a_e6_background_far_hn15_val/ | no | S6-5：E6 + train-split background_far hard negative 1.5x；训练、统一验证和 object-level 完成；mAP50-95=0.636090，AP_dark-small=0.515064，AP_tiny=0.556958，AP_low-contrast=0.638263，但 FP/image=1.606535、FPPI_dark=2.815341 升高，object-level AP_dark-small=0.096331 低于 E6 的 0.100028，不作为候选 |
| E22_2b | done_not_candidate | E22_1 done / train-split HN source | results/val/e22_2b_e6_background_far_hn2_val/ | no | S6-5：E6 + train-split background_far hard negative 2x；训练、统一验证和 object-level 完成；mAP50-95=0.632997，AP_dark-small=0.521550，AP_tiny=0.552714，AP_low-contrast=0.636456，但 FP/image=1.562968、FPPI_dark=2.843750 高于 E6，object-level AP_dark-small=0.095848 低于 E6 的 0.100028，不作为候选 |
| E23 | done | E6 weights / train-only thresholds | results/val/e23_object_level_evaluator/ | yes | S6-1 完成：E6 object-level metrics 输出；AP_dark-small_object=0.100028，Recall_dark-small_object=0.657039，AP_tiny_object=0.054049，Recall_tiny_object=0.500678，AP_low-contrast_object=0.246427，Recall_low-contrast_object=0.725633 |
| E14_1 | done_not_candidate | E23 done / CEBS implementation | results/val/e14_1_e6_cebs_a005_val/ | no | E6 + CEBS alpha=0.05；image-level：mAP50-95=0.637235，AP_dark-small=0.515630，FP/image=1.328114，FPPI_dark=2.258523；object-level AP_dark-small=0.087590，低于 E6 的 0.100028，因此不作为候选 |
| E14_2 | done_not_candidate | E14_1 reviewed | results/val/e14_2_e6_cebs_a010_val/ | no | E6 + CEBS alpha=0.10；image-level：mAP50-95=0.633557，AP_dark-small=0.505991，FP/image=1.238257，FPPI_dark=2.059659；object-level AP_dark-small=0.098110，仍低于 E6 的 0.100028，因此不作为候选 |
| E14_3 | done_not_candidate | E13_3b-light done / E14_1 reviewed | results/val/e14_3_e13_3b_light_cebs_a005_val/ | no | E13_3b-light + CEBS alpha=0.05；训练、统一验证和 object-level 完成；mAP50-95=0.633003，AP_dark-small=0.502155，FP/image=1.307692，FPPI_dark=2.196023，object-level AP_dark-small=0.087600 低于 E6 的 0.100028，不作为候选 |
| E14_4 | skipped_not_justified | E14_3 reviewed / train-split background_far HN | results/val/e14_4_e13_3b_light_cebs_hn15_val/ | no | E13_3b-light + CEBS + background_far HN 1.5x；E14_3、E22_2a、E22_2b 均未满足 image/object 候选条件，因此不再执行组合实验 |
| E24_0 | blocked_no_valid_candidate | E13_3b reviewed | results/val/e24_0_e13_3b_candidate_freeze/ |  | S6-4：冻结候选配置；当前没有满足 image/object 条件且 seed 状态有效的候选，E24 demo placeholder 不作为真实冻结实验执行 |
| E25_0 | rerun_seedfix_split_gpu0_gpu1_b48 | E13_3b config / GPU available | results/val/seedfix_e25_0_e13_3b_seed{42,43,44}_seedfix_b48_20260527* |  | 旧 seed42/43/44 产物已判定 seed_pipeline_failed 并删除；2026-05-27 已拆出 GPU1 batch48 加速队列，保持同一 seedfix 配置 |
| E25_1 | done_limited_cached_predictions | E6 cached train/val predictions | results/val/e25_1_e6_calibration_sweep/ | partial | preliminary：基于 E20 缓存 E6 预测做离线校准；最佳全局 conf=0.40，val FP/image=1.236896、FPPI_dark=1.934659、class_confusion FP=739；NMS 仅限缓存 IoU=0.70，object AP 图使用 cached dark-small object AP50 proxy |
| E25_1_full | done_not_candidate | E6 weights / train-only thresholds | results/val/e25_1_full_e6_calibration_sweep/ | no | S6.5-A2：完整重推理 calibration 完成；最佳 selected=illumination_wise dark0.35_other0.40 / NMS 0.50，FP/image=1.279101、FPPI_dark=2.196023、FPPI_low-contrast=1.420366，但 AP_dark-small_object=0.078896、AP_tiny_object=0.041289、AP_low-contrast_object=0.220138，未通过 object-level gate |
| E26_1 | done_limited_cached_predictions | E25_1 / E6 cached train/val predictions | results/val/e26_1_classwise_threshold_calibration/ | partial | preliminary：class-wise 阈值为 car=0.40、truck=0.50、bus=0.50、van=0.45、freight_car=0.45；val FP/image=1.147720、FPPI_dark=1.781250、class_confusion FP=661，dark-small object cached mAP50/precision/recall=0.134318/0.275575/0.726849 |
| E26_1_full | done_not_candidate | E25_1_full complete | results/val/e26_1_full_classwise_threshold_calibration/ | no | S6.5-A3：class-wise threshold 完整复核完成；FP/image=1.147720、FPPI_dark=1.781250、FPPI_low-contrast=1.273281、class_confusion FP=661，但 AP_dark-small_object=0.066720、AP_tiny_object=0.037388、AP_low-contrast_object=0.216065，未通过 object-level gate |
| E26_2a | done_not_candidate | E26_1_full reviewed / train-split class_confusion taxonomy | results/val/e26_2a_class_confusion_cls125_val/ | no | classification-only class_confusion BCE 正类加权 1.25x；训练、统一验证和 object-level 完成，但 image-level AP_dark-small=0.509768、FP/image=1.682097、FPPI_dark=2.960227，未通过第二批候选门槛 |
| E26_2a_run_b48 | off_plan_started_before_new_gate | E22_0/E26_1 reviewed | results/val/e26_2a_class_confusion_cls125/ | no | 已按 batch=48、workers=8 重启过；在新 gate 下属于第二批提前启动产物，第一批 full 版完成前不计入有效推进，不主动终止用户未授权进程 |
| E26_2b | done_not_candidate | E26_2a valid metrics | results/val/e26_2b_class_confusion_cls150_val/ | no | class_confusion classification-only loss 1.50x；训练、统一验证和 object-level 完成，但 image-level AP_dark-small=0.512533、FP/image=1.752212、FPPI_dark=3.085227，未通过第二批候选门槛 |
| E27_1 | done_limited_cached_predictions | E6 cached train/val predictions | results/val/e27_1_metadata_verifier/ | partial | preliminary：metadata verifier 离线完成；train TP vs background_far 负样本，排除 class_confusion/localization_error 负训练；holdout AUC=0.944833，score_final 最佳阈值 0.03，val FP/image=1.144997、FPPI_dark=1.781250、class_confusion FP=739、cached dark-small object AP50=0.164446 |
| E27_1_full | done_not_candidate | E25_1_full complete / E26_1_full reviewed | results/val/e27_1_full_metadata_verifier/ |  | full re-inference verifier completed; holdout AUC=0.967809, best score_final threshold=0.16, FP/image=1.230088, FPPI_dark=1.923295, cached score_final mAP50=0.829483, dark-small object=0.188110; not a candidate |

## 当前结论与下一步

- E6 multi-scale fusion 是当前主线基线：mAP50-95=0.635715，AP_small=0.586603，Recall_small=0.748135，AP_dark=0.585551，AP_dark-small=0.512464。
- E2 保留为暗弱支撑基线；E4 保留为低误报 WBF 参考线。
- S5 done：E20_0、E22_0、E13_loss_check、E18 多 seed、E12_1b 均已完成。普通 gate / P2 / 全局 loss 不再继续；主要 FP 来源为 background_far、class_confusion、localization_error。
- E13_3b 是当前最有希望的低误报候选：单 seed 同时做到 AP_dark-small 略高于 E6、FP/image 明显低于 E6、FPPI_dark 明显低于 E6。但 E13_3b 三 seed 汇总为 mAP50-95=0.635227±0.000547，AP_dark-small=0.509066±0.004227，AP_tiny=0.556394±0.002042，FP/image=1.329476±0.037730，FPPI_dark=2.265152±0.150898；稳定降低误报，但 AP_dark-small 均值低于 E6 三 seed 均值。
- S6 review：E23 object-level evaluator 已补齐，E18_check 已判定旧 E13_3b seed=1/2 指标完全重复，E22_1 已输出 per-taxonomy hard negative list；E18_6 seed=2 已完成单卡重跑但指标仍与 seed=1 完全一致，需要继续标记为 suspicious；E13_3b-light 已完成但 FP 显著升高且 object-level AP_dark-small 低于 E6，不作为候选；E14_1/E14_2/E14_3 CEBS 已完成 image-level 与 object-level，均未保住 dark-small object AP，不作为候选；E22_2a/E22_2b train-split background_far HN 已完成，但 FP 和 FPPI_dark 升高、object-level AP_dark-small 低于 E6，不作为候选；E14_4 无执行依据，E24_0 因无有效候选而阻塞。
- S6.5 第一批复核显示：完整重推理校准能明显降低 FP/image 和 FPPI_dark，但 E25_1_full 与 E26_1_full 都显著损伤 dark-small/tiny/low-contrast object-level AP，因此不作为候选。E25_0 多 seed 是当前唯一未完成的第一批任务，已拆出 GPU1 batch48 队列重跑 seed42/43/44；GPU0 继续 corrected multi-seed 主队列。
- S6 reproducibility audit 已通过：`python scripts/s6_repro_audit.py` 输出 `results/val/s6_repro_audit/metrics_summary.md`，status=pass、failures=0。审计覆盖 manifest 状态、训练 args/weights hash、required_metrics/object_metrics 与 leaderboard 一致性、train-split HN 来源、关键日志完成标记和 S6 demo。保留 3 个警告：E18 multi-seed invalid、E22_2a 训练日志含早期中断 traceback 但最终完成、当前 git worktree dirty。
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
| S6 | 诊断驱动的目标保持型背景抑制阶段 | E23、E18_check、E22_1、E13_3b-light、E22_2a/b、E14、E24_0 | awaiting_review | object-level 口径已落地；E13_3b 多 seed 当前 invalid；target-scoped light loss、train-split HN 和 CEBS 均未形成新候选 |
| S6.5 | 诊断驱动的分类混淆修正与候选可靠性校准阶段 | corrected multi-seed rerun；E25_1_full、E26_1_full；第二批为 E26_2a/b、E27、E28、E9_1、E16、E19 | seedfix_multiseed_completed_split_gpu0_gpu1_external_e13 | E25_1_full/E26_1_full 已完成但不通过 object-level gate；旧 E18/E25_0 多 seed 结果 invalid；GPU0/GPU1/E13 外部 seedfix 已收尾，E26_2b 与 E27_1_full 也已完成并进入评估收口，均未形成候选 |
| S7 | 论文完整验证阶段 | E15、E16、E18 full、E19、E20 full、E21、E24 full | not started | 暂不进入 test / 强模型 |

当前进度：旧 E18/E25_0 多 seed 结果 invalid；GPU0 的 corrected E6 seed0/1/2、E13_3b seed0/1/2 以及 GPU1 的 E25_0 seed42/43/44 都已完成训练、验证和导出。E25_1_full/E26_1_full 已完成但不作为候选；E26_2b 与 E27_1_full 也已完成且不作为候选，第二批只保留未执行项等待后续 gate。

## 当前阶段执行顺序

| 当前任务 | 编号 | 是否立即执行 | 说明 |
| --- | --- | --- | --- |
| corrected multi-seed 重跑 | seedfix/E25_0/E18 | completed_split_gpu0_gpu1_external_e13_b48 | 已修复 dataloader seed 逻辑；GPU0 batch48 顺序重跑 E6 seed0/1/2，E13_3b seed0/1/2 由另一台服务器执行，GPU1 batch48 拆出 E25_0 seed42/43/44；三路 seedfix 都已完成训练、验证与导出，日志 `results/val/logs/seedfix_multiseed_gpu0_b48_20260527.log` 与 `results/val/logs/seedfix_e25_gpu1_b48_20260527.log` |
| E6 完整重推理 calibration / threshold / NMS sweep | E25_1_full | done_not_candidate | 完整重推理完成，FP 降低但 AP_dark-small_object=0.078896，未过 gate |
| class-wise threshold 完整复核 | E26_1_full | done_not_candidate | 完整复核完成，FP/image=1.147720、FPPI_dark=1.781250，但 AP_dark-small_object=0.066720，未过 gate |
| E6 calibration 缓存版 | E25_1 | 已完成，preliminary | 输出 calibration_grid.csv、pareto_curve.csv/svg、pareto_ap_obj_vs_fppi_dark.png、best_operating_points.json、classwise_thresholds.json；NMS 仅覆盖缓存 IoU=0.70 |
| class-wise threshold 缓存版 | E26_1 | 已完成，preliminary | 输出 class_threshold_search.csv、per_class_metrics.csv、best_operating_points.json；当前最佳 class-wise 阈值显著降低 FP/class_confusion |
| class_confusion classification-only loss 1.25x | E26_2a | done_not_candidate | 训练、统一验证和 object-level 已完成；AP_dark-small=0.509768、FP/image=1.682097、FPPI_dark=2.960227 |
| metadata verifier full | E27_1_full | done_not_candidate | full re-inference verifier completed; holdout AUC=0.967809, best score_final threshold=0.16, FP/image=1.230088, FPPI_dark=1.923295 |
| metadata verifier 缓存版 | E27_1 | 已完成，preliminary | 输出 calibration_grid.csv、feature_schema.json、verifier_weights.json、best_operating_points.json、required_metrics.json |
| object-level evaluator | E23 | 已完成 | 输出 E6 的 AP/Recall_dark-small_object、tiny_object、low-contrast_object，并生成 scope 对比 |
| E13_3b seed 独立性核查 | E18_check | 已完成 | 关键指标完全一致，multi-seed invalid，需要重跑 seed=2 |
| hard negative list 构建与去重 | E22_1 | 已完成 | 只构建列表，不训练；background_far 是唯一 train-allowed taxonomy |
| E13_3b-light | E13_3b-light | 已完成，不作为候选 | FP/image 与 FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 1.5x | E22_2a | 已完成，不作为候选 | 使用 train-split background_far HN 来源；image-level AP_dark-small 提升但 FP/FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| background_far hard negative 2x | E22_2b | 已完成，不作为候选 | 使用 train-split background_far HN 来源；image-level AP_dark-small 提升但 FP/FPPI_dark 升高，object-level AP_dark-small 低于 E6 |
| CEBS alpha=0.05/0.10 | E14_1/E14_2 | 已完成，不作为候选 | E14_1 image-level 有提升但 object-level AP_dark-small 低于 E6；E14_2 降 FP 但 image/object dark-small AP 均低于 E6 |
| CEBS 组合候选 | E14_3/E14_4 | E14_3 已完成，不作为候选；E14_4 跳过 | 不强行使用 CEBS；E14_3 未超过 B 候选，且 HN 路线未提供组合依据 |
| candidate freeze | E24_0 | 阻塞，无有效候选 | 冻结配置、权重路径、metrics、protocol、commit、训练和验证参数；当前 E24 脚本仍是 demo placeholder |
| scale+center loss 后续扩展 | E13_4b / E13_4c | 暂停 | 误报明显升高，不作为当前最佳方向 |
| P2 普通扩展 | E11_2 / E11_3 | 暂停 | E11_1 已低于 E6 |
| 普通门控扩展 | E12_1c / E12_1d / E12_2 / E12_3 / E12_4 | 暂停 | gate 路线降 FP 但伤 AP_dark-small，S6 不继续 |
| E6 768 重采样扩展 | E10_3 / E10_4 | 暂停 | E10_2 没超过 E6 640 |
| 强模型/test | E15 / E21 | 禁止 | 不运行 test set，不启动 RT-DETR / YOLOv10 / YOLO11s |
| hard negative 3x/5x | E22_2 3x/5x variants | 禁止 | class_confusion 占比高，盲目强采样可能压掉真实车辆目标 |

当前阶段目标：以 E6 为 proposal generator，通过分类混淆修正、候选质量校准和 RGB/IR 可靠性重评分，在 AP_dark-small_object >= 0.098 且 AP_tiny_object / AP_low-contrast_object 不明显下降的前提下降低 FP/image、FPPI_dark 和 FPPI_low-contrast。

## Demo 脚本

- 全量编号 demo 入口：`scripts/train/run_dark_small_experiment_demos.sh`，覆盖 E0_1-E24_3，默认 dry-run。
- 当前 S6 阶段 demo：`scripts/train/run_dark_small_experiment_demos.sh S6`，会展开 E18_check / E22_1 / E23 / E13_3b_light / E22_2a / E22_2b / E14_1 / E14_2 / E14_3 / E14_4 / E24_0；其中 E14_4 和 E24_0 只输出跳过/阻塞原因。
- 当前 S6 可复现审计：`python scripts/s6_repro_audit.py`，报告写入 `results/val/s6_repro_audit/`；`--strict` 会在警告存在时返回非零。
- 当前 E6 主线快捷 demo：`scripts/train/run_e6_mainline_demos.sh`。
- 已实现的 manifest 实验可用 `RUN_MODE=run scripts/train/run_dark_small_experiment_demos.sh E22_2a` 或 runner 直接启动；未实现的后续分支只输出编号一致的命令模板，不会误跑 test set。

## 状态更新规则

- 每完成一个实验，先保存日志，再更新状态表。
- 实验失败必须标记为 failed，并保留错误日志。
- 只有验收通过后才能改为 done。
