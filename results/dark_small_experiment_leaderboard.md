# 暗弱小目标实验汇总表

表格按实验编号顺序排列，便于对应完整实验流程；指标比较时仍优先关注 AP_dark-small、AP_tiny、AP_low-contrast、FP/image、FPPI_dark 和 mAP50-95。

| 序号 | ID | 实验 | 状态 | mAP50 | mAP50-95 | AP_small | Recall_small | AP_dark | Recall_dark | AP_dark-small | AP_tiny | AP_low-contrast | AP_dark-small_obj | AP_tiny_obj | AP_low-contrast_obj | FP/image | FPPI_dark | FPPI_low-contrast | Params | GFLOPs | FPS | GPU MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E1 | E1 RGB-only | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 | E2 | E2 IR-only | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 | E3 | E3 Late NMS | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 | E4 | E4 Late WBF | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 | E5 | E5 single-layer fusion | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | E6 | E6 multi-scale fusion | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 7 | E7_1 | IR-only 768 | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 | E7_2 | IR-only 960 | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 9 | E7_3 | RGB-only 768 | running |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 10 | E7_4 | WBF RGB:IR = 0.4:0.6 | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 11 | E7_5 | WBF RGB:IR = 0.3:0.7 | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 12 | E7_6 | WBF RGB:IR = 0.5:0.5 | skipped_duplicate_E4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 13 | E8_1 | IR-only 768 + close_mosaic=20 | running |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 14 | E8_2 | IR-only 768 + dark-small resampling | pending |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 15 | E8_3 | IR-only 960 + dark-small resampling | paused_E8_2_not_better_than_E2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | E8_4 | Best WBF + dark-small resampled IR model | paused_E8_2_not_better_than_E2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 17 | E9_1 | 6-channel early fusion 640 | requires_model_support |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 18 | E9_2 | 6-channel early fusion 768 | requires_model_support |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 19 | E10_1 | E5 768 | stopped_deprioritized |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 20 | E10_2 | E6 768 | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 21 | E10_3 | E6 768 + small resampling | paused_until_E10_2_beats_E6_640 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 22 | E10_4 | E6 768 + dark-small resampling | paused_until_E10_2_beats_E6_640 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 23 | E11_1 | E6 + P2 detection head | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 24 | E11_2 | E6 + P2 head + P3/P4/P5 original fusion | paused_E11_1_not_better_than_E6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 25 | E11_3 | E6 + P2 head + 768 | paused_E11_1_not_better_than_E6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 26 | E12_1 | E6 + residual gated fusion | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 27 | E12_1b | E6 + weak residual gated fusion | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 28 | E12_2 | E6 + spatial gate | paused_E12_1_not_better_darksmall |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 29 | E12_3 | E6 + dark-aware reliability gate | paused_E12_1_not_better_darksmall |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 30 | E12_4 | E6 + residual gated fusion + P2 head | paused_E12_1_not_better_darksmall |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 | E13_1 | E6 baseline loss control | covered_by_E6_baseline |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | E13_2 | E6 + scale-aware loss | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 33 | E13_2b | E6 + small-scoped scale-aware loss | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34 | E13_3 | E6 + center-aware loss | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35 | E13_3b | E6 + small-scoped center-aware loss | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36 | E13_3b_light | E13_3b-light target-scoped center loss | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37 | E13_4b | E6 + small-scoped scale-center loss w0.05 | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 38 | E13_4c | E6 + small-scoped scale-center loss w0.10 | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 39 | E13_4 | E6 + scale-aware + center-aware loss | pending_gpu_busy |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40 | E14_1 | E6 + CEBS alpha=0.05 | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 41 | E14_2 | E6 + CEBS alpha=0.10 | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42 | E14_3 | E13_3b-light + CEBS alpha=0.05 | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43 | E14_4 | E13_3b-light + CEBS + background_far HN 1.5x | skipped_not_justified |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44 | E18_check | E13_3b seed integrity check | deleted_invalid_old_seed_logic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45 | E18_5 | E13_3b seed=1 stability | deleted_invalid_old_seed_logic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46 | E18_6 | E13_3b seed=2 stability | deleted_invalid_old_seed_logic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 47 | E22_1 | hard negative list build and dedup | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48 | E22_2a | E6 + train background_far hard negative 1.5x | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49 | E22_2b | E6 + train background_far hard negative 2x | done_not_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50 | E23 | object-level evaluator | done |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51 | E24_0 | candidate freeze | blocked_no_valid_candidate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 52 | E25_0 | E13_3b multi-seed valid rerun | done_not_candidate | 0.841995 | 0.635443 |  |  |  |  | 0.520173 | 0.553575 | 0.637531 | 0.097365 | 0.052771 | 0.242618 | 1.724302 | 3.063447 | 1.905715 |  |  |  |  |
| 53 | E25_1 | E6 calibration / threshold / NMS sweep | done_limited_cached_predictions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54 | E25_1_full | E6 full re-inference calibration / threshold / NMS sweep | done_not_candidate |  | 0.566849 |  |  |  |  |  |  |  | 0.078896 | 0.041289 | 0.220138 | 1.279101 | 2.196023 | 1.420366 |  |  |  |  |
| 55 | E26_1 | class-wise threshold calibration | done_limited_cached_predictions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56 | E26_1_full | class-wise threshold full re-inference validation | done_not_candidate |  |  |  |  |  |  |  |  |  | 0.066720 | 0.037388 | 0.216065 | 1.147720 | 1.781250 | 1.273281 |  |  |  |  |
| 57 | E26_2a | class_confusion classification-only loss 1.25x | failed_train_split_source_unverified | 0.840495 | 0.632659 | 0.581490 | 0.739554 | 0.583874 | 0.760564 | 0.509768 | 0.550148 | 0.634783 | 0.100160 | 0.052999 | 0.243073 | 1.682097 | 2.960227 | 1.847694 | 4153919.000000 | 10.199398 | 360.808467 | 2815.336914 |
| 58 | E26_2b | class_confusion classification-only loss 1.50x | failed_train_split_source_unverified | 0.842704 | 0.637563 | 0.589448 | 0.735260 | 0.584492 | 0.779490 | 0.512533 | 0.558463 | 0.640034 | 0.090241 | 0.052647 | 0.246311 | 1.752212 | 3.085227 | 1.936466 | 4153919.000000 | 10.199398 | 219.029754 | 2815.335449 |
| 59 | E27_1 | metadata verifier | done_limited_cached_predictions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 60 | E27_1_full | metadata verifier full re-inference validation | done_not_candidate |  | 0.565559 |  |  |  |  |  |  |  | 0.066380 | 0.037329 | 0.215091 | 1.230088 | 1.923295 | 1.395997 |  |  |  |  |
