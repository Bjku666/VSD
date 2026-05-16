# 暗弱小目标实验汇总表

表格按实验编号顺序排列，便于对应完整实验流程；指标比较时仍优先关注 AP_dark-small、Recall_small、FP/image 和 mAP50-95。

| 序号 | ID | 实验 | 状态 | mAP50 | mAP50-95 | AP_small | Recall_small | AP_dark | Recall_dark | AP_dark-small | FP/image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E1 | E1 RGB-only | done | 0.779699 | 0.556423 | 0.504289 | 0.672031 | 0.441884 | 0.664810 | 0.376499 |  |
| 2 | E2 | E2 IR-only | done | 0.777607 | 0.529912 | 0.497403 | 0.716214 | 0.558856 | 0.763623 | 0.510692 |  |
| 3 | E3 | E3 Late NMS | done | 0.774544 | 0.561468 | 0.519252 | 0.633062 | 0.522439 | 0.658037 | 0.472283 | 4.343771 |
| 4 | E4 | E4 Late WBF | done | 0.779689 | 0.559216 | 0.513805 | 0.630920 | 0.532153 | 0.658568 | 0.465549 | 1.530293 |
| 5 | E5 | E5 single-layer fusion | pending |  |  |  |  |  |  |  |  |
| 6 | E6 | E6 multi-scale fusion | pending |  |  |  |  |  |  |  |  |
| 7 | E7_1 | IR-only 768 | pending |  |  |  |  |  |  |  |  |
| 8 | E7_2 | IR-only 960 | pending |  |  |  |  |  |  |  |  |
| 9 | E7_3 | RGB-only 768 | pending |  |  |  |  |  |  |  |  |
| 10 | E7_4 | WBF RGB:IR = 0.4:0.6 | pending |  |  |  |  |  |  |  |  |
| 11 | E7_5 | WBF RGB:IR = 0.3:0.7 | pending |  |  |  |  |  |  |  |  |
| 12 | E7_6 | WBF RGB:IR = 0.5:0.5 | pending |  |  |  |  |  |  |  |  |
| 13 | E8_1 | IR-only 768 + close_mosaic=20 | pending |  |  |  |  |  |  |  |  |
| 14 | E8_2 | IR-only 768 + dark-small resampling | pending |  |  |  |  |  |  |  |  |
| 15 | E8_3 | IR-only 960 + dark-small resampling | pending |  |  |  |  |  |  |  |  |
| 16 | E8_4 | Best WBF + dark-small resampled IR model | pending |  |  |  |  |  |  |  |  |
| 17 | E9_1 | E5 fixed 640 | pending |  |  |  |  |  |  |  |  |
| 18 | E9_2 | E5 fixed 768 | pending |  |  |  |  |  |  |  |  |
| 19 | E9_3 | E6 fixed 640 | pending |  |  |  |  |  |  |  |  |
| 20 | E9_4 | E6 fixed 768 | pending |  |  |  |  |  |  |  |  |
| 21 | E9_5 | E6 fixed 768 + small resampling | pending |  |  |  |  |  |  |  |  |
| 22 | E9_6 | E6 fixed 768 + dark-small resampling | pending |  |  |  |  |  |  |  |  |
| 23 | E10_1 | 6-channel early fusion 640 | requires_model_support |  |  |  |  |  |  |  |  |
| 24 | E10_2 | 6-channel early fusion 768 | requires_model_support |  |  |  |  |  |  |  |  |
