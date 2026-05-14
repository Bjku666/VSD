# 暗弱小目标实验排行榜

排序规则：优先比较 AP_dark-small，其次 Recall_small，再比较更低 FP/image，最后比较 mAP50-95。

| 排名 | ID | 实验 | 状态 | mAP50 | mAP50-95 | AP_small | Recall_small | AP_dark | Recall_dark | AP_dark-small | FP/image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E2 | E2 IR-only | done | 0.766574 | 0.505324 | 0.019598 | 0.570700 | 0.536310 | 0.753658 | 0.023879 |  |
| 2 | E4 | E4 Late WBF | done | 0.776031 | 0.540385 | 0.022188 | 0.342082 | 0.519848 | 0.642643 | 0.022675 | 1.173587 |
| 3 | E3 | E3 Late NMS | done | 0.775085 | 0.544189 | 0.022444 | 0.341656 | 0.518022 | 0.642469 | 0.022597 | 3.336283 |
| 4 | E1 | E1 RGB-only | done | 0.770179 | 0.534859 | 0.022663 | 0.368336 | 0.452863 | 0.659743 | 0.013665 |  |
| 5 | E6 | E6 multi-scale fusion | done | 0.602380 | 0.421496 | 0.023412 | 0.225913 | 0.267129 | 0.410450 | 0.007973 |  |
| 6 | E5 | E5 single-layer fusion | done | 0.507211 | 0.352028 | 0.021935 | 0.207792 | 0.144986 | 0.176831 | 0.006807 |  |
| 7 | B1 | IR-only 768 | pending |  |  |  |  |  |  |  |  |
| 8 | B2 | IR-only 960 | pending |  |  |  |  |  |  |  |  |
| 9 | B3 | RGB-only 768 | pending |  |  |  |  |  |  |  |  |
| 10 | B4 | WBF RGB:IR = 0.4:0.6 | pending |  |  |  |  |  |  |  |  |
| 11 | B5 | WBF RGB:IR = 0.3:0.7 | pending |  |  |  |  |  |  |  |  |
| 12 | B6 | WBF RGB:IR = 0.5:0.5 | pending |  |  |  |  |  |  |  |  |
| 13 | B7 | IR-only 768 + close_mosaic=20 | pending |  |  |  |  |  |  |  |  |
| 14 | B8 | IR-only 768 + dark-small resampling | pending |  |  |  |  |  |  |  |  |
| 15 | F1 | 6-channel early fusion 640 | requires_model_support |  |  |  |  |  |  |  |  |
| 16 | F2 | 6-channel early fusion 768 | requires_model_support |  |  |  |  |  |  |  |  |
| 17 | F3 | E5 fixed 768 | pending |  |  |  |  |  |  |  |  |
| 18 | F4 | E6 fixed 768 | pending |  |  |  |  |  |  |  |  |
| 19 | F5 | E6 fixed 768 + small resampling | pending |  |  |  |  |  |  |  |  |
| 20 | F6 | E6 fixed 768 + dark-small resampling | pending |  |  |  |  |  |  |  |  |
