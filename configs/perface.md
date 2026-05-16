# configs

## 功能

存放数据集配置、子集配置和实验矩阵配置。训练脚本读取这里的配置，不在这里写结果。

## 主要内容

- `dronevehicle_resplit/`：当前 DroneVehicle RGB、IR、RGB-IR 三类数据配置。
- `dronevehicle_resplit/subsets/`：官方划分派生子集配置，包括 dark、small、dark-small、bright、normal、medium、large；后续需要补充 low-contrast 和 tiny-size 子集。
- `experiments/dark_small_next.yaml`：E7 之后的大规模实验矩阵。虽然文件名保留 `dark_small_next`，实际实验编号统一使用 E 编号。
- `visdrone.yaml`：VisDrone YOLO 数据配置。

## 实验配置规范

- 新实验优先通过 `configs/experiments/dark_small_next.yaml` 增加条目。
- 实验 ID 统一使用 `E7_1`、`E8_2`、`E16_3` 这种 E 编号。
- 新实验输出统一写入 `results/val/`。
- 新实验日志统一写入 `results/val/logs/`。
- 旧历史权重已删除；后融合和后续实验默认引用重新训练后位于 `results/val/<experiment>/weights/` 的权重。
- test set 只能用于最终锁定评测，不能在配置中作为反复调参目标。

## 后续需要同步到配置的实验

- E0：low-contrast 和 tiny-size 子集配置。
- E16：RGB-IR 配准扰动鲁棒性实验。
- E17：low-contrast 全模型评测。
- E18：多 seed 稳定性实验。
- E19：效率指标分析。
- E20：per-class、tiny-size、失败案例分析。
- E21：最终 test set 锁定评测。
