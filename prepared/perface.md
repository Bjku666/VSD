# prepared

## 功能

存放由原始数据转换得到的可训练数据和派生评测协议，不直接手工修改其中图像和标签。

## 主要内容

- `dronevehicle_resplit/`：当前 DroneVehicle 的 RGB/IR YOLO 格式数据；后续按 `DroneVehicle-DarkSmall` 协议继续补齐审计、阈值和子集统计。
  - `original_split/` 保留 train、val、test 三个官方划分。
  - `resplit_subsets/` 是基于官方划分派生的评测子集，包括 dark、small、dark-small 等。
- `visdrone_yolo/`：VisDrone 转 YOLO 后的数据，作为辅助数据或横向参考。

## DroneVehicle-DarkSmall 二次评测协议

二次构建不是重新随机切 train/val/test，而是在 DroneVehicle 官方 train、val、test 基础上做规范化转换和专项评测子集派生：

1. RGB 图像、IR 图像、RGB XML、IR XML 四者配对完整后才纳入转换。
2. YOLO 标签由 RGB XML 生成，并同步给 RGB/IR 两个模态，保证两模态标签完全一致。
3. `train/val/test` 使用官方原始划分，不混合样本，不重新随机打乱。
4. 原始 OBB/polygon 标注必须保留；当前 YOLO detect 训练使用转换后的 AABB 标签。
5. 必须先裁除四周白边并调整标注坐标，再计算亮度、尺寸和局部对比度。
6. `dark/small/dark-small/low-contrast/tiny` 阈值只能由 train split 统计得到，再应用到 val/test。
7. val/test 子集只用于评测，不能反向用于训练权重选择或重采样倍率调参。
8. 文件名 stem 在不同官方 split 内会重复，例如 train 和 val 都可能有 `00001`；判断泄漏时必须看真实路径，而不是只看 stem。

推荐阈值定义：

- `dark`：裁边后 RGB 灰度均值低于 train brightness p25。
- `small`：目标面积低于 train object area p25。
- `tiny`：目标面积低于 train object area p10。
- `low-contrast`：目标区域与外扩背景环的 Weber/local contrast 低于 train contrast p25。
- `dark-small`：同时保留 image-level 子集和 object-level 指标口径，避免指标含义不清。

## 后续数据协议补充

根据当前实验路线，E0 阶段需要补充：

- `low-contrast` 子集：衡量目标与背景弱显著性，不只看低亮度。
- `tiny` 尺寸分桶：建议 train 面积分位数 p10，或作为对照使用 `area < 16²`。
- 更细尺寸分桶：`tiny`、`small`、`medium`、`large`。
- 对应指标：`AP_low-contrast`、`Recall_low-contrast`、`FPPI_low-contrast`、`AP_tiny`、`Recall_tiny`。

## 审查产物要求

旧实验审查文件已经随旧结果清空。重新开始训练前，需要重新生成并保存：

- `results/dataset_audit/pair_audit.json`
- `results/dataset_audit/bad_pairs.csv`
- `results/dataset_audit/train_thresholds.json`
- `results/dataset_audit/subset_counts.csv`
- `results/dataset_audit/class_distribution.csv`
- `results/dataset_audit/object_size_distribution.csv`
- `results/dataset_audit/brightness_distribution.csv`
- `results/dataset_audit/contrast_distribution.csv`

重新生成后必须确认：

- RGB/IR train、val、test 图像与标签数量一致。
- YOLO 标签格式合法，无越界归一化框。
- 按真实路径检查，train/val/test 之间无交叉泄漏。
- RGB/IR 同一 split 的样本集合一致。
- RGB/IR 图像尺寸一致、标签数量一致、类别直方图一致。
- val/test 只被筛选成子集，不参与阈值学习。
