# data

## 功能

存放原始数据和原始压缩包，只作为数据源，不直接作为训练输入。

## 主要内容

- `DroneVehicle/raw/`：DroneVehicle 官方 train、val、test 原始目录。
- `DroneVehicle/zips/`：DroneVehicle 原始压缩包。
- `VisDrone2019/`：VisDrone 原始数据和压缩包。

## 规范

- 不在 `data/` 内生成训练结果。
- 不手工改动原始数据。
- 不重新打乱或覆盖 DroneVehicle 官方 train、val、test 原始划分。
- 训练使用 `prepared/` 中转换后的 YOLO 格式数据。
- `DroneVehicle-DarkSmall` 只作为派生评测协议生成在 `prepared/` 和 `results/dataset_audit/`，不写回 `data/`。
