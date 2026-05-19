# VSD 项目目录说明

本项目围绕 DroneVehicle/VisDrone 无人机目标检测实验展开，当前重点是 RGB-IR 暗弱小目标检测。根目录保留唯一 `README.md`；主要一级目录使用 `perface.md` 说明用途。

## 顶层目录

- `configs/`：数据集、子集和批量实验配置。
- `data/`：原始公开数据和压缩包。
- `prepared/`：转换后可直接用于 YOLO/Ultralytics 的数据。
- `scripts/`：数据处理、训练、验证、融合评估、结果清理和关键图重建脚本。
- `scripts/train/`：所有 shell 启动脚本。
- `results/`：实验过程、最终指标、权重、关键图和日志。
- `weights/`：预训练权重；旧历史训练权重已删除，后续按实验重新生成。

## 当前状态

- 旧训练结果、旧验证结果和旧历史权重已经清空，后续从 E0 数据协议和 E1/E2 基线开始重新训练。
- `weights/` 当前只保留预训练权重；新训练权重随实验保存在 `results/val/<experiment>/weights/`。
- 当前核心问题仍然是 `AP_small`、`AP_tiny`、`AP_dark-small` 和低对比度目标，而不是只追求总 mAP。
- 二次数据集协议采用 `DroneVehicle-DarkSmall` 思路：不重新打乱官方 train/val/test，只基于官方划分派生专项评测子集。

## 当前实验路线

完整实验路线以 `shiyan.md` 为准。当前实验路线为：

1. E0 固定数据协议，补充 `low-contrast` 和 `tiny` 子集。
2. E1/E2 重新训练 RGB-only 和 IR-only 基线。
3. E3/E4 重新评估 NMS/WBF 后融合。
4. E7 做 WBF 权重搜索和高分辨率强基线。
5. E9 建立 6 通道 early fusion 基线。
6. E10 做 E5/E6 高分辨率与重采样扩展。
7. E8、E11、E13 做小目标增强。
8. E14、E17 做暗弱和低对比度增强/评测。
9. E12、E16 做跨模态门控和配准扰动鲁棒性。
10. E15、E18、E19、E20、E21 完成强模型、多 seed、效率、失败案例和最终 test set 评测。

## 目录规范

- 新实验输出统一写入 `results/val/<experiment_id>.../`。
- 训练日志统一写入 `results/val/logs/`。
- shell 脚本统一放在 `scripts/train/`。
- 实验总表为 `results/val/dark_small_experiment_leaderboard.md`。
- 根目录不再保留 `logs/`、`old/`、`experiments/` 作为运行路径。
- 子目录不再放 `perface.md`，只在主要一级目录放说明文件。
