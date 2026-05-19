# scripts

## 功能

存放数据处理、训练、验证、融合评估、批量实验和结果汇总脚本。

## 主要内容

- `dataset_resplit_dronevehicle.py`：DroneVehicle 配对审计、YOLO 转换、数据报告和 val 子集生成。
- `dark_small_experiment_runner.py`：按实验矩阵执行 E7 之后实验并汇总指标。
- `e1_*` 到 `e6_*` 的 Python 文件：历史单实验训练、验证和融合评估逻辑。
- `render_key_result_figures.py`：重建关键结果图。
- `cleanup_result_artifacts.py`：清理冗余结果产物。
- `train/`：所有 shell 启动脚本，包括 E1/E2 训练、E3/E4 后融合验证、E5/E6 融合训练、E7/E8/E10 队列和 VisDrone baseline。

## 规范

- 新实验优先走 `dark_small_experiment_runner.py`。
- shell wrapper 统一放到 `scripts/train/`。
- 新输出统一写入 `results/val/`。
- 新日志统一写入 `results/val/logs/`。
- 不在 `scripts/` 下保存训练产物。
- 重新开始训练后，先运行 E0 数据审计，再训练 E1/E2；E3/E4/E7 后融合脚本默认引用 E1/E2 新生成的 `results/val/.../weights/best.pt`。
