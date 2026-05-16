# results

## 功能

存放实验过程、最终结果、可复现实验产物和统一日志，是论文表格、指标对比、关键图片、权重和训练记录的主目录。旧训练结果已经清空，后续结果从重新训练开始生成。

## 主要内容

- `val/`：当前验证集实验结果目录；重新训练后包含 E1、E2、E3 等实验的指标、权重和关键图。
- `val/dark_small_experiment_leaderboard.md`：按实验编号顺序排列的实验总表，由汇总脚本重新生成。
- `val/work/`：实验运行时生成的辅助配置，例如重采样训练列表；需要时由脚本重建。
- `val/logs/`：统一日志目录。
- `dataset_audit/`：数据构建脚本默认输出的数据审查报告目录。

## 规范

- 新训练输出统一放到 `results/val/<experiment_id>.../`。
- 每个训练实验应保留 `weights/best.pt`；需要断点续训时同时保留 `weights/last.pt`。
- 新日志统一放到 `results/val/logs/`。
- 旧日志不保留；重新训练后的所有日志只放在 `results/val/logs/`。
- test set 最终评测结果后续放到 `results/test/`，只用于最终锁定模型评估。
- 不再使用根目录 `logs/`、`experiments/` 或 `old/` 作为运行目录。
