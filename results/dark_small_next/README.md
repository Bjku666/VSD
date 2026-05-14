# 暗弱小目标下一批实验

本目录是下一批实验的统一输出根目录。新实验不再分散写到旧的 `experiments/` 和 `results/val/` 两套目录里，避免重复。

## 目录约定

- `runs/`：新训练实验的权重、日志和必要训练表；训练过程图片可清理。
- `val/`：每个实验的最终验证指标和关键图。
- `work/generated_data/`：自动生成的重采样训练 YAML/TXT。
- `dark_small_experiment_leaderboard.md`：按暗弱小目标优先规则排序的结果表。
- `dark_small_experiment_leaderboard.csv`：同一排行榜的 CSV 版本。

关键图可以保留在 `val/` 中；不要在 `runs/` 里再保留同一类验证图。

## 常用命令

列出第一批实验：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py list --stage batch1
```

预览命令，不实际训练：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py run B1 B4 B8 --dry-run
```

只跑第一批中无需重新训练的 WBF 权重搜索：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py run B4 B5 B6
```

跑指定训练和验证实验：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py run B1
```

只验证已经训练完成的实验：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py run B1 --only validate
```

重新汇总排行榜：

```bash
/mnt/disk2/lhr/conda_envs/vsd/bin/python scripts/dark_small_experiment_runner.py aggregate
```

## 当前优先级

1. 先跑 `B4/B5/B6`，它们只做 WBF 权重搜索，不需要重新训练。
2. 再跑 `B1/B2/B7/B8`，验证 IR-only 高分辨率、mosaic 关闭策略和 dark-small 重采样。
3. 第一批完成后再跑 `F3/F4/F5/F6`，用于判断 E5/E6 是否值得作为论文主线。

排序规则固定为：`AP_dark-small` 优先，其次 `Recall_small`，再比较更低 `FP/image`，最后比较 `mAP50-95`。
