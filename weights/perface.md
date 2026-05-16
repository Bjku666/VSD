# weights

## 功能

存放项目运行会引用的权重文件。

## 主要内容

- `pretrained/yolo11n.pt`：当前 YOLO11n 基线预训练权重。
- `trained/`：当前不使用。旧历史训练权重已经删除，后续实验从头重新训练。

## 规范

- 训练得到的实验权重统一随实验目录放在 `results/val/<experiment>/weights/`。
- 用于评估、推理、后融合的固定权重至少需要 `best.pt`。
- 用于断点继续训练或完整恢复训练状态的实验必须同时保留 `best.pt` 和 `last.pt`。
- 重新训练后，E3/E4/E7 的后融合默认引用 E1/E2 新训练输出中的 `best.pt`。
- 不再引用旧 `weights/trained/e1_rgb_only` 或 `weights/trained/e2_ir_only`。
