#!/bin/bash
# 在准备好的 VisDrone 数据集上训练 YOLO11n baseline。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd

yolo detect train \
    model=/mnt/disk2/lhr/VSD/weights/pretrained/yolo11n.pt \
    data=/mnt/disk2/lhr/VSD/configs/visdrone.yaml \
    imgsz=960 \
    epochs=100 \
    batch=16 \
    device=1 \
    workers=8 \
    project=/mnt/disk2/lhr/VSD/experiments/visdrone \
    name=yolo11n_baseline