# DroneVehicle-DarkSmall-v1 Protocol Summary

## Contract

- keep original DroneVehicle train/val/test splits
- thresholds are learned from train only
- crop a 100px border before all statistics and label conversion
- AP_dark-small is image-level subset evaluation
- low-contrast is derived from train object contrast statistics

## Train Thresholds

- brightness_dark_threshold: 33.503208
- object_area_small_threshold: 1288.011719
- object_area_tiny_threshold: 880.005127
- low_contrast_threshold: 0.084252

## Generated Subsets

- train / val / test
- dark
- small
- tiny
- dark-small
- low-contrast

## YAML Outputs

- /mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_dark.yaml
- /mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark.yaml
- /mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_ir_dark.yaml

## Metrics

Report these metrics in every experiment:
- mAP50
- mAP50-95
- Recall
- Precision
- AP_small
- Recall_small
- AP_dark
- Recall_dark
- AP_dark-small
- FPS / Params / FLOPs

Priority for method decisions:
- AP_dark
- Recall_small
- AP_dark-small

