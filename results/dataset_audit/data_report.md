# DroneVehicle Data Report

## Protocol

- protocol: DroneVehicle-DarkSmall-v1
- crop border: 100px
- thresholds are learned from train split only

## Train Thresholds

- brightness dark threshold: 33.503208
- object area small threshold: 1288.011719
- object area tiny threshold: 880.005127
- low-contrast threshold: 0.084252

## Summary

- total objects: 452568
- avg objects per image: 15.9136
- train dark images: 4498
- train dark objects: 76921

## Metrics Protocol

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

Priority metrics for method direction:
- AP_dark
- Recall_small
- AP_dark-small

