# Current Configuration

Reasoning: `sections/eda/eda_report.md`, section "Configuration (AI written)".

Target metric: Macro F1.

Selection metric: mAP.

Thresholds: per class $t_c$, tuned on validation.

Loss: `nn.BCEWithLogitsLoss`, full ratio `pos_weight` $w_c = (N - n_c)/n_c$ (`Config.pos_weight_power = 1.0`; code default 0.75 kept so exp01–exp08 configs reproduce). Reasoning: `sections/model/experiments/exp09_full_ratio_weight.md`, mAP 0.972 to 0.983, higher on both folds, smoother `val_map`. Config: `sections/model/configs/exp09_full_ratio_weight/resnet18.toml`.

Validation: multilabel stratified 5-fold CV, mean of per-fold mAP.

Model: ResNet-18, timm `resnet18.a1_in1k` (ImageNet pretrained), `img_size` 224. Reasoning: `sections/model/experiments/exp01_lecture_models.md`, ties EfficientNet-B0 on mAP, trains faster. Config: `sections/model/configs/exp01_lecture_models/resnet18.toml`.

Sampling: VIP oversampling ×8 (`Config.oversample = 8`), VIP rows drawn 8 times per epoch, VIP `pos_weight` divided by 8. Reasoning: `sections/model/experiments/exp04_oversampling_vip.md`, mAP 0.951 to 0.972, higher on both folds. Config: `sections/model/configs/exp04_oversampling_vip/resnet18.toml`.

Downsampling: non-VIP rows (Cat, Dog, none) random 20% per epoch (`Config.downsample = 0.2`), `pos_weight` full ratio counted over sampled draws. Reasoning: `sections/model/experiments/exp16_downsampling.md`, not worse (VIP AP equal or higher, mAP +0.007 / +0.011 on folds 0 / 1), ~4× fewer steps; gain within noise. Config: `sections/model/configs/exp16_downsampling/resnet18.toml`.
