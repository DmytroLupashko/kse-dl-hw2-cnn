# Experiment 36: Model comparison

## Request

create experiment to compare new models with each other and exp26; add table with ImageNet scores of these models; try to look on VIP animals GradCams visually and state what is the difference in feature selection for these models

## Question

On out-of-fold rows of folds 0–4, how do exp26 ResNet-18, EfficientNet-B0, RegNetY-016 and ConvNeXt-Tiny differ in AP per class and in image regions their Grad-CAM uses on VIP positives?

## Plan

1. Tabulate ImageNet top-1 and top-5 accuracy of four pretrained weights from timm `results-imagenet.csv`.
2. Plot precision-recall curve per class on pooled out-of-fold rows of folds 0–4, one panel per class (5), one line per model (4), AP in legend.
3. Plot Grad-CAM VIP validation figures of four models on same 16 random labeled positives per VIP class, maps of same image stacked, 3 figures.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp36_model_comparison.ipynb](../notebooks/experiments/exp36_model_comparison.ipynb)
- **Tracking:** none; reads saved results of exp26, exp32, exp33, exp34
- **What changes:** nothing trained; models compared side by side.
- **What stays fixed:** fold models and out-of-fold probs as saved, folds 0–4, sample seed 0.
- **Metric:** AP per class on pooled out-of-fold rows; Grad-CAM regions by eye.

## Fixes

add optimal thresholds for these models

## Results

| model | weights | ImageNet top-1 | top-5 |
|---|---|---|---|
| exp26 ResNet-18 | `resnet18.a1_in1k` | 71.5 | 90.1 |
| EfficientNet-B0 | `efficientnet_b0.ra_in1k` | 77.7 | 93.5 |
| RegNetY-016 | `regnety_016.tv2_in1k` | 80.7 | 95.3 |
| ConvNeXt-Tiny | `convnext_tiny.fb_in1k` | 82.1 | 95.9 |

timm `results-imagenet.csv`, evaluation at 224 px.

![Precision-recall curves](../figures/exp36_model_comparison_pr_curves.png)

Out-of-fold rows of folds 0–4 pooled, AP per class in legend.

| class | exp26 ResNet-18 | EfficientNet-B0 | RegNetY-016 | ConvNeXt-Tiny |
|---|---|---|---|---|
| Cat | 0.417 / 0.981 | 0.467 / 0.978 | 0.383 / 0.984 | 0.274 / 0.984 |
| Dog | 0.625 / 0.975 | 0.546 / 0.974 | 0.719 / 0.977 | 0.675 / 0.981 |
| Moris | 0.895 / 0.875 | 0.966 / 0.875 | 0.948 / 0.936 | 0.674 / 0.920 |
| Motya | 0.821 / 0.810 | 0.953 / 0.868 | 0.968 / 0.872 | 0.946 / 0.883 |
| Biatrix | 0.875 / 0.964 | 0.999 / 0.950 | 0.931 / 0.925 | 0.886 / 0.941 |
| Macro F1 | 0.921 | 0.929 | 0.939 | 0.942 |

Threshold / F1: best-F1 threshold per model and class on pooled out-of-fold rows of folds 0–4 (submission thresholds); F1 tuned and scored on same rows, optimistic.

![Moris compare](../figures/exp36_model_comparison_grad_cam_moris_compare.png)

![Motya compare](../figures/exp36_model_comparison_grad_cam_motya_compare.png)

![Biatrix compare](../figures/exp36_model_comparison_grad_cam_biatrix_compare.png)

16 random labeled positives per VIP class (seed 0), fold model that held image out; green at or above own best-F1 threshold, red below.

Grad-CAM by eye:
- ResNet-18: widest blob, whole body plus margin, centered on animal.
- EfficientNet-B0: tighter body blob; most often drifts to background (Biatrix snow, hallway floor, park ground; Moris shelf scene).
- RegNetY-016: close to ResNet-18 but tighter, body-centered, rarely off animal.
- ConvNeXt-Tiny: smallest, split spots on parts (head, paws, back edge); whole small body when animal far; some spots on background (Motya box, wall).
- Motya misses differ: new models miss far, curled and background cats ResNet-18 catches; ResNet-18 misses close-up face others catch.

## AI Summary

All three new models beat exp26 on Moris and Motya AP (RegNetY-016 best Moris, ConvNeXt-Tiny best Motya) and lose on Biatrix, Cat and Dog flat; AP order does not follow ImageNet order. Regions narrow from ResNet-18 (whole-body blob) to ConvNeXt-Tiny (part spots), and Motya misses fall on different images per model, so errors look complementary.

## Conclusion
