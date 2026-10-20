# Experiment 24: Head erase on non-VIP cats

## Request

let's assume that model actually looks at body after this boxes closing. what is next step?

create experiment 24

## Question

On generic (non-VIP) Cat positives of folds 0 and 1, does ResNet-18 with exp21 setup but Cat head boxes instead of face boxes keep more positives at or above own best-F1 threshold than exp21 when head and ears are erased, without lower VIP AP on out-of-fold rows of folds 0 and 1?

## Plan

1. Plot head boxes on 16 random generic Cat images of fold 0 training part, face and head outlines, share of Cat head boxes above 30% of image (skipped by erase) in title.
2. Plot recall at own best-F1 threshold, generic Cat positives of folds 0 and 1, per model (exp16, exp20, exp21, exp24) on clean images, head erased, and same-size box at random place off head.
3. Plot Grad-CAM Cat of four models on exp19 Cat images with head erased, maps of same image stacked.
4. Plot precision-recall curve per class on out-of-fold rows of folds 0 and 1, one panel per class (5), one line per model: exp20, exp21, exp24, AP in legend.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp24_head_erase.ipynb](../notebooks/experiments/exp24_head_erase.ipynb), config `configs/exp24_head_erase/resnet18.toml`, training `core.cv`
- **Tracking:** `cnn-model-head-erase`
- **What changes:** `augment = "head_erase"`: exp21 recipe, erased Cat box = head box: exp21 Cat face box (`data/processed/boxes_face.csv`) 2× wide around center, top edge raised by 1× face height, bottom edge lowered by 0.25× face height (covers ears above core), clipped to image (median 18.1% of image, 13.4% above 30%, skipped). Dog boxes unchanged (exp21 face boxes). New file `data/processed/boxes_head.csv` by `python -m core.boxes --head`, from face file, no new Grad-CAM. Same head box in exam (step 2, 3).
- **What stays fixed:** exp21 setup: erase on non-VIP rows only, p 0.5 per box, random color, box above 30% of image skipped; exp20 colored holes on non-VIP rows; `oversample = 8`, `downsample = 0.2`, full ratio `pos_weight`; 10 epochs, folds 0 and 1; exp16, exp20, exp21 results reused. Exam as exp23: fold model that held image out, fill gray 128, thresholds best-F1 on clean OOF rows, seed 0.
- **Metric:** Cat recall without head and ears, drop beyond random box (main); AP per class on out-of-fold rows of folds 0 and 1 (pooled), from PR curves (guard, VIP must not drop).

## Fixes

## Results

## AI Summary

## Conclusion

It did not work.