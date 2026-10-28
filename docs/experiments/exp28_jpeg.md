# Experiment 28: JPEG quality 10, non-VIP vs all

## Request

Another idea: keep size of image as 224px, but lower it's quality

build exp28 based on ImageCompression applied on not VIP. result should include precision recall curves, train/val gradcam visualizations for cats & dogs, train/val visualizations for VIP animals

remake experiment: I want to check image compression 10 only. what can be potential risks

try 2 setups: apply image compression only on non VIP and on all. in both cases, validation images stay untouched

## Question

On exp19 images (16 random Cat and 16 random Dog positives, folds 0 and 1), does exp20 setup plus JPEG quality 10 on training images (non-VIP rows only, or all rows) put Grad-CAM heat on whole body (not only face) in more images than exp20?

## Plan

1. Plot JPEG examples: 6 random Cat or Dog non-VIP images at 224 px, original and quality 10.
2. Plot Grad-CAM Cat and Dog validation figures of exp20, JPEG non-VIP and JPEG all on exp19 images, maps of same image stacked, caption with count of images with heat on body by eye.
3. Plot Grad-CAM Cat and Dog train figures (`GradCamFigures.positives`, `train=True`) of both JPEG models on exp19 images, 4 figures.
4. Plot Grad-CAM VIP figures (`GradCamFigures.positives`) of both JPEG models: per VIP class all labeled positives of folds 0 and 1, unseen and trained images, 12 figures.
5. Plot precision-recall curve per class on out-of-fold rows of folds 0 and 1, one panel per class (5), one line per model: exp20, JPEG non-VIP, JPEG all, AP in legend.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp28_jpeg.ipynb](../notebooks/experiments/exp28_jpeg.ipynb), configs `configs/exp28_jpeg/resnet18_non_vip.toml`, `resnet18_all.toml`, training `core.cv`
- **Tracking:** `cnn-model-jpeg`
- **What changes:** JPEG quality 10 on half of training draws, last before `Normalize`; validation images untouched.
  - `augment = "jpeg_non_vip"`: `NonVipImageCompression`, non-VIP rows only.
  - `augment = "jpeg_all"`: `ImageCompression`, all rows, VIP too.
  - Risk: in non-VIP setup blocks may become "not VIP" shortcut (all setup removes it); clean test images then lean to VIP (false positives). In all setup, coat details of VIP lost on half of VIP draws: VIP AP may drop. Colour shift at quality 10 (green patches) may cost Cat, Dog AP. Low-frequency outline kept, large faces too: heat may not move.
- **What stays fixed:** exp20 setup, 224 px; exp20 results reused.
- **Metric:** count of exp19 images with Grad-CAM heat on body, by eye (main); AP per class on out-of-fold rows of folds 0 and 1 (pooled), from PR curves (sanity check).

## Fixes

## Results

![JPEG quality 10 examples](../figures/exp28_jpeg_examples.png)

6 random non-VIP Cat or Dog training images at 224 px; quality 10: colour blocks, fur smeared, green patches on faces, outline kept.

![Cat, exp20 vs JPEG non-VIP vs JPEG all](../figures/exp28_jpeg_grad_cam_cat_compare.png)

![Dog, exp20 vs JPEG non-VIP vs JPEG all](../figures/exp28_jpeg_grad_cam_dog_compare.png)

exp19 images, 16 per class, out-of-fold rows of folds 0 and 1; caption: model, prob, green at or above own best-F1 threshold, red below. Maps almost identical to exp20 in all three rows. Heat on body, by eye: Cat about 3 of 16 in all three; Dog exp20 about 4, both JPEG about 5 (dog behind diamond fence: heat spreads along body; Boston terrier: non-VIP spreads over neck and chest). Leopard-print chair kitten still missed by all (p 0.043 / 0.042 / 0.072); JPEG non-VIP heat there on image corners. Boston terrier missed by both JPEG (p 0.536, 0.643).

![JPEG non-VIP, Cat train](../figures/exp28_jpeg_non_vip_grad_cam_cat_train.png)

![JPEG non-VIP, Dog train](../figures/exp28_jpeg_non_vip_grad_cam_dog_train.png)

![JPEG all, Cat train](../figures/exp28_jpeg_all_grad_cam_cat_train.png)

![JPEG all, Dog train](../figures/exp28_jpeg_all_grad_cam_dog_train.png)

Same 16 Cat and 16 Dog images, each by fold model that trained on it.

![JPEG non-VIP, Moris validation](../figures/exp28_jpeg_non_vip_grad_cam_moris_validation.png)

![JPEG non-VIP, Moris train](../figures/exp28_jpeg_non_vip_grad_cam_moris_train.png)

![JPEG non-VIP, Motya validation](../figures/exp28_jpeg_non_vip_grad_cam_motya_validation.png)

![JPEG non-VIP, Motya train](../figures/exp28_jpeg_non_vip_grad_cam_motya_train.png)

![JPEG non-VIP, Biatrix validation](../figures/exp28_jpeg_non_vip_grad_cam_biatrix_validation.png)

![JPEG non-VIP, Biatrix train](../figures/exp28_jpeg_non_vip_grad_cam_biatrix_train.png)

![JPEG all, Moris validation](../figures/exp28_jpeg_all_grad_cam_moris_validation.png)

![JPEG all, Moris train](../figures/exp28_jpeg_all_grad_cam_moris_train.png)

![JPEG all, Motya validation](../figures/exp28_jpeg_all_grad_cam_motya_validation.png)

![JPEG all, Motya train](../figures/exp28_jpeg_all_grad_cam_motya_train.png)

![JPEG all, Biatrix validation](../figures/exp28_jpeg_all_grad_cam_biatrix_validation.png)

![JPEG all, Biatrix train](../figures/exp28_jpeg_all_grad_cam_biatrix_train.png)

All labeled positives per VIP class in folds 0 and 1 (22 Moris, 17 Motya, 16 Biatrix). Validation misses at own best-F1 threshold, exp20 / non-VIP / all: Moris 3 / 2 / 4, Motya 2 / 5 / 6, Biatrix 0 / 0 / 2. False positives on non-VIP rows: Moris 0 / 10 / 1, Motya 0 / 0 / 1, Biatrix 0 / 1 / 0.

![Precision-recall curves](../figures/exp28_jpeg_pr_curves.png)

AP per class, out-of-fold rows of folds 0 and 1 pooled (4904 rows), exp20 / non-VIP / all: Cat 0.999 / 0.998 / 0.998, Dog 0.998 / 0.996 / 0.997, Moris 0.983 / 0.864 / 0.944, Motya 0.931 / 0.860 / 0.767, Biatrix 1.000 / 0.988 / 0.936.

## AI Summary

No: Cat and Dog maps almost unchanged in both setups, body heat about one Dog image more than exp20, Cat same. Both cost VIP AP; non-VIP setup shows predicted shortcut (10 Moris false positives on clean images), all setup loses most on Motya (0.931 to 0.767).

## Conclusion

The goal was to increase robustness of the model by aplying it.
