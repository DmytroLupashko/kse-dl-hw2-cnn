# Experiment 19: Grad-CAM cats and dogs

## Request

create experiment 19, where show GradCam for small sample of cats and dgos

## Question

For Cat and Dog, does exp16 ResNet-18 put Grad-CAM heat on animal in random labeled positives held out from training?

## Plan

1. Plot one figure per class (Cat, Dog): 16 random labeled positives from folds 0 and 1, original on top, Grad-CAM overlay of class logit below, by fold model that held image out, sorted by its probability.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp19_grad_cam_cats_dogs.ipynb](../notebooks/experiments/exp19_grad_cam_cats_dogs.ipynb), no training
- **Tracking:** none
- **What changes:** classes Cat and Dog instead of VIP; random sample of 16 positives per class, fixed seed, instead of all positives.
- **What stays fixed:** exp12 method, `core.gradcam.GradCamFigures.positives`; exp16 checkpoints and OOF, folds 0 and 1, Grad-CAM on `layer4`, `val_tfms` 224 px, thresholds best-F1 on OOF rows
- **Metric:** none, visual

## Fixes

## Results

![Cat validation](../figures/exp19_grad_cam_cats_dogs_cat_validation.png)

![Dog validation](../figures/exp19_grad_cam_cats_dogs_dog_validation.png)

16 random labeled positives per class (seed 0) from out-of-fold rows of folds 0 and 1, sorted by out-of-fold prob; caption: fold model, its prob, green at or above best-F1 threshold (Cat 0.666, Dog 0.392), red below. Misses: Cat 1 (kitten on leopard-print chair, p 0.101, heat on chair back), Dog 1 (Boston terrier side view, p 0.086, heat on red collar only).

## AI Summary

Mostly yes, about 14 of 16 per class: heat sits on face (cats: eyes and nose; dogs: snout), dogs behind fences or cages included. Weak hits and misses put heat off face: white cat neck (p 0.821), dog in cage on chain-link fence (p 0.955), leopard-print chair, red collar.

## Conclusion
