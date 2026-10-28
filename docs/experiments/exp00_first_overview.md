# Experiment 01: First overview

## Request

describe sizes of images, their format, contact sheets for all 5 classes, cat & dog combination, all 8 VIP animals combinations

## Question

What image sizes and formats appear, how many train images fall in each class (ratio), how often each of 2^5 label combinations appears, and how classes, images without cats or dogs, and Cat + Dog images look?

## Plan

1. Read header of every train and test image with PIL (no pixel decode) and count present (width, height, format, mode) combinations with ratio, in separate table for train and for test.
2. Plot histogram of train image ratio per class (`test.csv` has no labels).
3. Count all 2^5 = 32 label combinations in train with ratio, zero rows kept.
4. Show contact sheet of random sample, count in title, for each of 5 classes, "no cats or dogs" (all labels 0) and "cat and dog" (Cat = 1 and Dog = 1).

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp01_first_overview.ipynb](../notebooks/experiments/exp01_first_overview.ipynb)
- **Tracking:** `cnn-eda-first-overview` (counts, size stats, figures as artifacts)
- **What changes:** nothing, descriptive only
- **What stays fixed:** `SEED` from `core.config`, 16 images per contact sheet
- **Metric:** image counts; width, height, format, mode distributions

## Fixes

- class ratio plot should include 'no cats no dogs', plot should be more beautiful
- each image in contact sheet should be scaled by 0.5 by each axis, change n_cols to 8

## Results

All 12260 train and 18673 test images: 450 x 375, JPEG, RGB (ratio 1.0 each).

![Class ratio](../figures/exp01_first_overview_classes.png)

Present label combinations (24 of 32 have zero images):

| Cat | Dog | Moris | Motya | Biatrix | count | ratio |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 4989 | 0.4069 |
| 0 | 1 | 0 | 0 | 0 | 4984 | 0.4065 |
| 0 | 0 | 0 | 0 | 0 | 2162 | 0.1763 |
| 1 | 0 | 1 | 0 | 0 | 42 | 0.0034 |
| 0 | 1 | 0 | 0 | 1 | 37 | 0.0030 |
| 1 | 0 | 0 | 1 | 0 | 32 | 0.0026 |
| 1 | 0 | 1 | 1 | 0 | 10 | 0.0008 |
| 1 | 1 | 1 | 0 | 1 | 4 | 0.0003 |

Contact sheets:

![Cat](../figures/exp01_first_overview_sheet_cat.png)

![Dog](../figures/exp01_first_overview_sheet_dog.png)

![Moris](../figures/exp01_first_overview_sheet_moris.png)

![Motya](../figures/exp01_first_overview_sheet_motya.png)

![Biatrix](../figures/exp01_first_overview_sheet_biatrix.png)

![No cats or dogs](../figures/exp01_first_overview_sheet_no_cats_or_dogs.png)

![Cat and dog](../figures/exp01_first_overview_sheet_cat_and_dog.png)

Metric: image count and ratio; sizes on all train and test rows, classes and combinations on all 12260 train rows.

## AI Summary

Every image is 450 x 375 RGB JPEG: one fixed size, aspect 1.2 (not square); Cat and Dog each cover about 41% of train, 17.6% have no cat or dog, and each VIP animal has only 41 to 56 images. Only 8 of 32 label combinations occur, every VIP image also carries its parent Cat or Dog label, and Cat + Dog appears only in 4 images with Moris and Biatrix.

## Conclusion

- There are very few images for VIP animals.
- The only situation when there are more then 1 animal (cat / dog) on picture – moris and betrix show up together.
- Trying to classify only VIP animals and then extrapolate is really bad idea.
- Images for Beatrix are often showed up from the distance, so "Zooming is needed". In the same time, it is worth keeping not Zoomed version. (But what to do with images where there are Moris and Beatrix?)
- 