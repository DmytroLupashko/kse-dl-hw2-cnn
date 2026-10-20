# Experiment 29: Downsampling without coarse dropout, 5 folds

## Request

aggree, build submission with downsampling and without dropout

create an experiment card with conclusion: downsampling includes public score signifficantly, while it still showed worse result then using with dropout.

## Question

Does exp16 setup (downsampling, no coarse dropout) on 5 folds reach Kaggle public Macro F1 of exp26 (0.880), i.e. did exp26 gain over exp11 (0.828) come from downsampling?

## Plan

1. Tabulate Kaggle public Macro F1 of exp11, exp29 and exp26 submissions.

## Setup

- **Notebook:** [exp29_downsample_5fold.ipynb](../notebooks/submissions/exp29_downsample_5fold.ipynb), config `configs/exp29_downsample_5fold/resnet18.toml`, training `core.cv`
- **Tracking:** `cnn-model-downsample-5fold`
- **What changes:** exp16 setup on `folds = [0, 1, 2, 3, 4]`; vs exp26: no holes (`augment` default); vs exp11: `downsample = 0.2`, `pos_weight` counted over draws (VIP 29% of draws instead of 8%).
- **What stays fixed:** exp11 submission recipe: 5 fold models averaged, per-class best-F1 thresholds on pooled OOF rows; `oversample = 8`, 10 epochs, 224 px.
- **Metric:** Kaggle public Macro F1.

## Fixes

## Results

| submission | downsample | coarse dropout | public Macro F1 |
|---|---|---|---|
| exp11 | no | no | 0.828 |
| exp29 | 0.2 | no | 0.864 |
| exp26 | 0.2 | non-VIP rows | 0.880 |

Kaggle public leaderboard, Macro F1 over 5 classes.

## AI Summary

Mostly: downsampling alone gives +0.036 of exp26's +0.052 over exp11, coarse dropout adds remaining +0.016. Both parts help on test photos although CV could not separate them.

## Conclusion

downsampling includes public score signifficantly, while it still showed worse result then using with dropout.
