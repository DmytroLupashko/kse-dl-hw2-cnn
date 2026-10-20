# Mission: ML evaluation metrics

## Why
Understand classification metrics deeply, beyond this homework: know what each metric measures, when it misleads, which to use for model selection versus final score.

## Success looks like
- Explain why threshold-free metric (mAP, ROC-AUC) and thresholded metric (F1) disagree, with own example.
- Pick selection metric and threshold strategy for new task without looking it up.
- Spot misleading metric in someone else's report (AUC under heavy imbalance, F1 at untuned threshold).

## Constraints
- Grounded in `docs/metrics.md` and Cats and Dogs++ multilabel task.
- Clear, short writing; no tangents.

## Out of scope
- Detection and segmentation metrics (COCO mAP, IoU), unless asked.
