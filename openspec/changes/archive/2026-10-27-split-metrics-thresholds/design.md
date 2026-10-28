# Design

## Context

Motivation: see proposal.md.

Scoring today: `Scores(y_true, probs)` in `analysis/scores.py`, rows x classes arrays, index `CLASSES`. Callers in `src`: `cv` (`fold_ap`, `fold_scores`, `scores`), `train.RunCheckpoint` (`ap` to MLflow), `ensemble.Ensemble.thresholds` (`plateau_f1`), `submission.Single.thresholds` (`best_f1`), `gradcam.GradCamFigures` (`best_f1` thresholds), `plots.pr_curves` (`best_f1` dot; own sklearn PR curve, AP). Notebooks: `r.scores.best_f1()`, `Scores(labels, probs).<rule>()`, `cross_fold_f1(folds, f)` with `RULES` dict of callables, `fixed_f1`, `plateau_f1`, two reads of `scores.y_true`/`scores.probs`.

## Goals / Non-Goals

**Goals:**
- Measuring and choosing thresholds in separate modules, plain functions, no class.
- Both modules free of `core` imports: class names passed in.
- Every number equal to old code.

**Non-Goals:**
- `MeanAP` (torchmetrics) and checkpoint selection.
- Helper for `probs >= thresholds`: one expression, stays inline.
- Notebook direct sklearn calls, saved notebook outputs.

## Decisions

**Signatures: `(y, probs, classes, ...)`, arrays rows x classes.** Same order everywhere; `classes` names result index. Alternative: DataFrame in (labels and `p_` columns); rejected, ties metrics to OOF column naming.

`metrics.py`:
- `ap(y, probs, classes) -> Series` (name `ap`).
- `pr_curve(y, p) -> (precision, recall, thresholds, f1)` for one class, 1-D; `f1` without last point (old `_f1_curve`).
- `f1_at(y, probs, thresholds, classes) -> DataFrame` (threshold, precision, recall, f1): old `fixed_f1` plus `_scores_at`.

`thresholds.py`:
- `best_f1(y, probs, classes) -> DataFrame` (threshold, precision, recall, f1).
- `plateau_f1(y, probs, classes, margin=0.05) -> DataFrame` (threshold, low, high, f1).
- `cross_fold_f1(y, probs, folds, classes, margin=None) -> Series`: thresholds tuned on other folds by `best_f1` (`margin` None) or `plateau_f1(margin)`. No rule callables: notebook `RULES` dicts become `{"peak": None, "plateau": MARGIN}`.

**`map` dropped.** sklearn macro AP = mean of per-class AP; callers use `ap(...).mean()`. Checked equal (tolerance 1e-12) during verification.

**Place: `core/analysis/`.** Next to `gradcam`, `plots`. Alternative: own package `core/metrics/`; rejected, two files.

**`CVResult.scores` removed, `labels`, `probs` kept.** Call site: `best_f1(r.labels, r.probs, CLASSES)`. Alternative: keep property returning tuple; rejected, hides argument order.

**Notebook rewrite by script with fixed substitution table**, then manual edit of leftovers (`scores.y_true`, rule passing). Code cells parsed with `ast` after edit.

## Risks / Trade-offs

- Notebook substitution misses pattern. Mitigation: after rewrite, `grep -rn "Scores\|\.scores\b\|core.analysis.scores" notebooks` empty except words in markdown.
- Numbers drift from reordering. Mitigation: snapshot old outputs (`ap`, `map`, `best_f1`, `plateau_f1`, `fixed_f1`, `cross_fold_f1` with both rules, `fold_scores`, `Single.thresholds`) on all 33 runs before deleting `scores.py`; compare after.
- Saved notebook outputs show old code. Accepted: notebooks not rerun at full scale.
