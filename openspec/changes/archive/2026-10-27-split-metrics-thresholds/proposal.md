# Proposal

## Why

`Scores` (`src/core/analysis/scores.py`) mixes measuring (AP, F1 at threshold) with choosing thresholds (best, plateau, cross-fold tuning), and imports project `CLASSES`. `plots.pr_curves` recomputes PR curve and AP with sklearn on its own. Metrics are most reusable part of `core`, but tied to project.

## What Changes

- New `core.analysis.metrics`: pure functions of labels, probs, class names: `ap`, `pr_curve`, `f1_at` (was `fixed_f1`). No `core` imports.
- New `core.analysis.thresholds`: threshold rules `best_f1`, `plateau_f1`, plus `cross_fold_f1` (thresholds tuned on other folds; `margin=None`: best F1, number: plateau). Imports only `metrics`.
- **BREAKING**: `Scores` class and `scores.py` removed; `Scores.map()` replaced by `ap(...).mean()`.
- **BREAKING**: `CVResult.scores` property removed; callers use `CVResult.labels`, `CVResult.probs` with functions.
- Callers moved: `cv.py`, `train.py` (`RunCheckpoint`), `ensemble.py`, `kaggle/submission.py`, `analysis/gradcam.py`, `analysis/plots.py` (`pr_curves` via `pr_curve`, `ap`).
- Notebooks (15 in `experiments/`, `submissions/`) rewritten to function calls.
- Out of scope: `MeanAP` (torchmetrics, training-time checkpoint selection) unchanged; notebook direct sklearn calls (`precision_recall_curve`) kept.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. Pure refactor: `skip_specs: true`.

## Impact

- Code: `src/core/analysis/{metrics,thresholds}.py` new, `src/core/analysis/scores.py` deleted, callers above, `.claude/project-structure.md`, `CLAUDE.md` helper line (`Scores.cross_fold_f1`).
- Notebooks: code cells only; outputs stay as saved.
- Numbers unchanged: thresholds, F1, AP equal to old code.
