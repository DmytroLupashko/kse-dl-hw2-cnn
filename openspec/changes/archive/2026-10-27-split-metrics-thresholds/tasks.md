# Tasks

## 1. Snapshot

- [x] 1.1 With old `Scores`, pickle per run of all 33 runs: `ap`, `map`, `best_f1`, `plateau_f1`, `fixed_f1(best thresholds)`, `cross_fold_f1(folds)` with `best_f1` and `plateau_f1` rules, `fold_scores`, `fold_ap`, `Single.thresholds`, `Ensemble.thresholds` (exp11 + exp26) to scratchpad; verify pickle holds 33 runs

## 2. New modules

- [x] 2.1 Create `src/core/analysis/metrics.py` (`ap`, `pr_curve`, `f1_at`) and `src/core/analysis/thresholds.py` (`best_f1`, `plateau_f1`, `cross_fold_f1`), no `core` imports; add `tests/test_metrics.py` (hand-computed small case per function, `cross_fold_f1` with `margin=None` and `margin=0.05`); verify `python -m unittest discover -s tests -p "test_metrics.py"` passes and `grep -n "from core\|from \.\." src/core/analysis/{metrics,thresholds}.py` empty
- [x] 2.2 New functions vs snapshot on all 33 runs (`ap(...).mean()` vs `map` within 1e-12, rest exact); verify comparison script reports 0 diffs

## 3. Callers in `src`

- [x] 3.1 `cv.py`: `fold_ap`, `fold_scores`, `_per_fold` on functions; remove `scores` property; verify `fold_scores`, `fold_ap` equal snapshot
- [x] 3.2 `train.RunCheckpoint`, `ensemble.Ensemble.thresholds`, `submission.Single.thresholds`, `gradcam.GradCamFigures`, `plots.pr_curves` (`pr_curve`, `ap`) on functions; verify `Single.thresholds`, `Ensemble.thresholds` equal snapshot, `pr_curves({...})` and `GradCamFigures(<exp20 config>).pick("Moris", 4)` run without error, smoke fit `--trainer.max_epochs=1 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2` on throwaway experiment logs `best_ap_*`
- [x] 3.3 Delete `src/core/analysis/scores.py`; update `.claude/project-structure.md` (`analysis/` line) and `CLAUDE.md` helper line (`Scores.cross_fold_f1`); verify `grep -rn "Scores\|analysis.scores\|\.scores\b" src tests .claude CLAUDE.md` empty

## 4. Notebooks

- [x] 4.1 Rewrite code cells of `notebooks/experiments`, `notebooks/submissions` (imports, `r.scores.<rule>()`, `Scores(y, p).<rule>()`, `RULES` dicts to `{"peak": None, "plateau": MARGIN}` with `margin=` arg, `scores.y_true`/`probs` reads); verify all code cells parse (`ast`) and `grep -rn "Scores\|\.scores\b\|analysis.scores" notebooks/experiments notebooks/submissions` hits only markdown text
- [x] 4.2 Run one rewritten cell per pattern on real runs (`best_f1`, `cross_fold_f1` both margins, `fixed_f1`, `plateau_f1`) in throwaway script; verify output equals snapshot

## 5. Integration

- [x] 5.1 Full `python -m unittest discover -s tests`; verify only failure is pre-existing `test_custom_model.test_stage_shapes`

## Workflow follow-up

- Archive change after review.
