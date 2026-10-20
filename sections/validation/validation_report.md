# Validation Report

## What metric

Selection Metric: mAP

## CV or Handout

Two modes, same `MultilabelStratifiedKFold` folds:

(Fast) Folds 0 and 1, mean of 2 fold mAPs. Every experiment, data augmentation included.

(Full) 5 folds. Once for baseline winner (fold spread = real noise, check folds 0 and 1 typical), close finalists, final run.

## What stratification

Multilabeled Startification

## Mean over folds ot OOF

Mean over folds, because OOF pooling breaks independece of AP on scale, which is bad for later tuning.

## My Notes

I think I will stick to Stratified K-Fold and that's it.

**AI metric suggestion**. Moris, Motya, Biatrix have 41 to 56 train images, about 8 to 11 per validation fold, and drive 60% of Macro F1, so CV score noisy. Every VIP image also has parent label (Cat or Dog), i.e. not images where, for example, Moris = 1 and Cat = 0.

**My reponse**:
- I plan to use Handout, not CV. I am forced to CV because of small classes.


- Does it make sense to use Handout for choosing and tuning model, stratified k-fold for final evaluation?
- Should I use different metrics for train and validation?

**AI answer: holdout for selection, k-fold for final**

Holdout no collapse. Training identical; only score gets noisier. Question: is noise smaller than differences being compared?

Holdout = fold 0 of existing `MultilabelStratifiedKFold` (`fold=0` in DataModule): 20% validation, about 8 to 11 positives per rare class. AP on 8 positives has standard error roughly 0.1. Macro mAP averages 5 classes, 3 rare: $SE \approx \sqrt{3 \cdot 0.1^2}/5 \approx 0.035$. Cat and Dog add almost nothing (about 1000 positives each). So gap below about 0.07 mAP between two separately scored models not evidence.

Paired comparison shrinks this. Two models on same validation images share hard images, errors correlated, so noise of difference lower than noise of each score. Measure it free: paired bootstrap on holdout predictions (resample validation rows 1000 times, recompute mAP difference, take 2.5% and 97.5% quantiles). Interval excludes 0: decision made on holdout. Interval includes 0: escalate to 5-fold for those candidates only.

What holdout decides well: big switches (backbone family, pretrained vs scratch, augmentation on vs off, resolution), usually gaps above 0.05. What it decides badly: small tweaks (augmentation strength, LR fine steps), gaps of 0.01 to 0.02 drown in rare-class noise; those need k-fold or are not worth tuning.

Plan:

1. Iteration: holdout fold 0, mAP plus paired bootstrap interval vs current best.
2. Escalation: 5-fold only when interval includes 0 and choice matters.
3. Final: 5-fold on 1 to 3 final candidates, mean of fold mAP for choice, pooled OOF for per-class threshold tuning, 5 fold models averaged at inference.

Threshold tuning on holdout alone risky: 8 positives, 9 possible recall values. Thresholds from final 5-fold OOF only.

Cross-fold caveat: holdout fold 0 slightly easier or harder than average fold. Absolute holdout score not comparable to 5-fold mean; compare holdout only to holdout.

Second question: train loss BCE with `pos_weight`, validation metric mAP for selection and Macro F1 for final. Different by design: loss must be differentiable, metrics need not. Normal.

**AI answer: 2 folds instead of 1**

Meaning: train on folds 0 and 1 of existing 5-fold split, each with 80% training data, score mean of 2 fold mAPs. Not `n_splits=2`.

Effect: 2x cost (about 8 min for AlexNet). Validation positives double, about 16 to 22 per rare class. Noise drops by $\sqrt 2$: macro mAP $SE$ from about 0.035 to 0.025, smallest trustworthy gap between separately scored models from about 0.07 to 0.05. Paired per-fold differences also show if winner wins on both folds; with 1 fold no such check.

Diminishing returns: $SE \propto 1/\sqrt{m}$ for $m$ folds used. 1 to 2 folds cuts noise 29%, 2 to 5 folds cuts it further only 37% for 2.5x more cost.

Bonus: folds 0 and 1 are part of final 5-fold. If winning recipe unchanged, final CV needs only folds 2 to 4, so 2-fold iteration costs nothing extra for winner.

Not same as `n_splits=2` (50/50 split): each model trains on half data, rare classes left with about 20 to 28 training images, score pessimistic, ranking can favor models that learn from few samples rather than model best on full data. Avoid.

Choice: 1 fold for big switches with large gaps, 2 folds as default when gaps around 0.03 to 0.05 matter.

**AI answer: when 5 folds**

5 folds needed when every image must get out-of-fold prediction, or when choice too close for 2 folds.

1. Once, on baseline: fold spread of 5 fold mAPs measures real noise, replaces rough $SE$ estimates above. Also shows if folds 0 and 1 typical; outlier fold biases all 1- or 2-fold decisions.
2. Close finalists: 2-fold paired difference inside noise, choice matters (e.g. final backbone). 1 to 3 candidates only.
3. Threshold tuning: needs pooled OOF over all images, rare class 41 to 56 positives instead of 8 to 22. Only full 5 folds give it.
4. Ensemble: blend weights or stacking fit on OOF predictions of all images.
5. Submission: 5 fold models averaged at test time, free ensemble; OOF Macro F1 with tuned thresholds is final estimate.

Points 3 to 5 come from same run: final 5-fold of winner gives thresholds, ensemble inputs, submission models at once. Folds 0 and 1 reused from iteration if recipe unchanged.

**AI review: strategy before notes**

Metric, mAP: good choice. No threshold, smooth, less noisy than F1 on 8 positives, tracks F1 after threshold tuning. Gap: mAP blind to thresholds, so final decision also checks OOF Macro F1 with tuned thresholds.

CV or holdout: conclusion right for final run, reason too strong. Small classes make holdout noisy, not unusable; noise measurable with paired bootstrap. Section reads as "always 5-fold", which costs 20 min per experiment. Tiered plan (1 fold, 2 folds, 5 folds, above) belongs here.

Stratification, multilabel: correct, nothing to change. Parent labels (Moris implies Cat) preserved automatically: iterative stratification balances all labels at once.

Mean over folds: right choice, reason muddled. Pooling does not hurt later tuning; thresholds still tuned on pooled OOF. Real reason: pooled AP compares scores across fold models, so scale mismatch between them (different early-stopping epoch, `pos_weight`) lowers AP though each model ranks own fold well. Mean of fold APs measures ranking only.

Missing, biggest risk: leakage check. EDA has no duplicate or near-duplicate search. VIP pets likely photographed in sessions: near-identical photos of Moris in train and validation folds inflate rare-class AP, and then selection favours models that memorize backgrounds. Check: perceptual hash or embedding similarity on train images; if clusters found, `StratifiedGroupKFold`-style split keeping clusters whole (iterative stratification over groups).

Missing: decision rule. No statement when difference counts as real. Add: difference counts only if paired bootstrap interval (1 or 2 folds) or per-fold differences (5 folds) exclude 0.

**AI answer: 3 modes (1 fold, 2 folds, 5 folds)**

Idea sound, 3 modes too many. Simpler: 2 modes.

1. Fast: folds 0 and 1, every experiment. 8 min for AlexNet vs 4 min for 1 fold: 4 min saved not worth single-fold quirks deciding. Per-fold agreement check needs at least 2 folds.
2. Full: 5 folds, for baseline (noise calibration), close finalists, final run (thresholds, OOF, submission).

1 fold only for debugging, never for decisions.

Risks of any multi-mode plan:

1. Optional stopping: adding folds only when result disappoints (or stopping early when it pleases) biases selection toward lucky candidates. Fix: escalation rule decided before seeing result, applied to both candidates.
2. Mixed comparisons: 2-fold mean and 5-fold mean differ by fold difficulty. Compare only within same mode, same folds. Log mode (fold list) as MLflow param.
3. Recipe drift: folds 0 and 1 reusable in full run only if recipe identical (seed, epochs, transforms).

**AI answer: why 5 folds for baseline**

Purpose: calibrate fast mode once, not compare models.

1. Real noise. 0.035 above is formula guess (AP $SE \approx 0.1$ assumed). Spread of 5 fold mAPs measures actual noise: split, training randomness, rare-class luck together. 2 folds give 2 numbers, standard deviation from 2 points meaningless. Every later fast-mode decision ("gap above X counts") rests on this X.
2. Folds 0 and 1 typical? If fold 0 holds unusually easy Biatrix images, fast mode inherits that bias for whole project. Only 5 folds show it.

Cost: one model, once, folds 2 to 4 only (0 and 1 reused), 12 min for AlexNet. Not 5 folds for all 4 models of `exp01_lecture_models`: compare 4 models in fast mode, run folds 2 to 4 for winner only. Winner needs them anyway for final run.

Skippable: accept formula noise estimate, risk fold 0 and 1 atypical.
