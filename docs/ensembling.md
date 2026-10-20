# Ensembling in computer vision

Gain comes from models that err on different images: different architectures (CNN vs ConvNeXt), input views, seeds.

1. Pick diverse members: other backbone or resolution beats same model twice; check error overlap on OOF rows, Grad-CAM regions (exp36).
2. Add test-time augmentation per member: horizontal flip (plus small crops), mean of probs over views.
3. Combine per class: mean of sigmoid probs (default), weighted mean, or stacking meta-model fit on out-of-fold probs only.
4. Retune per-class thresholds on combined OOF probs; ensemble probs calibrated differently from any member.
5. Keep ensemble only if honest Macro F1 (thresholds from other folds) beats best single model beyond fold noise.

Pitfall: weights or meta-model chosen and scored on same OOF rows: optimistic. Choose on other folds, as thresholds.

## Averaging, no stacking

1. Stack OOF probs of each member: same rows, same class order.
2. Mean per class over members; optional weights per member, chosen on other folds.
3. Tune per-class thresholds with `best_f1` on mean OOF probs.
4. Test: per member mean of 5 fold models (`test_probs`), then same mean and weights, then thresholds.

## Stacking with logistic regression

1. Per class, features = that class's logit (`log(p / (1 - p))`, `p` clipped to `[1e-6, 1 - 1e-6]`) from each member; one `LogisticRegression` per class.
2. Honest OOF: for each fold, fit on other 4 folds' OOF rows, predict this fold.
3. Penalty L2 (scikit-learn default, `l1_ratio = 0`): shrinks correlated members' weights together. `C` not guessed: compare few values (e.g. 0.01, 0.1, 1, scikit-learn default) by honest Macro F1.
4. Tune thresholds on stacked OOF probs; score Macro F1, compare with plain mean.
5. Test: refit on all OOF rows, apply to members' test logits, then thresholds.

Caveat: test probs come from mean of 5 fold models, smoother than single-model OOF probs; meta-model weights may transfer imperfectly.

Learning materials:

- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning*, section 4.3 "Logistic Regression": intuition of logistic regression itself, logit and coefficients.
- Ting, Witten, "Issues in Stacked Generalization" (JAIR 10, 1999): why meta-model should take members' class probabilities, not hard labels, and simple linear meta-model works best.
- scikit-learn User Guide, [section 1.11.6 "Stacked generalization"](https://scikit-learn.org/stable/modules/ensemble.html#stacking): `StackingClassifier`, meta-model trained on `cross_val_predict` out-of-fold predictions, same idea as step 2.

## Sources

- Goodfellow, Bengio, Courville, *Deep Learning*, section 7.11 "Bagging and Other Ensemble Methods": why averaging models with uncorrelated errors cuts error.
- Lakshminarayanan, Pritzel, Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017): averaging independently trained image nets, accuracy and calibration gains.
- Ha, Liu, Liu, [winning solution to SIIM-ISIC Melanoma](https://arxiv.org/abs/2010.05351) (2020): practical Kaggle image ensemble, diverse backbones and input sizes, TTA, stable CV.
- NVIDIA, [Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data), hill climbing and stacking: how to pick members and weights on OOF; tabular examples, method same for images.
