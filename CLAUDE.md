# CLAUDE.md

@.claude/context/project-structure.md

## Project

- Multilabel image classification, 5 classes, Macro F1.
- Task, submission format: `docs/task/overview.md`.
- Rules: `docs/task/rules.md`.
- No external data, no relabeling.
- Pretrained weights only from timm or ImageNet.
- Chosen metric, loss, validation: `docs/neuroslope/methods/configuration.md`.
- Never run notebooks at full scale.
- Notebooks: little text.

## Experiments

- Tracking prefix `cnn`: MLflow experiments `cnn-expNN-<slug>`; older runs keep `cnn-<section>-<slug>`.
- MLflow database and artifacts: `data/mlflow/`.
- Jupyter: `.venv/bin/jupyter`
- Notebook helpers: `helpers.plots` (`use_style`, `pr_curves`, `save_figures`), `CVResult.fold_scores`, `.history`, `core.metrics.cross_fold_f1`.
- New experiments: Claude only drafts card from my request.
- Card Plan = exact text of plan shown in chat, same wording, same length.
