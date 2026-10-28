# Refactor: project layout

Uncommitted change on top of `138e18d`. Layout only: no training, metric or threshold logic changed.

## Code

Package `src/core/` split into three top-level packages: `src/dataset/` (was `core/data/`), `src/training/`, `src/analysis/`. Imports now `training.cv`, `dataset.augment`, `analysis.plots`, no `core.` prefix. Relative imports (`..training.config`) replaced by absolute. `pyproject.toml` lists three packages instead of `core`.

`training/config.py`: `ROOT` now `parents[2]`, one level less nesting. `training/model.py` keeps loading old checkpoints: module path starting `core.` mapped to `training.model`.

## MLflow

Database moved from `mlflow.db` at root to `data/mlflow/mlflow.db`; artifacts to `data/mlflow/artifacts/<experiment>/`, not `./mlruns` of current folder. New `tracking.artifact_location(experiment)`; `start_run` and `cli` create experiment with it. `.gitignore` drops `mlruns/`, `mlflow.db` (already under `/data/`), adds `.ruff_cache/`.

## Configs, notebooks

Configs moved `configs/models/expNN_<slug>/` to `configs/expNN_<slug>/`; `class_path` values lose `core.` prefix. Experiment and submission notebooks: imports and card links updated, nothing else.

## Docs

AI-written docs moved under `docs/neuroslope/`: `methods/`, `reports/`, `experiments/`, `code notes/`, `backlog.md`, `ai_usage/` (was root `ai_usage/`), `theory_answers.md` (was `docs/task/`). `docs/task/`, `docs/figures/` stay. `REPORT.md` became `README.md`, links pointed to `docs/neuroslope/`; old README tree removed.

## Tooling removed

OpenSpec gone: `openspec/` (specs, archived changes), `.claude/commands/opsx/`, `.claude/skills/openspec-*`. Trunk gone: `.trunk/`.

## Claude instructions

`CLAUDE.md`, `.claude/context/project-structure.md`, `.claude/rules/training.md`, experiment skill: new paths, new import names, MLflow location. Structure doc gains Branches section: `playground` (scratch, never merged into `main`), `playground-archive` (orphan, read-only); gitignored `notebooks/playground/`, `configs/models/playground/` dropped. Experiment skill smoke cleanup no longer mentions `checkpoints/<experiment>-smoke/`.

## Earlier commits

`1fd20e7` dropped code unused by exp04–exp38 (`core/boxes/`, unused augment recipes, model, ensemble, plots helpers), flattened `data/augment/` into one file. `138e18d` removed `tests/`. Plan behind both: [git-refactor.md](git-refactor.md).
