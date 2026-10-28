# Training on Kaggle T4 x2

Folds of one config split between Kaggle kernels and local machine. Push described by job file `configs/kgpu/<job>.yaml`: `kgpu` (`src/kgpu/`, project independent, ported from `Kaggle GPU` project) packs `code` in, attaches `data`, pastes `kgpu.runner` into notebook. Inside kernel runner installs packages, sets env, runs one task command per GPU (here: one `core.training.cli` fold process). Training knowledge lives only in YAML. Pulled run folders sit next to local ones, `CVResult.load` reads all.

## Once

1. Kaggle account phone-verified: GPU and internet in kernels need it.
2. Kaggle CLI on PATH (`pipx install kaggle`), `kaggle config view` shows username.
3. `poetry install`: registers `kgpu` package.

## Job file

```yaml
# configs/kgpu/exp42_f234.yaml
defaults: {run: cv}                 # links run/cv.yaml, job keys win
code: [src/core, configs]           # bundle
pythonpath: [src]                   # bundle folders on sys.path, PYTHONPATH
data: ["competition:cats-and-dogs"]
vars:
  config: configs/models/exp42_oversample_8_only/resnet18.yaml
tasks:
  items: [2, 3, 4]                  # one task per fold
```

`device` default `t4x2` (`--device=local`: background run here), `buffer` default 30 minutes. Keys: `kgpu.job` docstring.

- `run/cv.yaml`: folds. `setup` (`mkdir`, `pip` always, `pip_missing` only if module absent), `env` (`{find: train.csv}` = folder of match under kernel inputs), `tasks.command` `python -m core.training.cli fit --config {config} --data.fold={item} ... --trainer.precision=16-mixed`, `logs/fold{item}.log`, MLflow `mlruns/fold{item}.db` per fold.
- `run/fit.yaml`: final model on every training row. Same `setup`, `env`; `tasks.items: [full]`, `command` `... --data.fold=null ...`, `logs/full.log`, MLflow `mlruns/full.db`.

Final model job: `defaults: {run: fit}` plus `code`, `pythonpath`, `data`, `vars.config` (example `configs/kgpu/exp43_full.yaml`); `core.kaggle.pull` imports run folder `full/`.

Placeholders: `{item}`, `{gpu}`, `{workers}` (CPUs per GPU), `{workdir}`, `{items}`, `vars` keys. Config overrides: job `tasks.command` with extra `--trainer.max_epochs=1` etc. Kernel name: job file stem (slug), or `name:`. Box recipes (`AttentionDrop`, `BoxCrop`, `FaceErase`, `HeadErase`, ...): `code: [src/core, configs, data/processed/boxes.csv]` (or `boxes_face.csv`, `boxes_head.csv`).

## Run

1. Push: `.venv/bin/python -m kgpu push configs/kgpu/<job>.yaml [--key=value ...] [--dry-run]`. Overrides dotted, values YAML: `--device=local`, `'--tasks.items=[3]'`.
2. Meanwhile local folds, one call each: `.venv/bin/python -m core.training.cli fit --config <model config.yaml> --data.fold=0`
3. Watch: `kgpu status [name]`, `kgpu track name` (bars per fold, ETA), `kgpu log name [-f]`, `kgpu devices`, `kgpu history`.
4. Pull when COMPLETE (or ERROR, finished folds only): `kgpu pull <name>`
5. Import folds: `.venv/bin/python -m core.kaggle.pull configs/kgpu/<job>.yaml [--output=<folder>]` (model config `vars.config`, folds `tasks.items`, default output `data/kernels/<name>/output`).
6. Read: notebook `CVResult.load(<model config.yaml>)`: every finished fold folder, OOF from its `oof.parquet` (kernel predicts it).

`kgpu` = `.venv/bin/python -m kgpu`: `.venv/bin/kgpu` script broken, project path has spaces.

## Notes

- Push = new kernel version, starts run, spends weekly quota (`kaggle quota`: GPU 30 h, TPU 20 h). Push refused while same name queued or running. Kernel page: kaggle.com/code/<username>/<name>.
- Kernel folder: `data/kernels/<name>/` (built notebook, metadata, `runs.jsonl` history with resolved source, `output/` after pull). `--dry-run` writes it without upload. `[tool.kgpu]` of `pyproject.toml`: only `home` (`data/kernels`).
- Deadline: first cell sets `KGPU_DEADLINE` = session limit (GPU 12 h, TPU 9 h) minus job `buffer` minutes (default 30). `core.training.cli` `deadline_timer` (Lightning `Timer`) stops fit there, best checkpoint and OOF still written.
- Progress: `core.training.train` `Progress` callback prints `kgpu:progress {json}` per epoch; `kgpu status`/`track` parse it from kernel log.
- No GPU on `t4x2` (quota not granted): runner exits before tasks. TPU: `kgpu` supports it, `core` not (CUDA GPU needed).
- Pulled folds: run folder into `data/processed/cv/<experiment>/<name>/fold_<k>/`, MLflow run copied into local `mlflow.db` (new run id, `remote_run_id` in `run.json`).
- Kernel keeps Kaggle image torch (CUDA, 16-mixed precision); local MPS runs 32-bit. Other packages installed only if missing, versions differ from local pins. mlflow pinned to local version (`run/cv.yaml`, `run/fit.yaml` `setup.pip`): newer kernel database schema unreadable locally. Bump both together.
- `src/core/` must parse on Kaggle image Python (3.11+): no 3.12+ only syntax.
- Failed fold: `data/kernels/<name>/output/logs/fold<k>.log`, kernel log via `kgpu log <name>`.
