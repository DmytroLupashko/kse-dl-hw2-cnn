---
paths: ["configs/**", "src/**"]
---

# Training

- LightningCLI only, shell only, one run per fold: `python -m core.cli fit --config <config.yaml> --data.fold=<k> [--trainer.max_epochs=1 ...]`.
- Final model on every row: `--data.fold=null` (no validation, folder `full/`).
- Run folder: `data/processed/cv/<experiment>/<name>/fold_<k>/` (checkpoint, `config.yaml`, `oof.parquet`, `run.json`); rerun retrains, replaces folder.
- Notebook only reads: `CVResult.load(path)`, one cell per model.
