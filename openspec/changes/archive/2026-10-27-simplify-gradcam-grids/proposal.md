# Proposal

## Why

`core.analysis.gradcam` has three grid builders repeating same layout code. `positives` near-copy of `compare` with one model. Merge cuts code, keeps notebook API.

## What Changes

- `GradCamFigures.compare` gains `train=False`: per model, map by fold that held image out, or next fold (model trained on image).
- `GradCamFigures.positives` becomes thin call: `compare({config name: self}, cls, positive_rows(cls, n, seed), train=train)`.
- Overlay label in merged grid: `<name> · fold <k> · p <prob>` (was `fold <k> · p` in `positives`, `<name> · p` in `compare`).
- Suptitle: `<cls> · <split> · <N> labeled positives` in both views; `positives` loses config name prefix (name now on every overlay).
- No signature removed. `plot`, `pick`, `figures`, `load`, `probs`, `edit`/`edits`, `GradCam` untouched.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. Refactor plus figure label text; `skip_specs: true`.

## Impact

- `src/core/analysis/gradcam.py` only.
- Notebooks calling `positives` or `compare` (exp12-exp36) run unchanged; reruns show new label text. Saved figures in `docs/figures/` stay as is.
