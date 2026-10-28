# Tasks

## 1. Merge

- [x] 1.1 Add `train=False` to `GradCamFigures.compare`: fold per model held-out or next; label `<name> · fold <k> · p <prob>`; suptitle split from `train`; verify `python3.11 -m py_compile src/core/analysis/gradcam.py` passes
- [x] 1.2 Replace `positives` body with `compare({self.config["name"]: self}, cls, self.positive_rows(cls, n, seed), train=train)`; update docstrings; verify same signature as before

## 2. Docs

- [x] 2.1 Grep `docs/` for `positives`/`compare` behavior text; update if label described; verify grep shows no stale text
