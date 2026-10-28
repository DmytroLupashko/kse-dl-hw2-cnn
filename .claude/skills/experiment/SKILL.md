---
name: experiment
description: Create, run or report a numbered experiment card and notebook.
argument-hint: "<request> | build [NN] | report [NN]"
---

# Experiments

Headings come only from template.md.
Files: card, notebook, configs, figures at paths from CLAUDE.md (experiment files). One numbering for all experiments. Create missing folders. "Last" = highest NN in card folder.

Mode from `$ARGUMENTS`: first word `build` or `report`, otherwise a new request.

## Request

  1. Next NN = highest in card folder + 1. Copy template.md.
  2. Request: the user's words, verbatim. Question: one measurable sentence (ask if you can't). Plan and Setup: proposed. Plan: numbered steps, one per output (table, figure, sheet); step that only gathers data merges into step using it; each step one sentence starting with verb, never "label: details". Card fields follow examples.md.
  3. Stop. Wait for approval before building.

## Build [NN]

  1. Build the notebook for experiment NN (default: last).
  2. Notebook holds code only: title linking the card, section headers, one-line comments on choices that aren't obvious.
  3. Training experiments: one YAML config per model (path and training entry point from CLAUDE.md); notebook holds no training loop.
  4. Notebook holds no smoke code: no `SMOKE` flag, no smoke branches or overrides; constants hold full-scale values only. Project code handles `SMOKE=1` (shrunk training, `-smoke` tracking and results, no saved figures, no upload); see CLAUDE.md. Shared helpers (style, plots, scores) come from project code, never copied into notebook. Tracking experiment: `<prefix>-expNN-<slug>` (prefix from CLAUDE.md).
  5. Verify with `SMOKE=1 <jupyter> nbconvert --execute --to notebook --stdout <nb> > /dev/null` (`<jupyter>` from CLAUDE.md). Fix until it passes. Never run at full scale and never write outputs to the notebook.
  6. After it passes, delete smoke leftovers: smoke tracking experiment permanently (MLflow: `delete_experiment`, then `mlflow gc --tracking-uri <uri> --experiment-ids <id>`, otherwise the name stays taken and next smoke run fails) its checkpoint folder `checkpoints/<tracking experiment>-smoke/` and results folder `data/processed/cv/<tracking experiment>-smoke/`.


## Report [NN]

  1. Experiment NN (default: last). Run = code cells have outputs. Otherwise ask the user to run it and stop.
  2. Results: a figure for a trend or a table for a few values (not both for the same numbers, prefer figure), plus one line saying the metric and the rows it was scored on.
  3. AI Summary: up to 2 short sentences answering the Question. Leave Conclusion to the user.
