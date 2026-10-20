# Changelog

One line per change and the break it fixes (see "Experiment Skill: Approaches and Redesign", section 3).

- 2026-10-11: v4. Propose, approve, then run (idea from section 4); Request verbatim (breaks 1, 2); one-sentence Question, no AI Goal (2); smoke-only execution plus hook (5); report from MLflow or saved outputs, at most 5 bullets (5, 9); Corrections log (4); Conclusion is mine, open cards listed first (3); Context field (8); paths from CLAUDE.md, template.md is the only source of headings (6, 7).
- 2026-10-11: Notebook holds code only: a title line with a link to the card, section headers, one-line comments on non-obvious choices. Reports and descriptions live in the card.
- 2026-10-11: Results rule made concrete: figure for a trend, table for a few values, never both for the same numbers.
- 2026-10-11: No arrows or special symbols in cards and replies.
- 2026-10-11: Tool-independent: tracker named in a "Tools (fill in)" block at the top of SKILL.md instead of MLflow everywhere; template field "MLflow experiment" renamed "Tracking".
- 2026-10-12: Sections (any top-level folder): section argument or inferred, numbering and "last" per section, tracking `ashrae-<section>-<slug>`; removed the duplicate `experiment_NN.md` paragraph.
- 2026-10-12: Project details (section names, tracking prefix, Jupyter path) moved from SKILL.md to CLAUDE.md.
- 2026-10-16: Sections moved under `sections/<N>_<section>/`, name without number; folders created on first file.
- 2026-10-16: Section folders lose number prefix: `sections/<section>/`, order listed in CLAUDE.md.
- 2026-10-16: Plan as numbered steps, one per output, verb-first sentence, no "label: details".
- 2026-10-16: Fixes section after Setup: my fix requests after build, verbatim.
- 2026-10-17: Build step 5: smoke tracking experiment and smoke checkpoints deleted after verify; they piled up (6.8 GB) and were never read.
- 2026-10-17: Build step 3: training through `core.cv` and one TOML config per model (from playground); notebook passes smoke `OVERRIDES`, holds no training loop.
