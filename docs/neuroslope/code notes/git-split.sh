#!/usr/bin/env bash
# Split experiments into branches. One step per call, inspect between:
#   ./git-split.sh split    tags pre-split, skeleton commit on main, flat branch per experiment
#   ./git-split.sh archive  archive branch: skeleton plus merge of each experiment branch
# Run from repo root, clean tree. Nothing pushed.
set -euo pipefail

# Experiment numbers 00..43.
NUMS=$(seq -f '%02g' 0 43)

# Tracked files of experiment NN at pre-split, NUL separated.
files() {
  git ls-tree -r -z --name-only pre-split \
    | grep -zE "(^|/)exp$1_"
}

# Branch name expNN_<slug>: card name, else config folder (exp43 has no card).
branch() {
  {
    git ls-tree --name-only pre-split docs/neuroslope/experiments/
    git ls-tree --name-only pre-split configs/
  } \
    | grep -oE "/exp$1_[a-z0-9_]+" \
    | sed -n '1s#^/##p'
}

split() {
  git switch main

  # Save full state.
  git tag pre-split

  # Skeleton: main without experiment files.
  for n in $NUMS; do
    files "$n"
  done \
    | xargs -0 git rm -q --
  git commit -m "chore: skeleton, experiments moved to branches"
  git tag skeleton

  # One branch per experiment, own files only.
  for n in $NUMS; do
    b=$(branch "$n")
    git switch -q -c "$b" skeleton
    files "$n" \
      | xargs -0 git checkout pre-split --
    git commit -q -m "exp$n: restore files"
  done

  git switch main
}

archive() {
  git switch -c archive skeleton

  # Merge experiments in order. Disjoint files, no conflicts.
  for n in $NUMS; do
    git merge -q --no-ff --no-edit "$(branch "$n")"
  done

  git switch main

  # Check nothing lost.
  if git diff --quiet pre-split archive; then
    echo "archive matches pre-split"
  else
    echo "archive differs from pre-split"
  fi
}

# Run step named by first argument.
"$1"
