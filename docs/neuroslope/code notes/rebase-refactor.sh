#!/usr/bin/env bash
# Move experiment branches onto refactored main. One step per call, inspect between:
#   ./rebase-refactor.sh backup  tag backup/<branch> per experiment branch
#   ./rebase-refactor.sh prune   delete branches already merged into main
#   ./rebase-refactor.sh rebase  rebase rest onto main, stops on conflict; rerun after --continue
#   ./rebase-refactor.sh fix     old paths, imports, card links in own files; commit per branch
# Run from repo root, clean tree. Nothing pushed.
set -euo pipefail

# Experiment branches expNN_<slug>, sorted by number.
branches() {
  git for-each-ref --format='%(refname:short)' 'refs/heads/exp*' | sort
}

backup() {
  for b in $(branches); do
    git tag -f "backup/$b" "$b"
  done
}

prune() {
  # No own commits: content already in main.
  for b in $(branches); do
    if [ -z "$(git rev-list main.."$b")" ]; then
      git branch -d "$b"
    fi
  done
}

rebase() {
  for b in $(branches); do
    if git merge-base --is-ancestor main "$b"; then
      continue
    fi
    echo "rebase $b"
    # Added files follow renamed folders instead of conflict.
    git -c merge.directoryRenames=true rebase -q main "$b"
  done
  git switch -q main
}

# Old path to new path, same rules as layout refactor.
new_path() {
  sed -E \
    -e 's#^docs/(experiments|methods|reports|code notes)/#docs/neuroslope/\1/#' \
    -e 's#^docs/backlog\.md$#docs/neuroslope/backlog.md#' \
    -e 's#^ai_usage/#docs/neuroslope/ai_usage/#' \
    -e 's#^configs/models/#configs/#' \
    -e 's#^src/core/data/#src/dataset/#' \
    -e 's#^src/core/#src/#'
}

fix() {
  for b in $(branches); do
    git switch -q "$b"

    # Own files still at old paths.
    git diff --name-only --diff-filter=d main HEAD \
      | while IFS= read -r f; do
          to=$(printf '%s\n' "$f" | new_path)
          if [ "$to" != "$f" ]; then
            mkdir -p "$(dirname "$to")"
            git mv "$f" "$to"
          fi
        done

    # Imports, class_path, path mentions in own text files.
    # Against working tree: moved files at new paths.
    git diff --name-only -z --diff-filter=d main -- '*.md' '*.ipynb' '*.yaml' '*.py' \
      | xargs -0 -r perl -pi -e '
          s/(?<![\w.])core\.data\.augment\.(?:ops|recipes|rows)\b/dataset.augment/g;
          s/(?<![\w.])core\.data\./dataset./g;
          s/(?<![\w.])core\.(training|analysis)\./$1./g;
          s#src/core/data/#src/dataset/#g;
          s#src/core/#src/#g;
          s#(?<!\w)docs/(experiments|methods|reports|code notes)/#docs/neuroslope/$1/#g;
          s#(?<!\w)docs/backlog\.md#docs/neuroslope/backlog.md#g;
          s#configs/models/#configs/#g;
        '

    # Cards one folder deeper.
    git diff --name-only -z --diff-filter=d main -- 'docs/neuroslope/experiments/*.md' \
      | xargs -0 -r perl -pi -e '
          s#\]\(\.\./\.\./(notebooks|configs|src|data|README\.md)#](../../../$1#g;
          s#\]\(\.\./(figures|task)/#](../../$1/#g;
        '

    # Tracked only: moves already staged, untracked files stay out.
    git add -u
    if ! git diff --cached --quiet; then
      git commit -q -m "${b:0:5}: paths after layout refactor"
      echo "fixed $b"
    fi
  done
  git switch -q main
}

# Run step named by first argument.
"$1"
