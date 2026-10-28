#!/usr/bin/env bash
# Promote experiment to main, rebase later experiments onto it:
#   ./promote.sh 04    merge exp04_<slug> into main, rebase exp05.. onto main
# Run from repo root, clean tree. Nothing pushed.
set -euo pipefail

# Experiment branches expNN_<slug>, sorted by number.
branches() {
  git for-each-ref --format='%(refname:short)' 'refs/heads/exp*' | sort
}

n=$1
b=$(branches | grep "^exp${n}_")

git switch main
git merge --no-ff "$b" -m "exp$n: promote to main"

# Later experiments get new baseline.
for later in $(branches); do
  if [ "${later:3:2}" -gt "$n" ]; then
    git rebase -q main "$later"
  fi
done

git switch main
