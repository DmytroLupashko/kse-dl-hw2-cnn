# Git refactor file

## Idea

Goal: make working tree thiner (in terms of amount of content, not MBs)

What's now?

Now state contains all experiments and generic code that tries to satisfy all of them. As it turns out, most of these experiments are redandant for now and future. Also, they are redudant for doing other experiments (when doing experiment, we are comparing it with baseline. If it beats, then baseline becomes redundant). Moreover, I needed to onstantly update current best model in kind a "state" file.

What's new approach?

Each experiment / bunch of related experiments are contained in their special branch, which after finishing merges into archive branch which contains history of experimetns.****
If experiment is successful and beats baseline, then it moves to main.
The only thing that always is kept across experiments – mflow db, i.e. tracking results. I do not need to repredict everything, it is shitty job
Sometimes I will want to understand what experiments I did and what are their results. For this part I will used condensed notebooks which I will write myself.

## Plan

1. Save current state. It is lots of files, so it will take time.
2. Distribute experiments over branches. Goal is to separate them from each other. 
3. Refactor code inside each experiment. Rebase all things from main.

## Distribute experiments by branches

Goal: separate experiments from each other.

Experiment branches "communicate" with each other threw main (rebasing) and archive (cherrypicking some figures if needed).

Each experiment does not contain artifacts of previous experiments, if not needed.
