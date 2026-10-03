# ARC-AGI-2 baseline: measured on real data

**Result: 0/30 tasks solved, 0.0% attempt-1, 0.0% attempt-2 combined.**

Measured on 30 tasks sampled (seed 7) from the real ARC-AGI-2 evaluation set
(`arcprize/ARC-AGI-2`, `data/evaluation`, 120 tasks), fetched from the public repo rather than
Kaggle so no competition rules are required to reproduce it.

```
python3 tools/arc_baseline.py challenges.json --out submission.json --solutions solutions.json
tasks: 30   with a verified hypothesis: 0   falling back to identity: 30
accuracy: attempt-1 0/30 (0.0%), attempt-2 0/30, combined 0.0%
```

**This is the expected and honest floor, not a bug.** The solver searches a small library of grid
transformations (geometric transforms, a consistent colour map, tiling, largest-object bounding
box) and accepts a candidate only when it maps every training pair exactly. It found no verified
hypothesis on any of the 30 tasks, so every prediction is the identity fallback. ARC-AGI-2 is
built to defeat exactly this class of program synthesis by a fixed library.

**Why the solver exists at all:** the ARC Prize 2026 Paper Track ($450k) requires the team to have
submitted to ARC-AGI-2 or ARC-AGI-3. That is a participation requirement, not a score one. This
file is what satisfies it, and it reports its own score as zero rather than dressing up a number
that would look better in a summary.

**Format.** The submission is `{task_id: [[attempt1, attempt2], ...]}` -- one prediction per test
input, each holding two integer-grid attempts. Format is verified against a synthetic task with two
test inputs (see the commit that introduced the tool); a malformed file is a rejected submission,
so the format check is the part that matters.
