# ARC-AGI-2 baseline: measured, with a control that proves the tool works

Three runs, all on real ARC data fetched from the public repos (so they reproduce without
competition rules). Accuracy counts test inputs, not tasks.

| set | source | result |
|---|---|---|
| **ARC-AGI-2 evaluation** | `arcprize/ARC-AGI-2`, 30-task sample (seed 7) | **0/47 (0.0%)** |
| ARC-AGI-1 evaluation | `fchollet/ARC-AGI@master`, 40-task sample (seed 11) | 0/40 (0.0%) |
| **ARC-AGI-1 training** (control) | `fchollet/ARC-AGI@master`, 40-task sample (seed 3) | **4/41 (9.8%)** |

```
python3 tools/arc_baseline.py challenges.json --out submission.json --solutions solutions.json
```

## The control is what makes the zero meaningful

A solver that reports 0% is indistinguishable from a solver that is broken. The ARC-AGI-1
**training** set is the control: it contains easy tasks, and there the solver verifies a hypothesis
on 4 of 40 and scores **9.8%**. So the library works, the comparison works, and the zero on the
harder sets is a real measurement of a fixed-library approach rather than a defect.

The gradient is the expected shape: easy tasks score, evaluation sets do not. A fixed library of
geometric transforms cannot express ARC-AGI-2, which is constructed to defeat it.

## ★ A BUG THIS FOUND IN ITS OWN SCORER, AND WHY IT MATTERS

The first scoring pass reported **0% on all three sets, including the control.** The prediction
format is one entry *per test input*, each entry holding `[attempt1, attempt2]`; ground truth is a
list of gold grids. The scorer compared `prediction[0]` (a two-element list) against `gold[0]` (a
grid) -- a comparison that **can never succeed**.

Without the control this would have shipped as "the baseline scores zero", which reads as a fact
about the task. It was a fact about the scorer. **A scorer whose comparison cannot succeed still
reports a score, and that score is indistinguishable from a solver that fails** -- the same class
as a bare `==` in a `try/except` and a guard that only ever rejects.

## Why the solver exists

The ARC Prize 2026 Paper Track ($450k) requires the team to have submitted to ARC-AGI-2 or
ARC-AGI-3. That is a participation requirement, not a score one. This file satisfies it — and it
reports its own 0.0% rather than dressing up a number that would look better in a summary.

**Format.** `{task_id: [[attempt1, attempt2], ...]}` — one prediction per test input, each holding
two integer-grid attempts. A malformed file is a rejected submission, so the format is verified
against a synthetic task with two test inputs.
