#!/usr/bin/env python3
"""Build the ARC-AGI-2 submission notebook.

★ WHY THIS EXISTS. ARC-AGI-2 declares `onlyAllowKernelSubmissions: true` -- verified from the
competition metadata, not assumed. A hand-uploaded `submission.json` is not a valid entry; the
submission unit is a notebook that runs inside Kaggle and writes `submission.json` to the working
directory. `tools/arc_baseline.py` produces the right FORMAT but it is not, by itself, a submission.

★ THE SOLVER SOURCE IS EMBEDDED VERBATIM, not re-implemented. Kaggle does not have this repository,
so the notebook must be self-contained; and a second copy of the solver would drift from the one
whose measured scores are recorded in `tools/ARC-VALIDATION.md`. The source is read from the file
at build time, so there is exactly one implementation.

Usage: python3 tools/make_arc_notebook.py [out.ipynb]
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "tools", "arc_baseline.py")
OUT = os.environ.get("ARC_NB_OUT", os.path.join(REPO, "arc", "arc-agi-2-submission.ipynb"))

DRIVER = '''
# ─── Kaggle driver ────────────────────────────────────────────────────────────────────────────
# The competition is KERNELS-ONLY, so this cell is what actually constitutes a submission: it runs
# inside Kaggle, reads the test challenges, and writes submission.json to the working directory.
import glob, json, os, sys

# The mounted input directory has varied across ARC releases, so it is discovered rather than
# hardcoded -- a wrong hardcoded path produces an empty submission that still "runs successfully".
#
# ★★ THE 2026 MOUNT IS NESTED ONE LEVEL DEEPER: /kaggle/input/competitions/<slug>. A flat
# `glob("/kaggle/input/*arc*")` finds nothing there and the kernel then refuses (correctly) to write
# an empty submission -- which read for two runs as "the data is not mounting" when the data was
# right there under an extra directory. The search is RECURSIVE and the path is never assumed.
INPUT = None
for c in ("/kaggle/input/arc-prize-2026-arc-agi-2",
          "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"):
    if os.path.isdir(c):
        INPUT = c
        break
if INPUT is None:
    hits = sorted(glob.glob("/kaggle/input/**/*arc*", recursive=True))
    INPUT = hits[0] if hits else None
print("input dir:", INPUT)
if INPUT:
    print("  contents:", sorted(os.listdir(INPUT))[:20])

# Find the challenge file. ★ THE MOUNT HOLDS FIVE JSONs AND THREE OF THEM LOOK ELIGIBLE:
# training (1000 tasks), evaluation (120), and test (240) all carry "test" entries, so "the first
# one with a test key" picks EVALUATION -- the driver scored 120 tasks while the grader expects 240,
# and the artifact looked structurally valid at 0% instead of failing loudly.
#
# Two disambiguators, in order: the filename must name the test set, and the key set must match
# sample_submission.json when that file is present. The submission is checked against the sample,
# so deriving the task set from anything else is how a silent 120-vs-240 mismatch happens.
SAMPLE = None
for _s in glob.glob(os.path.join(INPUT or "/kaggle/input", "**", "sample_submission.json"),
                    recursive=True):
    try:
        SAMPLE = set(json.load(open(_s)))
    except Exception:
        SAMPLE = None
    break
print("sample keys:", len(SAMPLE) if SAMPLE else None)

def _eligible(p, d):
    return (isinstance(d, dict) and d
            and all(isinstance(v, dict) and "test" in v for v in d.values()))

cands = []
for p in sorted(glob.glob(os.path.join(INPUT, "**", "*.json"), recursive=True)):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if _eligible(p, d):
        cands.append((p, d))

challenge_path = None
# (1) the file whose name names the test set
for p, d in cands:
    if "test_challenges" in os.path.basename(p):
        challenge_path = p
        break
# (2) otherwise the one whose keys match the official sample
if challenge_path is None and SAMPLE:
    for p, d in cands:
        if set(d) == SAMPLE:
            challenge_path = p
            break
# (3) last resort: the largest eligible file, and say which rule was used
if challenge_path is None and cands:
    challenge_path = max(cands, key=lambda pd: len(pd[1]))[0]
    print("WARNING: matched by size, not by name or sample — verify the task set")
print("challenges:", challenge_path,
      "(%d tasks)" % len(json.load(open(challenge_path))) if challenge_path else "")

if challenge_path and SAMPLE and set(json.load(open(challenge_path))) != SAMPLE:
    raise SystemExit("challenge key set does not match sample_submission.json — refusing to "
                     "submit a task set the grader does not expect")

if challenge_path is None:
    # Fail LOUDLY. An empty submission that exits 0 is indistinguishable from a solved one at the
    # result level, which is the failure this whole repository is about.
    raise SystemExit("no challenge file found under %r -- refusing to write an empty submission" % INPUT)

tasks = json.load(open(challenge_path))
print("tasks:", len(tasks))

submission = {}
solved = 0
for tid, task in tasks.items():
    attempts, names = solve_task(task)
    if names:
        solved += 1
    # ★ WRAP INTO THE GRADER'S CONTRACT. `solve_task` returns raw [[a1,a2],...]; the grader wants
    # [{"attempt_1": g, "attempt_2": g}, ...] -- one DICT per test input. The library was fixed but
    # this driver kept writing the raw form, so the notebook emitted [[[grid]]] (120 keys, 1 entry
    # each) while the local tool emitted the correct dict shape. The self-check below did not catch
    # it because it was never reached: the write happened first and the check ran on a shape it
    # then had to interpret. Wrapping here fixes the artifact; the check now guards it.
    submission[tid] = [{"attempt_1": pair[0],
                        "attempt_2": pair[1] if len(pair) > 1 else pair[0]}
                       for pair in attempts]

out = "/kaggle/working/submission.json"
with open(out, "w") as fh:
    json.dump(submission, fh)
print("wrote %s  (%d tasks, %d with a verified hypothesis)" % (out, len(submission), solved))

# Format self-check before finishing: the grader rejects a malformed file, and a rejected file
# scores nothing regardless of the solver. Check the shape, do not assume it.
# ★ THE CONTRACT, taken from the competition's own sample_submission.json rather than assumed:
#   {"<task>": [{"attempt_1": grid, "attempt_2": grid}, ...]}  -- one dict per test input.
# An earlier version checked for a list of two grids, which is self-consistent and wrong; a format
# check written from an assumption validates the assumption, not the contract.
bad = []
for tid, preds in submission.items():
    if not isinstance(preds, list) or not preds:
        bad.append((tid, "no predictions")); continue
    for entry in preds:
        if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
            bad.append((tid, "entry is not {attempt_1,attempt_2}")); break
        for key in ("attempt_1", "attempt_2"):
            g = entry[key]
            if not isinstance(g, list) or not g or not all(isinstance(r, list) for r in g):
                bad.append((tid, "%s is not a grid" % key)); break
print("format check:", "PASS" if not bad else "FAIL %r" % bad[:5])
print("submission bytes:", os.path.getsize(out))
'''


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n")}


def main(argv):
    out = argv[1] if len(argv) > 1 else OUT
    if not os.path.exists(SRC):
        print("no solver at %s" % SRC)
        return 1
    solver = open(SRC, encoding="utf-8").read()

    # Take the library half -- everything before the CLI. The notebook supplies its own driver,
    # because the CLI reads a local path that does not exist inside a Kaggle kernel.
    cut = solver.find("def main(argv=None)")
    if cut == -1:
        cut = len(solver)
    lib = solver[:cut].rstrip()
    # Drop the module docstring's CLI usage line; the notebook has its own entry point.
    lib = re.sub(r"^Usage:\n.*\n", "", lib, flags=re.M)

    cells = [
        md("# ARC-AGI-2 — baseline submission\n\n"
           "**This is a participation entry, not a competitive one.** It is a program-synthesis "
           "baseline over a small library of grid transformations, built to satisfy the ARC Prize "
           "2026 Paper Track's requirement that the team make a submission to ARC-AGI-2 or "
           "ARC-AGI-3. Measured score on the evaluation set: **0.0%**, with a non-zero control on "
           "the easier ARC-AGI-1 training set (9.8%) confirming the solver and the scorer work.\n\n"
           "ARC-AGI-2 is constructed to defeat exactly this class of fixed-library approach. The "
           "notebook reports that rather than dressing it up."),
        md("## Solver\n\n"
           "Embedded verbatim from `tools/arc_baseline.py` at build time, so the implementation "
           "that produced the recorded scores is the one that runs here."),
        code(lib),
        md("## Run\n\nThe competition is kernels-only, so this cell is the submission: it reads the "
           "test challenges and writes `submission.json`."),
        code(DRIVER),
    ]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%d cells; solver %d chars)" % (out, len(cells), len(lib)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
