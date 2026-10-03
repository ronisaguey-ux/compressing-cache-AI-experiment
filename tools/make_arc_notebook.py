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
CANDIDATES = [
    "/kaggle/input/arc-prize-2026-arc-agi-2",
    "/kaggle/input/arc-prize-2026-arc-agi-2/",
]
INPUT = None
for c in CANDIDATES:
    if os.path.isdir(c):
        INPUT = c
        break
if INPUT is None:
    hits = glob.glob("/kaggle/input/*arc*")
    INPUT = hits[0] if hits else None
print("input dir:", INPUT)

# Find the challenge file: the test set is JSON with a "test" key and no "output".
challenge_path = None
if INPUT:
    for p in sorted(glob.glob(os.path.join(INPUT, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and d and all(isinstance(v, dict) and "test" in v for v in d.values()):
            challenge_path = p
            break
print("challenges:", challenge_path)

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
    submission[tid] = attempts

out = "/kaggle/working/submission.json"
with open(out, "w") as fh:
    json.dump(submission, fh)
print("wrote %s  (%d tasks, %d with a verified hypothesis)" % (out, len(submission), solved))

# Format self-check before finishing: the grader rejects a malformed file, and a rejected file
# scores nothing regardless of the solver. Check the shape, do not assume it.
bad = []
for tid, preds in submission.items():
    if not isinstance(preds, list) or not preds:
        bad.append((tid, "no predictions")); continue
    for p in preds:
        if not isinstance(p, list) or len(p) != 2:
            bad.append((tid, "prediction is not 2 attempts")); break
        for g in p:
            if not isinstance(g, list) or not g or not all(isinstance(r, list) for r in g):
                bad.append((tid, "attempt is not a grid")); break
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
