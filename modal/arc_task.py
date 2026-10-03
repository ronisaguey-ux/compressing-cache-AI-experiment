"""ARC-AGI-2 as a retention task, in the same shape as bug_task.

WHY THIS EXISTS
---------------
The paper's ARC claim is currently an argument: if a long search re-reads its context every attempt
and each attempt is 4x cheaper under the anchored policy, the search can try 4x more candidates in
the same budget. That step was labelled a hypothesis because it had never been run on ARC data.
This module runs it.

THE TASK SHAPE (deliberately identical to the fix task, so every policy, cache and metric path is
reused unchanged)
--------------------------------------------------------------------------------------------------
  * Turn 1 carries the whole puzzle brief: the example input/output pairs and the test input. This
    is the content that must survive N turns with no reminder, and it is large -- a 3-4 example ARC
    puzzle renders to roughly 1,000-1,500 tokens, so it is grid-heavy text, not a short brief.
  * Every later turn says only "Attempt k". No feedback about correctness is given, because the
    verifier signal would itself carry information and would let a policy that lost the examples
    still know where it stands.
  * Each turn the model rewrites `solve(test_input) -> grid`. The file is rewritten every turn, the
    same full-file-rewrite shape as the fix task, so the newest reply supersedes every earlier one
    and the anchored policy's "keep only the newest reply" rule is exercised.
  * Grading is PER TURN: after turn k, `solve()` is executed against the held-out gold output. The
    score is the fraction of turns on which the artifact was correct.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
This measures (a) the cost and prefix-cache difference between policies on a REAL ARC-AGI-2 task,
and (b) the per-turn solve rate under each policy. It does not attempt a competitive score: one
task at a time, no large candidate search. A low solve rate is a legitimate outcome and is reported
as such.

The gold output is written to a file in the working directory that the grader reads. It is never
placed in a prompt.
"""
import hashlib
import json
import os
import random

# Directory holding the public ARC task files. Two layouts are accepted:
#   <dir>/<split>/<task_id>.json          (the arcprize/ARC-AGI-2 and fchollet/ARC-AGI clones)
#   <dir>/<split>.json                    (a single-file bundle, if one is ever used)
ARC_DIR = os.environ.get("CCAI_ARC_DIR", "")
ARC_SPLIT = os.environ.get("CCAI_ARC_SPLIT", "training")
GOLD_NAME = "arc_gold.json"

# ★ THE TASK IS CHOSEN BY INDEX AND FIXED ACROSS ARMS. Both policies must attempt the identical
# puzzle, or the comparison is between two different problems. CCAI_ARC_INDEX selects it, and the
# harness passes the same value to every arm. If unset, the index is derived from the task salt so
# variance runs vary the puzzle while every arm within a run sees the same one.
#
# ★ READ AT CALL TIME, NOT AT IMPORT. The first version captured this at module load, so setting the
# variable after `import arc_task` had no effect and a scan over indices returned the same task
# every time -- a probe that silently measured one sample while reporting it had measured many.
def _task_files():
    d = os.path.join(ARC_DIR, ARC_SPLIT)
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    single = os.path.join(ARC_DIR, ARC_SPLIT + ".json")
    if os.path.isfile(single):
        return [single]
    raise FileNotFoundError("no ARC tasks under %r (looked in %r and %r)"
                            % (ARC_DIR, d, single))


def _env_index():
    return os.environ.get("CCAI_ARC_INDEX", "").strip()


def pick_index(n_tasks, salt=""):
    """The puzzle index. Fixed and identical across arms for a given salt."""
    idx = _env_index()
    if idx:
        return int(idx)
    # deterministic from the salt, so a variance run varies the puzzle and a repeated run repeats it
    h = hashlib.sha256(("arcidx:%s" % salt).encode()).hexdigest()
    return int(h[:8], 16) % max(1, n_tasks)


def load_task(salt=""):
    """Return (task_id, train_pairs, test_input, test_output)."""
    files = _task_files()
    idx = pick_index(len(files), salt)
    path = files[idx]
    with open(path) as f:
        d = json.load(f)
    task_id = os.path.splitext(os.path.basename(path))[0]
    train = [(e["input"], e["output"]) for e in d["train"]]
    test = d["test"][0]
    return task_id, train, test["input"], test["output"]


def _grid_lines(g):
    """Render a grid compactly: one line per row, digits with no separators. Dense and unambiguous."""
    return "\n".join("".join(str(c) for c in row) for row in g)


def _render_brief(task_id, train, test_input):
    parts = []
    parts.append("PUZZLE %s. This is an ARC grid-reasoning task. Study the examples, infer the "
                 "transformation, and apply it to the test input." % task_id)
    parts.append("")
    for i, (a, b) in enumerate(train, 1):
        parts.append("Example %d input (%d rows x %d cols):" % (i, len(a), len(a[0]) if a else 0))
        parts.append(_grid_lines(a))
        parts.append("Example %d output:" % i)
        parts.append(_grid_lines(b))
        parts.append("")
    parts.append("Test input (%d rows x %d cols):" % (len(test_input),
                                                      len(test_input[0]) if test_input else 0))
    parts.append(_grid_lines(test_input))
    parts.append("")
    parts.append("Write `def solve(grid):` so that solve(test_input) returns the correct output "
                 "grid as a list of lists of integers. The transformation is the same for every "
                 "example. This is the ONLY time the examples are given; they will not be repeated "
                 "and no reminder will restate them. Rewrite solve() in full on every turn.")
    # ★ KEEP THE REPLY SHORT. MEASURED: on the first ARC run the model wrote solve() with long
    # comment blocks, overran the 3072-token reply budget, and the JSON was cut mid-string, so the
    # turn was discarded and `solution.py` was never written. Every turn failed that way and the run
    # was vacuous. Comments are what consumed the budget, so they are forbidden here exactly as the
    # fix task forbids them.
    parts.append("")
    parts.append("CRITICAL OUTPUT RULES:\n"
                 "  * Reply with EXACTLY ONE json object and NOTHING else. Do NOT wrap it in "
                 "markdown fences.\n"
                 "  * Do NOT write comments in the code. Every comment spends reply budget that "
                 "your reply needs, and a reply cut off mid-string is DISCARDED ENTIRELY.\n"
                 "  * Keep solve() as short as it can be while still being correct.\n"
                 "  * Do NOT include any text before or after the json object.")
    return "\n".join(parts)


SEED_SOURCE = '''"""ARC solver. Rewrite solve() so it applies the transformation shown in the examples."""


def solve(grid):
    return grid
'''


def write_gold(work: str, salt="") -> str:
    """Write the held-out gold output to `work/GOLD_NAME` and return its path.

    The grader reads it there; it is never placed in a prompt. Written by the harness before the
    loop starts so the per-turn grader has an oracle from turn 0.
    """
    task_id, train, test_input, test_output = load_task(salt)
    path = os.path.join(work, GOLD_NAME)
    with open(path, "w") as f:
        json.dump({"test_input": test_input, "test_output": test_output, "task_id": task_id}, f)
    return path


def build(n, salt=""):
    """Same contract as bug_task.build: (seed_source, brief_lines, tests, reference_source).

    `tests` holds one boolean expression per turn. Every turn grades the SAME property -- whether
    solve() reproduces the gold output -- because the loop is iterative refinement of one artifact,
    not n independent sub-tasks. `n` is therefore the number of attempts, and per-turn success is
    the fraction of attempts on which the artifact was correct.
    """
    task_id, train, test_input, test_output = load_task(salt)
    brief = _render_brief(task_id, train, test_input)

    # ★ THE TESTS MUST BE `(name, expr)` PAIRS, matching bug_task.build's contract.
    # The first version returned bare strings; the harness does `[t for (_fn, t) in _ftests]`, so it
    # crashed at turn 0 with "too many values to unpack (expected 2)" AFTER loading a 24 GB model --
    # a full load wasted per arm. The interface is bug_task's; mirror it exactly, do not approximate.
    tests = [("solve", 'assert _arc_ok(m), "output grid does not match the expected output"')] * n

    # The reference solver proves the grader is SATISFIABLE: it reads the same gold file the grader
    # compares against, so a correct answer exists and is accepted.
    reference = (
        "import json as _json, os as _os\n"
        "_G = _json.load(open(_os.path.join(WORK, %r)))\n"
        "def solve(grid):\n"
        "    return _G['test_output']\n" % GOLD_NAME
    )
    return SEED_SOURCE, [brief], tests, reference


def turn_codes(n):
    """Per-turn unguessable codes, same mechanism as the fix task.

    WHY IDENTICAL: the codes are the retention probe that separated the arms in the fix task, and
    using the same construction keeps the two experiments comparable. A code is shown once, in that
    turn's instruction, forbidden from the file until the final turn, then required back in order.
    """
    out = []
    for i in range(n):
        h = hashlib.sha256(("arccode:%d" % i).encode()).hexdigest().upper()
        out.append("K%s%s" % (h[:3], h[8:11]))
    return out


def grade_codes(src, codes):
    """Count how many codes appear in the source, in order, without duplicates."""
    seen, ordered, last = set(), 0, -1
    for i, c in enumerate(codes):
        if c in src and c not in seen:
            seen.add(c)
            if i > last:
                ordered += 1
                last = i
    ordered_ok = ordered == len([c for c in codes if c in src])
    return len(seen), ordered_ok


def gate_secret(n):
    h = hashlib.sha256(("arcgate:%d" % n).encode()).hexdigest()
    return int(h[:6], 16)


GATE_FN = "gate_value"


def gate(n, codes=None):
    """A separate mid-session secret, mirroring bug_task.gate.

    Kept for interface parity. Note that the fix task's gate did NOT discriminate between policies
    (its value persists in the artifact), and the same caveat may apply here; the codes probe is the
    discriminating one.
    """
    val = gate_secret(n)
    src = ("def %s():\n    return 0\n" % GATE_FN)
    line = "def %s():\n    return %d" % (GATE_FN, val)
    test = "_g = m.%s()\nassert _g == %d" % (GATE_FN, val)
    return src, line, test, n // 2, val


# ── the grader prelude, assembled by the harness into the subprocess that checks a turn ──────────
# `_arc_ok` runs the model's solve() on the test input and compares to the gold with a normalised
# grid comparison. It fails loud: any exception is a failed turn, never a silent pass.
#
# ★ The prelude is a FUNCTION of the work directory, not a module constant with a %r placeholder.
# The first version formatted only the path and left GOLD_NAME as a bare name inside the emitted
# code, so the grader died with NameError on every turn -- which would have read as "the model
# failed" rather than "the harness is broken".
def arc_prelude(work: str) -> str:
    return (
        "import json as _json, os as _os\n"
        "_ARC_WORK = %r\n"
        "_ARC_GOLD = %r\n"
        "\n"
        "def _norm(g):\n"
        "    if not isinstance(g, (list, tuple)):\n"
        "        return None\n"
        "    out = []\n"
        "    for row in g:\n"
        "        if not isinstance(row, (list, tuple)):\n"
        "            return None\n"
        "        try:\n"
        "            out.append([int(c) for c in row])\n"
        "        except Exception:\n"
        "            return None\n"
        "    return out\n"
        "\n"
        "def _arc_ok(m):\n"
        "    with open(_os.path.join(_ARC_WORK, _ARC_GOLD)) as _f:\n"
        "        _g = _json.load(_f)\n"
        "    _got = _norm(m.solve(_g['test_input']))\n"
        "    _want = _norm(_g['test_output'])\n"
        "    return _got is not None and _got == _want\n"
    ) % (work, GOLD_NAME)
