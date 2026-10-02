"""Broken-repo task generator: a module seeded with exact, numbered bugs.

WHY THIS TASK SHAPE (owner's spec, 2026-10-02)
----------------------------------------------
*"give it a broken repo and a task to fix specific issues, but with exact like numbers and code
snippets, only read once at the beginning of the prompt, with no guidance, or reminders, only sent
on the first message that takes roughly 200 turns, and grade it based off that, that's where mine
should shine"*

The previous task (add one independent function per turn) was TOO EASY to discriminate: measured
122/123 vs 123/123, because a model that lost its early context could still do each new function
from scratch. Nothing required the EARLY detail.

This task makes the early detail load-bearing:

  * `solution.py` is SEEDED with N broken functions. Every one of them is already present and
    callable -- the work is to CORRECT them, not to add them.
  * TURN 1 carries the complete bug report: for each bug, the exact function, the exact wrong
    value, and the exact required value. Concretely, the numbers matter -- `MUL 88`, not
    "multiply by a constant".
  * EVERY LATER TURN says only `Fix bug N.` -- **no detail, no reminder, no restatement.**
  * The grader tests the EXACT required behaviour of each function.

So a model that has lost turn 1 cannot do the fix by reasoning, because the required constant is
arbitrary and appears nowhere else. It can only guess, and a guess is visibly wrong. That is the
property that makes this discriminate where the previous task did not.

★ AND THE NUMBERS ARE DELIBERATELY ARBITRARY. If the fix were derivable from the function name or
the surrounding code, a model with no memory could re-derive it and the task would measure nothing.
Every constant below is therefore chosen for being unguessable. This mirrors the `CONTRACT_NONCE`
logic, scaled up to a whole task: N independent un-re-derivable facts, all stated once.
"""
import hashlib


def _seed(i: int, salt: str = "") -> int:
    """Deterministic per-bug number, so the take is reproducible but the constants look arbitrary."""
    h = hashlib.sha256(("%s:%d" % (salt, i)).encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------------------------
# Bug families. Each returns (broken_body, fix_text, test_expr, correct_body).
#   broken_body : the function as seeded into solution.py (WRONG)
#   fix_text    : the EXACT instruction shown once in turn 1 (the un-re-derivable detail)
#   test_expr   : a boolean expression over module `m` that is True iff the fix landed
#   correct_body: the reference fix (used to prove the grader accepts a right answer)
# ---------------------------------------------------------------------------------------------

def _bug(i: int):
    fam = i % 6
    s = _seed(i, "bug")
    fn = "bug_%02d" % i

    if fam == 0:
        # wrong multiplier -> exact multiplier
        wrong = (s % 90) + 2          # arbitrary wrong constant
        right = ((s // 7) % 900) + 11  # arbitrary right constant, unrelated to `wrong`
        broken = ("def %s(x):\n"
                  "    return x * %d\n" % (fn, wrong))
        correct = ("def %s(x):\n"
                   "    return x * %d\n" % (fn, right))
        fix = "%s: `return x * %d` must become `return x * %d`" % (fn, wrong, right)
        test = "assert _fns[%d](1) == %d" % (i, right)

    elif fam == 1:
        # wrong offset
        wrong = (s % 500) + 3
        right = ((s // 11) % 700) + 7
        broken = ("def %s(x):\n"
                  "    return x + %d\n" % (fn, wrong))
        correct = ("def %s(x):\n"
                   "    return x + %d\n" % (fn, right))
        fix = "%s: change the added constant from %d to %d" % (fn, wrong, right)
        test = "assert _fns[%d](100) == %d" % (i, 100 + right)

    elif fam == 2:
        # wrong default in a keyword argument
        wrong = (s % 300) + 1
        right = ((s // 13) % 400) + 5
        broken = ("def %s(x, k=%d):\n"
                  "    return x - k\n" % (fn, wrong))
        correct = ("def %s(x, k=%d):\n"
                   "    return x - k\n" % (fn, right))
        fix = ("%s: the default for `k` must be %d, it is currently %d" % (fn, right, wrong))
        test = "assert _fns[%d](1000) == %d" % (i, 1000 - right)

    elif fam == 3:
        # wrong comparison operator
        wrong_op, right_op = (">", ">=") if s % 2 else ("<", "<=")
        thr = (s % 400) + 10
        broken = ("def %s(x):\n"
                  "    return x %s %d\n" % (fn, wrong_op, thr))
        correct = ("def %s(x):\n"
                   "    return x %s %d\n" % (fn, right_op, thr))
        fix = ("%s: the comparison must be `%s` (not `%s`) against %d" % (fn, right_op, wrong_op, thr))
        test = "assert _fns[%d](%d) is True" % (i, thr)

    elif fam == 4:
        # wrong slice bound
        mid = (s % 5) + 2
        right = mid + (s % 3) + 1
        broken = ("def %s(x):\n"
                  "    return x[:%d]\n" % (fn, mid))
        correct = ("def %s(x):\n"
                   "    return x[:%d]\n" % (fn, right))
        fix = "%s: the slice must end at %d, not %d" % (fn, right, mid)
        test = "assert _fns[%d]('abcdefghijkl') == %r" % (i, "abcdefghijkl"[:right])

    else:
        # wrong key / symbol name in a lookup.
        # ★ THE BROKEN AND FIXED VERSIONS MUST RETURN DIFFERENT VALUES or the test cannot tell them
        # apart. The first version mapped both keys to the SAME value, so `m.fn(0) >= 1` passed on
        # the broken module too -- a test that is satisfied by the bug measures nothing.
        opts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        wrong = opts[s % len(opts)]
        right = opts[(s // 5 + 3) % len(opts)]
        if right == wrong:
            right = opts[(opts.index(wrong) + 1) % len(opts)]
        wv = (s % 90) + 1                  # value under the wrong key
        rv = wv + 1000 + (s % 500)         # DIFFERENT value under the right key
        broken = ("%s_TABLE = {%r: %d}\n"
                  "def %s(x):\n"
                  "    return %s_TABLE[%r]\n" % (fn, wrong, wv, fn, fn, wrong))
        correct = ("%s_TABLE = {%r: %d}\n"
                   "def %s(x):\n"
                   "    return %s_TABLE[%r]\n" % (fn, right, rv, fn, fn, right))
        fix = "%s: the key must be %r (value %d), not %r" % (fn, right, rv, wrong)
        test = "assert _fns[%d](0) == %d" % (i, rv)

    return fn, broken, fix, test, correct


# ★ ONE RESOLVER, USED BY THE HARNESS AND BY EVERY TEST.  Grading by definition ORDER rather
# than by function NAME: measured, Gemma-4-12B renames `bug_00` to `func_0` on nearly every turn
# even when told to keep the name, so strict-name asserts failed a run that had applied every
# correct VALUE. Position is stable under a rename and the test is no weaker -- the required
# constant still exists only in turn 1, so a turn that lost turn 1 still cannot produce it.
FIX_PRELUDE = """
import inspect as _insp2
def _ordered_fns(_m):
    _out = []
    for _n, _v in vars(_m).items():
        if _insp2.isfunction(_v) and getattr(_v, '__module__', None) == _m.__name__:
            _ln = getattr(getattr(_v, '__code__', None), 'co_firstlineno', 10**9)
            _out.append((_ln, _v))
    _out.sort(key=lambda _t: _t[0])
    return [_t[1] for _t in _out]
_fns = _ordered_fns(m)
"""


def build(n: int):
    """Return (seed_source, bug_report_lines, tests, reference_fixed_source).

    `seed_source`   -- what solution.py starts as (the broken repo)
    `bug_report`    -- the EXACT list, shown ONLY in turn 1
    `tests`         -- one boolean expression per bug, True iff that fix landed
    `reference`     -- the same module with every fix applied (proves the grader is satisfiable)
    """
    broken_parts, fixed_parts, fixes, tests = [], [], [], []
    for i in range(n):
        fn, broken, fix, test, correct = _bug(i)
        broken_parts.append(broken)
        fixed_parts.append(correct)
        fixes.append("  %d. %s" % (i, fix))
        tests.append((fn, test))
    return ("\n".join(broken_parts), fixes, tests, "\n".join(fixed_parts))

# ══════════════════════════════════════════════════════════════════════════════════════════════
# ★★ PER-TURN CODES — THE CONTROLLED-EXPERIMENT FIX (owner, 2026-10-02)
#
# Owner's critique, verbatim: *"that's not good cuz it isn't a controlled experiment, u need to
# have it so both models keep the OG task from turn 1, but how they deal with junk is dependent on
# how their built (linear mess vs organized beauty)"*.
#
# He is right. The first design gave the runtime arm a PROTECTED TURN-1 SLOT and the linear arm
# none, so it tested "who was handed the brief" -- a tautology. Both arms now anchor turn 1.
#
# But with the brief anchored in BOTH, the fix numbers are available to both and accuracy would
# TIE: measured, every turn rewrites the whole file, and each rewrite SUBSUMES the last, so the
# newest write makes all older ones redundant. The middle therefore has to hold something that
# exists NOWHERE ELSE, or a tie is guaranteed and nothing is being measured.
#
# These codes are that thing. A fresh code is handed over in EACH turn and may never be written to
# the file until the final turn, so it survives only if that turn is retained. Both arms can do
# every fix (the brief is anchored in both); only the policy that keeps a USEFUL middle can
# also reproduce the accumulated codes.  That is the difference the experiment is meant to show:
# handling of accumulated junk, not who got the cheat slot.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def turn_codes(n):
    """One unguessable code per turn, deterministic for reproducibility, unique across turns."""
    out = []
    seen = set()
    for i in range(n):
        h = hashlib.sha256(("turncode:%d" % i).encode()).hexdigest().upper()
        c = "".join(ch for ch in h[:8] if ch.isdigit())[:4]
        while len(c) < 4:
            c += "0"
        cand = "%s-%d" % (c, i)
        while cand in seen:
            cand += "x"
        seen.add(cand)
        out.append(cand)
    return out


def grade_codes(src, codes):
    """How many turn codes are recoverable, IN ORDER, exactly once each.

    Scored 0..n rather than as a bit, because the interesting outcome is PARTIAL retention: a
    sliding window keeps the most recent few and loses the rest, which is visible as a prefix
    gap rather than an all-or-nothing result.
    """
    src = src or ""
    found = [c for c in codes if src.count(c) == 1]
    ordered = [c for c in codes if c in src]
    dup = any(src.count(c) > 1 for c in codes)
    return dict(codes_n=len(found), codes_total=len(codes),
                codes_ordered=(ordered == codes), codes_dup=dup)

