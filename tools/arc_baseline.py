#!/usr/bin/env python3
"""ARC-AGI-2 baseline solver and submission writer.

★ WHY THIS EXISTS AT ALL. The ARC Prize 2026 Paper Track requires the team to have made a
submission to ARC-AGI-2 or ARC-AGI-3. That requirement is about participation, not score, so a
legitimate minimal solver is what keeps the door open — and it must be honest about being minimal.

★ WHAT THIS IS AND IS NOT. It is a program-synthesis baseline over a small library of grid
transformations: it searches for a transformation consistent with every training pair of a task,
then applies it to the test inputs. It is NOT a learned system, NOT an LLM wrapper, and NOT a claim
about ARC capability. Reported scores from it should be read as a floor, and it will score near zero
on the 2026 set, which is designed to defeat exactly this class of approach.

★ THE FORMAT IS THE PART THAT MATTERS. A submission is JSON mapping each task id to a list of
attempts, each attempt a grid (list of rows, each a list of ints). ARC-AGI-2 allows two attempts
per test input, so a solver that is unsure can offer two candidates instead of one. Producing a
well-formed file is the whole requirement here; guessing badly is worse than guessing once.

Usage:
    python3 arc_baseline.py solve <challenges.json> [--out submission.json] [--solutions X.json]
"""
import argparse
import itertools
import json
import sys
from collections import Counter

Grid = list  # list[list[int]]


# ─── the transformation library ──────────────────────────────────────────────────────────────────
# Each candidate is a pure function Grid -> Grid, applied uniformly to every pair. A candidate is
# accepted only if it maps EVERY training input to its exact training output; a transformation that
# fits some pairs and not others is not a hypothesis, it is a coincidence.

def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def _flip_h(g):
    return [r[::-1] for r in g]


def _flip_v(g):
    return g[::-1]


def _transpose(g):
    return [list(r) for r in zip(*g)]


def _identity(g):
    return [list(r) for r in g]


def _rotations():
    return [_identity, _rot90, lambda g: _rot90(_rot90(g)), lambda g: _rot90(_rot90(_rot90(g)))]


def _flips():
    return [_identity, _flip_h, _flip_v, lambda g: _flip_v(_flip_h(g)), _transpose,
            lambda g: _rot90(_transpose(g)), lambda g: _rot90(_rot90(_transpose(g))),
            lambda g: _rot90(_rot90(_rot90(_transpose(g))))]


def _constant_color_map(pairs):
    """Map colours consistently: the modal (in -> out) colour correspondence across all pairs."""
    mapping = {}
    for gi, go in pairs:
        if len(gi) != len(go) or len(gi[0]) != len(go[0]):
            return None
        for ri, ro in zip(gi, go):
            for a, b in zip(ri, ro):
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    if not mapping:
        return None

    def f(g):
        return [[mapping.get(v, v) for v in row] for row in g]
    return f


def _outline_of_largest_object(pairs):
    """Crop to the bounding box of the largest non-background object, a common ARC motif."""
    def bbox(g):
        h, w = len(g), len(g[0])
        bg = Counter(v for r in g for v in r).most_common(1)[0][0]
        cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] != bg]
        if not cells:
            return None
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]
    return bbox


def _tile_to(g, kh, kw):
    """Tile the grid k times vertically and horizontally."""
    out = []
    for _ in range(kh):
        for row in g:
            out.append(row * kw)
    return out


def _tile_candidates(pairs):
    """Infer a tile factor from the first pair and verify on the rest."""
    (gi0, go0) = pairs[0]
    if not gi0 or not gi0[0]:
        return
    kh = len(go0) // len(gi0)
    kw = len(go0[0]) // len(gi0[0])
    if kh > 1 or kw > 1:
        yield lambda g, kh=kh, kw=kw: _tile_to(g, kh, kw)


def solve_task(task):
    """Return up to 2 candidate grids for each test input, best-first.

    Args:
        task: {"train": [{"input": Grid, "output": Grid}, ...], "test": [{"input": Grid}, ...]}
    """
    train = task.get("train") or []
    test = task.get("test") or []
    pairs = [(t["input"], t["output"]) for t in train if "input" in t and "output" in t]
    candidates = []

    # 1. geometric transforms, verified against every training pair
    for name, fn in zip(
            ["id", "rot", "rot180", "rot270", "fh", "fv", "fhfv", "tr", "tr_rot", "tr_r2", "tr_r3"],
            list(itertools.islice(itertools.chain(_rotations()), 4)) + _flips()[1:]):
        try:
            if all(fn(gi) == go for gi, go in pairs):
                candidates.append((name, fn))
        except Exception:
            continue

    # 2. consistent colour mapping
    cm = _constant_color_map(pairs)
    if cm is not None and all(cm(gi) == go for gi, go in pairs):
        candidates.append(("color_map", cm))

    # 3. tile factors
    if pairs:
        try:
            for tf in _tile_candidates(pairs):
                if all(tf(gi) == go for gi, go in pairs):
                    candidates.append(("tile", tf))
        except Exception:
            pass

    # 4. bounding box of the largest object
    try:
        bb = _outline_of_largest_object(pairs)
        if all(bb(gi) == go for gi, go in pairs):
            candidates.append(("bbox", bb))
    except Exception:
        pass

    # Build the attempts. With no verified transformation, fall back to identity -- offering a
    # well-formed guess costs nothing and a malformed file is a rejected submission.
    attempts = []
    seen = set()
    for test_case in test:
        ti = test_case.get("input")
        opts = []
        for _, fn in candidates:
            try:
                g = fn(ti)
            except Exception:
                continue
            key = json.dumps(g)
            if key not in seen:
                opts.append(g)
                seen.add(key)
            if len(opts) == 2:
                break
        while len(opts) < 2:
            opts.append([list(r) for r in ti])       # identity as the fill attempt
        attempts.append(opts)
    return attempts, [c[0] for c in candidates]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("challenges")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--solutions", default=None,
                    help="optional ground-truth file, to report accuracy instead of only producing output")
    args = ap.parse_args(argv)

    with open(args.challenges) as fh:
        tasks = json.load(fh)

    submission = {}
    stats = {"solved": 0, "total": 0, "no_hypothesis": 0}
    per_task_attempts = {}
    for tid, task in tasks.items():
        stats["total"] += 1
        attempts, names = solve_task(task)
        per_task_attempts[tid] = attempts
        submission[tid] = attempts
        if names:
            stats["solved"] += 1
        else:
            stats["no_hypothesis"] += 1
        print("[%s] hypotheses=%s" % (tid, ",".join(names) or "none -> identity fallback"))

    with open(args.out, "w") as fh:
        json.dump(submission, fh)

    print("\nwrote %s" % args.out)
    print("tasks: %d   with a verified hypothesis: %d   falling back to identity: %d"
          % (stats["total"], stats["solved"], stats["no_hypothesis"]))

    if args.solutions:
        # Score only if ground truth is supplied. Attempt 1 counts, attempt 2 counts only when the
        # first is wrong -- that is how the metric reads a two-attempt submission.
        #
        # ★ THE SHAPE HERE IS THE WHOLE CORRECTNESS ARGUMENT. A prediction is one entry PER TEST
        # INPUT, each entry holding [attempt1, attempt2]. Ground truth is a list of gold grids, one
        # per test input. So the comparison is `prediction[i][0] == gold[i]`. An earlier version
        # compared `prediction[0]` (a two-element list) against `gold[0]` (a grid) and therefore
        # reported 0% on every run -- including runs where a verified hypothesis had been applied
        # and the answer was right. A scorer whose comparison can never succeed reports a score, and
        # that score is indistinguishable from a solver that fails.
        sol = json.load(open(args.solutions))
        first = second = 0
        scored = 0
        for tid, s in sol.items():
            got = per_task_attempts.get(tid)
            if not got:
                continue
            gold = s if isinstance(s, list) else [s]
            for i, g in enumerate(gold):
                if i >= len(got):
                    break
                attempts = got[i]
                if not isinstance(attempts, list) or not attempts:
                    continue
                scored += 1
                if attempts[0] == g:
                    first += 1
                elif len(attempts) > 1 and attempts[1] == g:
                    second += 1
        n = scored or 1
        print("accuracy: attempt-1 %d/%d (%.1f%%), attempt-2 %d/%d, combined %.1f%%"
              % (first, n, 100.0 * first / n, second, n, 100.0 * (first + second) / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
