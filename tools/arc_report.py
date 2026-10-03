#!/usr/bin/env python3
"""ARC run report: turns the raw result JSON into the numbers the ARC paper needs.

WHY A SEPARATE TOOL
-------------------
The ARC task reuses the fix-mode harness, so the result schema is identical -- but what is worth
reporting is NOT. The fix task asks "did the agent apply the fix", and both policies scored 60/60, so
per-turn success saturates. The ARC task asks "did the agent produce the correct grid", which it will
mostly FAIL, so the interesting quantities are different:

  * solve rate        -- on how many of the N attempts was the artifact correct (expect low)
  * first solve turn  -- when did it first get it right, if ever
  * best streak       -- did it get it right and then lose it (a retention failure inside one task)
  * self-correction   -- did it fix itself after a wrong attempt, without feedback telling it to
  * cost / cache      -- the actual claim: what the two policies cost on a real ARC task

A tool that printed the fix-mode table here would show 0/60 twice and say nothing.

Usage:
    python3 tools/arc_report.py <runtime.json> <linear.json>
    python3 tools/arc_report.py results/arc/*.json
"""
import glob
import json
import os
import sys


def load_results(pats):
    paths = []
    for p in pats:
        paths.extend(glob.glob(p))
    out = {}
    for p in sorted(paths):
        try:
            d = json.load(open(p))
            out[d.get("arm") or os.path.basename(p)] = (p, d)
        except Exception as e:
            print("  skip %s: %s" % (p, e), file=sys.stderr)
    return out


def solve_stats(turn_ok):
    """Per-turn booleans -> the ARC-relevant summary."""
    n = len(turn_ok)
    solved = sum(1 for x in turn_ok if x)
    first = next((i for i, x in enumerate(turn_ok) if x), None)
    last = next((i for i in range(n - 1, -1, -1) if turn_ok[i]), None)

    # longest run of consecutive solves
    best = cur = 0
    for x in turn_ok:
        cur = cur + 1 if x else 0
        best = max(best, cur)

    # self-correction: a solve that follows at least one failure. No feedback is given between turns,
    # so this is genuinely the artifact improving from the model's own reasoning, not from a signal.
    corrections = sum(1 for i in range(1, n) if turn_ok[i] and not turn_ok[i - 1])
    # regressions: a failure that follows a solve -- the artifact got it right and then lost it.
    regressions = sum(1 for i in range(1, n) if not turn_ok[i] and turn_ok[i - 1])

    return {
        "turns": n,
        "solved": solved,
        "solve_rate": (solved / n) if n else 0.0,
        "first_solve_turn": first,
        "last_solve_turn": last,
        "best_streak": best,
        "self_corrections": corrections,
        "regressions": regressions,
        "final_ok": bool(turn_ok[-1]) if turn_ok else False,
    }


def cost_stats(d):
    m = d.get("metrics") or {}
    rows = d.get("cache_rows") or []
    return {
        "hit_rate": d.get("cache_hit_rate"),
        "cost_units": m.get("cost_units"),
        "cost_units_no_cache": m.get("cost_units_no_cache"),
        "saving": m.get("cost_saving_ratio"),
        "miss_tokens": d.get("cache_total_tokens", 0) - d.get("cache_hit_tokens", 0)
                        if d.get("cache_total_tokens") else None,
        "prompt_first": d.get("prompt_first"),
        "prompt_last": d.get("prompt_last"),
        "ttft_first": d.get("ttft_first"),
        "ttft_last": d.get("ttft_last"),
        "seconds": d.get("seconds"),
        "cache_rows": len(rows),
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    res = load_results(argv)
    if not res:
        print("no results loaded", file=sys.stderr)
        return 1

    print("=" * 78)
    print("ARC-AGI-2 RUN REPORT")
    print("=" * 78)

    for arm, (path, d) in sorted(res.items()):
        turn_ok = d.get("turn_ok") or []
        s = solve_stats(turn_ok)
        c = cost_stats(d)
        print("\narm=%s   (%s)" % (arm, os.path.basename(path)))
        print("  task/model: %s / %s   features=%s" % (
            (d.get("task_id") or d.get("arc_task") or "?"), d.get("model"), d.get("features")))
        print("  SOLVES      %d/%d   rate=%.3f   first=%s   last=%s   best_streak=%d   final_ok=%s" % (
            s["solved"], s["turns"], s["solve_rate"],
            ("turn %d" % s["first_solve_turn"]) if s["first_solve_turn"] is not None else "never",
            ("turn %d" % s["last_solve_turn"]) if s["last_solve_turn"] is not None else "never",
            s["best_streak"], s["final_ok"]))
        print("  self-corrections=%d   regressions=%d  (no feedback is given between turns)" % (
            s["self_corrections"], s["regressions"]))
        print("  COST        hit=%.3f  units=%s  saving=%s  prompt %s->%s  ttft %s->%s ms" % (
            c["hit_rate"] if c["hit_rate"] is not None else float("nan"),
            ("%.0f" % c["cost_units"]) if c["cost_units"] is not None else "?",
            ("%.2fx" % c["saving"]) if c["saving"] is not None else "?",
            c["prompt_first"], c["prompt_last"],
            ("%.0f" % c["ttft_first"]) if c["ttft_first"] is not None else "?",
            ("%.0f" % c["ttft_last"]) if c["ttft_last"] is not None else "?"))
        if d.get("stopped_early") is not None:
            print("  PARTIAL: stopped at turn %s" % d["stopped_early"])

    # paired comparison, when both arms are present
    arms = sorted(res)
    if len(arms) == 2:
        a1, a2 = arms[0], arms[1]
        _, d1 = res[a1]
        _, d2 = res[a2]
        c1, c2 = cost_stats(d1), cost_stats(d2)
        s1, s2 = solve_stats(d1.get("turn_ok") or []), solve_stats(d2.get("turn_ok") or [])
        print("\n" + "-" * 78)
        print("PAIRED COMPARISON")
        print("-" * 78)
        if c1["cost_units"] and c2["cost_units"]:
            hi = max(c1["cost_units"], c2["cost_units"])
            lo = min(c1["cost_units"], c2["cost_units"])
            cheaper = a1 if c1["cost_units"] == lo else a2
            print("  cost ratio %.2fx  (%s is cheaper: %.0f vs %.0f units)" % (
                hi / lo, cheaper, lo, hi))
        print("  solves: %s=%d/%d  %s=%d/%d" % (a1, s1["solved"], s1["turns"],
                                                a2, s2["solved"], s2["turns"]))
        print("\n  NOTE: a low solve count is the expected outcome on ARC-AGI-2 and is reported as such.")
        print("  The measured claim is the COST ratio on a real ARC task, not a competitive score.")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
