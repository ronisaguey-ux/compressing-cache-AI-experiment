#!/usr/bin/env python3
"""Compare two arms of the incremental-coding benchmark -- FAIRLY, or not at all.

Why this exists as a separate gate instead of a line in the runner
------------------------------------------------------------------
The two arms are deliberately different policies, so it is easy to compare them on a basis
that hands one an unearned win. Two of those traps are known and each one is checked here
BEFORE any number is reported:

1. TURN COUNT. The runtime arm keeps a small bounded context, so it is faster per turn. If
   the run is stopped by wall-clock rather than by turn count, runtime completes MORE
   features than linear and wins on volume rather than on policy. `features_completed` must
   match exactly; if it does not, this script prints the reason and exits non-zero instead
   of reporting a difference.

2. HARNESS FAILURE. A grader that crashes prints `PASSED 0/0` with an empty per_feature
   dict, which is indistinguishable from "ran fine, nothing passed". `harness_failed` is
   surfaced by the harness itself; a run carrying it is not a measurement of the model and
   is refused here.

Usage
-----
    python3 tools/run_compare.py linear.json runtime.json
    python3 tools/run_compare.py 'results/*.json'      # globs, matched by their `arm` field
"""
import glob
import json
import sys


def load(paths):
    out = {}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            try:
                with open(p) as fh:
                    r = json.load(fh)
            except Exception as e:
                print("  ! cannot read %s: %s" % (p, e))
                continue
            if isinstance(r, list):
                for item in r:
                    out[item.get("arm", p)] = item
            else:
                out[r.get("arm", p)] = r
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    runs = load(argv[1:])
    if len(runs) < 2:
        print("need two arms, got %d: %s" % (len(runs), list(runs)))
        return 2

    # ---- GATE 1: neither run may be a harness failure -------------------------------------
    bad = [a for a, r in runs.items() if r.get("harness_failed")]
    if bad:
        for a in bad:
            print("REFUSING: %s reported harness_failed -- not a model measurement" % a)
        return 1

    # ---- GATE 2: equal turn counts, or there is nothing to compare ------------------------
    counts = {a: r.get("features_completed") for a, r in runs.items()}
    if len(set(counts.values())) != 1:
        print("REFUSING: the arms did NOT run the same number of features -- the difference "
              "would be volume, not policy:")
        for a, c in counts.items():
            print("    %-10s features_completed=%s stopped_early=%s"
                  % (a, c, runs[a].get("stopped_early")))
        print("\n  Re-run with a turn-count stop (CCAI_TIME_BUDGET_S high enough that it never "
              "fires). A wall-clock stop is arm-dependent: the runtime arm is faster per turn.")
        return 1

    n = next(iter(counts.values()))
    truncated = [a for a, r in runs.items() if r.get("stopped_early") is not None]
    if truncated:
        print("NOTE: stopped early (time budget) in %s -- comparing the %d turns both ran.\n"
              % (", ".join(truncated), n))

    # ---- the comparison -------------------------------------------------------------------
    print("=" * 78)
    print("INCREMENTAL-CODING BENCHMARK  --  %d feature turns, both arms" % n)
    print("=" * 78)
    def _flag(r, name):
        """contract/nonce are written by the GRADER into per_feature, not the top level --
        reading r.get(name) returns None and reports blank retention on a real run."""
        pf = r.get("per_feature") or {}
        if name in pf:
            return pf[name]
        return r.get(name)

    hdr = "%-10s %10s %9s %9s %9s %10s %11s" % (
        "arm", "passed", "contract", "nonce", "TTFT 1st", "TTFT last", "prompt last")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(runs):
        r = runs[a]
        cont = _flag(r, "contract")
        print("%-10s %10s %9s %9s %9s %10s %11s" % (
            a,
            "%s/%s" % (r.get("passed"), r.get("total")),
            str(cont), str(_flag(r, "nonce")),
            "%.0f" % r.get("ttft_first", 0), "%.0f" % r.get("ttft_last", 0),
            r.get("prompt_last")))

    print()
    print("%-10s %12s %12s %10s" % ("arm", "TTFT growth", "prompt growth", "KV peak"))
    print("-" * 48)
    for a in sorted(runs):
        r = runs[a]
        f, l = r.get("ttft_first", 0), r.get("ttft_last", 0)
        pf, pl = r.get("prompt_first", 0), r.get("prompt_last", 0)
        print("%-10s %11s %12s %10s" % (
            a,
            ("%.2fx" % (l / f)) if f else "n/a",
            ("%.1fx" % (pl / pf)) if pf else "n/a",
            "%.0f MB" % (r.get("kv_peak", 0) / 1e6)))

    # ── ★★ RETENTION OF TURN 1: THE HEADLINE ──────────────────────────────────────────────
    # The manifest codes are stated ONCE in turn 1, are FORBIDDEN from the file until the final
    # turn, and cannot be inferred from the model's own recent output. So they are recoverable
    # exactly when turn 1 survived -- and there are eight, so partial retention scores partially.
    print()
    print("TURN-1 RETENTION  (8 manifest codes stated once at turn 1, required only at the end)")
    print("-" * 84)
    print("%-10s %12s %10s %10s" % ("arm", "manifest", "in order", "duplicated"))
    for a in sorted(runs):
        r = runs[a]
        n = r.get("manifest_n")
        tot = r.get("manifest_total", 8)
        if n is None:
            print("%-10s %12s" % (a, "(not reported)"))
            continue
        print("%-10s %8s/%-3s %10s %10s" % (
            a, n, tot, r.get("manifest_ordered"), r.get("manifest_dup")))

    # ── per-clause: WHICH KIND of requirement survives ─────────────────────────────────────
    clauses = sorted({k for r in runs.values() for k in (r.get("clauses") or {})})
    if clauses:
        print()
        print("CONTRACT CLAUSES  (C1-C3/C6 are exact values; C4/C5 are shapes re-applied as it writes)")
        print("-" * 84)
        print("%-10s %s" % ("arm", "  ".join("%-6s" % c for c in clauses)))
        for a in sorted(runs):
            cl = runs[a].get("clauses") or {}
            print("%-10s %s" % (a, "  ".join("%-6s" % cl.get(c) for c in clauses)))

    print()
    # ★ IN FIX MODE THERE IS NO NONCE OR MANIFEST -- the turn-1 payload is the bug report itself,
    # and per-bug correctness IS the retention signal. Printing "LOST (nonce=None)" here would
    # describe a test that was never run as a test that failed, which is the exact class of
    # mislabelled metric this file exists to prevent. Say "not applicable" and point at the number
    # that does measure it.
    for a in sorted(runs):
        r = runs[a]
        if r.get("per_feature", {}).get("task") == "fix":
            pf = r.get("per_feature", {})
            bugs = [k for k in pf if k.startswith("feature_")]
            ok = sum(1 for k in bugs if pf[k] is True)
            t_ok = r.get("turns_ok")
            t_tot = len(r.get("turn_ok") or []) or None
            # ★ THE CONTROLLED-EXPERIMENT LINE. Both arms hold the turn-1 brief, so this is the
            # number that isolates "how the policy handled accumulated junk" from "who was given
            # the brief". Printed ABOVE the fix counts on purpose -- it is the headline now.
            cn, ct = r.get("codes_n"), r.get("codes_total")
            if ct:
                print("  accumulated codes %-10s : %s/%s recovered, in order=%s, dup=%s"
                      % (a, cn, ct, r.get("codes_ordered"), r.get("codes_dup")))
            # ★ THE OWNER'S METRIC IS PER-TURN: did turn i succeed. The final-state count is kept
            # alongside it because the GAP between them is itself a finding -- a large gap means
            # fixes were applied and then clobbered by later full-file rewrites.
            if t_ok is not None and t_tot:
                print("  per-turn success  %-10s : %d/%d turns applied their own fix correctly"
                      % (a, t_ok, t_tot))
            print("  final state       %-10s : %d/%d bugs correct at the end"
                  % (a, ok, len(bugs)))
            if t_ok is not None and t_tot:
                lost = t_ok - ok
                if lost > 0:
                    print("  %-26s %d fix(es) applied then LOST to a later rewrite" % ("", lost))
    if any(r.get("per_feature", {}).get("task") == "fix" for r in runs.values()):
        return 0
    for a in sorted(runs):
        r = runs[a]
        nonce = _flag(r, "nonce")
        n = r.get("manifest_n")
        if n is None:
            verdict = "RETAINED" if nonce else "LOST"
            print("  turn-1 contract in %-10s : %s (nonce=%s)" % (a, verdict, nonce))
        else:
            verdict = "RETAINED" if (n == r.get("manifest_total", 8)) else "LOST"
            print("  turn-1 retention in %-10s : %-8s (%s/%s codes, in order=%s)"
                  % (a, verdict, n, r.get("manifest_total", 8), r.get("manifest_ordered")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
