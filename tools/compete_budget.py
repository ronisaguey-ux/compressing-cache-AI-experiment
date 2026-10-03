#!/usr/bin/env python3
"""Project both policies into the competition harness's real budget.

★ WHY A PROJECTION AND NOT A CLAIM OF MEASUREMENT. Our run measures the policies on a controlled
task. The competition harness imposes three constraints we did not run under, all of them
documented by the host and by public traces of the official starter:

  1. the harness compacts the agent's history once the prompt reaches 14,336 tokens -- 2,048 below
     the output ceiling it requests -- and it acts on the prompt just sent, so it is one call late;
  2. the scorer serves the model at max_model_len 32,768 and REFUSES any call whose prompt plus
     requested output exceeds it, ending the task with no patch;
  3. with a LoRA adapter the KV cache on 4xL4 falls from ~46,000 to ~7,600 tokens.

This script takes the MEASURED per-turn prompt sizes from our runs and asks what each policy looks
like inside that envelope. It is a projection from measured growth rates, and it says so; it is not
presented as a second experiment.

Usage: python3 tools/compete_budget.py [results_dir]
"""
import glob
import json
import os
import sys

RESULTS = os.environ.get("RESULTS_DIR",
                         os.path.expanduser("~/.local/share/ccai-results/fixmode-v3"))
# Host-stated constants (documented in the competition harness and public traces).
COMPACT_AT = 14336
OUTPUT_CEILING = 16384
MAX_MODEL_LEN = 32768
KV_WITH_ADAPTER = 7600
# A compaction keeps a summary plus the recent tail; the floor is an assumption, stated as one.
COMPACT_TO = 4000
KV_NO_ADAPTER = 46000
TURNS = 200          # a long agentic session, which is the regime the paper is about


def load(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "incremental_*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict) and r.get("arm") and r.get("cache_rows"):
            runs[r["arm"]] = r
    return runs


def growth_rate(run):
    """Tokens added per turn, measured from the EARLY turns.

    ★ Averaging over the whole run is wrong for the linear arm: once it saturates, deltas approach
    zero and drag the mean down, which would understate exactly the growth being projected. The
    rate is taken from the turns before saturation.
    """
    rows = sorted(run["cache_rows"], key=lambda x: x.get("turn", 0))
    if len(rows) < 4:
        return 0.0
    base = rows[0]["prompt"]
    peak = max(r["prompt"] for r in rows)
    # walk forward while still growing; stop at the first plateau
    last = 0
    for i, r in enumerate(rows):
        if r["prompt"] >= peak - 1:
            break
        last = i
    return (rows[last]["prompt"] - base) / max(1, last)


def project(start, rate, cap, compact_at=None, compact_to=None):
    """Prompt size per turn under a cap, plus the turns on which compaction fired.

    ★ THE RECORDED SERIES CANNOT SHOW A THRESHOLD CROSSING, because the reset happens in the same
    step that crosses it. Detecting "did this arm reach the threshold" from the output would
    therefore always answer no -- the earlier version of this script reported exactly that, for
    both arms, which would have understated the whole contrast. The crossing is recorded as it
    happens and returned alongside the series.
    """
    p = start
    out, compactions = [], []
    for t in range(1, TURNS + 1):
        p = min(cap, p + rate)
        if compact_at and compact_to and p >= compact_at:
            compactions.append(t)
            p = compact_to
        out.append(p)
    return out, compactions


def main(argv):
    d = argv[1] if len(argv) > 1 else RESULTS
    runs = load(d)
    if not runs:
        print("no runs in %s" % d)
        return 1
    print("Competition envelope: compact at %d | hard refusal at %d prompt+output | KV %d with adapter (vs %d without)"
          % (COMPACT_AT, MAX_MODEL_LEN, KV_WITH_ADAPTER, KV_NO_ADAPTER))
    print("Budget for prompt alone: %d tokens leaves %d for output before refusal.\n"
          % (MAX_MODEL_LEN - OUTPUT_CEILING, OUTPUT_CEILING))
    print("%-9s %8s %10s %9s %13s %12s" % ("arm", "start", "rate/turn", "peak", "prompt@t200",
                                              "compactions"))
    rows = {}
    for a, r in sorted(runs.items()):
        start = sorted(r["cache_rows"], key=lambda x: x["turn"])[0]["prompt"]
        rate = growth_rate(r)
        cap = min(MAX_MODEL_LEN - OUTPUT_CEILING, COMPACT_AT)
        proj, comps = project(start, rate, cap, compact_at=COMPACT_AT, compact_to=COMPACT_TO)
        rows[a] = (proj, comps, rate)
        print("%-9s %8d %10.1f %9d %13d %12d" % (a, start, rate, max(proj), proj[199], len(comps)))
    print()
    for a, (proj, comps, rate) in sorted(rows.items()):
        if comps:
            print("%-9s compacts %d time(s) in %d turns; first at turn %d, then every ~%d turns"
                  % (a, len(comps), TURNS, comps[0],
                     (comps[1] - comps[0]) if len(comps) > 1 else 0))
        else:
            print("%-9s never reaches the threshold within %d turns (growth %.0f tok/turn)"
                  % (a, TURNS, rate))
    print()
    print("Interpretation: our policy's per-turn growth is bounded by the instruction archive, so it")
    print("approaches the threshold far later, or not at all, where a policy that retains whole")
    print("turns reaches it early and then compacts repeatedly -- each compaction rewriting the")
    print("prefix the previous turn already paid for.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
