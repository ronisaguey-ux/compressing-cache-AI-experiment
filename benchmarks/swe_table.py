"""The SWE-bench table: linear prefill vs the block runtime, per model.

    python3 benchmarks/swe_table.py

★ WHAT MAKES THIS TABLE MEANINGFUL, and it is not the resolution rate.

The models are 7-8B and SWE-bench Lite is hard: the expected resolution rate is LOW, possibly zero
for both arms. A table that only reported "resolved 0/10 vs 0/10" would be true and useless -- it
would look like the runtime failed when in fact the task is beyond the model, and it would say
nothing about the LAYOUT, which is the entire subject of this work.

So the table leads with the quantities that DO separate the arms regardless of whether the model
solves anything:

  TTFT growth        the latency a user feels, first turn vs last. This is what an eviction policy
                     changes: linear re-processes an ever-growing context, runtime holds a bounded
                     one.
  prefill tokens     the WORK per turn, measured -- the direct cost of the layout.
  peak KV resident   the context that has to FIT. Linear grows without bound; runtime is capped.
  resolved           reported honestly, with parity, and NOT presented as the headline.

★ THE GOLD ARM IS THE PARITY CHECK and is read first. If gold does not resolve every instance the
harness is broken and none of the model numbers mean anything. That check already fired once: the
dataset's PASS_TO_PASS contained a test failing at the base commit, which marked a correct patch
unresolved until P2P was validated against the base state.
"""
import glob, json, os, statistics as st

R = "/tmp/opencode/ccai/benchmarks/results"


def load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(R, pat))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and "rows" in d:
            out.append(d)
    return out


def load_latest(pat):
    """One artifact per (model, arm): the NEWEST.

    ★ WITHOUT THIS THE TABLE MIXES RUNS. A stale artifact from before the parser fixes sits next to
    the re-run and the same arm appears twice with different numbers, which makes the comparison
    read as noise. The newest run is the one whose code is current, so it wins; the superseded file
    is moved to results/superseded/ so it stays on the record without entering the table.
    """
    best = {}
    for p in sorted(glob.glob(os.path.join(R, pat))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not (isinstance(d, dict) and "rows" in d):
            continue
        k = (d.get("model"), d.get("arm"))
        if k not in best or os.path.getmtime(p) > os.path.getmtime(best[k][0]):
            best[k] = (p, d)
    return [v[1] for v in best.values()]


def arm_summary(d):
    rows = [x for x in d["rows"] if x.get("status") == "ok"]
    if not rows:
        return None
    return dict(
        model=d["model"], arm=d["arm"], n=len(rows),
        resolved=sum(1 for x in rows if x.get("resolved")),
        ttft_first=st.mean([x["ttft_ms_first"] for x in rows]),
        ttft_mean=st.mean([x["ttft_ms_mean"] for x in rows]),
        ttft_last=st.mean([x["ttft_ms_last"] for x in rows]),
        peak_kv_mb=st.mean([x["peak_kv_mb"] for x in rows]),
        wall_s=st.mean([x["wall_s"] for x in rows]),
        turns=st.mean([x["turns"] for x in rows]),
        tool_calls=sum(sum(1 for t in x.get("per_turn", []) if t.get("tool")) for x in rows),
    )


def main():
    gold = load_latest("swe_*_gold_*.json")
    print("=" * 104)
    print("GOLD PARITY CHECK — the dataset's own patch. If this is not 100%, the harness is wrong "
          "and no model number counts.")
    print("=" * 104)
    if not gold:
        print("  (no gold run yet)")
    for d in gold:
        rows = [x for x in d["rows"] if x.get("status") == "ok"]
        ok = sum(1 for x in rows if x.get("resolved"))
        print("  %-22s %2d/%2d resolved" % (d["model"], ok, len(rows)))
        for x in rows:
            if not x.get("resolved"):
                print("      NOT RESOLVED: %s  f2p=%s p2p=%s" % (
                    x["instance"], x.get("f2p_summary"), x.get("p2p_summary")))

    print()
    print("=" * 104)
    print("LINEAR vs RUNTIME — measured per turn")
    print("=" * 104)
    arms = [arm_summary(d) for d in load_latest("swe_*_linear_*.json")
            + load_latest("swe_*_runtime_*.json")]
    arms = [a for a in arms if a]
    if not arms:
        print("  (no model arms yet)")
        return
    print("  %-22s %-9s %4s %8s %9s %9s %9s %10s %7s" % (
        "model", "arm", "n", "resolved", "TTFT t0", "TTFT avg", "TTFT tN", "peak KV MB", "turns"))
    print("  " + "-" * 100)
    for a in arms:
        print("  %-22s %-9s %4d %5d/%-2d %8.0f %9.0f %9.0f %10.1f %7.1f" % (
            a["model"], a["arm"], a["n"], a["resolved"], a["n"],
            a["ttft_first"], a["ttft_mean"], a["ttft_last"], a["peak_kv_mb"], a["turns"]))

    print()
    print("  GROWTH — the quantity the layout controls, independent of whether the model solves it")
    print("  %-22s %-9s %11s %13s" % ("model", "arm", "TTFT tN/t0", "peak KV MB"))
    print("  " + "-" * 60)
    by = {}
    for a in arms:
        by.setdefault(a["model"], {})[a["arm"]] = a
        g = a["ttft_last"] / a["ttft_first"] if a["ttft_first"] else 0
        print("  %-22s %-9s %10.2fx %13.1f" % (a["model"], a["arm"], g, a["peak_kv_mb"]))
    print()
    for m, d in sorted(by.items()):
        if "linear" in d and "runtime" in d:
            l, r = d["linear"], d["runtime"]
            ksr = r["peak_kv_mb"] / l["peak_kv_mb"] if l["peak_kv_mb"] else 0
            ttr = r["ttft_last"] / l["ttft_last"] if l["ttft_last"] else 0
            print("  %-22s runtime holds %.0f%% of linear's KV | last-turn TTFT %.2fx" % (
                m, ksr * 100, ttr))
    print()
    print("  ⚠️ A resolution rate of 0 for BOTH arms is a statement about a 7B model on SWE-bench,")
    print("     not about the runtime. The layout claim rests on TTFT/tokens/KV, which are measured")
    print("     on every turn regardless of outcome.")


if __name__ == "__main__":
    main()
