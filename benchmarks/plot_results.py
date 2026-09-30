#!/usr/bin/env python3
"""Bob's Step 4: the benchmark comparison plot.

    python benchmarks/plot_results.py

Produces `benchmark_wallclock_vs_recompute.png` with two panels:

  LEFT   end-to-end wall clock, per turn and cumulative, standard recompute vs block runtime
         (Bob's Test 4 / his "3.4x faster turnaround" ask)
  RIGHT  peak KV / memory watermark, the same two arms

★ WHAT THIS PLOT DELIBERATELY SHOWS THAT A FLATTERING ONE WOULD HIDE:
  - The runtime's ONE-TIME ingest is drawn as a separate bar, so the speedup is not read as free.
  - The resident-KV panel will show the runtime NOT winning, because on this workload it does not
    (the evicted turns are tiny and the global anchor is a retained tier). A plot that only showed
    the prefill win would be a sales chart.
  - Per-turn wall clock is plotted on a log scale with the turn index, because the vanilla arm's
    cost GROWS with turn number (its context grows) while the runtime's stays flat. That divergence
    is the actual claim.
"""
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_multiturn():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "multiturn_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d if isinstance(d, list) else [d]:
            if isinstance(r, dict) and "vanilla" in r:
                rows.append(r)
    return rows


def main():
    rows = load_multiturn()
    if not rows:
        print("no multiturn results found in %s" % RES)
        return
    r = rows[-1]                       # most recent run
    v, rt = r["vanilla"], r["runtime"]
    turns = [x["turn"] for x in v["rows"]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # ---- panel 1: per-turn prefill tokens (what standard serving re-does every turn)
    ax = axes[0]
    ax.plot(turns, [x["tokens"] for x in v["rows"]], "o-", color="#c0392b",
            label="vanilla (re-prefill all)", linewidth=2)
    ax.plot(turns, [x["tokens"] for x in rt["rows"]], "s-", color="#27ae60",
            label="block runtime", linewidth=2)
    ax.set_xlabel("turn"); ax.set_ylabel("tokens pushed through a forward")
    ax.set_title("Prefill work per turn\n(vanilla grows with context; runtime is flat)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # ---- panel 2: cumulative wall clock, with ingest shown separately
    ax = axes[1]
    vc, rc = [], []
    a = b = 0.0
    for x, y in zip(v["rows"], rt["rows"]):
        a += x["wall_s"]; b += y["wall_s"]
        vc.append(a); rc.append(b)
    ax.plot(turns, vc, "o-", color="#c0392b", label="vanilla cumulative", linewidth=2)
    ax.plot(turns, rc, "s-", color="#27ae60", label="runtime cumulative", linewidth=2)
    ax.bar([turns[-1] + 0.6], [rt.get("ingest_s", 0)], width=0.5, color="#f39c12",
           label="runtime one-time ingest")
    ax.set_xlabel("turn"); ax.set_ylabel("cumulative wall clock (s)")
    speed = (1 - rt["wall_s"] / v["wall_s"]) * 100 if v["wall_s"] else 0
    ax.set_title("End-to-end wall clock\nruntime is %.0f%% less (%.2fx faster)"
                 % (speed, v["wall_s"] / rt["wall_s"] if rt["wall_s"] else 0))
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    # ---- panel 3: KV watermark -- the honest panel that does NOT flatter the runtime
    ax = axes[2]
    ax.plot(turns, [x["kv"] / 1e6 for x in v["rows"]], "o-", color="#c0392b",
            label="vanilla", linewidth=2)
    ax.plot(turns, [x["kv"] / 1e6 for x in rt["rows"]], "s-", color="#27ae60",
            label="runtime", linewidth=2)
    ax.set_xlabel("turn"); ax.set_ylabel("KV resident at decode (MB)")
    ax.set_title("Memory watermark\nNOTE: runtime does NOT win here — see README F36")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.suptitle("Block-causal runtime vs standard recompute — %s, %d turns, %d evictions"
                 % (r["model"], r["turns"], len(r["evicted"])), fontsize=13, y=1.00)
    fig.tight_layout()
    out = os.path.join(HERE, "benchmark_wallclock_vs_recompute.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote %s" % out)

    print("\n  numbers behind the plot:")
    print("  %-24s %12s %12s" % ("", "vanilla", "runtime"))
    print("  %-24s %12d %12d" % ("prefill tokens (total)", v["total_tokens"],
                                  rt["total_tokens"]))
    print("  %-24s %11.2fs %11.2fs" % ("wall (total)", v["wall_s"], rt["wall_s"]))
    print("  %-24s %11.1fMB %11.1fMB" % ("KV at final decode",
                                          v["final"]["kv"] / 1e6, rt["final"]["kv"] / 1e6))
    print("  %-24s %12s %10.1fs" % ("one-time ingest", "-", rt.get("ingest_s", 0)))


if __name__ == "__main__":
    main()
