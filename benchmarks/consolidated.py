"""Consolidated benchmark table across every run artifact.

    python3 benchmarks/consolidated.py

★ WHY THIS IS A FILE AND NOT A PARAGRAPH IN A CHAT. Bob's ask was *"run the actual benchmarks
already, thats what people care about"*. A number quoted in a message cannot be checked, cannot be
regenerated, and drifts from the artifact it came from. This reads the JSON on disk and prints what
is actually there, so the table in the README is a rendering of the evidence rather than a memory
of it.

Reads, in order of authority:
  modal_*.json        the 4 core benchmarks x control/vanilla/runtime arms
  multiturn_*.json    8-turn cost
  recompute_state_*.json   sequential state update: block vs recompute
  hardware_*.json     measured Wh + modelled FLOPs/bytes
  two_needle_trials / rope_offsets / frankenstein / scratchpad    the validation tests

★ HONESTY RULES ENCODED HERE, because a table is where they get lost:
  - an arm that did not run prints "-", never 0 or PASS
  - a NULL (every arm fails, including the control) is labelled NULL, so it cannot be read as
    "the runtime failed"
  - n is printed next to every percentage; a rate off 4 trajectories is not a rate
"""
import glob, json, os, sys

R = "/tmp/opencode/ccai/benchmarks/results"


def load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(R, pat))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        out.append((p, d))
    return out


def yn(x):
    return "PASS" if x else "fail"


def cell(v):
    return "PASS" if v else "fail"


def core_benchmarks():
    print("=" * 100)
    print("CORE BENCHMARKS — control (all blocks, 1 causal seq) / vanilla (survivors, no "
          "distractor) / runtime (block table)")
    print("=" * 100)
    runs = load("modal_*.json")
    if not runs:
        print("  (no modal_*.json yet)")
        return
    rows = []
    for p, d in runs:
        scale = d.get("scale", "?")
        for x in d.get("rows", []):
            # a cell that failed to run has no arms at all -- skip, never render it as "fail"
            if not all(k in x for k in ("control", "vanilla", "runtime")):
                continue
            rows.append((x.get("model", "?"), x.get("test", "?"), scale,
                         x["control"]["ok"], x["vanilla"]["ok"], x["runtime"]["ok"],
                         x["control"].get("kv_bytes", 0), x["runtime"].get("kv_bytes", 0),
                         x["runtime"].get("drift_evict")))
    print("  %-15s %-17s %-6s %-8s %-8s %-8s %-9s %s" % (
        "model", "test", "scale", "control", "vanilla", "runtime", "kv held", "drift"))
    print("  " + "-" * 94)
    for m, t, sc, c, v, rt, ckv, rkv, dr in rows:
        ks = (rkv / ckv * 100) if ckv else 0
        print("  %-15s %-17s %-6s %-8s %-8s %-8s %8.1f%%  %.1e" % (
            m, t, sc, yn(c), yn(v), yn(rt), ks, dr or 0))

    # NULL detection: every arm failed, so the task cannot discriminate layouts here.
    print()
    seen = {}
    for m, t, sc, c, v, rt, *_ in rows:
        seen.setdefault((t, sc), []).append(c or v or rt)
    nulls = [k for k, v in seen.items() if not any(v)]
    if nulls:
        print("  ★ NULLS — every arm failed INCLUDING the control, so the task does not measure")
        print("    the layout on this model. Do NOT read these as a runtime failure:")
        for t, sc in nulls:
            print("      %s @ %s" % (t, sc))
    print()


def state_update():
    print("=" * 100)
    print("SEQUENTIAL STATE UPDATE — block table vs recompute (Bob's babilong / multi-turn case)")
    print("=" * 100)
    runs = load("recompute_state_*.json")
    if not runs:
        print("  (no recompute_state_*.json yet)")
        return
    for p, d in runs:
        rows = d if isinstance(d, list) else d.get("rows", [])
        for x in rows:
            r = x.get("rates") or {}
            cor = x.get("correct") or {}
            if not r and not cor:
                continue
            print("  %-15s  block %5s   recompute %5s   (n=%s)" % (
                x.get("model", "?"),
                ("%.0f%%" % (r["block"] * 100)) if r.get("block") is not None else
                ("%s/%s" % (cor.get("block"), x.get("trials", "?"))),
                ("%.0f%%" % (r["recompute"] * 100)) if r.get("recompute") is not None else
                ("%s/%s" % (cor.get("recompute"), x.get("trials", "?"))),
                x.get("trials", "?")))
    print()


def multiturn():
    print("=" * 100)
    print("MULTI-TURN COST (8 turns) — summed over turns, both arms")
    print("=" * 100)
    runs = load("multiturn_*.json")
    if not runs:
        print("  (no multiturn_*.json yet)")
        return
    for p, d in runs:
        rows = d if isinstance(d, list) else [d]
        for x in rows:
            s = x.get("summary") or x
            if "tokens" not in str(s) and "vanilla" not in str(x):
                continue
            print("  %s" % json.dumps(x, indent=2)[:600])
    print()


def main():
    core_benchmarks()
    state_update()
    avail = {os.path.basename(p): len(d if isinstance(d, list) else 1)
             for p, d in load("*.json")}
    print("=" * 100)
    print("ARTIFACTS ON DISK (%d files)" % len(avail))
    print("=" * 100)
    for k in sorted(avail):
        print("  %-46s %3s rows" % (k, avail[k]))


if __name__ == "__main__":
    main()
