#!/usr/bin/env python3
"""Compare the 7B depth sweep against the 0.5B one, side by side.

The question Bob set: does the conditioning wall scale away with parameters, or is
it architectural? The 0.5B answer was that each depth is passed by a DIFFERENT
selector, which is the signature of noise rather than a working policy. If 7B
passes consistently, that signature should disappear.
"""
import json, sys, os

def load(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("error"):
            rows[(d.get("depth"), d.get("mode"))] = {"error": d["error"]}
        elif "depth" in d:
            rows[(d["depth"], d["mode"])] = d
    return rows

def table(title, rows, depths, modes):
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("  (no data)")
        return
    print("  %-14s %s" % ("mode", "".join("%-16s" % ("depth %.0f%%" % (d * 100)) for d in depths)))
    for m in modes:
        cells = []
        for d in depths:
            r = rows.get((d, m))
            if not r:
                cells.append("%-16s" % "--")
            elif "error" in r:
                cells.append("%-16s" % "ERROR")
            else:
                cells.append("%-16s" % ("%s %4.1f%%" % ("PASS" if r["pass_"] else "fail",
                                                        r.get("kv_saved_pct", 0))))
        print("  %-14s %s" % (m, "".join(cells)))
    passes = sum(1 for r in rows.values() if r.get("pass_"))
    print("  -> %d/%d combinations pass" % (passes, len(rows)))

def consistency(rows, depths, modes):
    """Does any single mode pass every depth? That is what 'robust' means."""
    print()
    print("ROBUSTNESS: does one mode pass at EVERY measured depth?")
    for m in modes:
        got = [(d, rows.get((d, m))) for d in depths]
        have = [(d, r) for d, r in got if r]
        if not have:
            continue
        ok = all(r.get("pass_") for _, r in have)
        detail = " ".join("%.0f%%:%s" % (d * 100, "P" if r.get("pass_") else "f") for d, r in have)
        print("  %-14s %s   %s" % (m, detail, "ALL PASS" if ok else ""))
    print()
    print("  A mode passing all depths = the wall scaled away.")
    print("  Each depth passed by a different mode = noise, as on the 0.5B.")


if __name__ == "__main__":
    depths = [0.15, 0.35, 0.55, 0.75]
    modes = ["baseline", "line", "posnorm"]
    a = load("data/run_7b.jsonl")
    b = {}
    for f in ("data/depth_sweep.jsonl", "data/baseline_causal.jsonl"):
        for k, v in load(f).items():
            b[k] = v
    table("7B (8-bit)  Qwen/Qwen2.5-7B", a, depths, modes)
    print()
    table("0.5B (fp32) Qwen/Qwen2.5-0.5B", b, depths, modes)
    print()
    print("=" * 78)
    print("THE QUESTION: did 7B make the wall go away?")
    print("=" * 78)
    consistency(a, depths, modes)
    if a:
        print()
        print("  needle retained in %d/%d 7B combos (selection worked)"
              % (sum(1 for r in a.values() if r.get("needle_kept")), len(a)))
        print("  answers actually correct in %d/%d" % (
            sum(1 for r in a.values() if r.get("pass_")), len(a)))
        print("  NOTE: 8-bit changes numerics (Finding 10), so this is not a")
        print("        like-for-like rerun of the fp32 0.5B arm.")
