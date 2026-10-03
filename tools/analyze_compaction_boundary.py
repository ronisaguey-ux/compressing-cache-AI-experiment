#!/usr/bin/env python3
"""Analyze prefix cache hit rate and cost penalties across compaction boundaries.

Measures the exact cache hit rate cliff at turn 30 and turn 60 compaction events
in the prune arm vs the continuous prefix preservation of the runtime arm.
"""
import glob
import json
import sys

def analyze(paths):
    runs = {}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            try:
                with open(p) as f:
                    d = json.load(f)
                arm = d.get("arm", p)
                runs[arm] = d
            except Exception as e:
                print(f"Error loading {p}: {e}", file=sys.stderr)

    if not runs:
        print("No valid runs loaded.", file=sys.stderr)
        return 1

    print("=" * 80)
    print("COMPACTION BOUNDARY PREFIX CACHE ANALYSIS")
    print("=" * 80)

    for arm in sorted(runs):
        r = runs[arm]
        rows = sorted(r.get("cache_rows") or [], key=lambda x: x.get("turn", 0))
        if not rows:
            print(f"[{arm}] No cache_rows available.")
            continue

        print(f"\n--- Arm: {arm} (Total turns: {len(rows)}) ---")
        print(f"{'turn':<6} {'prompt_tok':<12} {'hit_tok':<12} {'miss_tok':<12} {'hit_rate':<10} {'event':<20}")
        print("-" * 72)

        prev_hit_rate = None
        for row in rows:
            t = row.get("turn", 0)
            prompt = row.get("prompt", 0)
            hit = row.get("hit", 0)
            miss = max(0, prompt - hit)
            hr = hit / prompt if prompt else 0.0

            event = ""
            if arm == "prune":
                if t in (29, 30, 31):
                    event = "★ BOUNDARY 30" if t == 30 else ("pre-prune" if t == 29 else "post-prune")
                elif t in (59, 60, 61):
                    event = "★ BOUNDARY 60" if t == 60 else ("pre-prune" if t == 59 else "post-prune")
            elif arm == "runtime":
                if t in (29, 30, 31, 59, 60):
                    event = "continuous flat"

            # Print key window around turn 30 and turn 60, or full if requested
            if t in range(25, 36) or t in range(55, 65):
                drop = ""
                if prev_hit_rate is not None and hr < prev_hit_rate - 0.05:
                    drop = f" (DROP: -{(prev_hit_rate - hr)*100:.1f}%)"
                print(f"{t:<6d} {prompt:<12d} {hit:<12d} {miss:<12d} {hr*100:<9.1f}% {event + drop:<20}")
            prev_hit_rate = hr

    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_json_path_or_glob> ...")
        sys.exit(2)
    sys.exit(analyze(sys.argv[1:]))
