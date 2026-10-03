#!/usr/bin/env python3
"""Sweep prefix-cache hit discount multipliers (0.05, 0.10, 0.20, 0.50, 1.00).

Demonstrates sensitivity of cost units and cost saving ratios to the assumed
cache discount constant across commercial serving engine regimes.
"""
import glob
import json
import sys

def sweep(paths, multipliers=(0.05, 0.10, 0.20, 0.50, 1.00)):
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
    print(f"PREFIX CACHE SENSITIVITY SWEEP  ({len(runs)} arms: {', '.join(sorted(runs))})")
    print("=" * 80)

    # Base token accounting
    print(f"{'arm':<12} {'total_tokens':<14} {'hit_tokens':<14} {'miss_tokens':<14} {'hit_rate':<10}")
    print("-" * 66)
    stats = {}
    for arm in sorted(runs):
        r = runs[arm]
        tot = r.get("cache_total_tokens", 0)
        hit = r.get("cache_hit_tokens", 0)
        miss = tot - hit
        hr = hit / tot if tot else 0.0
        stats[arm] = (tot, hit, miss, hr)
        print(f"{arm:<12} {tot:<14d} {hit:<14d} {miss:<14d} {hr:<10.3f}")

    print("\n" + "=" * 80)
    print("COST UNITS & SAVINGS BY HIT MULTIPLIER (miss*1.0 + hit*mult)")
    print("=" * 80)

    base_arm = "runtime" if "runtime" in stats else sorted(stats.keys())[0]

    header = f"{'multiplier':<12}"
    for arm in sorted(stats):
        header += f" {arm + ' (units)':<18}"
    for arm in sorted(stats):
        if arm != base_arm:
            header += f" {arm + '/' + base_arm:<14}"
    print(header)
    print("-" * len(header))

    for m in multipliers:
        row = f"{m:<12.2f}"
        costs = {}
        for arm in sorted(stats):
            tot, hit, miss, _ = stats[arm]
            cu = miss * 1.0 + hit * m
            costs[arm] = cu
            row += f" {cu:<18.1f}"
        for arm in sorted(stats):
            if arm != base_arm:
                ratio = costs[arm] / costs[base_arm] if costs[base_arm] else 0.0
                row += f" {ratio:<14.2f}x"
        print(row)
    print("=" * 80)
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_json_path_or_glob> ...")
        sys.exit(2)
    sys.exit(sweep(sys.argv[1:]))
