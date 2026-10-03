#!/usr/bin/env python3
"""Aggregate multi-instance variance runs across task salts.

Computes mean ± standard deviation and 95% Student's t confidence intervals
for cache hit rate, cost units, saving ratio, and code recall across arms.
Enforces the stopping-rule invariant (equal feature turns).
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

def t_critical_95(df):
    """Two-tailed 95% t-critical values for small degrees of freedom."""
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return t_table.get(df, 1.96)

def stats(vals):
    n = len(vals)
    if n == 0:
        return 0, 0, 0, 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, mean, mean
    variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)
    ci95 = se * t_critical_95(n - 1)
    return mean, sd, mean - ci95, mean + ci95

def main(patterns):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))

    if not paths:
        print("Usage: python3 tools/analyze_variance.py 'results/*_salt*.json'", file=sys.stderr)
        return 1

    runs_by_arm = defaultdict(list)
    for p in sorted(paths):
        try:
            with open(p) as f:
                d = json.load(f)
            arm = d.get("arm", "unknown")
            runs_by_arm[arm].append(d)
        except Exception as e:
            print(f"Skipping {p}: {e}", file=sys.stderr)

    print("=" * 84)
    print(f"VARIANCE ANALYSIS REPORT — {len(runs_by_arm)} ARMS")
    print("=" * 84)

    # Invariant check: stopping rule
    for arm, runs in runs_by_arm.items():
        turns = [r.get("features_completed", len(r.get("turn_ok", []))) for r in runs]
        if len(set(turns)) > 1:
            print(f"WARNING: Unequal turn counts detected for arm {arm}: {turns}")

    metrics = [
        ("cache_hit_rate", lambda r: r.get("cache_hit_rate", 0.0), "{:.3f}"),
        ("cost_units", lambda r: (r.get("metrics") or {}).get("cost_units", 0.0), "{:.0f}"),
        ("cost_saving_ratio", lambda r: (r.get("metrics") or {}).get("cost_saving_ratio", 0.0), "{:.2f}x"),
        ("codes_recall", lambda r: (r.get("metrics") or {}).get("codes_recall", 0.0), "{:.3f}"),
        ("per_turn_success", lambda r: (r.get("metrics") or {}).get("per_turn_success", 0.0), "{:.3f}")
    ]

    for label, fn, fmt in metrics:
        print(f"\nMetric: {label}")
        print(f"{'arm':<14} {'N':<4} {'Mean ± SD':<24} {'95% Conf. Interval':<24}")
        print("-" * 68)
        for arm in sorted(runs_by_arm):
            vals = [fn(r) for r in runs_by_arm[arm]]
            mean, sd, low, high = stats(vals)
            msd = f"{fmt.format(mean)} ± {fmt.format(sd)}"
            ci = f"[{fmt.format(low)}, {fmt.format(high)}]"
            print(f"{arm:<14} {len(vals):<4d} {msd:<24} {ci:<24}")

    print("\n" + "=" * 84)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
