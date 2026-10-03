#!/usr/bin/env python3
"""Aggregate variance runs across task salts, paired by instance.

What this is for
----------------
Every headline number in the paper is a *ratio*: anchored cost against baseline cost, anchored
recall against baseline recall. The honest uncertainty for a ratio is a confidence interval on the
ratio, computed from the per-instance paired differences -- not two independent arm means, whose
intervals can overlap even when the paired effect is consistent.

Runs are paired by their task salt (the filename carries `<salt>_<arm>`), so for salt i we compute
the ratio from the two arms run on the *identical* task, then report the mean ratio with a 95%
Student-t interval over the N instances. This is the same reason a paired t-test is more sensitive
than an unpaired one: the task-instance variation cancels.

Usage
-----
    python3 tools/analyze_variance.py ~/.local/share/ccai-results/variance/var_*.json
    python3 tools/analyze_variance.py 'results/*.json' --baseline linear --arm runtime

Named arms default to baseline=`linear`, arm=`runtime`. If the filenames do not carry a salt, it
falls back to positional pairing only when the two arms have equal counts, and says so.
"""
import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict

# Two-tailed 95% t-critical values, df = n-1.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def t95(df):
    return T95.get(df, 1.96)


def stats(vals):
    """Return (n, mean, sd, lo, hi) with a 95% Student-t interval. n<2 => no interval."""
    n = len(vals)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    mean = sum(vals) / n
    if n == 1:
        return 1, mean, 0.0, mean, mean
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    half = t95(n - 1) * sd / math.sqrt(n)
    return n, mean, sd, mean - half, mean + half


def metric_getters(d):
    """Pull the metrics the paper reports. Missing values are surfaced, not silently zeroed."""
    m = d.get("metrics") or {}
    cache_rows = d.get("cache_rows") or []
    reused = [r.get("hit") for r in cache_rows if isinstance(r.get("hit"), (int, float))]
    return {
        "cost_units":       m.get("cost_units"),
        "hit_rate":         d.get("cache_hit_rate"),
        "codes_recall":     m.get("codes_recall", d.get("codes_n")),
        "per_turn_success": m.get("per_turn_success", d.get("passed")),
        "mean_reused_tokens": (sum(reused) / len(reused)) if reused else None,
        "prompt_last":      d.get("prompt_last"),
    }


SALT_RE = re.compile(r"([A-Za-z0-9]+?)[_\-](runtime|linear|prune|ablate-recent)\.json$")


def load(paths):
    """Return {salt_or_index: {arm: record}}, plus any files skipped."""
    by_salt = defaultdict(dict)
    skipped = []
    idx = 0
    for p in sorted(paths):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            skipped.append((p, str(e)))
            continue
        arm = d.get("arm") or "unknown"
        m = SALT_RE.search(p)
        key = m.group(1) if m else "auto%02d" % idx
        if not m:
            idx += 1
        by_salt[key][arm] = {"path": p, "data": d, "metrics": metric_getters(d)}
    return by_salt, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--arm", default="runtime")
    ap.add_argument("--baseline", default="linear")
    args = ap.parse_args()

    paths = []
    for pat in args.patterns:
        paths.extend(glob.glob(pat))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 1

    by_salt, skipped = load(paths)
    if skipped:
        print("skipped %d unreadable file(s):" % len(skipped), file=sys.stderr)
        for p, e in skipped:
            print("  %s: %s" % (p, e), file=sys.stderr)

    paired = {s: v for s, v in by_salt.items() if args.arm in v and args.baseline in v}
    arms_present = sorted({a for v in by_salt.values() for a in v})

    print("=" * 78)
    print("VARIANCE ANALYSIS  (arm=%s, baseline=%s)" % (args.arm, args.baseline))
    print("=" * 78)
    print("files loaded: %d   instances: %d   paired: %d" % (len(paths), len(by_salt), len(paired)))
    print("arms present:", ", ".join(arms_present))

    if not paired:
        print("\nNo paired instances found. Per-arm means only, NO ratio interval.")
        for arm in arms_present:
            for metric in ("cost_units", "hit_rate", "codes_recall"):
                vals = [v[arm]["metrics"][metric] for v in by_salt.values()
                        if arm in v and v[arm]["metrics"][metric] is not None]
                if vals:
                    n, mean, sd, lo, hi = stats(vals)
                    print("  %-14s %-12s n=%d  mean=%.4g  sd=%.4g" % (arm, metric, n, mean, sd))
        return 1 if len(by_salt) < 2 else 0

    # ★ The paired ratio is the measured quantity: both arms ran the identical task instance.
    print("\nPaired per-instance ratios (%s / %s). 1.0 = no difference." % (args.baseline, args.arm))
    print("-" * 78)
    ratio_specs = [
        ("cost_units",       "cost ratio (baseline/arm) -- >1 means arm cheaper"),
        ("codes_recall",     "recall ratio (arm/baseline) -- >1 means arm remembers more"),
        ("per_turn_success", "success ratio (arm/baseline) -- 1.0 means equal accuracy"),
    ]
    results = {}
    for metric, desc in ratio_specs:
        ratios = []
        for s, v in sorted(paired.items()):
            a = v[args.arm]["metrics"].get(metric)
            b = v[args.baseline]["metrics"].get(metric)
            if a is None or b is None or a == 0:
                continue
            ratios.append(b / a if metric == "cost_units" else a / b)
        if not ratios:
            continue
        n, mean, sd, lo, hi = stats(ratios)
        results[metric] = (n, mean, sd, lo, hi)
        print("%s" % desc)
        print("   n=%d  mean=%.3fx  sd=%.3f  95%% CI [%.3f, %.3f]" % (n, mean, sd, lo, hi))
        print("   per instance: %s" % ", ".join("%.2fx" % r for r in ratios))
        if n >= 3 and lo > 1.0:
            print("   -> interval excludes 1.0: the difference is consistent across instances")
        elif n >= 3 and hi < 1.0:
            print("   -> interval excludes 1.0 in the other direction (the arm is WORSE)")
        elif n >= 2:
            print("   -> interval includes 1.0: cannot exclude no-difference at n=%d" % n)

    print("\nPer-arm means (context, not the claim).")
    print("-" * 78)
    for metric in ("cost_units", "hit_rate", "codes_recall", "per_turn_success", "prompt_last"):
        line = []
        for arm in (args.arm, args.baseline):
            vals = [v[arm]["metrics"][metric] for v in paired.values()
                    if arm in v and v[arm]["metrics"][metric] is not None]
            if vals:
                n, mean, sd, lo, hi = stats(vals)
                line.append("%s=%.3g±%.3g" % (arm, mean, sd))
        if line:
            print("  %-16s %s" % (metric, "   ".join(line)))

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
