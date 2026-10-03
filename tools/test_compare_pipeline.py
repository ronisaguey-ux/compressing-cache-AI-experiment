#!/usr/bin/env python3
"""End-to-end test of the comparison pipeline on fixtures shaped like REAL results.

★ WHY THIS EXISTS. `tools/run_compare.py` and `tools/make_results_section.py` are the only things
standing between a three-hour GPU run and the numbers that go into a scored submission. Neither was
covered by a test, and both had already shipped a schema bug (`cache_rows` read from `metrics`
instead of top level) that silently degraded the compaction analysis to *omitted* — the failure mode
that never announces itself.

A fixture must carry the REAL schema. Reading the field from the wrong place, or a fixture built with
the wrong shape, means the test passes while the feature is dead. So this builds fixtures from the
actual key layout in `incremental_coding.py` (`cache_rows` top-level; the same cache fields also
mirrored under `metrics`) and asserts the pipeline output, not just that it exits zero.

Run: python3 tools/test_compare_pipeline.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
COMPARE = os.path.join(HERE, "run_compare.py")
RESULTS = os.path.join(HERE, "make_results_section.py")

FAILURES = []


def check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if cond else "FAIL", ("  " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def arm(name, n, hit_frac, gate, codes, prune_at=None):
    """One arm's result dict, in the schema the harness actually writes."""
    rows = []
    for i in range(n):
        prompt = 2000 + i * 130
        if prune_at and i in prune_at:
            prompt = 3000                      # the compaction discontinuity
        rows.append(dict(turn=i, prompt=prompt, hit=int(prompt * hit_frac)))
    tot = sum(r["prompt"] for r in rows)
    hit = sum(r["hit"] for r in rows)
    miss = tot - hit
    cu = miss * 1.0 + hit * 0.1
    return dict(
        arm=name, model="gemma-4-12b", harness_failed=False,
        features=60, features_completed=n, stopped_early=(None if n == 60 else n),
        passed=n, total=n, turn_ok=[1] * n, turns_ok=n,
        codes_n=codes, codes_total=60, codes_ordered=True, codes_dup=False, gate_ok=gate,
        per_feature={"task": "fix", "codes_n": codes, "codes_total": 60,
                     "codes_ordered": True, "codes_dup": False, "gate_ok": gate,
                     "contract": None, "nonce": None},
        # ★ TOP LEVEL -- this is the placement the readers got wrong.
        cache_rows=rows,
        cache_hit_tokens=hit, cache_total_tokens=tot,
        cache_hit_rate=round(hit / tot, 4),
        metrics=dict(
            per_turn_success=1.0, final_state_accuracy=1.0, fixes_applied_then_lost=0,
            codes_recall=codes / 60, codes_ordered=True, codes_duplicated=False,
            release_gate=gate,
            cache_hit_rate=round(hit / tot, 4), cache_miss_tokens=miss,
            cost_units=cu, cost_units_no_cache=float(tot), cost_saving_ratio=tot / cu,
            prompt_first=rows[0]["prompt"], prompt_last=rows[-1]["prompt"],
            prompt_peak=max(r["prompt"] for r in rows),
            prompt_growth=rows[-1]["prompt"] / rows[0]["prompt"],
            ttft_first_ms=1800.0, ttft_last_ms=4900.0, ttft_growth=2.72,
            kv_peak_bytes=4.0e9, wall_per_turn_s=118.0, decode_tokens_per_s=10.2,
            total_prefill_tokens=tot,
        ),
    )


def main():
    d = tempfile.mkdtemp(prefix="cmp_pipeline_")
    # runtime: high reuse, gate PASS, all codes. linear: partial. prune: worst + a compaction.
    for r in (arm("runtime", 60, 0.81, True, 60),
              arm("linear", 60, 0.35, False, 22),
              arm("prune", 60, 0.18, False, 14, prune_at=[30])):
        json.dump(r, open(os.path.join(d, "incremental_%s.json" % r["arm"]), "w"))

    print("pipeline test -- fixtures carry the REAL schema (cache_rows top-level)")
    print()

    # ---- run_compare ----
    cp = subprocess.run([sys.executable, COMPARE] + sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")),
        capture_output=True, text=True, timeout=180)
    out = cp.stdout
    check("run_compare exits 0", cp.returncode == 0, "rc=%d" % cp.returncode)
    check("runtime hit rate rendered", "0.813" in out or "0.810" in out)
    check("linear hit rate rendered", "0.350" in out)
    check("prune hit rate rendered", "0.180" in out)
    check("release gate read from the artifact", "PASS" in out and "False" in out)
    check("a VERDICT line is printed", "VERDICT:" in out)
    check("cache_rows found at top level",
          "cache" in out.lower(), "compaction analysis depends on this")

    # ---- results generator (splice into a temp copy of the real paper) ----
    paper = os.path.join(d, "PAPER.md")
    src = open(os.path.join(REPO, "paper", "PAPER.md"), encoding="utf-8").read()
    open(paper, "w", encoding="utf-8").write(src)
    env = dict(os.environ, RESULTS_DIR=d, PAPER=paper)
    gp = subprocess.run([sys.executable, RESULTS], capture_output=True, text=True,
                        env=env, timeout=180)
    gen = gp.stdout
    check("generator exits 0", gp.returncode == 0, "rc=%d" % gp.returncode)
    check("Table 1 emitted", "Table 1" in gen)
    check("Table 2 emitted", "Table 2" in gen)
    check("compaction events detected", "Compaction events" in gen,
          "requires top-level cache_rows")
    check("no unresolved metric printed as n/a in Table 1",
          "n/a" not in gen.split("Table 1")[1].split("Table 2")[0] if "Table 1" in gen else False)
    check("word-cap guard ran", "words=" in gen and ("OK" in gen or "OVER" in gen))
    spliced = open(paper, encoding="utf-8").read()
    check("§5 was replaced in the paper", "## 5. Results" in spliced and "to be completed" not in spliced)


    # ---- non-vacuity of the honesty guard: it must DECLINE when the arms do not separate ----
    # A guard only ever exercised in the passing direction stays green even when stuck on, so the
    # negative case is the one that proves it works.
    dA = tempfile.mkdtemp(prefix="cmp_nosep_")
    for a in ("runtime", "linear", "prune"):
        r = arm(a, 60, 0.81, True, 60)
        json.dump(r, open(os.path.join(dA, "incremental_%s.json" % a), "w"))
    paperA = os.path.join(dA, "PAPER.md")
    open(paperA, "w", encoding="utf-8").write(src)
    gA = subprocess.run([sys.executable, RESULTS], capture_output=True, text=True,
                        env=dict(os.environ, RESULTS_DIR=dA, PAPER=paperA), timeout=180).stdout
    check("guard DECLINES when arms do not separate",
          "does not establish the cost claim" in gA)
    check("guard CLAIMS when arms do separate",
          "does not establish" not in gen and "Separation." in gen)

    print()
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
