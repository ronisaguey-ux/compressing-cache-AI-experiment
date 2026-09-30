#!/usr/bin/env python3
"""Needle-preservation rate per sifter, across depths and budgets.

WHY THIS METRIC. Finding 27 showed the answer survives any selection that retains the needle
and its local context, and two sifters with only 19% overlap both succeeded. So the binding
constraint is not "does the sifter select well" but "does it drop the evidence". That makes
needle preservation the metric that actually predicts success, and it is far cheaper to
measure than end-to-end retrieval -- no generation, no consumer model, just the selected set.

Sifters compared:

  attention-05b    the 0.5B's own self-attention scores over the evictable region
  random           a uniform random selection at the same budget (the null)
  first-k          keep the leading run (what a naive prefix-cache policy does)
  last-k           keep the trailing run (what a tail-pruning policy does)
  laya-<variant>   Laya's per-line score, if a label file is present

Each is evaluated at several budgets and depths. `needle_preserved` is the headline; the
selection rate is reported alongside because a sifter that keeps everything trivially
preserves the needle, and that difference has to be visible.

Usage:
    python src/needle_preservation.py --model Qwen/Qwen2.5-0.5B
    python src/needle_preservation.py --model Qwen/Qwen2.5-0.5B --laya-scores data/laya_line_scores.json
"""
import sys, os, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--budgets", default="0.30,0.45,0.60,0.75",
                    help="fraction of the evictable region kept")
    ap.add_argument("--seeds", type=int, default=5, help="random replicates")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "needle_preservation")
    h = T.Harness(args.model)
    depths = [float(x) for x in args.depths.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]

    rows = []
    print("=" * 94)
    print("NEEDLE PRESERVATION PER SIFTER  |  model=%s" % args.model)
    print("=" * 94)

    for d in depths:
        text = PROMPT_PREFIX + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0
        cache, qcap, ids = T.prefill_capture_q(h, text)
        N = ids.shape[1]
        sc = T.self_attn_scores(h, cache, qcap, c0, c1)
        del cache

        print()
        print("depth %.2f  c1=%d  needle at chunk-1 index %s (%.0f%% of the region)" % (
            d, c1, rel, 100 * rel / c1 if rel is not None else -1))

        for keep_frac in budgets:
            keep = max(1, int(c1 * keep_frac))
            order = torch.argsort(sc, descending=True)
            picks = {
                "attention-05b": sorted(int(x) for x in order[:keep].tolist()),
                "first-k": list(range(keep)),
                "last-k": list(range(c1 - keep, c1)),
            }
            for s in range(args.seeds):
                rnd = random.Random(1000 + s)
                picks["random#%d" % s] = rnd.sample(range(c1), keep)

            for name, pick in picks.items():
                preserved = (rel is not None and rel in set(pick))
                rows.append(dict(depth=d, keep_frac=keep_frac, sifter=name,
                                 keep=keep, c1=c1, needle_rel=rel,
                                 needle_preserved=bool(preserved)))
            if keep_frac == budgets[0]:
                pass

        # summarise this depth
        print("   %-14s %s" % ("sifter", "  ".join("k=%d%%" % (100 * b) for b in budgets)))
        names = ["attention-05b", "first-k", "last-k"] + ["random#0"]
        for nm in names:
            cells = []
            for b in budgets:
                sub = [r for r in rows if r["depth"] == d and r["keep_frac"] == b
                       and (r["sifter"] == nm or (nm == "random#0" and r["sifter"].startswith("random")))]
                if nm == "random#0":
                    sub = [r for r in rows if r["depth"] == d and r["keep_frac"] == b
                           and r["sifter"].startswith("random")]
                    p = sum(r["needle_preserved"] for r in sub) / max(1, len(sub))
                    cells.append("%s" % ("YES" if p == 1 else "%d/%d" % (
                        sum(r["needle_preserved"] for r in sub), len(sub))))
                else:
                    cells.append("YES" if (sub and sub[0]["needle_preserved"]) else "no ")
            label = "random" if nm == "random#0" else nm
            print("   %-14s %s" % (label, "   ".join(cells)))

    # aggregate
    print()
    print("=" * 94)
    print("AGGREGATE over %d depths x %d budgets" % (len(depths), len(budgets)))
    print("=" * 94)
    for nm in ["attention-05b", "first-k", "last-k"]:
        sub = [r for r in rows if r["sifter"] == nm]
        p = sum(r["needle_preserved"] for r in sub)
        print("  %-16s %2d/%2d preserved (%.0f%%)" % (nm, p, len(sub), 100 * p / len(sub)))
    sub = [r for r in rows if r["sifter"].startswith("random")]
    p = sum(r["needle_preserved"] for r in sub)
    print("  %-16s %2d/%2d preserved (%.0f%%)  [%d seeds]" % (
        "random", p, len(sub), 100 * p / len(sub), args.seeds))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
