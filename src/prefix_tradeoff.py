#!/usr/bin/env python3
"""How much re-prefill can a forced prefix buy, and what does it cost in retrieval?

Finding 21 measured the pristine prefix at only 8-12% of the re-prefill, because scattered
top-k evicts early. The lever is the SELECTION pattern, so this tests the trade directly:
force-keep the first `f` of chunk 1, allocate the remaining budget by attention over the
rest, and reuse the resulting prefix.

    f = 0.00   current behaviour (scattered top-k)
    f = 0.15 / 0.30 / 0.45   progressively longer forced prefix

For each f and depth: prefix length, share of the re-prefill reusable, retrieval verdict.
The interesting output is where retrieval breaks, because the budget left for the region
containing the needle shrinks as f grows (keep is fixed at 60% of c1).

Usage:
    python src/prefix_tradeoff.py --model Qwen/Qwen2.5-0.5B
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at
from prefix_cache import prefix_len_of, gen_from_prefix


def select_forced(c1, scores_c1, keep, f):
    """Keep the first f*c1 tokens outright; fill the rest of the budget by attention."""
    force = int(c1 * f)
    forced = list(range(force))
    rest_budget = max(0, keep - force)
    if rest_budget > 0:
        tail_scores = scores_c1[force:]
        order = torch.argsort(tail_scores, descending=True)
        chosen = [force + int(x) for x in order[:rest_budget].tolist()]
    else:
        chosen = []
    return sorted(set(forced) | set(chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--fracs", default="0.0,0.15,0.30,0.45")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "prefix_tradeoff %s" % args.model)
    h = T.Harness(args.model)
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    rows = []

    fracs = [float(x) for x in args.fracs.split(",")]
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        ids_all = h.tok(text, add_special_tokens=False)["input_ids"]
        c1 = len(ids_all) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0

        cache, qcap, ids = T.prefill_capture_q(h, text)
        N = ids.shape[1]
        sc = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        row = ids[0].tolist()

        for f in fracs:
            sel1 = select_forced(c1, sc, keep, f)
            kept = sorted(set(range(c0)) | set(c0 + i for i in sel1))
            p = prefix_len_of(kept, N)
            rem = [j for j in kept if j >= p]
            rem_ids = [row[j] for j in rem]
            kept_needle = None if rel is None else (rel in set(sel1))

            pc = h.clone_cache(cache)
            for L in pc.layers:
                L.keys = L.keys[:, :, :p, :].contiguous()
                L.values = L.values[:, :, :p, :].contiguous()
            t = time.time()
            ans, _ = gen_from_prefix(h, pc, rem_ids, p, qids)
            dt = time.time() - t
            ok = "9AF4" in ans.upper().replace(" ", "")
            del pc

            r = dict(depth=d, force_frac=f, c1=c1, N=N, keep=len(kept), prefix_len=p,
                     re_prefill=len(rem_ids), reusable_pct=round(100 * p / len(kept), 1),
                     needle_kept=kept_needle, pass_=bool(ok), sec=round(dt, 2),
                     answer=ans[:50])
            rows.append(r)
            print(json.dumps(r), flush=True)
        del cache

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print()
    print("%-7s %-7s %-9s %-11s %-12s %-9s %s" % (
        "depth", "force", "prefix", "re-prefill", "reusable", "needle", "verdict"))
    for r in rows:
        print("%-7s %-7s %-9d %-11d %-12s %-9s %s" % (
            r["depth"], r["force_frac"], r["prefix_len"], r["re_prefill"],
            "%.1f%%" % r["reusable_pct"], r["needle_kept"],
            "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 74)
    print("THE TRADE: re-prefill reduction vs retrieval")
    print("=" * 74)
    for f in fracs:
        sub = [r for r in rows if r["force_frac"] == f]
        p = sum(1 for r in sub if r["pass_"])
        reuse = sum(r["reusable_pct"] for r in sub) / len(sub)
        reprefill = sum(r["re_prefill"] for r in sub) / len(sub)
        pat = " ".join("P" if r["pass_"] else "F" for r in sub)
        print("  force=%-5s reuse %5.1f%% of re-prefill | avg re-prefill %6.0f tok | %d/%d  %s" % (
            f, reuse, reprefill, p, len(sub), pat))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
