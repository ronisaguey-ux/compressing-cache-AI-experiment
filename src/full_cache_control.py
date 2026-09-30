#!/usr/bin/env python3
"""THE MISSING CONTROL: does the UNCOMPACTED cache also fail at shallow depth?

Every result so far compares one COMPACTION mode against another. Nothing in the sweep
compares compaction against NO compaction at the same needle position. Without that, a
failure at depth 15% cannot be attributed to compaction at all -- it may simply be a
property of the prompt.

This runs the identical prompt and needle placement with a FULL cache, no eviction, no
rotation, and reports the answer. Three arms per depth:

  full      every token kept, no rotation            <- the control that was missing
  identity  every token kept, rotated by R(0)=I      <- proves the rotation pipeline is
                                                        a no-op when nothing is removed
  evict     the normal 60% top-k + re-rotation       <- what the sweep does

Reading it:

  full PASS, identity PASS, evict fail  -> compaction is the cause. The depth failure
                                           is genuinely ours.
  full fail                             -> compaction is INNOCENT. The depth failure is
                                           a property of the prompt/position, and every
                                           Finding 8/9/12 conclusion about "the
                                           compaction recipe" needs restating.
  full PASS, identity fail              -> the rotation path itself is broken even at
                                           zero displacement, which would be a bug.

Usage:
    python src/full_cache_control.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
"""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def run_arm(h, cache_full, ids, arm, c0, c1, rel, keep_frac=0.60):
    tail = list(range(c0 + c1, ids.shape[1]))
    if arm == "full":
        ecache = h.clone_cache(cache_full)
        saved = 0.0
        kept = None
        rank = None
    elif arm == "identity":
        ecache = T.build_arm_cache(h, cache_full,
                                   dict(kind="identity"), c0, c0, c1, tail)
        saved = 0.0
        kept = None
        rank = None
    else:
        scores = T.self_attn_scores(h, cache_full, {}, c0, c1)  # qcap consumed below
        raise SystemExit("evict arm handled separately")

    ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
    ok = "9AF4" in ans.upper().replace(" ", "")
    del ecache
    return dict(arm=arm, kv_saved_pct=round(saved, 2), pass_=bool(ok), answer=ans[:90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "full_cache_control %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0

        t0 = time.time()
        cache, qcap, ids = T.prefill_capture_q(h, text)
        prefill_s = time.time() - t0

        # ---- full: untouched cache ----
        r = run_arm(h, cache, ids, "full", c0, c1, rel)
        r.update(depth=d, arm="full", c1=c1, needle_rel=rel, sec=round(time.time() - t0, 1))
        rows.append(r); print(json.dumps(r), flush=True)

        # ---- identity: keep everything, rotate by zero displacement ----
        r = run_arm(h, cache, ids, "identity", c0, c1, rel)
        r.update(depth=d, arm="identity", c1=c1, needle_rel=rel, sec=round(time.time() - t0, 1))
        rows.append(r); print(json.dumps(r), flush=True)

        # ---- evict: normal top-k + rotation (needs its own q capture) ----
        cache2, qcap2, ids2 = T.prefill_capture_q(h, text)
        scores = T.self_attn_scores(h, cache2, qcap2, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        tail = list(range(c0 + c1, ids2.shape[1]))
        ecache = T.build_arm_cache(h, cache2, dict(kind="evict", keep=idx, rot=True),
                                   c0, c0, c1, tail)
        ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
        ok = "9AF4" in ans.upper().replace(" ", "")
        saved = 100 * (1 - h.cache_len(ecache) / ids2.shape[1])
        del ecache, cache2
        r = dict(depth=d, arm="evict", c1=c1, needle_rel=rel, rank=rank, keep=keep,
                 needle_kept=(None if rel is None else rel in set(idx)),
                 kv_saved_pct=round(saved, 2), pass_=bool(ok), answer=ans[:90],
                 sec=round(time.time() - t0, 1))
        rows.append(r); print(json.dumps(r), flush=True)

        del cache

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print()
    print("%-7s %-11s %-9s %-9s %-10s %s" % (
        "depth", "arm", "saved", "needle", "rank", "verdict"))
    for r in rows:
        print("%-7s %-11s %-9s %-9s %-10s %s" % (
            r["depth"], r["arm"], r["kv_saved_pct"], r.get("needle_kept"),
            r.get("rank"), "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    for d in sorted(set(r["depth"] for r in rows)):
        fr = next(r for r in rows if r["depth"] == d and r["arm"] == "full")
        ir = next(r for r in rows if r["depth"] == d and r["arm"] == "identity")
        er = next(r for r in rows if r["depth"] == d and r["arm"] == "evict")
        tag = ""
        if not fr["pass_"]:
            tag = "  <- UNCOMPACTED ALSO FAILS: compaction is innocent here"
        elif fr["pass_"] and not er["pass_"]:
            tag = "  <- compaction is the cause"
        if fr["pass_"] != ir["pass_"]:
            tag += "  ** identity != full: rotation path is not a no-op **"
        print("  depth %-5s full=%s identity=%s evict=%s%s" % (
            d, "PASS" if fr["pass_"] else "fail",
            "PASS" if ir["pass_"] else "fail",
            "PASS" if er["pass_"] else "fail", tag))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
