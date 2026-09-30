#!/usr/bin/env python3
"""Does the LAYOUT decide whether eviction is safe? The decisive inversion test.

Bob's research agent (Hypothesis 13) proposed the falsifying test directly: invert the
build log so the needle sits early instead of at 75%, and check whether the failure
mode inverts. This runs it.

Finding 23 showed the conflict: tail eviction gives a PERFECT pristine prefix (100% of
retained tokens reusable) and FAILS retrieval, because the needle lives in the discarded
tail. Front eviction gives ZERO reusable prefix and PASSES. Cache efficiency and
retrieval are in direct conflict -- and the conflict is decided by WHERE the critical
content sits, not by the eviction policy.

So: hold the eviction policy fixed, move the needle, watch the verdict invert.

    needle 10%   -> tail eviction should now KEEP the needle and PASS
    needle 75%   -> tail eviction should drop it and FAIL

If that holds, the actionable conclusion is that the layout is the knob: put critical
content early and the tail is free to prune, which is the inverse of the hypothesis
under test (hoisting critical content to the suffix), not a confirmation of it.

Usage:
    python src/needle_layout.py --model Qwen/Qwen2.5-0.5B
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


def build(h, needle_frac):
    text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(needle_frac) + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n", add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    return text, c0, c1


def run(h, cache, ids, kept, qids, N):
    row = ids[0].tolist()
    p = prefix_len_of(kept, N)
    rem = [row[j] for j in kept if j >= p]
    pc = h.clone_cache(cache)
    for L in pc.layers:
        L.keys = L.keys[:, :, :p, :].contiguous()
        L.values = L.values[:, :, :p, :].contiguous()
    t = time.time()
    ans, _ = gen_from_prefix(h, pc, rem, p, qids)
    dt = time.time() - t
    del pc
    ok = "9AF4" in ans.upper().replace(" ", "")
    return dict(prefix_len=p, reuse_pct=round(100 * p / len(kept), 1), pass_=bool(ok),
                sec=round(dt, 2), answer=ans[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--fracs", default="0.10,0.25,0.45,0.65,0.85")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--force-frac", type=float, default=0.15)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--only-policy", default=None,
                    choices=[None, "middle", "tail"],
                    help="run ONE policy instead of all three. On the 7B a full "
                         "3-policy x 5-position sweep is ~2.5 h; the shipped config "
                         "is middle scatter, so that is usually the only row needed.")
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "needle_layout %s" % args.model)
    h = T.Harness(args.model)
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    rows = []

    print("=" * 92)
    print("LAYOUT IS THE KNOB  |  model=%s  depth=%.2f  (the needle moves, the policy does not)"
          % (args.model, args.depth))
    print("=" * 92)
    print()
    print("%-8s %-8s %-7s %-10s | %-22s | %-22s | %s" % (
        "needle", "c1", "needle", "needle", "SUFFIX eviction (tail)", "MIDDLE scatter f=%.2f" % args.force_frac,
        "full (ceiling)"))
    print("%-8s %-8s %-7s %-10s | %-22s | %-22s | %s" % (
        "", "", "index", "in tail?", "prefix  reuse  verdict", "prefix  reuse  verdict", "verdict"))
    print("-" * 92)

    for nf in [float(x) for x in args.fracs.split(",")]:
        text, c0, c1 = build(h, nf)
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0
        keep = max(1, int(c1 * args.keep_frac))
        cache, qcap, ids = T.prefill_capture_q(h, text)
        N = ids.shape[1]
        sc = T.self_attn_scores(h, cache, qcap, c0, c1)

        full = list(range(N))
        tail = list(range(keep))
        force = int(c1 * args.force_frac)
        budget = max(0, keep - c0 - force)
        order = torch.argsort(sc[force:], descending=True)
        sel1 = sorted(set(range(force)) | {force + int(x) for x in order[:budget].tolist()})
        mid = sorted(set(range(c0)) | {c0 + i for i in sel1})

        only = args.only_policy
        r_full = run(h, cache, ids, full, qids, N) if only is None else {"pass_": None, "prefix_len": 0, "reuse_pct": 0.0}
        r_tail = run(h, cache, ids, tail, qids, N) if only in (None, "tail") else {"pass_": None, "prefix_len": 0, "reuse_pct": 0.0}
        r_mid = run(h, cache, ids, mid, qids, N) if only in (None, "middle") else {"pass_": None, "prefix_len": 0, "reuse_pct": 0.0}

        in_tail = (rel is not None and rel + c0 < keep)
        def verdict(r):
            return "-" if r["pass_"] is None else ("PASS" if r["pass_"] else "fail")
        print("%-8s %-8d %-7s %-10s | %-6d %-6s %-8s | %-6d %-6s %-8s | %s" % (
            nf, c1, rel, "YES" if in_tail else "no",
            r_tail["prefix_len"], "%.0f%%" % r_tail["reuse_pct"], verdict(r_tail),
            r_mid["prefix_len"], "%.0f%%" % r_mid["reuse_pct"], verdict(r_mid),
            verdict(r_full)), flush=True)

        rows.append(dict(needle_frac=nf, c1=c1, needle_rel=rel, keep=keep, N=N,
                         needle_in_tail=bool(in_tail),
                         tail_prefix=r_tail["prefix_len"], tail_reuse=r_tail["reuse_pct"],
                         tail_pass=r_tail["pass_"],
                         mid_prefix=r_mid["prefix_len"], mid_reuse=r_mid["reuse_pct"],
                         mid_pass=r_mid["pass_"], full_pass=r_full["pass_"]))
        del cache

    print()
    tpass = sum(1 for r in rows if r["tail_pass"])
    tkill = sum(1 for r in rows if r["needle_in_tail"])
    print("  suffix eviction: %d/%d PASS; the needle was inside the retained tail in %d/%d rows"
          % (tpass, len(rows), tkill, len(rows)))
    print("  middle scatter : %d/%d PASS" % (sum(1 for r in rows if r["mid_pass"]), len(rows)))
    print()
    print("  If the tail verdict tracks needle_in_tail, the LAYOUT decides and the policy")
    print("  merely follows it -- put critical content early and the tail is free to prune.")
    print("  That is the INVERSE of hoisting critical content to the suffix.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
