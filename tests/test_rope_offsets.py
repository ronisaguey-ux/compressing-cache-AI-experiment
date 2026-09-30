#!/usr/bin/env python3
"""TEST 1 (Bob 2026-09-30): RoPE positional invariance under eviction.

THE QUESTION. When a middle span is evicted, the survivors could be re-indexed so their
position ids are contiguous with the anchor (400-600 becomes 200-400), or they could keep
their ORIGINAL absolute ids (400-600 stays 400-600). Re-indexing rotates every query and key
by a different RoPE angle, which is a semantic change on top of the eviction itself.

★ PASS METRIC (Bob): retaining original absolute positional coordinates preserves generation
coherence without requiring continuous re-indexing.

★ WHAT THIS MEASURES, and it is a comparison rather than a verdict. Three caches for the SAME
surviving tokens:

    full        the original sequence, nothing evicted       (the reference)
    static      survivors forwarded with ORIGINAL positions
    compact     survivors forwarded with CONTIGUOUS positions

Both are compared against `full` at the surviving indices, so the eviction damage is held
constant and the ONLY difference is the position ids. Whatever gap exists between them is
attributable to re-indexing alone.

★ AN HONEST PREDICTION, so the result cannot be read as a post-hoc story. Finding 23 already
established that preserving positions removes the POSITIONAL error but not the CONTENT error:
the survivors attended to the evicted tokens and that is baked into their weights. So `static`
is expected to beat `compact` clearly, while neither is expected to be exact. If `compact`
were as good as `static`, that would contradict Finding 23 and is worth knowing.

Usage:
    python tests/test_rope_offsets.py
    python tests/test_rope_offsets.py --model Qwen/Qwen2.5-0.5B --evict-from 300 --evict-to 600
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"


def fwd(h, ids, positions):
    dev = h.model.device
    with torch.no_grad():
        return h.model(input_ids=torch.tensor([ids], device=dev), use_cache=True,
                       position_ids=torch.tensor([positions], device=dev),
                       cache_position=torch.tensor(positions, device=dev))


def kv_pairs(out, lo=None, hi=None):
    if lo is None:
        return [(l.keys.clone(), l.values.clone()) for l in out.past_key_values.layers]
    return [(l.keys[:, :, lo:hi, :].clone(), l.values[:, :, lo:hi, :].clone())
            for l in out.past_key_values.layers]


def rel_error(a, b):
    """max|a-b| normalised by the reference's own scale -- comparable across layers."""
    d = float((a - b).abs().max())
    s = float(a.abs().max())
    return d, (d / s if s > 0 else 0.0)


def decode(h, out, qids, start_pos, max_new=20):
    dev = h.model.device
    c = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]
    p = start_pos
    for _ in range(max_new - 1):
        with torch.no_grad():
            o = h.model(input_ids=nxt, past_key_values=c, use_cache=True,
                        cache_position=torch.tensor([p], device=dev))
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        p += 1
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--evict-from", type=int, default=0,
                    help="relative to chunk 1; 0 means auto (~1/3 in)")
    ap.add_argument("--evict-to", type=int, default=0,
                    help="relative to chunk 1; 0 means auto (~1/2 in)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "rope_offsets %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    text = PROMPT_PREFIX + log_at(args.depth) + "\n</build_log><|im_end|>\n"
    ids = h.tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
    c1 = n - c0
    ef = args.evict_from or (c0 + c1 // 3)
    et = args.evict_to or (c0 + c1 // 2)
    if ef >= et or et > n:
        raise SystemExit("bad eviction range %d..%d for n=%d" % (ef, et, n))

    print("=" * 92)
    print("TEST 1 — RoPE POSITIONAL INVARIANCE UNDER EVICTION")
    print("  model=%s  depth=%.2f" % (args.model, args.depth))
    print("=" * 92)
    print("  sequence %d tokens; evicting [%d, %d) (%d tokens, %.0f%% of the evictable region)"
          % (n, ef, et, et - ef, 100 * (et - ef) / c1))
    print()

    # reference: nothing evicted, natural positions
    full = fwd(h, ids, list(range(n)))
    kept_idx = list(range(0, ef)) + list(range(et, n))
    kept_ids = [ids[i] for i in kept_idx]

    # static: survivors keep their ORIGINAL absolute positions
    static = fwd(h, kept_ids, kept_idx)
    # compact: survivors re-indexed contiguously from 0
    compact = fwd(h, kept_ids, list(range(len(kept_ids))))

    # compare at the surviving indices. In `static` the survivor slices are contiguous in the
    # new sequence and line up 1:1; in `compact` likewise -- only the position ids differ.
    ref_pre = kv_pairs(full, 0, ef)
    sur_pre = kv_pairs(static, 0, ef)
    com_pre = kv_pairs(compact, 0, ef)
    ref_post = kv_pairs(full, et, n)
    sur_post = kv_pairs(static, ef, ef + (n - et))
    com_post = kv_pairs(compact, ef, ef + (n - et))

    def agg(refl, otherl):
        worst_d, worst_r = 0.0, 0.0
        for (rk, rv), (ok, ov) in zip(refl, otherl):
            d, r = rel_error(rk, ok)
            if r > worst_r:
                worst_d, worst_r = d, r
            d2, r2 = rel_error(rv, ov)
            if r2 > worst_r:
                worst_d, worst_r = d2, r2
        return worst_d, worst_r

    s_pre_d, s_pre_r = agg(ref_pre, sur_pre)
    c_pre_d, c_pre_r = agg(ref_pre, com_pre)
    s_post_d, s_post_r = agg(ref_post, sur_post)
    c_post_d, c_post_r = agg(ref_post, com_post)

    print("  %-26s %-16s %-16s" % ("region", "STATIC (original)", "COMPACT (re-indexed)"))
    print("  " + "-" * 62)
    print("  %-26s %-16s %-16s" % ("prefix, before eviction",
                                   "%.3e" % s_pre_d, "%.3e" % c_pre_d))
    print("  %-26s %-16s %-16s" % ("suffix, after eviction",
                                   "%.3e" % s_post_d, "%.3e" % c_post_d))
    print("  %-26s %-16s %-16s" % ("  (relative to scale)",
                                   "%.3e" % s_post_r, "%.3e" % c_post_r))
    print()
    ratio = (c_post_d / s_post_d) if s_post_d > 0 else float("inf")
    print("  re-indexing damages the suffix %.1fx more than preserving positions" % ratio)
    print()

    # generation coherence -- Bob's stated pass metric
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    dev = h.model.device
    with torch.no_grad():
        o_static = h.model(input_ids=torch.tensor([qids], device=dev),
                           past_key_values=static.past_key_values, use_cache=True,
                           cache_position=torch.arange(n - (et - ef), n - (et - ef) + len(qids), device=dev))
    ans_static = decode(h, o_static, qids[:0] if False else qids, n - (et - ef) + len(qids))
    with torch.no_grad():
        o_compact = h.model(input_ids=torch.tensor([qids], device=dev),
                            past_key_values=compact.past_key_values, use_cache=True,
                            cache_position=torch.arange(len(kept_ids), len(kept_ids) + len(qids), device=dev))
    ans_compact = decode(h, o_compact, qids, len(kept_ids) + len(qids))
    full_pos = fwd(h, ids + qids, list(range(n + len(qids))))
    ans_full = decode(h, full_pos, [], n + len(qids))

    ok_s = "9AF4" in ans_static.upper().replace(" ", "")
    ok_c = "9AF4" in ans_compact.upper().replace(" ", "")
    ok_f = "9AF4" in ans_full.upper().replace(" ", "")
    print("  %-26s %-8s %r" % ("full sequence (reference)", "PASS" if ok_f else "fail", ans_full[:44]))
    print("  %-26s %-8s %r" % ("STATIC positions", "PASS" if ok_s else "fail", ans_static[:44]))
    print("  %-26s %-8s %r" % ("COMPACT positions", "PASS" if ok_c else "fail", ans_compact[:44]))
    print()
    print("=" * 92)
    if ok_s and not ok_c:
        print("  PASS METRIC MET: original absolute positions preserve coherence where")
        print("  re-indexing does not. No continuous re-indexing required.")
    elif ok_s and ok_c:
        print("  both decode correctly: re-indexing is harmless at THIS eviction size.")
        print("  The KV error gap above still shows the positional cost is real, it just")
        print("  did not change the answer here.")
    elif ok_f and not ok_s:
        print("  eviction itself breaks coherence regardless of positions -- the position")
        print("  question is not the binding constraint at this eviction size.")
    else:
        print("  inconclusive: the reference also failed, so the comparison proves nothing.")
    print("=" * 92)

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                model=args.model, depth=args.depth, n=n, evict_from=ef, evict_to=et,
                static_prefix_err=s_pre_d, compact_prefix_err=c_pre_d,
                static_suffix_err=s_post_d, compact_suffix_err=c_post_d,
                static_suffix_rel=s_post_r, compact_suffix_rel=c_post_r,
                damage_ratio=ratio,
                full_pass=bool(ok_f), static_pass=bool(ok_s), compact_pass=bool(ok_c),
                static_answer=ans_static[:70], compact_answer=ans_compact[:70],
                full_answer=ans_full[:70])) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
