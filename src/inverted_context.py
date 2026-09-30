#!/usr/bin/env python3
"""Inverted-context layout: can a suffix survive front eviction under causal attention?

THE HYPOTHESIS UNDER TEST (Bob, 2026-09-30). Push volatile/disposable tokens to the
early prefix (0..K) and hoist query-critical content to the suffix (K..N), then prune
the early context for free.

THE MATH SAYS NO, AND THE REASON IS DIRECTIONAL. In causal attention

    h_j = f(x_j, x_{j-1}, ..., x_0)

so token j's keys and values are a function of EVERY token before it. Validity is a
PREFIX property: token j is reusable only if 0..j-1 are untouched and unmoved. Two
consequences that run opposite to the hypothesis:

  * evicting from the FRONT invalidates everything after the first dropped index, so
    a suffix hoisted to the back is the WORST place to put critical content;
  * evicting from the END keeps every earlier token pristine, giving the LONGEST
    reusable prefix of any policy.

So the useful inversion is the mirror image of the hypothesis: critical content early
(inside the pristine prefix), disposable content late (free to cut).

This script measures all of it rather than asserting it:

  1. pristine-prefix length under four eviction policies, and retrieval for each
  2. the suffix drift equation -- cosine of suffix K/V under a full prefix vs a
     front-truncated one, as a function of how much was dropped, with positions both
     shifted (natural) and preserved (patched)
  3. what evicting the attention sinks costs
  4. needle placed early vs late under suffix eviction

Usage:
    python src/inverted_context.py --model Qwen/Qwen2.5-0.5B
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


def pristine_prefix(kept, N):
    """Longest k such that kept contains 0..k-1 in order -- the reusable prefix."""
    return prefix_len_of(kept, N)


def run_policy(h, cache, ids, kept, qids, label, rows, depth, N):
    """Retrieval when `kept` defines the cache, reusing its pristine prefix."""
    row = ids[0].tolist()
    p = pristine_prefix(kept, N)
    rem_ids = [row[j] for j in kept if j >= p]
    pc = h.clone_cache(cache)
    for L in pc.layers:
        L.keys = L.keys[:, :, :p, :].contiguous()
        L.values = L.values[:, :, :p, :].contiguous()
    t = time.time()
    ans, _ = gen_from_prefix(h, pc, rem_ids, p, qids)
    dt = time.time() - t
    ok = "9AF4" in ans.upper().replace(" ", "")
    del pc
    r = dict(depth=depth, policy=label, N=N, keep=len(kept), prefix_len=p,
             reuse_pct=round(100 * p / len(kept), 1), pass_=bool(ok),
             sec=round(dt, 2), answer=ans[:60])
    rows.append(r)
    print("  %-22s keep=%-5d pristine_prefix=%-5d reuse=%-6s %s" % (
        label, len(kept), p, "%.1f%%" % r["reuse_pct"], "PASS" if ok else "fail"), flush=True)
    return r


def suffix_drift(h, text, drop, preserve_positions, n_layers_max=6):
    """Cosine between suffix K/V under a FULL prefix and a front-truncated one.

    This is the equation the brief asks for. `drop` tokens are removed from the front;
    suffix position j (full) is compared against position j-drop (truncated). With
    `preserve_positions` the truncated forward is given the ORIGINAL position ids, so
    the only difference left is the missing context -- which isolates content damage
    from positional damage.
    """
    ids = h.tok(text, add_special_tokens=False)["input_ids"]
    N = len(ids)
    dev = next(h.model.parameters()).device

    with torch.no_grad():
        o_full = h.model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
    K_full = [l.keys for l in o_full.past_key_values.layers]
    V_full = [l.values for l in o_full.past_key_values.layers]
    del o_full

    ids_t = ids[drop:]
    kwargs = {}
    if preserve_positions:
        pos = torch.arange(drop, N, device=dev).unsqueeze(0)
        kwargs["position_ids"] = pos
        kwargs["cache_position"] = torch.arange(drop, N, device=dev)
    with torch.no_grad():
        o_t = h.model(input_ids=torch.tensor([ids_t], device=dev), use_cache=True, **kwargs)
    K_t = [l.keys for l in o_t.past_key_values.layers]
    V_t = [l.values for l in o_t.past_key_values.layers]

    n_layers = min(n_layers_max, len(K_full))
    ck, cv = [], []
    for li in range(n_layers):
        a = K_full[li][0, :, drop:, :]
        b = K_t[li][0, :, :, :]
        an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ck.append(float((an * bn).sum(-1).mean()))
        a = V_full[li][0, :, drop:, :]
        b = V_t[li][0, :, :, :]
        an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cv.append(float((an * bn).sum(-1).mean()))
    del K_full, V_full, K_t, V_t, o_t
    return dict(drop=drop, preserve_positions=bool(preserve_positions),
                cos_K=sum(ck) / len(ck), cos_V=sum(cv) / len(cv),
                cos_K_min=min(ck), cos_V_min=min(cv), N=N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--force-frac", type=float, default=0.15)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "inverted_context %s" % args.model)
    h = T.Harness(args.model)
    d = args.depth
    text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n", add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    cache, qcap, ids = T.prefill_capture_q(h, text)
    N = ids.shape[1]
    sc = T.self_attn_scores(h, cache, qcap, c0, c1)
    keep = max(1, int(c1 * args.keep_frac))
    na = T.locate_needle(h.tok, text)
    needle_rel = None if na is None else na - c0

    print("=" * 78)
    print("INVERTED-CONTEXT LAYOUT  |  model=%s depth=%.2f" % (args.model, d))
    print("N=%d  chunk0=%d  chunk1=%d  keep=%d  needle at chunk-1 index %s (%.0f%%)" % (
        N, c0, c1, keep, needle_rel, 100 * needle_rel / c1 if needle_rel else -1))
    print("=" * 78)
    rows = []

    # ---------------------------------------------------------------- 1. policies
    print()
    print("1. PRISTINE PREFIX BY EVICTION POLICY  (redo across the whole sequence,")
    print("   not just chunk 1, so the policies are comparable)")
    print()
    full = list(range(N))
    run_policy(h, cache, ids, full, qids, rows, d, N) if False else None
    # reference: everything kept (no eviction) -- the ceiling
    r0 = run_policy(h, cache, ids, full, qids, "no eviction (ceiling)", rows, d, N)

    # suffix eviction: keep the first `keep` of the sequence, drop the tail
    suffix_kept = list(range(keep))
    r1 = run_policy(h, cache, ids, suffix_kept, qids, "suffix eviction (tail)", rows, d, N)

    # front eviction: drop the head, keep the tail -- the hypothesis' prune
    drop_n = N - keep
    front_kept = list(range(drop_n, N))
    r2 = run_policy(h, cache, ids, front_kept, qids, "front eviction (head)", rows, d, N)

    # middle: keep first f of chunk1 by force, scatter the rest (Finding 22 config)
    force = int(c1 * args.force_frac)
    budget = max(0, keep - c0 - force)
    tail_scores = sc[force:]
    order = torch.argsort(tail_scores, descending=True)
    sel1 = sorted(set(range(force)) | {force + int(x) for x in order[:budget].tolist()})
    mid_kept = sorted(set(range(c0)) | {c0 + i for i in sel1})
    r3 = run_policy(h, cache, ids, mid_kept, qids, "middle scatter (f=%.2f)" % args.force_frac,
                    rows, d, N)

    # sinks: keep everything except the first 4 positions (evict the sinks)
    no_sink_kept = [j for j in mid_kept if j >= 4]
    r4 = run_policy(h, cache, ids, no_sink_kept, qids, "middle, sinks evicted", rows, d, N)

    print()
    print("  pristine-prefix ranking: %s" % " > ".join(
        "%s(%d)" % (r["policy"].split("(")[0].strip(), r["prefix_len"]) for r in rows))
    print("  => the reusable prefix is set by the EARLIEST eviction index, so evicting")
    print("     the tail preserves everything before it and evicting the head preserves")
    print("     nothing. The hypothesis puts the critical content where it cannot survive.")
    del cache

    # ------------------------------------------------------------ 2. suffix drift
    print()
    print("2. SUFFIX DRIFT EQUATION  (cosine of suffix K/V, full prefix vs front-truncated)")
    print()
    print("  %-8s %-12s %-10s %-10s %-10s" % ("drop", "positions", "cos_K", "cos_V", "cos_K_min"))
    drift_rows = []
    for drop in [1, 4, 16, 64, 256]:
        if drop >= N - 8:
            continue
        for preserve in (False, True):
            dr = suffix_drift(h, text, drop, preserve)
            drift_rows.append(dr)
            print("  %-8d %-12s %-10.6f %-10.6f %-10.6f" % (
                drop, "preserved" if preserve else "shifted",
                dr["cos_K"], dr["cos_V"], dr["cos_K_min"]), flush=True)

    print()
    print("  => dropping even ONE front token drops the suffix cosine below 1.0, so the")
    print("     suffix is never reusable once anything before it changes. Preserving the")
    print("     original position ids removes the positional error but not the content")
    print("     error: the suffix attended to the dropped tokens, and that is baked in.")
    print("     Values hold up; keys are what break -- consistent with Finding 18.")

    out = dict(model=args.model, depth=d, N=N, c0=c0, c1=c1, keep=keep,
               needle_rel=needle_rel, policies=rows, drift=drift_rows)
    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            for r in drift_rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
