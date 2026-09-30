#!/usr/bin/env python3
"""Block-causal (document-masking) prefill: does block-diagonal attention make
eviction LOSSLESS?

WHY THIS IS MORE THAN A SPIKE. Findings 23-24 established that validity is a PREFIX
property: if you evict a token in the middle, every token after it is invalidated,
because they all attended to it. That is why tail eviction gives a perfect reusable
prefix and front eviction gives none, and why cache surgery (rotation) corrupted the
keys in Finding 18.

Block-diagonal attention dissolves that problem by construction. If token j may only
attend to tokens inside its own block, then block 5 never saw block 3 -- so evicting
block 3 does not touch block 5's keys or values at all. Eviction becomes lossless
rather than approximate.

THIS SCRIPT TESTS THREE CLAIMS, in order of strength:

  1. SURVIVAL.  Prefill with block-causal masking, then prefill again with one whole
     block removed, giving the survivors their ORIGINAL position ids. Compare the
     survivors' K/V to the full run. Expect torch.equal -- bit-identical.

     Original positions are required: RoPE is a function of absolute position, so
     without them the survivors shift and their keys change even though they attended
     nowhere new. That is why the position argument is not an optimisation here, it is
     load bearing.

  2. RETRIEVAL UNDER EVICTION. The needle sits inside one block. Evict a DIFFERENT
     block and check the answer still comes back -- where the same eviction under full
     causal attention breaks it.

  3. THE COST. A block that needs something from another block can no longer see it.
     That is a real capability loss, not a free win, so it is measured rather than
     asserted: the same needle question is asked with the needle's own block evicted,
     and it should fail. If it does not fail, the test is not exercising the mask.

Claim 3 is the control. Without it a passing test could mean the mask was never applied.

Usage:
    python src/block_causal.py --model Qwen/Qwen2.5-0.5B
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

NEG = float("-inf")


def split_blocks(ids, n_blocks):
    """Contiguous blocks of roughly equal token length."""
    n = len(ids)
    bounds = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_blocks)]


def block_causal_mask(blocks, dtype=torch.float32):
    """4D additive mask: token in block b attends only within block b, causally.

    Shape (1, 1, N, N). Position (i, j) is 0.0 when j may be attended to by i.
    """
    n = blocks[-1][1]
    m = torch.full((1, 1, n, n), NEG, dtype=dtype)
    for (lo, hi) in blocks:
        for i in range(lo, hi):
            m[0, 0, i, lo:i + 1] = 0.0
    return m


def block_diag_mask_for(kept_blocks, n_total, dtype=torch.float32):
    """Mask over a CONCATENATED sequence of kept blocks, block-diagonal.

    The new sequence is the kept blocks laid end to end; each token may attend only
    to tokens in its own (new) block, causally within it.
    """
    n_new = sum(hi - lo for lo, hi in kept_blocks)
    m = torch.full((1, 1, n_new, n_new), NEG, dtype=dtype)
    cur = 0
    for (lo, hi) in kept_blocks:
        L = hi - lo
        for i in range(L):
            m[0, 0, cur + i, cur:cur + i + 1] = 0.0
        cur += L
    return m


def causal_mask(n, dtype=torch.float32):
    m = torch.full((1, 1, n, n), NEG, dtype=dtype)
    for i in range(n):
        m[0, 0, i, :i + 1] = 0.0
    return m


def prefill(h, ids, mask, position_ids=None):
    dev = h.model.device
    kw = {}
    if position_ids is not None:
        kw["position_ids"] = torch.tensor([position_ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = h.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
                      attention_mask=mask.to(dev), use_cache=True, **kw)
    return out


def gen(h, out, qids, max_new=24, position_ids=None):
    dev = h.model.device
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen_ids = [nxt]
    n_past = out.logits.shape[1]
    start = (position_ids[-1] + 1) if position_ids else n_past
    for k in range(max_new - 1):
        pos = torch.tensor([[start + k]], dtype=torch.long, device=dev)
        with torch.no_grad():
            out2 = h.model(input_ids=nxt, past_key_values=cache, use_cache=True,
                           position_ids=pos, attention_mask=torch.ones((1, 1, 1, n_past + 1 + k), device=dev))
        cache = out2.past_key_values
        nxt = out2.logits[:, -1, :].argmax(-1, keepdim=True)
        gen_ids.append(nxt)
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(gen_ids, dim=1)[0], skip_special_tokens=True)


def kv_of(out):
    return [(l.keys.clone(), l.values.clone()) for l in out.past_key_values.layers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--evict", type=int, default=3, help="which block to remove (0-based)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "block_causal %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(args.depth) + "\n</build_log><|im_end|>\n"
    ids = h.tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    blocks = split_blocks(ids, args.blocks)
    na = T.locate_needle(h.tok, text)
    needle_block = None
    if na is not None:
        needle_block = next(b for b, (lo, hi) in enumerate(blocks) if lo <= na < hi)

    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    victim = args.evict
    if needle_block is not None and victim == needle_block:
        victim = (victim + 1) % args.blocks

    print("=" * 84)
    print("BLOCK-CAUSAL PREFILL  |  model=%s  depth=%.2f" % (args.model, args.depth))
    print("=" * 84)
    print("  tokens=%d  blocks=%d  block sizes=%s" % (
        n, args.blocks, [hi - lo for lo, hi in blocks]))
    print("  needle at token %s -> block %s ; evicting block %d" % (na, needle_block, victim))
    print()

    # ---------------------------------------------------------------- reference
    t = time.time()
    full_causal = prefill(h, ids, causal_mask(n))
    full_kv = kv_of(full_causal)
    t_causal = time.time() - t
    bc_full = prefill(h, ids, block_causal_mask(blocks))
    bc_kv = kv_of(bc_full)
    print("  prefill: full-causal %.1fs   block-causal %.1fs" % (
        t_causal, time.time() - t - t_causal))

    # ------------------------------------------------- 1. survival, bit-identical
    print()
    print("1. SURVIVAL — evict one whole block, keep original positions, compare tensors")
    kept = [b for b in blocks if b != blocks[victim]]
    kept_ids, kept_pos = [], []
    for (lo, hi) in kept:
        kept_ids.extend(ids[lo:hi])
        kept_pos.extend(range(lo, hi))
    bc_ev = prefill(h, kept_ids, block_diag_mask_for(kept, n), position_ids=kept_pos)
    ev_kv = kv_of(bc_ev)

    # map each surviving block to its slice in the evicted run
    exact = True
    detail = []
    cur = 0
    for b, (lo, hi) in enumerate(blocks):
        L = hi - lo
        if b == victim:
            continue
        ok_k = all(torch.equal(bc_kv[li][0][:, :, lo:hi, :], ev_kv[li][0][:, :, cur:cur + L, :])
                   for li in range(len(bc_kv)))
        ok_v = all(torch.equal(bc_kv[li][1][:, :, lo:hi, :], ev_kv[li][1][:, :, cur:cur + L, :])
                   for li in range(len(bc_kv)))
        exact &= (ok_k and ok_v)
        detail.append((b, ok_k, ok_v))
        cur += L
    for b, ok_k, ok_v in detail:
        print("     block %d: K bit-equal %s   V bit-equal %s" % (b, ok_k, ok_v))
    print("     => %s" % ("ALL SURVIVING BLOCKS BIT-IDENTICAL (torch.equal)" if exact
                          else "NOT bit-identical -- block-diagonality did not isolate them"))

    # --------------------------------------------- 2. retrieval under eviction
    print()
    print("2. RETRIEVAL — same eviction, both maskings")
    ans_bc, _ = gen(h, bc_ev, qids, position_ids=kept_pos)
    ok_bc = "9AF4" in ans_bc.upper().replace(" ", "")
    print("     block-causal, block %d evicted : %s   %r" % (
        victim, "PASS" if ok_bc else "fail", ans_bc[:56]))

    # the same eviction under FULL causal attention, for contrast
    fc_ev = prefill(h, kept_ids, causal_mask(len(kept_ids)), position_ids=kept_pos)
    ans_fc, _ = gen(h, fc_ev, qids, position_ids=kept_pos)
    ok_fc = "9AF4" in ans_fc.upper().replace(" ", "")
    print("     full-causal,  block %d evicted : %s   %r" % (
        victim, "PASS" if ok_fc else "fail", ans_fc[:56]))

    # --------------------------------------------- 3. the control: evict the needle
    print()
    print("3. CONTROL — evict the needle's OWN block; this MUST fail, or the mask is inert")
    if needle_block is None:
        print("     needle not located; control skipped")
    else:
        kept2 = [b for b in blocks if b != blocks[needle_block]]
        k2_ids, k2_pos = [], []
        for (lo, hi) in kept2:
            k2_ids.extend(ids[lo:hi])
            k2_pos.extend(range(lo, hi))
        bc_nd = prefill(h, k2_ids, block_diag_mask_for(kept2, n), position_ids=k2_pos)
        ans_nd, _ = gen(h, bc_nd, qids, position_ids=k2_pos)
        ok_nd = "9AF4" in ans_nd.upper().replace(" ", "")
        print("     block-causal, needle block %d evicted : %s   %r" % (
            needle_block, "PASS" if ok_nd else "fail", ans_nd[:56]))
        print("     => %s" % ("mask is doing work: removing the evidence loses the answer"
                              if not ok_nd else
                              "WARNING: the answer survived without the needle -- the mask "
                              "may not be applied, or the answer is guessable"))

    # --------------------------------------------- ceiling
    ans_full, _ = gen(h, bc_full, qids)
    ok_full = "9AF4" in ans_full.upper().replace(" ", "")
    print()
    print("   ceiling (block-causal, nothing evicted): %s   %r" % (
        "PASS" if ok_full else "fail", ans_full[:56]))

    print()
    print("=" * 84)
    print("  bit-identical survivors : %s" % exact)
    print("  block-causal evict %-2d   : %s" % (victim, "PASS" if ok_bc else "fail"))
    print("  full-causal  evict %-2d   : %s" % (victim, "PASS" if ok_fc else "fail"))
    print("  ceiling                 : %s" % ("PASS" if ok_full else "fail"))
    print("=" * 84)

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                model=args.model, depth=args.depth, blocks=args.blocks, N=n,
                block_sizes=[hi - lo for lo, hi in blocks],
                needle_token=na, needle_block=needle_block, victim_block=victim,
                bit_identical=bool(exact),
                block_causal_evict_pass=bool(ok_bc), full_causal_evict_pass=bool(ok_fc),
                ceiling_pass=bool(ok_full),
                block_causal_answer=ans_bc[:70], full_causal_answer=ans_fc[:70],
                ceiling_answer=ans_full[:70],
            )) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
