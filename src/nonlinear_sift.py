#!/usr/bin/env python3
"""Sifter comparison in a NON-LINEAR context arrangement.

THE READING. Until now context has been one ordered sequence: a linear arrangement where a
selection is a subset of positions, and eviction invalidates everything downstream (Findings
23-24). Here the context is a TABLE of independent blocks (Findings 28-30) -- non-linear in
the sense that no block depends on another's presence or order. Eviction becomes a set
operation: drop the row.

That changes what a sifter has to do. It is no longer "which tokens survive" but **"which
blocks are worth a row"**, and the failure mode is different: the question is not whether a
token's rank fits the budget but whether the block holding the evidence gets kept.

Sifters compared, all scored at BLOCK granularity on the same table:

  attention-05b   sum of the 0.5B's self-attention over each block's tokens
  line            same, but pooled per line then summed (coherence-aware, per Finding 33)
  first-k         keep the leading blocks (a naive prefix policy)
  last-k          keep the trailing blocks
  random          the null

Each is asked to keep a budget of blocks, and two things are measured:

  needle_block_kept   is the block containing the evidence still in the table?
  retrieval           does the query still answer, with the query as its own tier (Tier 3)

Block-level selection is coarse -- a block is ~1/K of the context, so the budget granularity
is much wider than tokens -- which is exactly why this is worth measuring separately rather
than assuming the token-level results carry over.

Usage:
    python src/nonlinear_sift.py --model Qwen/Qwen2.5-0.5B
"""
import sys, os, json, argparse, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"


def build_table(h, global_text, chunks, block_causal=True):
    """Global as Tier 1, each chunk as its own Tier-2 block. Returns (g_ids, g_cache, blocks, G)."""
    g_ids = h.tok(global_text, add_special_tokens=False)["input_ids"]
    G = len(g_ids)
    dev = h.model.device
    with torch.no_grad():
        o = h.model(input_ids=torch.tensor([g_ids], device=dev), use_cache=True,
                    position_ids=torch.arange(G, device=dev).unsqueeze(0))
    from transformers import DynamicCache
    g_cache = DynamicCache(ddp_cache_data=[
        (l.keys.clone(), l.values.clone()) for l in o.past_key_values.layers])

    from tiered_cache import slice_cache
    blocks, cur = [], G
    for ch in chunks:
        c_ids = h.tok(ch, add_special_tokens=False)["input_ids"]
        if not c_ids:
            blocks.append(None); continue
        p = list(range(cur, cur + len(c_ids)))
        o = h.model(input_ids=torch.tensor([c_ids], device=dev),
                    past_key_values=slice_cache(g_cache, 0, G), use_cache=True,
                    position_ids=torch.tensor([p], device=dev),
                    cache_position=torch.tensor(p, device=dev))
        blocks.append(slice_cache(o.past_key_values, G, G + len(c_ids)))
        cur += len(c_ids)
    return g_ids, g_cache, blocks, G, cur


def ask(h, g_cache, blocks, keep_idx, qids, qpos):
    from tiered_cache import concat_caches, slice_cache, cache_len
    keep = [blocks[i] for i in sorted(keep_idx) if blocks[i] is not None]
    full = concat_caches([g_cache] + keep)
    full = slice_cache(full, 0, cache_len(full))
    dev = h.model.device
    with torch.no_grad():
        o = h.model(input_ids=torch.tensor([qids], device=dev), past_key_values=full,
                    use_cache=True, cache_position=torch.arange(qpos, qpos + len(qids), device=dev))
    c = o.past_key_values
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]; p = qpos + len(qids)
    for _ in range(23):
        with torch.no_grad():
            o = h.model(input_ids=nxt, past_key_values=c, use_cache=True,
                        cache_position=torch.tensor([p], device=dev))
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt); p += 1
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--keep-blocks", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "nonlinear_sift")
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    log = log_at(args.depth)
    lines = log.split("\n")
    per = max(1, len(lines) // args.blocks)
    chunks = ["\n".join(lines[i:i + per]) for i in range(0, len(lines), per)]
    needle_line = next((i for i, l in enumerate(lines) if "STACK_FAIL" in l or "9AF4" in l), None)
    needle_chunk = None if needle_line is None else needle_line // per

    # token-level attention for block scoring: reuse the linear scorer on whole chunks
    g_ids, g_cache, blocks, G, total = build_table(h, PROMPT_PREFIX, chunks)
    nblocks = len([b for b in blocks if b is not None])
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    qpos = total

    print("=" * 88)
    print("NON-LINEAR CONTEXT — which sifter picks the right BLOCKS?")
    print("  model=%s  depth=%.2f  %d blocks, keeping %d" % (
        args.model, args.depth, nblocks, args.keep_blocks))
    print("=" * 88)
    print("  needle line %s -> block %s" % (needle_line, needle_chunk))
    print()

    # score each block by the total attention its tokens receive (recomputed per chunk)
    block_scores, block_line_scores = [], []
    for ci, ch in enumerate(chunks):
        if not ch.strip():
            block_scores.append(0.0); block_line_scores.append(0.0); continue
        text = PROMPT_PREFIX + ch + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        if c1 <= 0:
            block_scores.append(0.0); block_line_scores.append(0.0); continue
        h.chunk1_len = c1
        cache, qcap, ids = T.prefill_capture_q(h, text)
        sc = T.self_attn_scores(h, cache, qcap, c0, c1)
        del cache
        block_scores.append(float(sc.sum()))
        # line pooling: mean per line, then sum, so a coherent line is a unit
        nlines = max(1, ch.count("\n") + 1)
        block_line_scores.append(float(sc.mean()) * nlines)

    order = sorted(range(nblocks), key=lambda i: -block_scores[i])
    line_order = sorted(range(nblocks), key=lambda i: -block_line_scores[i])

    picks = {
        "attention-05b": sorted(order[:args.keep_blocks]),
        "line": sorted(line_order[:args.keep_blocks]),
        "first-k": list(range(args.keep_blocks)),
        "last-k": list(range(nblocks - args.keep_blocks, nblocks)),
    }
    for s in range(args.seeds):
        rnd = random.Random(7 + s)
        picks["random#%d" % s] = sorted(rnd.sample(range(nblocks), args.keep_blocks))

    print("  %-16s %-16s %-12s %s" % ("sifter", "blocks kept", "needle block", "verdict"))
    rows = []
    for name, keep in picks.items():
        kept_needle = (needle_chunk is None) or (needle_chunk in keep)
        ans = ask(h, g_cache, blocks, keep, qids, qpos)
        ok = "9AF4" in ans.upper().replace(" ", "")
        rows.append(dict(sifter=name, keep=keep, needle_block_kept=bool(kept_needle),
                         pass_=bool(ok), answer=ans[:60], scores=block_scores))
        print("  %-16s %-16s %-12s %s  %r" % (
            name, str(keep), "KEPT" if kept_needle else "LOST",
            "PASS" if ok else "fail", ans[:34]), flush=True)

    print()
    print("=" * 88)
    for nm in ["attention-05b", "line", "first-k", "last-k"]:
        r = next(x for x in rows if x["sifter"] == nm)
        print("  %-16s needle_block=%-5s retrieval=%s" % (
            nm, r["needle_block_kept"], "PASS" if r["pass_"] else "fail"))
    rnd_rows = [r for r in rows if r["sifter"].startswith("random")]
    print("  %-16s needle_block=%d/%d   retrieval=%d/%d" % (
        "random",
        sum(r["needle_block_kept"] for r in rnd_rows), len(rnd_rows),
        sum(r["pass_"] for r in rnd_rows), len(rnd_rows)))
    print()
    print("  block scores:", " ".join("%.2f" % s for s in block_scores))
    print("  needle block score: %.2f  (rank %d of %d by attention)" % (
        block_scores[needle_chunk] if needle_chunk is not None else -1,
        order.index(needle_chunk) + 1 if needle_chunk is not None else -1, nblocks))

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                depth=args.depth, model=args.model, nblocks=nblocks,
                keep_blocks=args.keep_blocks, needle_chunk=needle_chunk,
                block_scores=block_scores,
                results=[{k: v for k, v in r.items() if k != "scores"} for r in rows],
            )) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
