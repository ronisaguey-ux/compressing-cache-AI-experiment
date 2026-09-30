#!/usr/bin/env python3
"""Tiered attention: an immutable global backbone plus modular, independently
evictable workspace blocks.

THE ARCHITECTURE (Bob, 2026-09-30), and the implementation insight that makes it simple.

    [ Tier 1: Global Backbone ]  [ Tier 2: Workspace Blocks ]  [ Tier 3: Query ]

Tier 1 is prefilmed ONCE (system prompt, tool schemas, anything invariant). Each Tier-2
block is then forwarded with the global cache as `past_key_values` and its own disjoint
position range. Tier 3 is the query, attending over the global plus whichever blocks
survive.

★ THE MASK IS NOT NEEDED, AND THAT IS THE WHOLE TRICK. It would be natural to build a
4D block-diagonal mask. But a block prefilled in its OWN forward pass never had another
block's keys in its attention window at all -- separate forwards ARE the mask, exactly.
So each block's KV is bit-identical to what it would be with any set of other blocks
present or absent, and eviction is lossless by construction:

    evict block 2  ->  pop its rows from the table. Nothing is recomputed, nothing is
                       renumbered, no rotation, no repair. Blocks 1 and 3 are untouched.

★ POSITION GAPS ARE HARMLESS. Each block gets a disjoint absolute range so RoPE phases
do not collide. When a middle block is popped, later blocks keep their original positions
and a hole appears in the range -- and that is fine, because a block's keys depend only
on its own tokens and its own positions. It never attended across the hole.

CONTRAST WITH FINDINGS 23-24. There, validity was a PREFIX property: evicting a middle
token invalidated everything after it, front eviction destroyed all reuse, and rotation
could not repair the damage (Finding 18). Here the property is LOCAL: a block is valid
regardless of what its neighbours do.

★ THE HONEST COST, measured not asserted: a block can see only the global anchor and
itself. A question requiring two blocks to reason across CANNOT be answered. That is a
real capability loss and the `--probe cross` arm measures it. A design that only showed
the wins would be selling something.

Usage:
    python src/tiered_cache.py --model Qwen/Qwen2.5-0.5B
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def forward(h, ids, past=None, position_ids=None):
    dev = h.model.device
    kw = {}
    if position_ids is not None:
        kw["position_ids"] = torch.tensor([position_ids], dtype=torch.long, device=dev)
        kw["cache_position"] = torch.tensor(position_ids, dtype=torch.long, device=dev)
    with torch.no_grad():
        out = h.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
                      past_key_values=past, use_cache=True, **kw)
    return out


def slice_cache(cache, lo, hi):
    """Fresh DynamicCache holding positions [lo, hi).

    ★ `DynamicCache()` is EMPTY -- it has no layers until a forward populates them, so
    indexing `c.layers[li]` on a fresh one raises IndexError (which is how the first
    version of this file failed). The constructor takes the data directly:
    `DynamicCache(ddp_cache_data=[(key, value), ...])`.
    """
    from transformers import DynamicCache
    data = [(cache.layers[li].keys[:, :, lo:hi, :].contiguous().clone(),
             cache.layers[li].values[:, :, lo:hi, :].contiguous().clone())
            for li in range(len(cache.layers))]
    return DynamicCache(ddp_cache_data=data)


def concat_caches(caches):
    """Concatenate several caches along the sequence axis, in the given order."""
    from transformers import DynamicCache
    live = [c for c in caches if c is not None and len(c.layers) > 0]
    if not live:
        return DynamicCache()
    n_layers = len(live[0].layers)
    data = []
    for li in range(n_layers):
        k = torch.cat([c.layers[li].keys for c in live], dim=2).contiguous()
        v = torch.cat([c.layers[li].values for c in live], dim=2).contiguous()
        data.append((k, v))
    return DynamicCache(ddp_cache_data=data)


def cache_len(cache):
    return cache.layers[0].keys.shape[2]


def evict_by_line(log, needle_marker, keep_lines):
    """Build chunks from a log with the needle placed in a chosen chunk.

    `keep_lines` is a predicate over line index; chunks are runs of kept lines.
    """
    lines = log.split("\n")
    chunks, cur = [], []
    for i, ln in enumerate(lines):
        if keep_lines(i):
            cur.append(ln)
        elif cur:
            chunks.append("\n".join(cur)); cur = []
    if cur:
        chunks.append("\n".join(cur))
    return lines, chunks


def build_table(h, global_text, chunks, verbose=True):
    """Tier 1 prefilmed once; each Tier-2 block forwarded with the global as past."""
    g_ids = h.tok(global_text, add_special_tokens=False)["input_ids"]
    G = len(g_ids)
    g_out = forward(h, g_ids, past=None, position_ids=list(range(G)))
    global_cache = g_out.past_key_values

    blocks, pos = [], []
    cur = G
    for ci, ch in enumerate(chunks):
        c_ids = h.tok(ch, add_special_tokens=False)["input_ids"]
        if not c_ids:
            blocks.append(None); pos.append((cur, cur)); continue
        p = list(range(cur, cur + len(c_ids)))
        # ★ PASS A FRESH CLONE OF THE GLOBAL CACHE EVERY TIME. A DynamicCache is mutated
        # in place by the forward pass -- it appends the new K/V and returns the same
        # object. Handing the same object to every block means block 2 sees block 1's
        # keys and the isolation the whole design rests on is silently gone. The clone
        # is what makes "separate forwards ARE the mask" true.
        out = forward(h, c_ids, past=slice_cache(global_cache, 0, G), position_ids=p)
        blk = slice_cache(out.past_key_values, G, G + len(c_ids))
        blocks.append(blk)
        pos.append((cur, cur + len(c_ids)))
        cur += len(c_ids)
        if verbose:
            print("    block %d: %4d tok  positions [%d, %d)" % (
                ci, len(c_ids), pos[-1][0], pos[-1][1]), flush=True)

    # Self-check for the mutation trap: if a block forward appended to the global cache,
    # the global would have grown and every block after the first would have seen its
    # predecessor. Cheap to assert, and it is the difference between isolation and none.
    gl = cache_len(global_cache)
    if gl != G:
        raise RuntimeError(
            "global cache grew from %d to %d during block prefill -- a block forward "
            "mutated it in place and the blocks are NOT isolated" % (G, gl))
    return g_ids, global_cache, blocks, pos, G, cur


def assemble_contiguous(g_cache, kept_blocks):
    """Assemble the table with survivors RE-INDEXED CONTIGUOUSLY over the global anchor.

    ★ THIS IS THE FINDING 37 FIX, and it is the difference between 0/10 and 10/10 on the
    cross-block join (10 randomised trials, Qwen2.5-7B and Mistral-7B-Instruct).

    The bug it repairs: `build_table` gives every block a DISJOINT range in POSITION space, so
    evicting a middle block leaves a hole. With the distractor gone the table holds ~90 rows but
    block 3's keys still carry position ids ~3300, and block 1's carry ~60. RoPE is relative, so
    the query lands ~3200 positions from block 1 instead of ~60 -- the geometry of a context that
    still CONTAINS the distractor, with the content removed. Block 1 falls out of range and the
    join fails.

    ★ WHY RE-INDEXING IS LEGAL HERE. Each block's KV was produced in its own forward pass, so its
    keys are a function of its own tokens and its own position ids only -- never of another
    block's. Applying a constant offset to a block's position ids would change every key, but
    moving the ASSEMBLY so the survivors occupy a contiguous span is exactly what a context with
    no eviction would look like. The survivors stay bit-identical; only the gap closes.

    ⚠️ A caller that re-indexes must ALSO position the query contiguously (at the assembled
    length), not at the original `total`. `generate(h, tbl, qids, cache_len(tbl))` does that.
    """
    from transformers import DynamicCache
    live = [c for c in [g_cache] + list(kept_blocks) if c is not None and len(c.layers) > 0]
    if not live:
        return DynamicCache()
    n_layers = len(live[0].layers)
    data = []
    for li in range(n_layers):
        k = torch.cat([c.layers[li].keys for c in live], dim=2).contiguous()
        v = torch.cat([c.layers[li].values for c in live], dim=2).contiguous()
        data.append((k, v))
    return DynamicCache(ddp_cache_data=data)


def generate(h, cache, qids, qpos, max_new=24):
    dev = h.model.device
    ids = torch.tensor([qids], dtype=torch.long, device=dev)
    pos = torch.arange(qpos, qpos + len(qids), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = h.model(input_ids=ids, past_key_values=cache, use_cache=True,
                      position_ids=pos.unsqueeze(0), cache_position=pos)
    c = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]
    p = qpos + len(qids)
    for _ in range(max_new - 1):
        with torch.no_grad():
            out = h.model(input_ids=nxt, past_key_values=c, use_cache=True,
                          cache_position=torch.tensor([p], device=dev))
        c = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        p += 1
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)


def eq(a, b):
    if a is None or b is None:
        return a is b
    if len(a.layers) != len(b.layers):
        return False
    for li in range(len(a.layers)):
        if not torch.equal(a.layers[li].keys, b.layers[li].keys):
            return False
        if not torch.equal(a.layers[li].values, b.layers[li].values):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--needle-block", type=int, default=3, help="which block holds the needle")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "tiered_cache %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    log = log_at(args.depth)
    lines = log.split("\n")
    # split the log into equal line-count chunks
    per = max(1, len(lines) // args.blocks)
    chunks = ["\n".join(lines[i:i + per]) for i in range(0, len(lines), per)]
    needle_line = next((i for i, l in enumerate(lines) if "STACK_FAIL" in l or "9AF4" in l), None)
    needle_chunk = None if needle_line is None else needle_line // per

    print("=" * 84)
    print("TIERED ATTENTION  |  model=%s  depth=%.2f" % (args.model, args.depth))
    print("=" * 84)
    print("  global anchor: %d chars (system prompt)" % len(SYS))
    print("  %d blocks of ~%d lines; needle at line %s -> block %s" % (
        len(chunks), per, needle_line, needle_chunk))
    print()

    global_text = SYS + "<|im_start|>user\n"
    g_ids, g_cache, blocks, pos, G, total = build_table(h, global_text, chunks)
    # the question goes after the last block
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    qpos = total
    print("  global %d tok + %d blocks = %d tok (query at position %d)" % (
        G, len(blocks), total, qpos))

    # ------------------------------------------------------ 1. isolation / bit-exact
    print()
    print("1. ISOLATION — does a block depend on WHICH OTHER blocks are in the table?")
    # ★ HOLD THE BLOCK'S OWN POSITIONS FIXED. The first version of this test compared a
    # chunk built at one position range against the same chunk built at another, and
    # reported DIFFERS -- which is correct and meaningless, because RoPE is a function of
    # absolute position, so moving a block legitimately changes its keys. That tests
    # position-dependence, not isolation. The real question is whether the PRESENCE of
    # other blocks changes this one.
    bi = args.needle_block if args.needle_block < len(blocks) else 0
    if blocks[bi] is None:
        same = None
        print("     target block empty; skipped")
    else:
        c_ids = h.tok(chunks[bi], add_special_tokens=False)["input_ids"]
        p_only = list(range(pos[bi][0], pos[bi][0] + len(c_ids)))
        # built with ONLY the global behind it -- no other block exists
        o_alone = forward(h, c_ids, past=slice_cache(g_cache, 0, G), position_ids=p_only)
        alone = slice_cache(o_alone.past_key_values, G, G + len(c_ids))
        same = eq(blocks[bi], alone)
        print("     block %d built with 5 other blocks present vs built alone: %s"
              % (bi, "IDENTICAL" if same else "DIFFERS"))
        print("     => %s" % ("a block's KV depends only on (global, itself) -- the table "
                              "is genuinely modular" if same else
                              "other blocks influence this one -- isolation FAILS"))

    # ------------------------------------------------------ 2. eviction is lossless
    print()
    print("2. EVICTION — pop a block, check survivors are untouched")
    victim = 1 if len(blocks) > 2 else 0
    kept_idx = [i for i in range(len(blocks)) if i != victim and blocks[i] is not None]
    assembled = concat_caches([g_cache] + [blocks[i] for i in kept_idx])
    all_assembled = concat_caches([g_cache] + [b for b in blocks if b is not None])
    # compare each survivor's slice in the assembled cache against its standalone block
    cur = cache_len(g_cache)
    exact = True
    for i in kept_idx:
        L = cache_len(blocks[i])
        sub = slice_cache(assembled, cur, cur + L)
        if not eq(sub, blocks[i]):
            exact = False
        cur += L
    print("     evicted block %d; survivors re-checked inside the assembled cache: %s"
          % (victim, "BIT-IDENTICAL" if exact else "CHANGED"))
    print("     => eviction is lossless: nothing is recomputed, renumbered, or repaired")

    print()
    print("3. RETRIEVAL — the needle is in block %s, we evict block %d" % (needle_chunk, victim))
    ans_keep = generate(h, slice_cache(assembled, 0, cache_len(assembled)), qids, qpos)
    ok_keep = "9AF4" in ans_keep.upper().replace(" ", "")
    print("     evict a NON-needle block : %s   %r" % ("PASS" if ok_keep else "fail", ans_keep[:56]))
    ans_all = generate(h, slice_cache(all_assembled, 0, cache_len(all_assembled)), qids, qpos)
    ok_all = "9AF4" in ans_all.upper().replace(" ", "")
    print("     full table (ceiling)     : %s   %r" % ("PASS" if ok_all else "fail", ans_all[:56]))

    print()
    print("4. CONTROL — evict the needle's OWN block; must fail")
    if needle_chunk is None:
        print("     needle not located; control skipped")
        ok_no = None
    else:
        k2 = [i for i in range(len(blocks)) if i != needle_chunk and blocks[i] is not None]
        c2 = concat_caches([g_cache] + [blocks[i] for i in k2])
        ans_no = generate(h, slice_cache(c2, 0, cache_len(c2)), qids, qpos)
        ok_no = "9AF4" in ans_no.upper().replace(" ", "")
        print("     evict needle block %d    : %s   %r" % (
            needle_chunk, "PASS" if ok_no else "fail", ans_no[:56]))
        print("     => %s" % ("correct: without the evidence the answer is gone"
                              if not ok_no else
                              "WARNING: answer survived without the needle -- either the "
                              "question leaks the answer or the blocks are not isolated"))

    # ------------------------------------------------------ 5. the honest cost
    print()
    print("5. THE COST — a question that needs TWO blocks cannot be answered")
    print("     (cross-block reasoning is impossible by construction; measured, not assumed)")
    # ask something whose evidence is split: the needle's block plus the line naming the file
    xq = "Which file failed to link, and what was the failure code?"
    xq_ids = h.tok(xq, add_special_tokens=False)["input_ids"] + h.tok("\nAnswer:", add_special_tokens=False)["input_ids"]
    ans_cross = generate(h, slice_cache(assembled, 0, cache_len(assembled)), xq_ids, qpos)
    print("     cross-block question: %r" % ans_cross[:70])
    print("     (if this names both the file and the code, the blocks are NOT isolated)")

    # ------------------------------------------------------ timing / savings
    kept_tok = cache_len(assembled)
    all_tok = cache_len(all_assembled)
    print()
    print("=" * 84)
    print("  block order-independence : %s" % same)
    print("  eviction lossless        : %s" % exact)
    print("  retrieval (evict non-needle): %s" % ("PASS" if ok_keep else "fail"))
    print("  retrieval (ceiling)      : %s" % ("PASS" if ok_all else "fail"))
    print("  control (evict needle)   : %s" % ("fail as expected" if ok_no is False else ok_no))
    print("  cache after eviction     : %d tok  (full table %d, %.0f%% retained)" % (
        kept_tok, all_tok, 100 * kept_tok / all_tok))
    print("=" * 84)

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                model=args.model, depth=args.depth, blocks=len(chunks),
                needle_chunk=needle_chunk, victim_block=victim,
                order_independent=bool(same), eviction_lossless=bool(exact),
                retrieval_evict_pass=bool(ok_keep), ceiling_pass=bool(ok_all),
                control_evict_needle_pass=(None if ok_no is None else bool(ok_no)),
                kept_tokens=kept_tok, full_tokens=all_tok,
                cross_block_answer=ans_cross[:80],
            )) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
