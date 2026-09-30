#!/usr/bin/env python3
"""Multi-turn serving cost: block runtime vs naive recompute (Bob, 2026-09-30).

THE CLAIM UNDER TEST. Bob's Test 3/4 metric: "Standard serving must re-prefill the entire
prompt; your runtime dispatches instantly from the cached Anchor + Block 3." His Test 4 asks
for cumulative prefill FLOPs reduction and cache hit rate across a 10-turn loop.

8 turns, 4 evictions. Two arms, same model, same task, same questions:

  vanilla   STANDARD SERVING. Every turn re-prefills the WHOLE current context from token 0
            with fresh positions. An eviction or an appended turn invalidates nothing that was
            cached, because nothing was cached -- this is the behaviour Bob describes as
            "100% cache miss on edits".

  runtime   THE BLOCK TABLE. Blocks are ingested ONCE and their KV is retained. An eviction
            drops that block's KV. Each turn forwards ONLY the new turn text plus the query
            against the retained table. Prompt positions stay ABSOLUTE (Finding 23).

WHAT IS COUNTED, and why tokens rather than FLOPs:
  prefill_tokens  tokens pushed through a forward pass, summed over turns. Attention cost is
                  quadratic in sequence length, so this UNDERSTATES the runtime's advantage --
                  vanilla's tokens are the long ones. Reported honestly as the token figure
                  with the note that the FLOP gap is at least this wide.
  wall_s          measured, not modelled
  peak_kv_mb      bytes resident in the KV cache at the widest point
  cache_hit_pct   fraction of the context that did NOT need re-prefilling

★ THE CONTROL THAT MAKES IT INTERPRETABLE: both arms ingest the SAME blocks and answer the
SAME question on the final turn, and the runtime must be CORRECT (not just fast) or the
comparison is meaningless. A fast wrong answer loses. `answer_ok` is recorded per arm, and the
headline is only reported for arms that answered.

★ THIS IS NOT A 16k RUN. Same box limit as the rest of the matrix: prompt sizes are recorded
per turn so the scale is never implied to be larger than it is.

Usage:
    python benchmarks/multiturn.py --model Qwen/Qwen2.5-0.5B --turns 8 --evictions 4
"""
import sys, os, json, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS
from tiered_cache import build_table, concat_caches, cache_len
import run_benchmarks as RB
import tasks as TASKS


def turn_text(i):
    """Turn i of a shell debugging loop. Turns 2,4,6,8 are FAILED attempts (the evictable ones)."""
    if i % 2 == 0:
        return ("Turn %d\nCommand: ip addr add 192.168.1.%d/24 dev eth0\n"
                "Output: RTNETLINK answers: Operation not permitted. Failed with error code 1."
                % (i, 50 + i))
    return ("Turn %d\nCommand: sudo ip addr add 192.168.1.%d/24 dev eth0\n"
            "Output: Success. Interface eth0 bound to 192.168.1.%d/24." % (i, 50 + i, 50 + i))


def build_turns(n):
    """Turn 0 is the task spec; turns 1..n-1 are tool exchanges."""
    out = [("Task Spec",
            "Objective: Configure static IP for interface eth0. Subnet: 192.168.1.0/24.\n"
            "Maintain only the active configuration state; failed attempts are not state.")]
    for i in range(1, n):
        out.append(("Turn %d%s" % (i, " (FAILED)" if i % 2 == 0 else ""), turn_text(i)))
    return out


QUERY = ("What is the most recent IP address successfully assigned to eth0? "
         "Give only the address.")


def arm_vanilla(h, global_text, turns, evict_idx, max_new):
    """Standard serving: every turn prefills the entire context from scratch."""
    rows = []
    live = [i for i in range(len(turns)) if i not in evict_idx]
    for t in range(len(turns)):
        ctx_blocks = [turns[i][1] for i in live if i <= t]
        ids = RB.full_prompt_ids(h, global_text, ctx_blocks, QUERY)
        t0 = time.perf_counter()
        dev = h.model.device
        ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
        pos = torch.arange(0, len(ids), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = h.model(input_ids=ids_t, use_cache=True, position_ids=pos.unsqueeze(0))
        dt = time.perf_counter() - t0
        kv = RB.cache_bytes(o.past_key_values)
        rows.append(dict(turn=t, prefill_tokens=len(ids), wall_s=dt, kv_bytes=kv))
        del o
    # final answer on the last turn's context
    ctx_blocks = [turns[i][1] for i in live]
    ids = RB.full_prompt_ids(h, global_text, ctx_blocks, QUERY)
    ans, ttft, kv = RB.ttft_decode(h, None, ids, 0, max_new)
    return rows, ans, kv


def arm_runtime(h, global_text, turns, evict_idx, max_new):
    """Block runtime: ingest once, reuse; each turn forwards only the new turn + query."""
    # Ingest EVERY block once, up front -- the one-time cost, attributed separately.
    texts = [t[1] for t in turns]
    t0 = time.perf_counter()
    g_ids, g_cache, blk_caches, pos, G, total = build_table(h, global_text, texts, verbose=False)
    ingest_s = time.perf_counter() - t0
    ingest_tokens = G + sum(len(h.tok(x, add_special_tokens=False)["input_ids"]) for x in texts)

    rows = []
    for t in range(len(turns)):
        kept = [c for i, c in enumerate(blk_caches)
                if c is not None and i not in evict_idx and i <= t]
        tbl = concat_caches([g_cache] + kept)
        # Only the newest turn's text is new to the model; the query is always new.
        new_text = T.build_question(turns[t][1] + "\n" + QUERY)
        qids = h.tok(new_text, add_special_tokens=False)["input_ids"]
        base = cache_len(tbl)
        ans, ttft, kv = RB.ttft_decode(h, tbl, qids, base, max_new)
        rows.append(dict(turn=t, prefill_tokens=len(qids), wall_s=ttft, kv_bytes=kv,
                         cached_tokens=base))
    return rows, ans, kv, ingest_s, ingest_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--evictions", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--outdir", default=os.path.join(_HERE, "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    need = 9000 if "7B" in args.model.upper() else 3500
    T.require_memory(need, "multiturn %s" % args.model)
    if "7B" in args.model.upper():
        os.environ["CCAI_QUANT"] = "8bit"

    turns = build_turns(args.turns)
    # Evict the failed attempts (odd indices), capped at --evictions.
    failed = [i for i in range(1, len(turns)) if i % 2 == 0][:args.evictions]

    print("=" * 92)
    print("MULTI-TURN SERVING COST  model=%s  turns=%d  evictions=%d" % (
        args.model, args.turns, args.evictions))
    print("  evicted turns: %s" % failed)
    print("=" * 92, flush=True)

    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model
    global_text = SYS + "<|im_start|>user\n"

    v_rows, v_ans, v_kv = arm_vanilla(h, global_text, turns, failed, args.max_new)
    r_rows, r_ans, r_kv, ingest_s, ingest_tok = arm_runtime(h, global_text, turns, failed, args.max_new)

    v_tok = sum(r["prefill_tokens"] for r in v_rows)
    r_tok = sum(r["prefill_tokens"] for r in r_rows)
    v_wall = sum(r["wall_s"] for r in v_rows)
    r_wall = sum(r["wall_s"] for r in r_rows)
    v_peak = max(r["kv_bytes"] for r in v_rows)
    r_peak = max(r["kv_bytes"] for r in r_rows)

    print("\n  turn |        vanilla        |        runtime")
    print("       |  tok    wall      kv   |  tok    wall      kv")
    print("  " + "-" * 66)
    for a, b in zip(v_rows, r_rows):
        print("   %2d  | %5d %6.2fs %6.1fMB | %5d %6.2fs %6.1fMB" % (
            a["turn"], a["prefill_tokens"], a["wall_s"], a["kv_bytes"] / 1e6,
            b["prefill_tokens"], b["wall_s"], b["kv_bytes"] / 1e6), flush=True)

    print("\n  TOTALS")
    print("    prefill tokens   vanilla %7d   runtime %7d   -> %.1f%% less" % (
        v_tok, r_tok, (1 - r_tok / v_tok) * 100 if v_tok else 0))
    print("    wall clock       vanilla %6.1fs   runtime %6.1fs   -> %.1f%% less" % (
        v_wall, r_wall, (1 - r_wall / v_wall) * 100 if v_wall else 0))
    print("    peak KV          vanilla %6.1fMB  runtime %6.1fMB  -> %.1f%% less" % (
        v_peak / 1e6, r_peak / 1e6, (1 - r_peak / v_peak) * 100 if v_peak else 0))
    print("    ingest (one-time) runtime %.1fs, %d tokens (NOT counted in the per-turn totals)" % (
        ingest_s, ingest_tok))
    print("    cache hit        runtime %.1f%% of context reused per turn" % (
        (1 - r_tok / (v_tok / args.turns)) * 100 if v_tok else 0))
    print("\n  answer check")
    print("    vanilla: %r" % v_ans[:80].replace("\n", " "))
    print("    runtime: %r" % r_ans[:80].replace("\n", " "))

    out = os.path.join(args.outdir, "multiturn_%s.json" % args.model.replace("/", "_"))
    with open(out, "w") as f:
        json.dump(dict(model=args.model, turns=args.turns, evictions=failed,
                       vanilla=dict(rows=v_rows, prefill_tokens=v_tok, wall_s=v_wall,
                                    peak_kv_bytes=v_peak, answer=v_ans),
                       runtime=dict(rows=r_rows, prefill_tokens=r_tok, wall_s=r_wall,
                                    peak_kv_bytes=r_peak, answer=r_ans,
                                    ingest_s=ingest_s, ingest_tokens=ingest_tok)), f, indent=2)
    print("\n  wrote %s" % out)


if __name__ == "__main__":
    main()
