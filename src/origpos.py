#!/usr/bin/env python3
"""POSITION-INDEPENDENT EVICTION: keep ORIGINAL positions, do not re-rotate.

Why this is a different idea from everything already tested.

`apply_delta` rotates every surviving key from its original position to its new DENSE
position, and `generate` lets transformers derive the query position from the compacted
cache length. The geometry is then internally consistent -- but the RELATIVE DISTANCES are
not the ones the model was trained on. A needle 485 tokens away in the full cache becomes
230 tokens away in a 60% compaction, and RoPE encodes exactly that distance.

The rotation-off arm (`rot=False`) is worse, not better: the keys keep their original
rotations while the query sits at the compacted length, so the relative angle is simply
inconsistent. Measured failing.

This arm keeps the key rotations AND puts the query back at its ORIGINAL position:

    keys:  rotate(survivor, original_position)     <- no re-rotation at all
    query: cache_position = original_sequence_len  <- explicit, not derived

Relative angle for a retained key is then `orig_len - original_position`, which is
EXACTLY the full-cache angle. The model sees every retained token at the distance it was
trained to see it, while unretained tokens are simply absent. That is the
"position-independent KV cache" idea (SGLang RFC #30928) and it is the one arm the harness
has never run.

Prediction if the conditioning-loss account is right: `origpos` recovers the full-cache
verdict at every depth, because nothing about the surviving tokens' geometry changed.

Arms per depth:
    full      no eviction, no rotation              (upper control)
    evict     dense re-rotation -- the current method
    origpos   eviction, original positions preserved

Usage:
    python src/origpos.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
"""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

MAX_NEW = 24


def generate_at_positions(h, prompt_text, cache, query_pos0, max_new=MAX_NEW):
    """Greedy decode with an EXPLICIT query start position.

    transformers otherwise derives the query position from the cache length, which is
    the compacted length and is the whole reason the relative distances move.
    """
    enc = h.tok(prompt_text, return_tensors="pt")
    ids = enc["input_ids"]
    q_len = ids.shape[1]

    pos = torch.arange(query_pos0, query_pos0 + q_len, dtype=torch.long)
    out = h.model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    generated = [nxt]
    p = query_pos0 + q_len
    for _ in range(max_new - 1):
        cp = torch.tensor([p], dtype=torch.long)
        out = h.model(input_ids=nxt, past_key_values=cache, use_cache=True, cache_position=cp)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(nxt)
        p += 1
        if nxt.item() == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "origpos %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        orig_len = c0 + c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0
        qtext = T.build_question(T.Q2_TEXT)

        t0 = time.time()
        cache_full, qcap, ids = T.prefill_capture_q(h, text)
        full_len = ids.shape[1]

        # ---- full control ----
        ans_full, _ = h.generate(qtext, h.clone_cache(cache_full))
        ok_full = "9AF4" in ans_full.upper().replace(" ", "")
        rows.append(dict(depth=d, arm="full", c1=c1, needle_rel=rel, kv_saved_pct=0.0,
                         pass_=bool(ok_full), answer=ans_full[:80],
                         sec=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1]), flush=True)

        # ---- selection (shared by both eviction arms) ----
        scores = T.self_attn_scores(h, cache_full, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        kept = None if rel is None else (rel in set(idx))
        tail = list(range(c0 + c1, ids.shape[1]))

        # ---- evict: dense re-rotation (the current method) ----
        ce = T.build_arm_cache(h, cache_full, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
        ans_e, _ = h.generate(qtext, ce)
        ok_e = "9AF4" in ans_e.upper().replace(" ", "")
        saved_e = 100 * (1 - h.cache_len(ce) / full_len)
        del ce
        rows.append(dict(depth=d, arm="evict", c1=c1, needle_rel=rel, rank=rank,
                         keep=keep, needle_kept=kept, kv_saved_pct=round(saved_e, 2),
                         pass_=bool(ok_e), answer=ans_e[:80], sec=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1]), flush=True)

        # ---- origpos: rotate NOTHING, query at the ORIGINAL position ----
        co = T.build_arm_cache(h, cache_full, dict(kind="evict", keep=idx, rot=False),
                               c0, c0, c1, tail)
        ans_o = generate_at_positions(h, qtext, co, full_len)
        ok_o = "9AF4" in ans_o.upper().replace(" ", "")
        saved_o = 100 * (1 - h.cache_len(co) / full_len)
        del co
        rows.append(dict(depth=d, arm="origpos", c1=c1, needle_rel=rel, rank=rank,
                         keep=keep, needle_kept=kept, kv_saved_pct=round(saved_o, 2),
                         pass_=bool(ok_o), answer=ans_o[:80], sec=round(time.time() - t0, 1)))
        print(json.dumps(rows[-1]), flush=True)

        del cache_full

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print()
    print("%-7s %-9s %-7s %-8s %-9s %s" % ("depth", "arm", "saved", "kept", "rank", "verdict"))
    for r in rows:
        print("%-7s %-9s %-7s %-8s %-9s %s" % (
            r["depth"], r["arm"], r["kv_saved_pct"], r.get("needle_kept"),
            r.get("rank"), "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 72)
    print("HEAD-TO-HEAD")
    print("=" * 72)
    print("  %-7s %-7s %-9s %-9s %s" % ("depth", "full", "evict", "origpos", "verdict"))
    wins = 0
    for d in sorted(set(r["depth"] for r in rows)):
        f = next(r for r in rows if r["depth"] == d and r["arm"] == "full")
        e = next(r for r in rows if r["depth"] == d and r["arm"] == "evict")
        o = next(r for r in rows if r["depth"] == d and r["arm"] == "origpos")
        tag = ""
        if o["pass_"] and not e["pass_"]:
            tag = "  <- origpos FIXES what evict fails"; wins += 1
        elif o["pass_"] == e["pass_"] == f["pass_"]:
            tag = "  (both match full)"
        elif not o["pass_"] and e["pass_"]:
            tag = "  <- origpos WORSE"
        print("  %-7s %-7s %-9s %-9s%s" % (
            d, "PASS" if f["pass_"] else "fail", "PASS" if e["pass_"] else "fail",
            "PASS" if o["pass_"] else "fail", tag))
    print()
    print("origpos fixed %d depth(s) that dense re-rotation failed." % wins)

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
