#!/usr/bin/env python3
"""
One (depth, mode) combination per process, so memory is released between runs.

The earlier version evaluated every combination twice inside one process, keeping the
biggest caches alive throughout. At the deepest layout the chunk is ~2200 tokens and a
single materialised attention matrix is 268 MB per layer, which is what took the box
into an OOM. A process per combination makes one bad run cost one run.

Modes:
  baseline   all-layer self-attention, top-k tokens      (the Finding 7/8 recipe)
  posnorm    + divide by the causal horizon (owner fix a)
  line       + keep the whole \n-delimited line          (owner fix b)
  both       + both fixes

Usage:
    python src/depth_sweep.py --depth 0.15 --mode baseline
"""
import sys, os, io, json, time, math, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T

SYS = ("<|im_start|>system\nYou are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
       "Always answer with the exact value requested, quoting it verbatim.\n<|im_end|>\n")
PRE = T.PRE_LINES * 3
POST = T.POST_LINES * 3
NEEDLE = T.NEEDLE_LINES


def log_at(frac):
    n = len(PRE) + len(NEEDLE) + len(POST)
    want = int(n * frac)
    before = min(len(PRE), max(0, want))
    after = max(0, n - before - len(NEEDLE))
    return "\n".join(PRE[:before] + NEEDLE + POST[:after])


def causal_scores(h, cache, qcap, c0, c1, block=256):
    """Same as self_attn_scores but WITH a causal mask, used only to prove the
    real scorer is non-causal (owner fix c claims the mask is the bias source)."""
    acc = torch.zeros(c1, dtype=torch.float32)
    sl = slice(c0, c0 + c1)
    mask = torch.triu(torch.ones(c1, c1, dtype=torch.bool), diagonal=1)
    for i in sorted(qcap):
        q_all = qcap.pop(i)
        k = cache.layers[i].keys[:, :, sl, :]
        if h.n_q_heads != h.n_kv_heads:
            k = k.repeat_interleave(h.n_q_heads // h.n_kv_heads, dim=1)
        for s in range(0, c1, block):
            e = min(s + block, c1)
            q = q_all[:, c0 + s:c0 + e]
            q = q.reshape(1, e - s, h.n_q_heads, h.head_dim).permute(0, 2, 1, 3).contiguous()
            q = h.rope.apply(q, list(range(c0 + s, c0 + e)))
            sc = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(h.head_dim)
            sc = sc.masked_fill(mask[s:e].unsqueeze(0).unsqueeze(0), float("-inf"))
            # key axis is global; a block of queries contributes a full [c1] vector
            acc += torch.softmax(sc, dim=-1)[0].float().sum(0).sum(0)
            del sc, q
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, required=True)
    ap.add_argument("--mode", required=True,
                    choices=["baseline", "posnorm", "line", "both"])
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--check-causal", action="store_true",
                    help="also score WITH a causal mask, to prove the scorer is non-causal")
    args = ap.parse_args()

    T.require_memory(3500, "depth=%.2f mode=%s" % (args.depth, args.mode))

    h = T.Harness()
    log = log_at(args.depth)
    text = SYS + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS, add_special_tokens=False)["input_ids"])
    ids_len = h.tok(text, add_special_tokens=False)["input_ids"]
    c1 = len(ids_len) - c0
    h.chunk1_len = c1
    na = T.locate_needle(h.tok, text)
    rel = None if na is None else na - c0

    t0 = time.time()
    cache, qcap, ids = T.prefill_capture_q(h, text)
    prefill_s = time.time() - t0

    scores = T.self_attn_scores(h, cache, qcap, c0, c1)
    keep = max(1, int(c1 * args.keep_frac))
    scores_used = scores
    spans = None
    if args.mode in ("posnorm", "both"):
        scores_used = T.sel_position_normalized(scores_used, c1)
    if args.mode in ("line", "both"):
        spans = T.line_spans(h.tok, text, c0, c1)
    if spans is not None:
        idx = T.sel_line_pooled(scores_used, spans, keep)
    else:
        order = torch.argsort(scores_used, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())

    order = torch.argsort(scores_used, descending=True)
    rank = None if rel is None else (order == rel).nonzero().flatten().item()
    kept = None if rel is None else (rel in set(idx))

    tail = list(range(c0 + c1, ids.shape[1]))
    ecache = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
    ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
    ok = "9AF4" in ans.upper().replace(" ", "")
    saved = 100 * (1 - h.cache_len(ecache) / ids.shape[1])

    out = dict(depth=args.depth, mode=args.mode, c1=c1, c0=c0, keep_frac=args.keep_frac,
               keep=keep, rank=rank, rank_frac=(None if rank is None else round(rank / c1, 3)),
               needle_kept=kept, kv_saved_pct=round(saved, 2), pass_=bool(ok),
               answer=ans[:100], prefill_s=round(prefill_s, 1))

    if args.check_causal:
        # Re-run the prefill for a fresh q capture, then score WITH the mask.
        cache2, qcap2, _ = T.prefill_capture_q(h, text)
        cs = causal_scores(h, cache2, qcap2, c0, c1)
        co = torch.argsort(cs, descending=True)
        out["causal_rank"] = None if rel is None else int((co == rel).nonzero().flatten().item())
        out["causal_keeps_needle"] = None if rel is None else bool(rel in set(co[:keep].tolist()))
        del cache2

    del cache, ecache
    print(json.dumps(out))


if __name__ == "__main__":
    main()
