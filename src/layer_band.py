#!/usr/bin/env python3
"""Does restricting saliency scoring to the mid-layer RETRIEVAL BAND help?

Both research agents, and this harness's own Finding 1 per-layer rank table (needle 303rd
at layer 0, 46th at layer 16), point the same way: summing attention across ALL layers
mixes high-entropy early-layer syntactic attention into a sparse mid-layer retrieval
signal. The proposal is to score only the band where retrieval actually happens.

Bands, for L layers:
    all      every layer                 (current behaviour)
    mid      [0.35L, 0.75L)              the retrieval band both agents name
    late     [0.65L, L)                  the readout end
    early    [0, 0.35L)                  the syntactic end

Each band is tested with the same keep budget on the same rows, so any difference is the
band and nothing else. A band that lifts depth 0.15 from fail to PASS while holding the
others is the actionable result; a band that only reshuffles ranks is not.

Usage:
    python src/layer_band.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def band_scores(h, cache, qcap, c0, c1, lo, hi, block=256):
    """self_attn_scores restricted to layers [lo, hi)."""
    import math
    acc = torch.zeros(c1, dtype=torch.float32)
    sl = slice(c0, c0 + c1)
    for i in sorted(qcap):
        q_all = qcap.pop(i)
        if not (lo <= i < hi):
            continue
        k = cache.layers[i].keys[:, :, sl, :]
        if h.n_q_heads != h.n_kv_heads:
            k = k.repeat_interleave(h.n_q_heads // h.n_kv_heads, dim=1)
        for s in range(0, c1, block):
            e = min(s + block, c1)
            q = q_all[:, c0 + s:c0 + e]
            q = q.reshape(1, e - s, h.n_q_heads, h.head_dim).permute(0, 2, 1, 3).contiguous()
            q = h.rope.apply(q, list(range(c0 + s, c0 + e)))
            sc = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(h.head_dim)
            acc += torch.softmax(sc, dim=-1)[0].float().sum(0).sum(0)
            del sc, q
    return acc


def bands(L):
    return [("all", 0, L),
            ("mid", int(0.35 * L), int(0.75 * L)),
            ("late", int(0.65 * L), L),
            ("early", 0, int(0.35 * L))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "layer_band %s" % args.model)
    h = T.Harness(args.model)
    L = len(h.model.model.layers)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0
        tail_base = None
        for bname, lo, hi in bands(L):
            cache, qcap, ids = T.prefill_capture_q(h, text)
            sc = band_scores(h, cache, qcap, c0, c1, lo, hi)
            keep = max(1, int(c1 * args.keep_frac))
            order = torch.argsort(sc, descending=True)
            idx = sorted(int(x) for x in order[:keep].tolist())
            rank = None if rel is None else int((order == rel).nonzero().flatten().item())
            kept = None if rel is None else (rel in set(idx))
            tail = list(range(c0 + c1, ids.shape[1]))
            ec = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                                   c0, c0, c1, tail)
            ans, _ = h.generate(T.build_question(T.Q2_TEXT), ec)
            ok = "9AF4" in ans.upper().replace(" ", "")
            saved = 100 * (1 - h.cache_len(ec) / ids.shape[1])
            r = dict(depth=d, band=bname, layers="%d-%d" % (lo, hi), c1=c1,
                     rank=rank, needle_kept=kept, kv_saved_pct=round(saved, 2),
                     pass_=bool(ok), answer=ans[:70])
            rows.append(r)
            print(json.dumps(r), flush=True)
            del cache, ec

    print()
    print("model=%s  L=%d  keep=%.2f" % (args.model, L, args.keep_frac))
    print()
    print("%-7s %-8s %-9s %-8s %-9s %-8s %s" % (
        "depth", "band", "layers", "rank", "saved", "needle", "verdict"))
    for r in rows:
        print("%-7s %-8s %-9s %-8s %-9s %-8s %s" % (
            r["depth"], r["band"], r["layers"], r["rank"], r["kv_saved_pct"],
            r["needle_kept"], "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 70)
    print("PASS COUNT BY BAND")
    print("=" * 70)
    for bname, _, _ in bands(L):
        sub = [r for r in rows if r["band"] == bname]
        p = sum(1 for r in sub if r["pass_"])
        pat = " ".join("P" if r["pass_"] else "F" for r in sub)
        print("  %-7s %d/%d   %s" % (bname, p, len(sub), pat))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
