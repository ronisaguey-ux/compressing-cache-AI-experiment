#!/usr/bin/env python3
"""Context density around the needle, and a finer depth sweep to test the alternation.

Two open questions from Finding 14:

1. The evict verdict alternated F P F P across 0.15/0.35/0.55/0.75. Four points cannot
   distinguish periodicity from 0.5B noise. This sweeps EIGHT evenly spaced depths.

2. No scalar about the needle predicts the verdict. The remaining candidate is a property
   of the neighbourhood rather than the token: a model asked to reproduce an identifier
   needs the tokens that establish it as one. This measures retained-token density in
   windows centred on the needle:

       density_pm16 / pm64 / pm256   fraction of retained tokens inside the window
       kept_before                   retained tokens ahead of the needle in the log

   If density separates pass from fail while rank/mass/contiguity did not, the usable-set
   property is contextual.

Usage:
    python src/density_sweep.py --model Qwen/Qwen2.5-0.5B --depths 0.10,0.20,...,0.80
"""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def density(kept_glob, needle_global, half, lo, hi):
    """Fraction of positions retained inside [needle-half, needle+half] intersect [lo,hi)."""
    a = max(lo, needle_global - half)
    b = min(hi, needle_global + half)
    if b <= a:
        return None, 0, 0
    n = b - a
    got = sum(1 for t in range(a, b) if t in kept_glob)
    return round(got / n, 4), got, n


def run_one(h, depth, mode="baseline", keep_frac=0.60):
    text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(depth) + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n", add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    na = T.locate_needle(h.tok, text)
    rel = None if na is None else na - c0

    cache, qcap, ids = T.prefill_capture_q(h, text)
    scores = T.self_attn_scores(h, cache, qcap, c0, c1)
    keep = max(1, int(c1 * keep_frac))
    used = scores
    spans = None
    if mode in ("line", "both"):
        spans = T.line_spans(h.tok, text, c0, c1)
    if mode == "posnorm":
        used = T.sel_position_normalized(scores, c1)
    if spans is not None:
        idx = T.sel_line_pooled(used, spans, keep)
    else:
        order = torch.argsort(used, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())

    order = torch.argsort(used, descending=True)
    rank = None if rel is None else int((order == rel).nonzero().flatten().item())
    kept_glob = set(c0 + i for i in idx)
    ng = None if rel is None else c0 + rel

    d16 = density(kept_glob, ng, 16, c0, c0 + c1) if ng is not None else (None, 0, 0)
    d64 = density(kept_glob, ng, 64, c0, c0 + c1) if ng is not None else (None, 0, 0)
    d256 = density(kept_glob, ng, 256, c0, c0 + c1) if ng is not None else (None, 0, 0)
    kept_before = (sum(1 for t in range(c0, ng) if t in kept_glob) if ng is not None else None)

    tail = list(range(c0 + c1, ids.shape[1]))
    ecache = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
    ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
    ok = "9AF4" in ans.upper().replace(" ", "")
    saved = 100 * (1 - h.cache_len(ecache) / ids.shape[1])
    del cache, ecache
    return dict(depth=depth, mode=mode, c1=c1, needle_rel=rel, rank=rank, keep=keep,
                needle_kept=(None if rel is None else rel in set(idx)),
                density_pm16=d16[0], density_pm64=d64[0], density_pm256=d256[0],
                kept_before=kept_before, kept_before_frac=(
                    None if not kept_before else round(kept_before / max(1, ng - c0), 3)),
                kv_saved_pct=round(saved, 2), pass_=bool(ok), answer=ans[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "density_sweep %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        t0 = time.time()
        r = run_one(h, d, args.mode, args.keep_frac)
        r["model"] = args.model
        r["sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print(json.dumps(r), flush=True)

    print()
    print("model=%s  mode=%s  keep=%.2f" % (args.model, args.mode, args.keep_frac))
    print("%-7s %-6s %-7s %-9s %-7s %-9s %-9s %-9s %-9s %s" % (
        "depth", "c1", "needle", "rank", "kept", "d16", "d64", "d256", "kept_before", "verdict"))
    for r in rows:
        print("%-7s %-6d %-7s %-9s %-7s %-9s %-9s %-9s %-9s %s" % (
            r["depth"], r["c1"], r["needle_rel"], r["rank"], r["needle_kept"],
            r["density_pm16"], r["density_pm64"], r["density_pm256"],
            r["kept_before_frac"], "PASS" if r["pass_"] else "fail"))

    pat = "".join("P" if r["pass_"] else "F" for r in rows)
    print()
    print("verdict pattern over %d depths: %s" % (len(rows), " ".join(pat)))
    print("fail count: %d of %d" % (sum(1 for r in rows if not r["pass_"]), len(rows)))

    print()
    print("=" * 74)
    print("DOES CONTEXT DENSITY SEPARATE PASS FROM FAIL?")
    print("=" * 74)
    for key in ["density_pm16", "density_pm64", "density_pm256", "kept_before_frac"]:
        ps = [r[key] for r in rows if r["pass_"] and r.get(key) is not None]
        fs = [r[key] for r in rows if not r["pass_"] and r.get(key) is not None]
        if not ps or not fs:
            continue
        print("  %-17s PASS mean %.3f (n=%d)   FAIL mean %.3f (n=%d)   delta %+.3f" % (
            key, sum(ps) / len(ps), len(ps), sum(fs) / len(fs), len(fs),
            sum(ps) / len(ps) - sum(fs) / len(fs)))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
