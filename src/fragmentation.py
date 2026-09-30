#!/usr/bin/env python3
"""Is the failure SEQUENCE FRAGMENTATION rather than retrieval?

The stored rows are unambiguous about the symptom and silent about the cause. Among the
38 rows where the needle WAS retained, 20 passed; every genuine failure is the correct
sentence FRAME with a wrong VALUE:

    "The failure code in the build log is 1."
    "The failure code in the build log is 0x9999999999999"
    "The failure code in the build log was 00:03:41."

The needle is the multi-token sequence `0x9AF4_STACK_FAIL`. Copying it verbatim requires
the model to see it INTACT. Token-level top-k keeps the highest-scoring individual tokens
and drops the tokens between them, so the code can arrive as fragments -- and a language
model handed `0x` and `4` produces a plausible digit rather than reporting a gap.

This measures that directly. For the needle's own line it reports:

    needle_tok_kept   is the marker token itself retained
    line_kept_frac    fraction of the needle LINE's tokens retained
    max_run           longest CONTIGUOUS retained run inside the needle line
    line_intact       the whole line retained, in order, no holes

and correlates them with the verdict. Prediction if the hypothesis holds: `line_intact` /
`max_run` separate pass from fail, while `needle_tok_kept` does not (Finding 9).

Usage:
    python src/fragmentation.py --model Qwen/Qwen2.5-0.5B --modes baseline,line
"""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def needle_line_span(tok, text, c0, c1):
    """Global token span of the needle's OWN line (the FATAL line)."""
    marker = T.NEEDLE_LINES[-1]                      # 'FATAL: link aborted, failure code 0x9AF4...'
    ci = text.find(marker)
    if ci < 0:
        return None
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    lo = ci
    hi = ci + len(marker)
    span = [t for t, (a, b) in enumerate(offs)
            if b > lo and a < hi and c0 <= t < c0 + c1]
    return (min(span), max(span) + 1) if span else None


def longest_run(kept_set, span):
    """Longest contiguous run of kept tokens within [a, b)."""
    a, b = span
    best = cur = 0
    for t in range(a, b):
        if t in kept_set:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_one(h, depth, mode, keep_frac=0.60):
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
    if mode == "posnorm":
        used = T.sel_position_normalized(scores, c1)
    if mode in ("line", "both"):
        spans = T.line_spans(h.tok, text, c0, c1)
        if mode == "both":
            used = T.sel_position_normalized(scores, c1)
    if spans is not None:
        idx = T.sel_line_pooled(used, spans, keep)
    else:
        order = torch.argsort(used, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())

    order = torch.argsort(used, descending=True)
    rank = None if rel is None else int((order == rel).nonzero().flatten().item())
    kept_glob = set(c0 + i for i in idx)

    span = needle_line_span(h.tok, text, c0, c1)
    if span is None:
        line_tok_kept = line_kept_frac = maxrun = line_intact = None
    else:
        a, b = span
        n = b - a
        got = sum(1 for t in range(a, b) if t in kept_glob)
        line_tok_kept = got
        line_kept_frac = round(got / n, 3)
        maxrun = longest_run(kept_glob, span)
        line_intact = bool(got == n)

    tail = list(range(c0 + c1, ids.shape[1]))
    ecache = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
    ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
    ok = "9AF4" in ans.upper().replace(" ", "")
    saved = 100 * (1 - h.cache_len(ecache) / ids.shape[1])
    del cache, ecache
    return dict(depth=depth, mode=mode, c1=c1, needle_rel=rel, rank=rank, keep=keep,
                needle_kept=(None if rel is None else rel in set(idx)),
                line_span=(None if span is None else [span[0] - c0, span[1] - c0]),
                line_tokens=(None if span is None else span[1] - span[0]),
                line_tokens_kept=line_tok_kept, line_kept_frac=line_kept_frac,
                max_run=maxrun, line_intact=line_intact,
                kv_saved_pct=round(saved, 2), pass_=bool(ok), answer=ans[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--modes", default="baseline,line,posnorm")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "fragmentation %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        for m in args.modes.split(","):
            t0 = time.time()
            r = run_one(h, d, m, args.keep_frac)
            r["model"] = args.model
            r["sec"] = round(time.time() - t0, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print("%-7s %-9s %-8s %-9s %-11s %-8s %-9s %s" % (
        "depth", "mode", "ntok", "keptfrac", "max_run", "intact", "needle", "verdict"))
    for r in rows:
        print("%-7s %-9s %-8s %-9s %-11s %-8s %-9s %s" % (
            r["depth"], r["mode"], r["line_tokens"], r["line_kept_frac"],
            r["max_run"], r["line_intact"], r["needle_kept"],
            "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 74)
    print("DOES CONTIGUITY PREDICT SUCCESS WHERE RETENTION DOES NOT?")
    print("=" * 74)
    for label, key, thresh in [("line_intact", "line_intact", True),
                               ("max_run >= 3", "max_run", 3),
                               ("needle_kept", "needle_kept", True)]:
        hit = [r for r in rows if r.get(key) is not None and
               (r[key] is True if thresh is True else (r[key] is not False and r[key] >= thresh))]
        miss = [r for r in rows if r not in hit]
        ph = sum(1 for r in hit if r["pass_"])
        pm = sum(1 for r in miss if r["pass_"])
        print("  %-14s  TRUE : %2d/%2d pass (%.0f%%)   FALSE/n-a: %2d/%2d pass (%.0f%%)" % (
            label, ph, len(hit), 100 * ph / max(1, len(hit)),
            pm, len(miss), 100 * pm / max(1, len(miss))))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
