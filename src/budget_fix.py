#!/usr/bin/env python3
"""Does a larger keep budget fix the 7B shallow-depth selection failure?

THE OPEN DEFECT. On the 7B at depth 0.15 the attention selector misses the needle:
measured, needle rank 564 against a keep budget of 543 -- **short by 21**. On the 0.5B the
same selector preserves the needle at every depth once the budget reaches 45% (Finding 31).

Finding 31 reframed this as UNDER-BUDGETING rather than a broken scorer. If that is right,
raising `keep_frac` from 0.60 to ~0.64 should pull rank 564 inside the budget and the
retrieval should pass -- with no change to the selector at all. If it does not, the
"under-budgeting" explanation is wrong and the defect is real.

That is a cheap, decisive test and it is run entirely through the CORRECTED pipeline
(fresh forward over the shortened sequence), so nothing here is contaminated by the
rotation defect of Finding 18.

Also re-tests line pooling at this depth. The old 7B sweep showed line pooling passing where
baseline failed -- but that was through the broken pipeline, and Finding 20 concluded the
advantage was an artefact of reconstructing from neighbouring tokens. Running both here
settles whether line pooling still helps once the pipeline is correct.

Usage:
    python src/budget_fix.py --model Qwen/Qwen2.5-7B --depth 0.15
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"


def run_budget(h, text, c0, c1, keep_frac, mode, qids):
    """Corrected pipeline: score -> keep -> FRESH forward over the survivors -> decode."""
    h.chunk1_len = c1
    cache, qcap, ids = T.prefill_capture_q(h, text)
    sc = T.self_attn_scores(h, cache, qcap, c0, c1)
    N = ids.shape[1]
    keep = max(1, int(c1 * keep_frac))
    if mode == "line":
        # pool scores over the line each token belongs to, so a whole line survives together
        toks = h.tok(text, add_special_tokens=False)["input_ids"]
        # group chunk-1 tokens into lines by newline boundaries
        groups, cur = [], []
        for i in range(c1):
            cur.append(i)
            try:
                if "\n" in h.tok.decode([toks[c0 + i]]):
                    groups.append(cur); cur = []
            except Exception:
                pass
        if cur:
            groups.append(cur)
        gscore = [(sum(float(sc[i]) for i in g) / len(g), g) for g in groups]
        gscore.sort(key=lambda x: -x[0])
        picked, used = [], 0
        for _, g in gscore:
            if used + len(g) > keep:
                continue
            picked.extend(g); used += len(g)
        sel1 = sorted(picked)
    else:
        order = torch.argsort(sc, descending=True)
        sel1 = sorted(int(x) for x in order[:keep].tolist())
    kept = sorted(set(range(c0)) | set(c0 + i for i in sel1))

    na = T.locate_needle(h.tok, text)
    rel = None if na is None else na - c0
    needle_kept = (rel is not None and rel in set(sel1))
    rank = None
    if rel is not None:
        order_all = torch.argsort(sc, descending=True).tolist()
        rank = order_all.index(rel) + 1 if rel in order_all else None

    # FRESH forward over the survivors -- no rotation, no cache surgery
    row = ids[0].tolist()
    kept_ids = [row[j] for j in kept]
    dev = h.model.device
    with torch.no_grad():
        out = h.model(input_ids=torch.tensor([kept_ids + qids], device=dev), use_cache=True)
    c = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]
    for _ in range(23):
        with torch.no_grad():
            out = h.model(input_ids=nxt, past_key_values=c, use_cache=True)
        c = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    ans = h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
    del cache
    return dict(mode=mode, keep_frac=keep_frac, keep=keep, c1=c1, rank=rank,
                needle_kept=bool(needle_kept),
                kv_saved_pct=round(100 * (1 - len(kept) / N), 2),
                pass_=("9AF4" in ans.upper().replace(" ", "")), answer=ans[:70])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--depth", type=float, default=0.15)
    ap.add_argument("--fracs", default="0.60,0.63,0.66")
    ap.add_argument("--modes", default="baseline,line")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "budget_fix %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    text = PROMPT_PREFIX + log_at(args.depth) + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    na = T.locate_needle(h.tok, text)
    rel = None if na is None else na - c0

    print("=" * 86)
    print("BUDGET FIX  |  model=%s  depth=%.2f" % (args.model, args.depth))
    print("=" * 86)
    print("  c1=%d  needle at chunk-1 index %s (%.0f%% of the region)" % (
        c1, rel, 100 * rel / c1 if rel is not None else -1))
    print()
    print("  %-9s %-7s %-7s %-7s %-9s %-8s %s" % (
        "mode", "keep%", "keep", "rank", "needle", "saved", "verdict"))

    rows = []
    for mode in args.modes.split(","):
        for kf in [float(x) for x in args.fracs.split(",")]:
            r = run_budget(h, text, c0, c1, kf, mode, qids)
            rows.append(r)
            print("  %-9s %-7s %-7d %-7s %-9s %-8s %s" % (
                r["mode"], r["keep_frac"], r["keep"], r["rank"],
                "KEPT" if r["needle_kept"] else "LOST",
                "%.1f%%" % r["kv_saved_pct"],
                "PASS" if r["pass_"] else "fail"), flush=True)

    print()
    print("=" * 86)
    fixed = [r for r in rows if r["pass_"]]
    print("  any PASS: %s" % ("yes -- " + ", ".join(
        "%s@%.2f" % (r["mode"], r["keep_frac"]) for r in fixed) if fixed else "NO"))
    if fixed:
        best = min(fixed, key=lambda r: r["keep"])
        print("  cheapest passing config: %s keep=%.2f (keep %d of %d, %.1f%% KV saved)" % (
            best["mode"], best["keep_frac"], best["keep"], best["c1"], best["kv_saved_pct"]))
    print()
    print("  If raising the budget turns this PASS, Finding 31's 'under-budgeting'")
    print("  explanation is confirmed and there is no scorer defect to fix.")
    print("  If it does not, the shallow-depth miss is a real selection failure.")

    if args.json_out:
        with open(args.json_out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, depth=args.depth, model=args.model)) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
