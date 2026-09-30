#!/usr/bin/env python3
"""Is the depth failure about needle POSITION, or about log LENGTH?

Finding 8 concluded "the recipe is not robust to needle position", but depth_sweep
changes TWO things at once: it moves the needle AND grows the log (c1 906 -> 1960 ->
2189). So "position" was never isolated, and Finding 12's clean two-cause split
inherits that confound.

This holds the log length FIXED and moves only the needle. Same content, same token
count, same keep budget -- only the needle's index changes. Compare against the
recorded depth_sweep row at the same fraction:

  pos_vs_len PASS, depth_sweep fail  -> position alone is not the problem; LENGTH is
  pos_vs_len fail, depth_sweep pass  -> length alone is not the problem; POSITION is
  both fail                          -> neither; something else
  both pass                          -> the recorded failure was a length artifact

Context lines are drawn from the same pools depth_sweep uses (PRE before, POST after)
so the only thing that varies is how many of each. Total context lines stay at
CTX_LINES, which is the count the 0.15 log already has -- so every row here is the
same size as the shallow failing case.

Usage:
    python src/pos_vs_len.py --model Qwen/Qwen2.5-0.5B --fracs 0.15,0.35,0.55,0.75
"""
import sys, os, io, json, time, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, PRE, POST, NEEDLE

CTX_LINES = 56            # 17 PRE + 39 POST at the 0.15 layout -- held constant


def take(pool, n):
    return [pool[i % len(pool)] for i in range(n)]


def log_fixed(frac, ctx_lines=CTX_LINES):
    """Constant-length log with the needle at `frac` of the context."""
    k = int(round(ctx_lines * frac))
    k = max(1, min(ctx_lines - 1, k))
    before = take(PRE, k)
    after = take(POST, ctx_lines - k)
    return "\n".join(before + NEEDLE + after), k


def run_one(h, frac, keep_frac=0.60):
    log, k = log_fixed(frac)
    text = SYS + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n", add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    na = T.locate_needle(h.tok, text)
    rel = None if na is None else na - c0

    cache, qcap, ids = T.prefill_capture_q(h, text)
    scores = T.self_attn_scores(h, cache, qcap, c0, c1)
    keep = max(1, int(c1 * keep_frac))
    order = torch.argsort(scores, descending=True)
    idx = sorted(int(x) for x in order[:keep].tolist())
    rank = None if rel is None else int((order == rel).nonzero().flatten().item())
    kept = None if rel is None else (rel in set(idx))

    tail = list(range(c0 + c1, ids.shape[1]))
    ecache = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
    ans, _ = h.generate(T.build_question(T.Q2_TEXT), ecache)
    ok = "9AF4" in ans.upper().replace(" ", "")
    saved = 100 * (1 - h.cache_len(ecache) / ids.shape[1])
    del cache, ecache
    return dict(frac=frac, ctx_before_lines=k, c0=c0, c1=c1, needle_rel=rel, rank=rank,
                rank_frac=(None if rank is None else round(rank / c1, 3)),
                needle_kept=kept, kv_saved_pct=round(saved, 2), pass_=bool(ok),
                answer=ans[:90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fracs", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "pos_vs_len %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for f in [float(x) for x in args.fracs.split(",")]:
        t0 = time.time()
        r = run_one(h, f, args.keep_frac)
        r["model"] = args.model
        r["sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print(json.dumps(r), flush=True)

    # recorded depth_sweep rows for the same fractions, baseline mode
    rec = {}
    try:
        for line in open("data/run_7b.jsonl" if "7B" in args.model else "data/depth_sweep.jsonl"):
            d = json.loads(line)
            if d.get("mode") == "baseline":
                rec[round(d["depth"], 3)] = d
    except Exception:
        pass

    print()
    print("model=%s  keep=%.2f  log length HELD CONSTANT at %d context lines"
          % (args.model, args.keep_frac, CTX_LINES))
    print()
    print("%-7s %-8s %-8s %-11s %-8s %-7s %-8s %s" % (
        "frac", "c1", "needle", "rank", "kept", "saved", "posonly", "recorded(depth_sweep)"))
    for r in rows:
        d = rec.get(round(r["frac"], 3))
        got = ("%s r=%s" % ("PASS" if d["pass_"] else "fail", d.get("rank"))) if d else "--"
        print("%-7s %-8d %-8s %-11s %-8s %-7s %-8s %s" % (
            r["frac"], r["c1"], r["needle_rel"], r["rank"], r["needle_kept"],
            r["kv_saved_pct"], "PASS" if r["pass_"] else "fail", got))

    npass = sum(1 for r in rows if r["pass_"])
    print()
    print("fixed-length: %d of %d pass" % (npass, len(rows)))
    lens = sorted(set(r["c1"] for r in rows))
    print("c1 spread: %s  (%s)" % (lens, "constant" if len(lens) == 1 else "VARIES - test invalid"))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
