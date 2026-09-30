#!/usr/bin/env python3
"""Does physically displacing the needle from the attention sink fix depth 15%?

The sink measurement showed 8 tokens absorb ~54% of all attention at every depth and
that position 0 is the top-1 key everywhere -- but ALSO that the needle receives MORE
mass at the depth where it fails (0.0014 at 15%) than where it passes (0.0006 at 55%).
So mass alone does not explain the failure. This test asks the causal question instead
of the correlational one: if the needle's DISTANCE from the sink basin is what matters,
inserting neutral filler between them should flip the verdict even though the needle's
own attention mass barely changes.

Two placements, because they distinguish two different stories:

  sys     filler PREPENDED before the system chunk. This moves the sink basin into the
          filler and shifts the whole prompt away from it. If this fixes retrieval, the
          problem is proximity to the sink at position 0.
  needle  filler inserted immediately BEFORE the needle inside the log. The sink stays
          where it is; only the needle moves. If THIS fixes it but `sys` does not, the
          effect is local phase/adjacency, not the global basin.

Baseline mode only (all-layer self-attention, top-k), keep 0.60, rotation ON -- the exact
configuration that fails at 0.15 and passes at 0.55. If padding makes 0.15 behave like
0.55, the mechanism is positional.

Usage:
    python src/sink_pad.py --model Qwen/Qwen2.5-0.5B --depth 0.15 --pads 0,16,32
"""
import sys, os, io, json, time, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

FILLER = " filler"          # neutral: carries no task-relevant content


def build(depth, pad, where, h):
    """Return (text, note) with `pad` filler tokens placed as `where`."""
    log = log_at(depth)
    body = "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    fill = FILLER * pad
    if where == "sys":
        # prepended before the system prompt: the sink basin lands in the filler
        return fill + SYS + body, "prepended %d filler tokens before the system chunk" % pad
    if where == "needle":
        # insert immediately BEFORE the needle line, inside the log
        pre, needle, post = log_at_parts(depth)
        log2 = pre + fill + "\n" + needle + "\n" + post
        return SYS + "<|im_start|>user\n<build_log>\n" + log2 + "\n</build_log><|im_end|>\n", \
               "inserted %d filler tokens immediately before the needle" % pad
    return SYS + body, "no padding"


def log_at_parts(frac):
    """Mirror depth_sweep.log_at but return the three pieces separately.

    PRE_LINES/POST_LINES must be multiplied by 3 here exactly as depth_sweep does
    -- omitting it silently builds a log a third of the right length, which makes
    the needle arm incomparable to the sys arm and to the recorded sweep.
    """
    PRE = T.PRE_LINES * 3
    POST = T.POST_LINES * 3
    n = len(PRE) + len(T.NEEDLE_LINES) + len(POST)
    want = int(n * frac)
    before = min(len(PRE), max(0, want))
    after = max(0, n - before - len(T.NEEDLE_LINES))
    return ("\n".join(PRE[:before]),
            "\n".join(T.NEEDLE_LINES),
            "\n".join(POST[:after]))


def run_one(h, depth, pad, where, keep_frac=0.60):
    text, note = build(depth, pad, where, h)
    # c0 is everything before the log: compute it from the ACTUAL text (the filler
    # moves it), never from SYS, or the chunk-1 offset is wrong in the sys arm.
    marker = "<|im_start|>user\n<build_log>\n"
    head = text.split(marker)[0] + marker
    c0 = len(h.tok(head, add_special_tokens=False)["input_ids"])
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
    return dict(depth=depth, pad=pad, where=where, c0=c0, c1=c1, needle_rel=rel,
                rank=rank, needle_kept=kept, kv_saved_pct=round(saved, 2),
                pass_=bool(ok), answer=ans[:90], note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depth", type=float, default=0.15)
    ap.add_argument("--pads", default="0,16,32")
    ap.add_argument("--where", default="both", choices=["sys", "needle", "both"])
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    need = 9000 if "7B" in args.model else 3500
    T.require_memory(need, "sink_pad %s" % args.model)
    h = T.Harness(args.model)

    wheres = ["sys", "needle"] if args.where == "both" else [args.where]
    rows = []
    for where in wheres:
        for pad in [int(x) for x in args.pads.split(",")]:
            t0 = time.time()
            r = run_one(h, args.depth, pad, where, args.keep_frac)
            r["model"] = args.model
            r["sec"] = round(time.time() - t0, 1)
            rows.append(r)
            print(json.dumps(r), flush=True)

    print()
    print("model=%s  depth=%.2f  keep=%.2f" % (args.model, args.depth, args.keep_frac))
    print("%-9s %-6s %-7s %-9s %-8s %-7s %s" % (
        "where", "pad", "needle_rel", "rank", "kept", "saved", "verdict"))
    for r in rows:
        print("%-9s %-6d %-11s %-9s %-8s %-7s %s" % (
            r["where"], r["pad"], r["needle_rel"], r["rank"], r["needle_kept"],
            r["kv_saved_pct"], "PASS" if r["pass_"] else "fail"))
    print()
    fail0 = [r for r in rows if r["pad"] == 0]
    base_fail = all(not r["pass_"] for r in fail0) if fail0 else None
    flipped = [r for r in rows if r["pad"] > 0 and r["pass_"]]
    print("baseline (pad=0) all fail: %s" % base_fail)
    print("padding flipped a fail -> PASS: %s" % (bool(flipped),))
    if flipped:
        print("=> sink displacement is CAUSAL: distance from the basin, not attention mass.")
    else:
        print("=> padding does NOT fix it; sink proximity is not the mechanism.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
