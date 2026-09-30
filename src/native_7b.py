#!/usr/bin/env python3
"""Does the corrected method (re-prefill, no rotation) work on the 7B?

Finding 18 established on the 0.5B that the retained tokens are sufficient when computed
natively and insufficient through the rotation pipeline. The 7B gave the cleanest signal in
the whole project (Finding 12), so the finding has to be confirmed there before it is the
headline.

Runs three arms at each depth, 8-bit, one process:
    full            native, all tokens
    compact_native  native over the SELECTED tokens only   <- the corrected method
    compact_cache   shipped select+rotate path            <- the old method

Usage:
    python src/native_7b.py --depths 0.15,0.35,0.55
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def gen_native(h, token_ids, max_new=24):
    with torch.no_grad():
        out = h.model.generate(torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
                               max_new_tokens=max_new, do_sample=False,
                               pad_token_id=h.tok.eos_token_id)
    return h.tok.decode(out[0][len(token_ids):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--depths", default="0.15,0.35,0.55")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000, "native_7b %s" % args.model)
    h = T.Harness(args.model)
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        ids_all = h.tok(text, add_special_tokens=False)["input_ids"]
        c1 = len(ids_all) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0

        t0 = time.time()
        cache, qcap, ids = T.prefill_capture_q(h, text)
        full_len = ids.shape[1]
        ans_full = gen_native(h, ids_all + qids)
        ok_full = "9AF4" in ans_full.upper().replace(" ", "")

        scores = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        kept = None if rel is None else (rel in set(idx))

        row = ids[0].tolist()
        compact = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]
        ans_nat = gen_native(h, compact + qids)
        ok_nat = "9AF4" in ans_nat.upper().replace(" ", "")

        tail = list(range(c0 + c1, ids.shape[1]))
        ec = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True), c0, c0, c1, tail)
        ans_cc, _ = h.generate(T.build_question(T.Q2_TEXT), ec)
        ok_cc = "9AF4" in ans_cc.upper().replace(" ", "")
        saved = 100 * (1 - h.cache_len(ec) / full_len)
        del ec, cache

        r = dict(depth=d, c1=c1, keep=keep, needle_rel=rel, rank=rank, needle_kept=kept,
                 kv_saved_pct=round(saved, 2), full=bool(ok_full),
                 compact_native=bool(ok_nat), compact_cache=bool(ok_cc),
                 a_full=ans_full[:70], a_native=ans_nat[:70], a_cache=ans_cc[:70],
                 sec=round(time.time() - t0, 1))
        rows.append(r)
        print(json.dumps(r), flush=True)

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print("%-7s %-8s %-7s %-6s %-8s %-9s %s" % (
        "depth", "saved", "rank", "kept", "full", "native", "cache"))
    for r in rows:
        print("%-7s %-8s %-7s %-6s %-8s %-9s %s" % (
            r["depth"], r["kv_saved_pct"], r["rank"], r["needle_kept"],
            "PASS" if r["full"] else "fail", "PASS" if r["compact_native"] else "fail",
            "PASS" if r["compact_cache"] else "fail"))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
