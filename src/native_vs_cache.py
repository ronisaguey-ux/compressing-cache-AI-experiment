#!/usr/bin/env python3
"""Is the failure the ROTATION ARITHMETIC, or the TOKEN LOSS itself?

Every eviction result so far runs the kept tokens through a MANIPULATED cache:
original KV, index-selected, keys re-rotated. If any step of that pipeline is subtly
wrong, the model sees a cache no honest forward pass would produce.

This compares three ways of asking the same question:

  full            all tokens, native forward                        (upper control)
  compact_native  ONLY the kept tokens, native forward, fresh      <- never tested
  compact_cache   only the kept tokens, via cache rotation          (the method)

`compact_native` is the honest version of the same content: the model is given exactly
the tokens we keep, at dense positions, computed the way it computes everything else. It
isolates content loss from cache surgery.

  full PASS, compact_native PASS, compact_cache FAIL  -> the ROTATION PIPELINE is wrong.
                                                         The content is sufficient; our
                                                         arithmetic is losing it.
  full PASS, compact_native FAIL                      -> losing 40% of the tokens is
                                                         inherently fatal for this task.
  all three PASS                                      -> the failure is elsewhere again.

Usage:
    python src/native_vs_cache.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.55
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def generate_native(h, token_ids, prompt_text, max_new=24):
    """Plain generate() on an explicit token id sequence -- no cache surgery at all."""
    with torch.no_grad():
        out = h.model.generate(
            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=h.tok.eos_token_id)
    new = out[0][len(token_ids):]
    return h.tok.decode(new, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.55")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "native_vs_cache %s" % args.model)
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

        cache, qcap, ids = T.prefill_capture_q(h, text)
        full_len = ids.shape[1]

        # 1. full, native
        ans_full = generate_native(h, ids_all + qids, T.build_question(T.Q2_TEXT))
        ok_full = "9AF4" in ans_full.upper().replace(" ", "")

        # 2. select
        scores = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        kept = None if rel is None else (rel in set(idx))

        # 3. compact_native: the kept tokens only, as an honest sequence
        row = ids[0].tolist()
        compact = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]
        ans_cn = generate_native(h, compact + qids, T.build_question(T.Q2_TEXT))
        ok_cn = "9AF4" in ans_cn.upper().replace(" ", "")

        # 4. compact_cache: same kept set, via the rotation pipeline
        tail = list(range(c0 + c1, ids.shape[1]))
        ec = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
        ans_cc, _ = h.generate(T.build_question(T.Q2_TEXT), ec)
        ok_cc = "9AF4" in ans_cc.upper().replace(" ", "")
        saved = 100 * (1 - h.cache_len(ec) / full_len)
        del ec, cache

        print("=" * 78)
        print("depth=%.2f  c1=%d  keep=%d (%.0f%% saved)  needle_rel=%s rank=%s kept=%s" % (
            d, c1, keep, saved, rel, rank, kept))
        print("  full          : %-4s %r" % ("PASS" if ok_full else "fail", ans_full[:58]))
        print("  compact_native: %-4s %r" % ("PASS" if ok_cn else "fail", ans_cn[:58]))
        print("  compact_cache : %-4s %r" % ("PASS" if ok_cc else "fail", ans_cc[:58]))
        if ok_full and ok_cn and not ok_cc:
            print("  ==> ROTATION PIPELINE is the defect: the same tokens work natively.")
        elif ok_full and not ok_cn:
            print("  ==> content loss is fatal: 40% fewer tokens is enough to break it.")
        elif ok_cn and not ok_cc:
            print("  ==> cache surgery loses information a native forward keeps.")
        print()

        rows.append(dict(depth=d, c1=c1, keep=keep, needle_rel=rel, rank=rank,
                         needle_kept=kept, kv_saved_pct=round(saved, 2),
                         full=bool(ok_full), compact_native=bool(ok_cn),
                         compact_cache=bool(ok_cc),
                         a_full=ans_full[:80], a_native=ans_cn[:80], a_cache=ans_cc[:80]))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in rows:
        print("  depth %-5s full=%-4s compact_native=%-4s compact_cache=%s" % (
            r["depth"], r["full"], r["compact_native"], r["compact_cache"]))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
