#!/usr/bin/env python3
"""Re-score the selectors against the CORRECTED pipeline.

line pooling (Finding 12) and the mid-layer band (Finding 17) were both discovered inside
the broken rotation pipeline. Once the pipeline is the corrected one -- select, then
re-prefill the compacted sequence natively -- their advantage may change, because they were
never being measured against clean tensors.

Selectors, all at the same keep budget, all through the corrected path:
    baseline   all-layer self-attention (the Finding 7/8 recipe)
    line       line pooling over baseline scores
    mid        baseline restricted to layers [0.35L, 0.75L)
    random     uniform random, seeded    (control: must not pass)

Also runs the OLD cache path for `baseline` at each depth, so the two pipelines are visible
side by side on identical rows.

A selector that passes where baseline fails, natively, is a real result. A selector that only
won through the broken pipeline was measuring the defect.

Usage:
    python src/selectors_corrected.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
"""
import sys, os, json, math, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def band_scores(h, cache, qcap, c0, c1, lo, hi, block=256):
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


def gen_native(h, token_ids, max_new=24):
    with torch.no_grad():
        out = h.model.generate(torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
                               max_new_tokens=max_new, do_sample=False,
                               pad_token_id=h.tok.eos_token_id)
    return h.tok.decode(out[0][len(token_ids):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "selectors_corrected %s" % args.model)
    h = T.Harness(args.model)
    L = len(h.model.model.layers)
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
        row = None

        # scores per family (each needs its own q capture; qcap is consumed)
        cache, qcap, ids = T.prefill_capture_q(h, text)
        row = ids[0].tolist()
        full_len = ids.shape[1]
        base_scores = T.self_attn_scores(h, cache, qcap, c0, c1)

        cache2, qcap2, _ = T.prefill_capture_q(h, text)
        mid_scores = band_scores(h, cache2, qcap2, c0, c1, int(0.35 * L), int(0.75 * L))
        spans = T.line_spans(h.tok, text, c0, c1)
        del cache2

        keep = max(1, int(c1 * args.keep_frac))
        tail = list(range(c0 + c1, ids.shape[1]))

        def sel_for(name):
            if name == "baseline":
                return base_scores, None
            if name == "mid":
                return mid_scores, None
            if name == "line":
                return base_scores, "line"
            return None, None

        for name in ["baseline", "line", "mid", "random"]:
            if name == "random":
                rng = random.Random(1234)
                idx = sorted(rng.sample(range(c1), min(keep, c1)))
                rank = None
            else:
                sc, lp = sel_for(name)
                order = torch.argsort(sc, descending=True)
                rank = None if rel is None else int((order == rel).nonzero().flatten().item())
                if lp == "line":
                    idx = T.sel_line_pooled(sc, spans, keep)
                else:
                    idx = sorted(int(x) for x in order[:keep].tolist())
            kept = None if rel is None else (rel in set(idx))
            compact = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]
            ans = gen_native(h, compact + qids)
            ok = "9AF4" in ans.upper().replace(" ", "")
            r = dict(depth=d, selector=name, pipeline="native", c1=c1, keep=keep,
                     rank=rank, needle_kept=kept, pass_=bool(ok), answer=ans[:70])
            rows.append(r)
            print(json.dumps(r), flush=True)

        # old pipeline on the same rows, for contrast
        idxb = sorted(int(x) for x in torch.argsort(base_scores, descending=True)[:keep].tolist())
        ec = T.build_arm_cache(h, cache, dict(kind="evict", keep=idxb, rot=True), c0, c0, c1, tail)
        ans_old, _ = h.generate(T.build_question(T.Q2_TEXT), ec)
        ok_old = "9AF4" in ans_old.upper().replace(" ", "")
        rows.append(dict(depth=d, selector="baseline", pipeline="cache", c1=c1, keep=keep,
                         rank=None, needle_kept=None, pass_=bool(ok_old), answer=ans_old[:70]))
        print(json.dumps(rows[-1]), flush=True)
        del cache, ec

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print()
    print("%-7s %-10s %-9s %-8s %-7s %-8s %s" % (
        "depth", "selector", "pipeline", "rank", "kept", "saved", "verdict"))
    for r in rows:
        print("%-7s %-10s %-9s %-8s %-7s %-8s %s" % (
            r["depth"], r["selector"], r["pipeline"], r["rank"], r["needle_kept"],
            "", "PASS" if r["pass_"] else "fail"))

    print()
    print("=" * 72)
    print("PASS COUNT BY SELECTOR, CORRECTED PIPELINE ONLY")
    print("=" * 72)
    for s in ["baseline", "line", "mid", "random"]:
        sub = [r for r in rows if r["selector"] == s and r["pipeline"] == "native"]
        p = sum(1 for r in sub if r["pass_"])
        pat = " ".join("P" if r["pass_"] else "F" for r in sub)
        print("  %-10s %d/%d   %s" % (s, p, len(sub), pat))
    old = [r for r in rows if r["pipeline"] == "cache"]
    print("  %-10s %d/%d   %s   (old rotation pipeline)" % (
        "cache", sum(1 for r in old if r["pass_"]), len(old),
        " ".join("P" if r["pass_"] else "F" for r in old)))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
