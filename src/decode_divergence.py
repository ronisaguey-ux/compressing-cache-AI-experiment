#!/usr/bin/env python3
"""WHERE does decode diverge? Compare the output distribution, pass vs fail.

Every instrument so far measured the CACHE (what is retained, how ranked, how dense, how
rotated). Four of them are now refuted: attention mass, sequence contiguity, context
density, RoPE geometry. None predicts the verdict.

This measures what the model DOES with the cache. For one depth, it runs the identical
question against the full cache and against the compacted cache for the first N generated
tokens and compares, at each step:

    top1 / top5 ids and the token strings     what was chosen
    top1_prob, entropy, margin                how confident it was
    kl(full || evict) over the vocab          how far the distributions moved

then it finds the FIRST step where the two disagree and reports the layer-by-layer
divergence at that step, so a failure can be localised rather than described.

Usage:
    python src/decode_divergence.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35
"""
import sys, os, io, json, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

STEPS = 16


def decode_trace(h, prompt_text, cache, steps=STEPS):
    """Greedy decode, recording the full distribution at every step."""
    enc = h.tok(prompt_text, return_tensors="pt")
    ids = enc["input_ids"]
    trace = []
    out = h.model(input_ids=ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :].float()
    for _ in range(steps):
        p = torch.softmax(logits, dim=-1)[0]
        top = torch.topk(p, 5)
        nxt = top.indices[0].view(1, 1)
        ent = float(-(p * (p + 1e-12).log()).sum())
        trace.append(dict(
            id=int(nxt.item()),
            tok=h.tok.decode(nxt[0], skip_special_tokens=False),
            top5=[h.tok.decode(torch.tensor([int(i)]), skip_special_tokens=False)
                  for i in top.indices.tolist()],
            top1_prob=round(float(top.values[0]), 5),
            top2_prob=round(float(top.values[1]), 5),
            margin=round(float(top.values[0] - top.values[1]), 5),
            entropy=round(ent, 4),
            probs=p.detach().clone(),
        ))
        out = h.model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return trace


def kl(a, b):
    a = a / a.sum()
    b = b / b.sum()
    return float((a * ((a + 1e-12).log() - (b + 1e-12).log())).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "decode_divergence %s" % args.model)
    h = T.Harness(args.model)

    out_rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        rel = T.locate_needle(h.tok, text)
        rel = None if rel is None else rel - c0
        qtext = T.build_question(T.Q2_TEXT)

        cache_full, qcap, ids = T.prefill_capture_q(h, text)
        full_len = ids.shape[1]

        tf = decode_trace(h, qtext, h.clone_cache(cache_full))

        scores = T.self_attn_scores(h, cache_full, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        idx = sorted(int(x) for x in torch.argsort(scores, descending=True)[:keep].tolist())
        tail = list(range(c0 + c1, ids.shape[1]))
        ce = T.build_arm_cache(h, cache_full, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
        te = decode_trace(h, qtext, ce)

        # first divergence
        first = None
        for i in range(min(len(tf), len(te))):
            if tf[i]["id"] != te[i]["id"]:
                first = i
                break

        print("=" * 78)
        print("depth=%.2f  c1=%d  needle_rel=%s  rank=%s  keep=%d  kv_saved=%.1f%%" % (
            d, c1, rel, int((torch.argsort(scores, descending=True) == rel).nonzero().flatten().item())
            if rel is not None else None, keep, 100 * (1 - h.cache_len(ce) / full_len)))
        print("=" * 78)
        print("%-4s %-26s %-26s %-9s %-9s %s" % (
            "step", "FULL top1", "EVICT top1", "kl", "d_top1p", "same"))
        for i in range(min(len(tf), len(te), 14)):
            a, b = tf[i], te[i]
            k = kl(a["probs"], b["probs"])
            same = a["id"] == b["id"]
            print("%-4d %-26s %-26s %-9.4f %-9.4f %s" % (
                i, repr(a["tok"])[:24], repr(b["tok"])[:24], k,
                a["top1_prob"] - b["top1_prob"], "yes" if same else "NO  <-- diverge"))
        print()
        print("full  text: %r" % h.tok.decode(torch.tensor([t["id"] for t in tf]), skip_special_tokens=True)[:70])
        print("evict text: %r" % h.tok.decode(torch.tensor([t["id"] for t in te]), skip_special_tokens=True)[:70])
        print("first divergence at step: %s" % first)
        print()

        out_rows.append(dict(
            depth=d, c1=c1, needle_rel=rel, keep=keep, first_divergence=first,
            full_top1=[t["tok"] for t in tf], evict_top1=[t["tok"] for t in te],
            kl=[round(kl(a["probs"], b["probs"]), 5) for a, b in zip(tf, te)],
            full_prob=[t["top1_prob"] for t in tf], evict_prob=[t["top1_prob"] for t in te],
            full_entropy=[t["entropy"] for t in tf], evict_entropy=[t["entropy"] for t in te],
        ))
        del cache_full, ce

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in out_rows:
        print("  depth %.2f  first divergence step %s  mean kl %.4f  full p1 %.3f -> evict p1 %.3f" % (
            r["depth"], r["first_divergence"],
            sum(r["kl"]) / max(1, len(r["kl"])),
            sum(r["full_prob"]) / max(1, len(r["full_prob"])),
            sum(r["evict_prob"]) / max(1, len(r["evict_prob"]))))
        print("            entropy full %.2f  evict %.2f" % (
            sum(r["full_entropy"]) / max(1, len(r["full_entropy"])),
            sum(r["evict_entropy"]) / max(1, len(r["evict_entropy"]))))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
