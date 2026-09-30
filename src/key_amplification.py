#!/usr/bin/env python3
"""Is the needle's signal present but BELOW THRESHOLD?

decode_divergence showed the compacted cache tracks the full cache exactly through
"The failure code in the build log is 0x" and then diverges at the first digit: full
emits '9', compacted emits '0'. Entropy at that step is 1.96 (compacted) vs 0.65 (full).
The model has the frame, has the '0x' prefix, and falls back to a generic digit.

If the needle's contribution is present but too weak to outvote the model's prior, then
AMPLIFYING it should recover the answer without adding any information. This scales the
needle's cached key vectors before attention, which scales its pre-softmax logit:

    logit = q . (alpha * k_needle) / sqrt(d)

alpha=1.0 is the current behaviour. If some alpha>1 flips the verdict to PASS, the
information was always there and the failure is a signal-strength/readout problem, not a
missing-content problem. If no alpha recovers it, the needle is not merely weak -- it is
not being read.

Measured per depth, and it also reports WHERE the first divergence moves as alpha rises.

Usage:
    python src/key_amplification.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.55
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

ALPHAS = [1.0, 1.5, 2.0, 3.0, 5.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.55")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "key_amplification %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0

        cache, qcap, ids = T.prefill_capture_q(h, text)
        scores = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        # the needle's index inside the compacted, renumbered cache
        needle_new = (sorted(idx).index(rel) + c0) if (rel is not None and rel in idx) else None
        tail = list(range(c0 + c1, ids.shape[1]))

        for a in ALPHAS:
            ec = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                                   c0, c0, c1, tail)
            if needle_new is not None and a != 1.0:
                for li in range(len(ec.layers)):
                    ec.layers[li].keys[:, :, needle_new, :] *= a
            ans, _ = h.generate(T.build_question(T.Q2_TEXT), ec)
            ok = "9AF4" in ans.upper().replace(" ", "")
            r = dict(depth=d, alpha=a, c1=c1, rank=rank, needle_new=needle_new,
                     pass_=bool(ok), answer=ans[:70])
            rows.append(r)
            print(json.dumps(r), flush=True)
            del ec

    print()
    print("model=%s  keep=%.2f" % (args.model, args.keep_frac))
    print()
    print("%-7s %-7s %-8s %s" % ("depth", "alpha", "rank", "verdict / answer"))
    for r in rows:
        print("%-7s %-7s %-8s %-4s %r" % (
            r["depth"], r["alpha"], r["rank"], "PASS" if r["pass_"] else "fail", r["answer"][:52]))

    print()
    print("=" * 68)
    print("DID AMPLIFICATION RECOVER IT?")
    print("=" * 68)
    for d in sorted(set(r["depth"] for r in rows)):
        sub = [r for r in rows if r["depth"] == d]
        base = sub[0]
        rec = [r for r in sub[1:] if r["pass_"]]
        print("  depth %-5s alpha=1.0 %s  ->  %s" % (
            d, "PASS" if base["pass_"] else "fail",
            ("RECOVERED at alpha=%s" % rec[0]["alpha"]) if rec else "no alpha recovers it"))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
