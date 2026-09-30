#!/usr/bin/env python3
"""Do the shipped cache's tensors MATCH what a native compact forward produces?

native_vs_cache showed the same kept tokens PASS natively and FAIL through the cache
path. The direct question is therefore a tensor question: after resize + apply_delta, is
the cache the model receives the same one it would have built itself?

No rotation gymnastics -- take the needle's new index in both caches and compare:

    cos(shipped_K[needle_new], native_K[needle_new])
    cos(shipped_V[needle_new], native_V[needle_new])

per layer, plus the mean over all retained positions. A cosine well below 1 at the
needle while the rest of the cache matches localises the defect to the needle's key,
which is exactly where dense re-rotation acts.

Usage:
    python src/cache_vs_native.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.55
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def cos(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    na, nb = a.norm(), b.norm()
    return None if (na == 0 or nb == 0) else float((a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.55")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "cache_vs_native %s" % args.model)
    h = T.Harness(args.model)
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
        scores = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        idx = sorted(int(x) for x in torch.argsort(scores, descending=True)[:keep].tolist())
        rank = None if rel is None else int((torch.argsort(scores, descending=True) == rel).nonzero().flatten().item())
        row = ids[0].tolist()
        compact_ids = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]

        tail = list(range(c0 + c1, ids.shape[1]))
        shipped = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True),
                                    c0, c0, c1, tail)
        with torch.no_grad():
            out = h.model(input_ids=torch.tensor(compact_ids).unsqueeze(0), use_cache=True)
        native = out.past_key_values

        # needle's index in the compacted layout
        npos = (c0 + sorted(idx).index(rel)) if (rel is not None and rel in idx) else None
        print("=" * 78)
        print("depth=%.2f c1=%d keep=%d rank=%s needle_in_keep=%s compact_len=%d native_len=%d"
              % (d, c1, keep, rank, rel in idx if rel is not None else None,
                 h.cache_len(shipped), h.cache_len(native)))
        kl, vl, kl_all, vl_all = [], [], [], []
        for li in range(len(shipped.layers)):
            sk = shipped.layers[li].keys; nk = native.layers[li].keys
            sv = shipped.layers[li].values; nv = native.layers[li].values
            n = min(sk.shape[2], nk.shape[2])
            kl.append(cos(sk[:, :, :n, :], nk[:, :, :n, :]))
            vl.append(cos(sv[:, :, :n, :], nv[:, :, :n, :]))
            if npos is not None and npos < n:
                kl_all.append((li, cos(sk[:, :, npos, :], nk[:, :, npos, :]),
                               cos(sv[:, :, npos, :], nv[:, :, npos, :])))
        km = sum(x for x in kl if x is not None) / max(1, len([x for x in kl if x is not None]))
        vm = sum(x for x in vl if x is not None) / max(1, len([x for x in vl if x is not None]))
        print("  ALL retained positions: cos K=%.5f  V=%.5f" % (km, vm))
        if kl_all:
            nk_m = sum(x[1] for x in kl_all) / len(kl_all)
            nv_m = sum(x[2] for x in kl_all) / len(kl_all)
            print("  NEEDLE position %d:      cos K=%.5f  V=%.5f" % (npos, nk_m, nv_m))
            worst = sorted(kl_all, key=lambda x: x[1])[:3]
            print("  worst needle-K layers: %s" % [(l, round(k, 4)) for l, k, _ in worst])
        rows.append(dict(depth=d, c1=c1, keep=keep, rank=rank, npos=npos,
                         k_match_all=round(km, 5), v_match_all=round(vm, 5),
                         k_match_needle=(round(nk_m, 5) if kl_all else None),
                         v_match_needle=(round(nv_m, 5) if kl_all else None)))
        del cache, shipped, native

    print()
    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
