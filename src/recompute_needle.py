#!/usr/bin/env python3
"""THE DECISIVE TEST: recompute the needle's KV under the COMPACTED prefix.

Both research agents converged on the value cache. Agent 1's proposed probe (cosine
between V_compacted and V_full) is VACUOUS IN THIS HARNESS: `resize` does index_select
on keys AND values, and `apply_delta` rotates only keys, so V is byte-identical by
construction. Cosine would be exactly 1.0 and it would prove nothing.

The hypothesis is still alive in the form Agent 2 stated it: V is unchanged, but it was
COMPUTED under the original prefix, and a token's KV encodes its antecedents
(KV(B|A) != KV(B|nothing)). Removing the antecedent tokens does not un-bake that
conditioning -- it leaves a value that was written for a context that no longer exists.
That is Kamera's conditioning loss, and it is testable.

The test: take the kept tokens, run a fresh forward pass over the COMPACTED sequence, and
splice in the needle's K/V as computed in that compacted context. Dense re-rotation makes
the positions agree, so this isolates conditioning from geometry.

    recompute OFF -> the depth-sweep behaviour (expected fail at 0.15)
    recompute ON  -> conditioning repaired; if this still fails, conditioning loss
                     is REFUTED as the mechanism

Also reports, per layer, the cosine between the cached needle KV and the recomputed
needle KV, so the SIZE of the conditioning shift is measured rather than assumed.

Usage:
    python src/recompute_needle.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35
"""
import sys, os, io, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def fresh_cache_on(h, token_ids_row):
    """Forward pass over an explicit token id sequence, returning a fresh cache."""
    with torch.no_grad():
        out = h.model(input_ids=token_ids_row.unsqueeze(0), use_cache=True)
    return out.past_key_values


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return None
    return float((a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "recompute_needle %s" % args.model)
    h = T.Harness(args.model)

    out_rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        text = SYS + "<|im_start|>user\n<build_log>\n" + log_at(d) + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(SYS + "<|im_start|>user\n<build_log>\n",
                       add_special_tokens=False)["input_ids"])
        ids_all = h.tok(text, add_special_tokens=False)["input_ids"]
        c1 = len(ids_all) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = None if na is None else na - c0
        ng = None if rel is None else c0 + rel

        t0 = time.time()
        cache_full, qcap, ids = T.prefill_capture_q(h, text)
        full_len = ids.shape[1]

        scores = T.self_attn_scores(h, cache_full, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        order = torch.argsort(scores, descending=True)
        idx = sorted(int(x) for x in order[:keep].tolist())
        rank = None if rel is None else int((order == rel).nonzero().flatten().item())
        kept = None if rel is None else (rel in set(idx))
        tail = list(range(c0 + c1, ids.shape[1]))

        # ---- A: normal evict + re-rotation ----
        ce = T.build_arm_cache(h, cache_full, dict(kind="evict", keep=idx, rot=True),
                               c0, c0, c1, tail)
        ans_a, _ = h.generate(T.build_question(T.Q2_TEXT), ce)
        ok_a = "9AF4" in ans_a.upper().replace(" ", "")

        # ---- B: recompute the needle's KV under the COMPACTED prefix ----
        # keep order must be ascending; needle's new index is its position in idx
        pos_in_keep = rel if rel in idx else None
        row = ids[0].tolist()
        compact_ids = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]
        needle_new = (c0 + sorted(idx).index(rel)) if (rel is not None and rel in idx) else None

        kcos = vcos = None
        ce_b = T.build_arm_cache(h, cache_full, dict(kind="evict", keep=idx, rot=True),
                                 c0, c0, c1, tail)
        if needle_new is not None:
            cf = fresh_cache_on(h, torch.tensor(compact_ids, dtype=torch.long))
            per_layer = []
            for li in range(len(ce_b.layers)):
                k_old = ce_b.layers[li].keys[:, :, needle_new, :]
                k_new = cf.layers[li].keys[:, :, needle_new, :]
                v_old = ce_b.layers[li].values[:, :, needle_new, :]
                v_new = cf.layers[li].values[:, :, needle_new, :]
                kc = cos(k_old, k_new)
                vc = cos(v_old, v_new)
                per_layer.append((li, kc, vc))
                # splice the RECOMPUTED k and v into the working cache
                ce_b.layers[li].keys[:, :, needle_new, :] = k_new
                ce_b.layers[li].values[:, :, needle_new, :] = v_new
            kcos = round(sum(x[1] for x in per_layer) / len(per_layer), 5)
            vcos = round(sum(x[2] for x in per_layer) / len(per_layer), 5)
            mid = [x for x in per_layer if 8 <= x[0] <= 18]
            mid_k = round(sum(x[1] for x in mid) / max(1, len(mid)), 5)
            mid_v = round(sum(x[2] for x in mid) / max(1, len(mid)), 5)
            del cf

        ans_b, _ = h.generate(T.build_question(T.Q2_TEXT), ce_b)
        ok_b = "9AF4" in ans_b.upper().replace(" ", "")

        print("=" * 78)
        print("depth=%.2f  c1=%d  needle_rel=%s  rank=%s  keep=%d  kv_saved=%.1f%%" % (
            d, c1, rel, rank, keep, 100 * (1 - h.cache_len(ce) / full_len)))
        print("  needle in keep set: %s   new index: %s" % (kept, needle_new))
        print("  mean cos(orig_KV, recomputed_KV)  all layers: K=%s  V=%s" % (kcos, vcos))
        if needle_new is not None:
            print("  mean cos  mid layers 8-18:                   K=%s  V=%s" % (mid_k, mid_v))
        print("  A evict+rerotate  : %-4s  %r" % ("PASS" if ok_a else "fail", ans_a[:60]))
        print("  B + recompute KV  : %-4s  %r" % ("PASS" if ok_b else "fail", ans_b[:60]))
        if not ok_a and ok_b:
            print("  => CONDITIONING LOSS CONFIRMED: recomputing the needle KV recovers it.")
        elif not ok_a and not ok_b:
            print("  => conditioning loss REFUTED: the needle's own KV is not the problem.")
        print()

        out_rows.append(dict(depth=d, c1=c1, needle_rel=rel, rank=rank, keep=keep,
                             needle_kept=kept, k_cos=kcos, v_cos=vcos,
                             evict_pass=bool(ok_a), recompute_pass=bool(ok_b),
                             evict_answer=ans_a[:80], recompute_answer=ans_b[:80]))
        del cache_full, ce, ce_b

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in out_rows:
        print("  depth %-5s evict=%s recompute=%s   cos K=%s V=%s" % (
            r["depth"], "PASS" if r["evict_pass"] else "fail",
            "PASS" if r["recompute_pass"] else "fail", r["k_cos"], r["v_cos"]))

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
