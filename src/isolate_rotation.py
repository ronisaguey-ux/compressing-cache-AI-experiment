#!/usr/bin/env python3
"""WHICH step of the cache surgery loses the needle?

native_vs_cache established: the same kept tokens PASS when computed natively and FAIL
through the cache path. So neither selection nor content loss is the cause. One of these
three operations is:

    resize()      index_select on keys and values
    apply_delta() rotate surviving keys by (new_pos - orig_pos)
    the ask        generating against a cache built by surgery

This isolates them by building caches from a NATIVE forward over the compacted sequence
(which is already correct and already passes) and then applying each operation on top. If
the native cache passes and adding ONE operation breaks it, that operation is the defect:

    native_compact        forward the compacted tokens, ask        expect PASS
    + pivot/append query  ask against that cache                   expect PASS
    + apply_delta         rotate the ALREADY-CORRECT keys           <- if this fails,
                                                                    apply_delta is wrong
    cache_path            the shipped evict path                    expect fail

The `+ apply_delta` arm is the sharp one: the native compact cache's keys are already at
their correct positions, so rotating them by the delta schedule DOUBLE-rotates them and
must fail. That would prove the coordinate frame in apply_delta does not match the frame
the cache actually lives in.

Usage:
    python src/isolate_rotation.py --model Qwen/Qwen2.5-0.5B --depths 0.15
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "isolate_rotation %s" % args.model)
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
        full_len = ids.shape[1]
        scores = T.self_attn_scores(h, cache, qcap, c0, c1)
        keep = max(1, int(c1 * args.keep_frac))
        idx = sorted(int(x) for x in torch.argsort(scores, descending=True)[:keep].tolist())
        rank = None if rel is None else int((torch.argsort(scores, descending=True) == rel).nonzero().flatten().item())
        row = ids[0].tolist()
        compact_ids = row[:c0] + [row[c0 + i] for i in idx] + row[c0 + c1:]
        qtext = T.build_question(T.Q2_TEXT)

        def ask(cache_obj, label):
            ans, _ = h.generate(qtext, cache_obj)
            ok = "9AF4" in ans.upper().replace(" ", "")
            print("  %-22s %-4s %r" % (label, "PASS" if ok else "fail", ans[:52]), flush=True)
            return ok, ans

        print("=" * 78)
        print("depth=%.2f  c1=%d keep=%d rank=%s" % (d, c1, keep, rank))

        # A. native compact cache (correct by construction)
        with torch.no_grad():
            out = h.model(input_ids=torch.tensor(compact_ids).unsqueeze(0), use_cache=True)
        nat = out.past_key_values
        okA, ansA = ask(nat, "A native_compact")

        # B. native compact cache + apply_delta (double rotation)
        cB = T.build_arm_cache.__self__ if False else None
        natB = h.clone_cache(nat)
        # deltas are over the KEEP set, not over c1: apply_delta's own convention is
        # wanted = arange(len(keep_idx)) vs orig = keep_idx (chunk1-relative). Using c1
        # here was the bug -- 899 vs 539.
        wanted = torch.arange(len(idx), dtype=torch.float32)
        orig = torch.as_tensor(idx, dtype=torch.float32)
        deltas = (wanted - orig).tolist()
        for layer in natB.layers:
            k = layer.keys
            b, hh, t, dd = k.shape
            # tail tokens are at the end; rotate only the first c0+c1 of them
            head = k[:, :, :c0 + c1, :]
            hb = head.reshape(b * hh, 1, c0 + c1, dd)
            # chunk0 unrotated (delta 0), chunk1 rotated by its schedule
            dl = [0.0] * c0 + deltas
            rot = h.rope.apply(hb.contiguous(), dl)
            layer.keys = torch.cat([rot.reshape(b, hh, c0 + c1, dd), k[:, :, c0 + c1:, :]], dim=2)
        okB, ansB = ask(natB, "B native + apply_delta")

        # C. the shipped cache path
        tail = list(range(c0 + c1, ids.shape[1]))
        cC = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=True), c0, c0, c1, tail)
        okC, ansC = ask(cC, "C shipped cache path")

        # D. shipped resize WITHOUT rotation
        cD = T.build_arm_cache(h, cache, dict(kind="evict", keep=idx, rot=False), c0, c0, c1, tail)
        okD, ansD = ask(cD, "D shipped, no rotation")

        print()
        if okA and okB and not okC:
            print("  ==> apply_delta is the defect: correct keys break when re-rotated.")
        elif okA and not okB:
            print("  ==> apply_delta breaks an ALREADY-CORRECT cache (double rotation).")
        elif okA and okD and not okC:
            print("  ==> rotation is needed but mis-scheduled.")
        print()

        rows.append(dict(depth=d, c1=c1, keep=keep, rank=rank,
                         native=bool(okA), native_plus_delta=bool(okB),
                         shipped=bool(okC), shipped_norot=bool(okD),
                         aA=ansA[:70], aB=ansB[:70], aC=ansC[:70], aD=ansD[:70]))
        del cache, nat, natB, cC, cD

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
