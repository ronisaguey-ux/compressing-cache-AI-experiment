#!/usr/bin/env python3
"""Are the causal mask and position normalisation COUPLED?

Bob's fix (a) removes the causal mask from chunk-1 scoring.
Bob's fix (c) divides received attention by each position's causal horizon.

Those two fixes target the same bias, so applying (c) to a scorer that already has
(a) may be actively wrong: with no mask, every position can be attended by all T
queries, so the causal horizon is no longer the number of valid contributors and
dividing by it penalises early positions for nothing.

Prediction: posnorm HELPS on the causal scorer and HURTS on the non-causal one.
If that holds, the earlier "posnorm hurts" result is explained rather than
contradicted, and the two fixes must be chosen together, not independently.
"""
import sys, os, io, math, contextlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
torch.set_num_threads(4)
import test_2d_kv_cache as T
import depth_sweep as DS


def scores(h, cache, qcap, c0, c1, causal):
    if causal:
        return DS.causal_scores(h, cache, qcap, c0, c1)
    return T.self_attn_scores(h, cache, qcap, c0, c1)


def main():
    h = T.Harness()
    print("%-8s %-7s | %-24s %-24s %s" % ("depth", "c1", "causal raw -> posnorm",
                                          "non-causal raw -> posnorm", "prediction"))
    print("-" * 100)
    for frac in (0.15, 0.35, 0.55, 0.75):
        log = DS.log_at(frac)
        text = DS.SYS + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
        c0 = len(h.tok(DS.SYS, add_special_tokens=False)["input_ids"])
        c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
        h.chunk1_len = c1
        na = T.locate_needle(h.tok, text)
        rel = na - c0
        cache, qcap, _ = T.prefill_capture_q(h, text)

        out = {}
        for causal in (True, False):
            raw = scores(h, cache, dict(qcap), c0, c1, causal)
            pn = T.sel_position_normalized(raw, c1)
            oraw = torch.argsort(raw, descending=True)
            opn = torch.argsort(pn, descending=True)
            r_raw = (oraw == rel).nonzero().flatten().item()
            r_pn = (opn == rel).nonzero().flatten().item()
            out[causal] = (r_raw, r_pn)
        del cache

        c_raw, c_pn = out[True]
        n_raw, n_pn = out[False]
        c_effect = c_pn - c_raw      # negative = posnorm improved the causal rank
        n_effect = n_pn - n_raw
        verdict = ("posnorm helps causal, hurts non-causal" if c_effect < 0 <= n_effect
                   else ("both hurt" if c_effect >= 0 and n_effect >= 0
                         else ("both help" if c_effect < 0 and n_effect < 0 else "mixed")))
        print("%-8.2f %-7d | rank %4d -> %4d (%+5d)   | rank %4d -> %4d (%+5d)   | %s"
              % (frac, c1, c_raw, c_pn, c_effect, n_raw, n_pn, n_effect, verdict))


if __name__ == "__main__":
    main()
