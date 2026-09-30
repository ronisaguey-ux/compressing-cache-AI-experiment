#!/usr/bin/env python3
"""Two-speed cascade, run in TWO SEQUENTIAL PHASES so only one model is ever resident.

THE IDEA (Bob, 2026-09-30). The cheap model does an attention pass over the FULL context
and emits a shorter, CONTIGUOUS prompt. The expensive model never does cache surgery --
it prefills that prompt from scratch. Because a sifted prompt has no holes, it is a
complete valid sequence and therefore its own full prefix, so every later turn reuses
100% of it. The cascade sidesteps the corruption problem rather than repairing it.

WHY TWO PHASES. The first version loaded both models in one process. Two problems:

  1. MEMORY. 0.5B fp32 (~2 GB) + 7B 8-bit (~8.1 GB) + torch overhead lands near 11 GB
     against a watchdog that SIGSTOPs at 85% of 15.7 GB -- and a SIGSTOP does not free
     the paused process's RSS, so it deadlocks rather than failing. Measured live: the
     pair peaked at 86%. Bob's instruction was "strictly sequential execution to respect
     the host memory watchdog", and holding both models at once contradicts it.
  2. SEPARATION. A sift produced while the consumer is also resident is not evidence that
     the sift works on its own; the phases must be independently runnable.

`--phase sift`    loads the sifter, writes the sifted ids, exits (freeing all of it).
`--phase consume` loads the consumer, reads the ids, runs the three arms, exits.

THREE ARMS, so a failure is attributable:

    A  full prompt -> consumer          the ceiling
    B  sifter's selection -> consumer   does cross-model salience transfer?
    C  consumer's own selection         does self-sifting work?

    B fails, C passes  => cross-model transfer is the problem
    both pass          => a cheap two-speed front end
    both fail          => the sift itself is insufficient, not the transfer

Also measured: the sift ratio, the sifter's cost against a consumer full prefill, and
determinism across two sifts -- determinism is what makes the "100% reuse on turn 2"
claim real rather than nominal.

Usage:
    python src/cascade_sifter.py --phase sift    --depth 0.55 --json data/cascade_sifter.jsonl
    python src/cascade_sifter.py --phase consume --depth 0.55 --json data/cascade_sifter.jsonl
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"


def build_text(depth):
    return PROMPT_PREFIX + log_at(depth) + "\n</build_log><|im_end|>\n"


def select(h, text, force_frac, keep_frac, want_ids=True):
    """Score the evictable region by attention; return the surviving ids, CONTIGUOUS.

    Survivors are emitted in original order, so the result is a gap-free sequence the
    consumer can prefill as an ordinary prompt.
    """
    c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    t = time.time()
    cache, qcap, ids = T.prefill_capture_q(h, text)
    prefill_s = time.time() - t
    t = time.time()
    sc = T.self_attn_scores(h, cache, qcap, c0, c1)
    score_s = time.time() - t
    N = ids.shape[1]
    keep = max(1, int(c1 * keep_frac))
    force = int(c1 * force_frac)
    budget = max(0, keep - force)
    order = torch.argsort(sc[force:], descending=True)
    sel1 = sorted(set(range(force)) | {force + int(x) for x in order[:budget].tolist()})
    kept = sorted(set(range(c0)) | {c0 + i for i in sel1})
    row = ids[0].tolist()
    full = list(row)
    sifted = [row[j] for j in kept]
    del cache
    return (sifted if want_ids else None), dict(
        N=N, c0=c0, c1=c1, keep=len(kept), full_ids=full, sifted_ids=sifted,
        prefill_s=round(prefill_s, 2), score_s=round(score_s, 2))


def answer_from(h, prompt_ids, qids, max_new=24):
    """Greedy decode from a prompt the model prefills itself. No cache surgery."""
    dev = h.model.device
    ids = torch.tensor([prompt_ids + qids], dtype=torch.long, device=dev)
    t = time.time()
    with torch.no_grad():
        out = h.model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]
    for _ in range(max_new - 1):
        with torch.no_grad():
            out = h.model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        if int(nxt.item()) == h.tok.eos_token_id:
            break
    return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True), round(time.time() - t, 2)


def sift_file(depth, sifter):
    tag = sifter.split("/")[-1]
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "cascade_sift_d%.2f_%s.json" % (depth, tag))


def phase_sift(args):
    text = build_text(args.depth)
    print("=" * 84)
    print("PHASE 1: SIFT  |  sifter=%s  depth=%.2f  (consumer NOT resident)" % (
        args.sifter, args.depth))
    print("=" * 84)
    hs = T.Harness(args.sifter)
    hs.model_id = args.sifter
    sift_ids, meta = select(hs, text, args.force_frac, args.keep_frac)
    qids_s = hs.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    ans_s, t_s = answer_from(hs, sift_ids, qids_s)
    ok_s = "9AF4" in ans_s.upper().replace(" ", "")
    sift_ids2, _ = select(hs, text, args.force_frac, args.keep_frac)
    det = sift_ids == sift_ids2

    print()
    print("  sift           : %d -> %d tokens (%.1f%% retained contiguously)" % (
        meta["N"], meta["keep"], 100 * meta["keep"] / meta["N"]))
    print("  sifter cost    : prefill %.1fs + score %.1fs" % (meta["prefill_s"], meta["score_s"]))
    print("  deterministic  : %s  (two sifts, identical ids)" % det)
    print("  sifter alone   : %s  on its own sifted prompt (%.2fs)" % (
        "PASS" if ok_s else "fail", t_s))
    print("                    %r" % ans_s[:70])

    out = dict(sifter=args.sifter, depth=args.depth, deterministic=bool(det),
               sifter_alone_pass=bool(ok_s), sifter_alone_answer=ans_s[:70],
               sifter_alone_sec=t_s, **meta)
    path = sift_file(args.depth, args.sifter)
    with open(path, "w") as f:
        json.dump(out, f)
    print("\n  wrote %s" % path)
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps({k: v for k, v in out.items()
                                if k not in ("full_ids", "sifted_ids")}) + "\n")


def phase_consume(args):
    path = sift_file(args.depth, args.sifter)
    with open(path) as f:
        s = json.load(f)
    text = build_text(args.depth)
    print("=" * 84)
    print("PHASE 2: CONSUME  |  consumer=%s  depth=%.2f  (sifter NOT resident)" % (
        args.consumer, args.depth))
    print("=" * 84)
    hc = T.Harness(args.consumer)
    hc.model_id = args.consumer
    qids = hc.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]

    print()
    print("  A  full prompt (%d tok) ..." % len(s["full_ids"]), flush=True)
    ansA, tA = answer_from(hc, s["full_ids"], qids)
    okA = "9AF4" in ansA.upper().replace(" ", "")

    print("  B  sifter-selected prompt (%d tok) ..." % len(s["sifted_ids"]), flush=True)
    ansB, tB = answer_from(hc, s["sifted_ids"], qids)
    okB = "9AF4" in ansB.upper().replace(" ", "")

    print("  C  consumer self-sift ...", flush=True)
    _, cmeta = select(hc, text, args.force_frac, args.keep_frac)
    ansC, tC = answer_from(hc, cmeta["sifted_ids"], qids)
    okC = "9AF4" in ansC.upper().replace(" ", "")

    print()
    print("  %-34s %-8s %-9s %s" % ("arm", "verdict", "seconds", "answer"))
    print("  %-34s %-8s %-9s %s" % ("A full prompt (ceiling)", "PASS" if okA else "fail", tA, ansA[:44]))
    print("  %-34s %-8s %-9s %s" % ("B sifter-selected (cross-model)", "PASS" if okB else "fail", tB, ansB[:44]))
    print("  %-34s %-8s %-9s %s" % ("C consumer self-sift", "PASS" if okC else "fail", tC, ansC[:44]))
    print()
    print("  overlap: sifter kept %d, consumer kept %d, both %d" % (
        len(s["sifted_ids"]), len(cmeta["sifted_ids"]),
        len(set(s["sifted_ids"]) & set(cmeta["sifted_ids"]))))

    print()
    print("=" * 84)
    if okA and okB and okC:
        print("  => CASCADE WORKS. Cross-model sifting holds: a cheap model can pick what")
        print("     an expensive one needs, and the consumer prefills a prompt %.0f%% shorter" % (
            100 * (1 - len(s["sifted_ids"]) / len(s["full_ids"]))))
        print("     with no cache surgery at all.")
    elif okA and not okB and okC:
        print("  => CROSS-MODEL TRANSFER FAILS. Self-sifting works, cross-model does not:")
        print("     the sifter must be the consumer, or distilled from it.")
    elif okA and not okB and not okC:
        print("  => THE SIFT ITSELF FAILS at this budget -- not a transfer problem.")
    else:
        print("  => the ceiling failed; raise the budget before reading B and C.")
    print("=" * 84)
    print()
    print("  cost: sifter %.1fs + consumer sifted prefill %.1fs = %.1fs  vs  consumer full %.1fs" % (
        s["prefill_s"] + s["score_s"], tB, s["prefill_s"] + s["score_s"] + tB, tA))
    print("  turn-2 reuse: 100%% of the sifted prompt, contingent on determinism = %s" % s["deterministic"])

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                phase="consume", depth=args.depth, consumer=args.consumer,
                A_full_pass=bool(okA), A_sec=tA, A_answer=ansA[:70],
                B_cross_pass=bool(okB), B_sec=tB, B_answer=ansB[:70],
                C_self_pass=bool(okC), C_sec=tC, C_answer=ansC[:70],
                sifted_len=len(s["sifted_ids"]), full_len=len(s["full_ids"]),
                overlap=len(set(s["sifted_ids"]) & set(cmeta["sifted_ids"])),
            )) + "\n")
        print("\nwrote", args.json_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["sift", "consume"])
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--force-frac", type=float, default=0.15)
    ap.add_argument("--sifter", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--consumer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    if args.phase == "sift":
        phase_sift(args)
    else:
        phase_consume(args)


if __name__ == "__main__":
    main()
