#!/usr/bin/env python3
"""Generate training labels for a tuned Laya edge-sifter.

★ THE LABEL SOURCE IS THE WHOLE PROBLEM, AND THE OBVIOUS CHOICE IS WRONG.

Distilling from the consumer's own attention looks attractive -- it is one forward pass
and the labels are free. But the open defect is precisely that the 7B's attention MISSES
the needle at shallow depth (Finding 19: needle rank 555 against a keep budget of 539;
Bob's H13: the shutdown lines after a mid-log error never attend back to it). Training
Laya on those labels teaches it to reproduce the failure we are trying to fix.

So there are two label modes, and they answer different questions:

  --mode distil    label = top-K by the consumer's attention. CHEAP. Measures whether a
                   small bidirectional model can imitate a big causal one. If the tuned
                   Laya fails the shallow-depth task, that is the teacher's blind spot
                   inherited, not a Laya limitation.

  --mode ablation  label = 1 if REMOVING the line breaks retrieval, 0 otherwise.
                   EXPENSIVE (one generation per line) but GROUND TRUTH -- it does not
                   care what attention looked like, only what the answer needed. This is
                   the only label source that can teach Laya to beat the attention sifter
                   on the one case where that sifter is known to fail.

The honest comparison in Bob's roster therefore needs the ablation labels, at least for
the eval split, or arm 5 is measuring imitation rather than improvement.

Token-to-line mapping uses the tokenizer's offset mapping so a line's score is the mean
over its own tokens -- not a naive split, which drifts whenever a line tokenizes across
a boundary.

Usage:
    python src/laya_labels.py --mode distil   --model Qwen/Qwen2.5-7B --depth 0.15
    python src/laya_labels.py --mode ablation --model Qwen/Qwen2.5-7B --depth 0.15 --max-lines 40
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
PROMPT_SUFFIX = "\n</build_log><|im_end|>\n"


def line_spans(h, text):
    """(line_index, char_start, char_end) for each line, from the real tokenizer."""
    enc = h.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    lines, pos = [], 0
    for i, ln in enumerate(text.split("\n")):
        start, end = pos, pos + len(ln)
        span = [k for k, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
        if span:
            lines.append((i, start, end, span[0], span[-1] + 1))
        pos = end + 1
    return lines, enc["input_ids"]


def distil_scores(h, text, keep_frac=0.60):
    """Per-line mean attention the evictable region pays to that line."""
    c0 = len(h.tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    cache, qcap, ids = T.prefill_capture_q(h, text)
    sc = T.self_attn_scores(h, cache, qcap, c0, c1)
    del cache
    spans, _ = line_spans(h, text)
    out = []
    for (li, cs, ce, t0, t1) in spans:
        a, b = max(0, t0 - c0), max(0, t1 - c0)
        if b <= a:
            out.append((li, text.split("\n")[li], 0.0))
            continue
        out.append((li, text.split("\n")[li], float(sc[a:b].mean())))
    return out, c0, c1


def generate(h, prompt_ids, qids, max_new=24):
    dev = h.model.device
    ids = torch.tensor([prompt_ids + qids], dtype=torch.long, device=dev)
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
    return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["distil", "ablation"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--max-lines", type=int, default=40,
                    help="ablation mode only: cap the number of generations")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "laya_labels %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    log = log_at(args.depth)
    text = PROMPT_PREFIX + log + PROMPT_SUFFIX
    lines = log.split("\n")
    qids = h.tok(T.build_question(T.Q2_TEXT), add_special_tokens=False)["input_ids"]
    needle_line = next((i for i, l in enumerate(lines) if "STACK_FAIL" in l or "9AF4" in l), None)

    print("=" * 84)
    print("LAYA SIFTER LABELS  |  mode=%s  model=%s  depth=%.2f" % (
        args.mode, args.model, args.depth))
    print("=" * 84)
    print("  %d lines, needle at line %s" % (len(lines), needle_line))
    print()

    if args.mode == "distil":
        scored, c0, c1 = distil_scores(h, text, args.keep_frac)
        n = len(scored)
        keep = max(1, int(n * args.keep_frac))
        order = sorted(range(n), key=lambda i: -scored[i][2])
        pos = set(order[:keep])
        rows = [dict(line=li, text=txt[:160], score=round(s, 4), label=int(i in pos))
                for i, (li, txt, s) in enumerate(scored)]
        nl = sum(1 for r in rows if r["line"] == needle_line)
        print("  labeled %d lines, %d positive (%.0f%%)" % (n, keep, 100 * keep / n))
        print("  needle line %s -> label %s, score %.4f, attention rank %d" % (
            needle_line, rows[needle_line]["label"] if needle_line is not None else None,
            rows[needle_line]["score"] if needle_line is not None else -1,
            order.index(needle_line) + 1 if needle_line is not None else -1))
        print()
        print("  ★ NOTE: if the needle's rank is worse than the keep budget, these labels")
        print("    mark the evidence as NOT NEEDED. That is the teacher's blind spot being")
        print("    written into the training data -- use --mode ablation to avoid it.")

        if args.json_out:
            with open(args.json_out, "w") as f:
                for r in rows:
                    f.write(json.dumps(dict(r, mode="distil", depth=args.depth)) + "\n")
            print("\nwrote %s" % args.json_out)

    else:
        # ablation: remove one line, see whether the answer survives
        base = generate(h, h.tok(text, add_special_tokens=False)["input_ids"], qids)
        base_ok = "9AF4" in base.upper().replace(" ", "")
        print("  baseline (full prompt): %s  %r" % ("PASS" if base_ok else "fail", base[:56]))
        if not base_ok:
            print("  !! the baseline fails; ablation labels are meaningless. Raise the budget.")
            return
        print()
        cand = [i for i in range(len(lines)) if lines[i].strip()]
        if len(cand) > args.max_lines:
            step = len(cand) / args.max_lines
            cand = [cand[int(i * step)] for i in range(args.max_lines)]
        print("  ablating %d lines (one generation each) ..." % len(cand))
        rows = []
        t0 = time.time()
        for j, li in enumerate(cand):
            kept = "\n".join(l for k, l in enumerate(lines) if k != li)
            txt = PROMPT_PREFIX + kept + PROMPT_SUFFIX
            ans = generate(h, h.tok(txt, add_special_tokens=False)["input_ids"], qids)
            ok = "9AF4" in ans.upper().replace(" ", "")
            label = 0 if ok else 1        # 1 == removing this line broke the answer
            rows.append(dict(line=li, text=lines[li][:160], label=label,
                             ablated_ok=bool(ok), ablated_answer=ans[:60]))
            print("    [%2d/%2d] line %-4d %s  %.0fs" % (
                j + 1, len(cand), li, "NEEDED" if label else "removable",
                time.time() - t0), flush=True)
        pos = sum(r["label"] for r in rows)
        print()
        print("  %d/%d lines are load-bearing (%.0f%%)" % (pos, len(rows), 100 * pos / len(rows)))
        nl_row = next((r for r in rows if r["line"] == needle_line), None)
        if nl_row:
            print("  needle line %s -> %s" % (
                needle_line, "NEEDED (label 1)" if nl_row["label"] else "not needed"))
        if args.json_out:
            with open(args.json_out, "w") as f:
                for r in rows:
                    f.write(json.dumps(dict(r, mode="ablation", depth=args.depth)) + "\n")
            print("\nwrote %s" % args.json_out)


if __name__ == "__main__":
    main()
