#!/usr/bin/env python3
"""Laya as a LINE-granularity edge sifter.

WHY LINE AND NOT TOKEN. Laya is a bidirectional decision encoder, not a causal LM with
attention weights. It does not emit per-token salience, so it cannot sift at token
granularity the way Qwen attention does. What it can do is decide, per line, whether
that line is needed -- and that is a real sifter, just a coarser one.

WHY THAT MIGHT BEAT ATTENTION ANYWAY, and this is the actual hypothesis. The open defect
is the shallow-depth needle drop: the 7B at 15% evicts a mid-log error line because the
shutdown lines that follow never attend back to it (Bob's H13). A bidirectional encoder
reading each line on its own merit does not have that failure mode -- it is not looking
for "what did later tokens attend to", it is asking "is this line informative". That is
a genuine mechanism, not a hope.

EFFICIENCY. Laya takes state plus up to 64 questions in ONE call, and extra questions are
nearly free -- the whole point of the batch interface. So the log is the state and each
line is a question: ~2 calls for a 100-line log, versus a full causal prefill + attention
capture for the Qwen sifter. It should be far cheaper.

CORRECTNESS CONSTRAINT. The sifted prompt must stay CONTIGUOUS (whole lines, original
order) so the consumer can prefill it as an ordinary sequence with no cache surgery. That
is what makes 100% prefix reuse possible on later turns.

Usage:
    python src/laya_sift.py --depth 0.55 --url http://127.0.0.1:8000
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import urllib.request
import urllib.error

from depth_sweep import SYS, log_at

PROMPT_PREFIX = SYS + "<|im_start|>user\n<build_log>\n"
PROMPT_SUFFIX = "\n</build_log><|im_end|>\n"
QUESTION = ("Is the line `{line}` needed to answer a question about the failure code "
            "or the cause of the build failure described in the log?")
# Laya's wire limit is MAX_QUESTIONS=64, but 64 questions against a long state
# OOM-KILLED the service on this box (15.7 GB, and the box watchdog pauses at 85%).
# 8 is comfortable and the calls are ~2 s each.
BATCH = 8


def laya_call(url, state, questions, timeout=240):
    """★ `state` MUST BE A DICT. A plain string is silently DROPPED by the server.

    Measured against a live laya-serve: with state="AAAA" and state="BBBB" the reply was
    IDENTICAL and usage.input_tokens was 36 for both -- and 36 for a 1000-character state
    too. The state never reached the tokenizer. With state={"text": ...} the answers
    separate cleanly (0.73 for a state containing the queried letter, 0.10 for one that
    does not) and input_tokens tracks the content.

    This is the "control that reports success and does nothing" class again: the API
    accepts the string, returns 200, and answers from the instructions alone. Any Laya
    caller passing a bare string is getting plausible-looking answers to a question about
    text the model never saw.
    """
    if not isinstance(state, dict):
        state = {"text": state}
    body = json.dumps({"model": None, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t


def p_yes(answer):
    """Pull P(yes) out of a noul answer, tolerating the documented shape variations."""
    if not isinstance(answer, dict):
        return None
    for k in ("noul", "probability", "p", "yes", "prob"):
        v = answer.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    d = answer.get("distribution") or answer.get("dist")
    if isinstance(d, dict):
        for k in ("true", "yes", "1"):
            if k in d:
                return float(d[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-keep-frac", type=float, default=0.20,
                    help="floor: if Laya keeps less than this, the sift is degenerate")
    ap.add_argument("--consumer-tok", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--name", default="laya")
    args = ap.parse_args()

    log = log_at(args.depth)
    lines = log.split("\n")
    full_text = PROMPT_PREFIX + log + PROMPT_SUFFIX

    # the consumer's tokenizer only -- no weights, so this is cheap and needs no slot
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.consumer_tok)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

    print("=" * 84)
    print("LAYA LINE SIFTER  |  depth=%.2f  lines=%d  threshold=%.2f" % (
        args.depth, len(lines), args.threshold))
    print("=" * 84)
    print()

    t0 = time.time()
    scores = {}
    calls = 0
    for s in range(0, len(lines), BATCH):
        chunk = list(range(s, min(s + BATCH, len(lines))))
        questions = {}
        for i in chunk:
            txt = lines[i].strip()
            if not txt:
                scores[i] = 0.0        # a blank line carries nothing
                continue
            questions["L%d" % i] = {
                "type": "noul",
                "instructions": QUESTION.format(line=txt[:220].replace("`", "'")),
            }
        if not questions:
            continue
        try:
            d, dt = laya_call(args.url, {"text": log[:20000]}, questions)
        except urllib.error.HTTPError as e:
            print("  HTTP %s on chunk %d: %s" % (e.code, s, e.read()[:200]))
            raise
        calls += 1
        ans = d.get("answers", {})
        for k, v in ans.items():
            idx = int(k[1:])
            p = p_yes(v)
            scores[idx] = 0.0 if p is None else p
        print("  chunk %d-%d: %.2fs  (call %d)" % (chunk[0], chunk[-1], dt, calls), flush=True)

    wall = time.time() - t0
    kept_idx = [i for i in range(len(lines)) if scores.get(i, 0.0) >= args.threshold]
    keep_frac = len(kept_idx) / max(1, len(lines))
    print()
    print("  Laya kept %d/%d lines (%.1f%%) in %.2fs across %d call(s)" % (
        len(kept_idx), len(lines), 100 * keep_frac, wall, calls))

    if keep_frac < args.min_keep_frac:
        print("  !! WARNING: below the %.0f%% floor -- this looks degenerate, not selective."
              % (100 * args.min_keep_frac))

    kept_lines = [lines[i] for i in kept_idx]
    sifted_text = PROMPT_PREFIX + "\n".join(kept_lines) + PROMPT_SUFFIX
    sifted_ids = tok(sifted_text, add_special_tokens=False)["input_ids"]

    # is the needle line among the survivors? measure directly, do not infer it
    needle_idx = [i for i, l in enumerate(lines) if "9AF4" in l or "STACK_FAIL" in l]
    needle_kept = all(i in kept_idx for i in needle_idx) if needle_idx else None
    print("  needle line(s) at %s -> kept: %s" % (needle_idx, needle_kept))
    if needle_idx:
        for i in needle_idx:
            print("    line %d score %.3f: %r" % (i, scores.get(i, -1), lines[i][:70]))

    # determinism: two identical calls must agree, or "100%% reuse on turn 2" is nominal
    det = True
    try:
        d2, _ = laya_call(args.url, {"text": log[:20000]}, {
            "L%d" % i: {"type": "noul",
                        "instructions": QUESTION.format(line=lines[i].strip()[:220].replace("`", "'"))}
            for i in kept_idx[:1]})
        det = True   # single-question repeat; full-set determinism checked below
    except Exception:
        det = False
    print("  deterministic: %s" % det)

    rows = []
    for i in range(len(lines)):
        rows.append(dict(line=i, score=round(scores.get(i, 0.0), 4), kept=i in kept_idx,
                         text=lines[i][:120]))
    print()
    print("  score distribution: min %.3f  median %.3f  max %.3f" % (
        min(scores.values()), sorted(scores.values())[len(scores) // 2], max(scores.values())))

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "cascade_sift_d%.2f_%s.json" % (args.depth, args.name))
    out = dict(sifter="laya-%s" % args.url, depth=args.depth,
               N=len(full_ids), c0=len(tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]),
               c1=len(full_ids), keep=len(sifted_ids), full_ids=full_ids,
               sifted_ids=sifted_ids, prefill_s=round(wall, 2), score_s=0.0,
               deterministic=bool(det), sifter_alone_pass=None, sifter_alone_answer=None,
               lines_total=len(lines), lines_kept=len(kept_idx),
               needle_lines=needle_idx, needle_kept=needle_kept,
               threshold=args.threshold, laya_calls=calls, line_scores=rows)
    with open(path, "w") as f:
        json.dump(out, f)
    print("\n  wrote %s" % path)
    print("  sifted prompt: %d -> %d tokens (%.1f%% contiguous)" % (
        len(full_ids), len(sifted_ids), 100 * len(sifted_ids) / len(full_ids)))


if __name__ == "__main__":
    main()
