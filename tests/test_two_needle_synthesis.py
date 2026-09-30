#!/usr/bin/env python3
"""TEST 3 (Bob 2026-09-30): can the query synthesise a fact split across TWO surviving
blocks when a block between them is evicted?

    Block 1   Server Alpha port is <PORT>.            <- survives
    Block 2   2,000 lines of unrelated compiler spam  <- EVICTED
    Block 3   Server Alpha password is '<PASS>'.      <- survives
    Query     Connect to Server Alpha, format <port>:<password>
    Expect    <PORT>:<PASS>

★ WHY THIS IS THE TEST THAT MATTERS. Everything measured so far is single-fact: find the
needle, report the needle. A join across two blocks is a strictly harder question, because
neither block contains the answer on its own. Block 1 knows the port and not the password;
block 3 knows the password and not the port. The answer exists only in the QUERY's attention
over both.

And there is a specific reason to expect trouble, not just "this is harder". Under
block-diagonal masking (Finding 28) a block attends only within itself, so neither block can
carry a representation of the other. The join has to happen entirely in the query tier. If
the query cannot do it, the tiered architecture cannot do multi-block reasoning at all, and
that is a real limitation of the design rather than an implementation bug.

THREE ARMS, so a failure is attributable:

  A  both blocks + query tier                  the actual test
  B  both blocks + query tier, NO eviction     control: is the task answerable at all?
  C  both blocks CONCATENATED as one ordinary causal sequence (the cascade shape)
                                               is any failure the block layout, or the task?

If C passes and A fails, the block layout is the cause. If B fails, the task is unanswerable
and A tells us nothing. Both controls are required or the result is uninterpretable.

10 trials with randomised port/password, because a single constant pair cannot distinguish a
working join from one the model happens to have memorised.

Usage:
    python tests/test_two_needle_synthesis.py
    python tests/test_two_needle_synthesis.py --model Qwen/Qwen2.5-0.5B --trials 10
"""
import sys, os, json, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS
from tiered_cache import (build_table, concat_caches, slice_cache, cache_len, generate)

SPAM_LINES = [
    "gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
    "gcc -O2 -Wall -Wextra -c src/net/tls.c -o build/tls.o",
    "src/core/pool.h:88:12: warning: unused parameter 'flags'",
    "ccache: cache miss for src/vm/stack.o (stats reset)",
    "ninja: build stopped: subcommand failed at phase 0x%02x",
    "make[2]: Entering directory '/build/obj'",
    "make[2]: Leaving directory '/build/obj'",
    "ar rcs build/libcore.a build/pool.o build/emit.o",
    "ld: warning: ignoring duplicate libraries: '-lc'",
    "install -m 0644 build/libcore.a /usr/local/lib/",
]


def make_spam(target_lines):
    out = []
    for i in range(target_lines):
        out.append(SPAM_LINES[i % len(SPAM_LINES)] % (i % 256) if "%" in SPAM_LINES[i % len(SPAM_LINES)]
                   else SPAM_LINES[i % len(SPAM_LINES)])
    return "\n".join(out)


def build_case(h, port, password, spam_lines):
    """Return the block texts and the query, for one trial."""
    b1 = ("Build artefacts for the fleet were assembled in this run.\n"
          "The deployment manifest lists the following service endpoint.\n"
          "Server Alpha port is %s.\n"
          "Further topology follows in later sections." % port)
    b2 = make_spam(spam_lines)
    b3 = ("The operations handbook records access material separately from endpoints.\n"
          "Server Alpha password is '%s'.\n"
          "Rotate credentials on the usual schedule." % password)
    query = ("Connect to Server Alpha using the format <port>:<password>. "
             "Reply with only that string, nothing else.")
    return [b1, b2, b3], query


def concat_prompt(global_text, chunks, query, h):
    """Arm C: one ordinary causal sequence -- the cascade shape.

    ★ THE QUERY MUST BE IN THE PROMPT AND MUST BE A PROPER TURN. The first version of this
    function took `query` and never used it, so arm C was asked nothing and answered with a
    continuation. The query is wrapped by `build_question`, which supplies
    `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` -- without that turn
    structure the model has no boundary and simply keeps writing the log.
    """
    body = "\n".join(chunks)
    text = (global_text + "<build_log>\n" + body + "\n</build_log><|im_end|>\n"
            + T.build_question(query))
    return h.tok(text, add_special_tokens=False)["input_ids"]


def run_trial(h, port, password, spam_lines, global_text, arms=("A", "B", "C"), max_new=16):
    chunks, query = build_case(h, port, password, spam_lines)
    # ★ WRAP THE QUERY AS A TURN. A raw question string leaves the model mid-document with no
    # assistant header, so it continues the log instead of answering -- which is exactly what
    # the first run did in EVERY arm, controls included. `build_question` supplies the
    # user/assistant structure.
    qids = h.tok(T.build_question(query), add_special_tokens=False)["input_ids"]
    want = "%s:%s" % (port, password)
    res = {}

    # ---- arms A and B: block table, query as its own tier
    g_ids, g_cache, blocks, pos, G, total = build_table(h, global_text, chunks, verbose=False)
    qpos = total

    if "A" in arms:
        # evict block 2 (index 1), keep 1 and 3
        keep = [blocks[0], blocks[2]]
        tbl = concat_caches([g_cache] + keep)
        tbl = slice_cache(tbl, 0, cache_len(tbl))
        ans = generate(h, tbl, qids, qpos, max_new=max_new)
        res["A"] = dict(answer=ans, ok=(want in ans.replace(" ", "")))

    if "B" in arms:
        # no eviction: all three blocks present
        tbl = concat_caches([g_cache] + [b for b in blocks if b is not None])
        tbl = slice_cache(tbl, 0, cache_len(tbl))
        ans = generate(h, tbl, qids, qpos, max_new=max_new)
        res["B"] = dict(answer=ans, ok=(want in ans.replace(" ", "")))

    if "C" in arms:
        # one causal sequence, no blocks
        ids = concat_prompt(global_text, chunks, query, h)
        dev = h.model.device
        with torch.no_grad():
            o = h.model(input_ids=torch.tensor([ids], device=dev), use_cache=True)
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt]
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = h.model(input_ids=nxt, past_key_values=c, use_cache=True)
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            if int(nxt.item()) == h.tok.eos_token_id:
                break
        ans = h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        res["C"] = dict(answer=ans, ok=(want in ans.replace(" ", "")))

    return res, want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--spam-lines", type=int, default=120,
                    help="lines of unrelated log between the two needles")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    T.require_memory(9000 if "7B" in args.model else 3500, "two_needle %s" % args.model)
    h = T.Harness(args.model)
    if not hasattr(h, "model_id"):
        h.model_id = args.model

    global_text = SYS + "<|im_start|>user\n"
    rng = random.Random(args.seed)
    arms = tuple(args.arms.split(","))

    print("=" * 88)
    print("TEST 3 — TWO-NEEDLE CROSS-BLOCK SYNTHESIS")
    print("  model=%s  trials=%d  spam=%d lines" % (args.model, args.trials, args.spam_lines))
    print("=" * 88)
    print("  A  both needles, middle block EVICTED      <- the actual test")
    print("  B  both needles, nothing evicted           <- is the task answerable at all?")
    print("  C  concatenated as one causal sequence     <- is a failure the layout or the task?")
    print()
    print("  %-5s %-14s %-24s %-24s %-24s" % (
        "trial", "want", "A evicted middle", "B no eviction", "C concatenated"))
    print("  " + "-" * 86)

    rows = []
    for i in range(args.trials):
        port = rng.randint(1024, 65535)
        password = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        res, want = run_trial(h, port, password, args.spam_lines, global_text, arms)
        def cell(k):
            if k not in res:
                return "-"
            return ("PASS " if res[k]["ok"] else "fail ") + repr(res[k]["answer"][:15])
        print("  %-5d %-14s %-24s %-24s %-24s" % (
            i + 1, want, cell("A"), cell("B"), cell("C")), flush=True)
        rows.append(dict(trial=i + 1, want=want, port=port, password=password,
                         **{k: v for k, v in res.items()}))

    print()
    print("=" * 88)
    summary = {}
    for k in arms:
        n = sum(1 for r in rows if r.get(k, {}).get("ok"))
        summary[k] = n
        label = {"A": "A  evicted middle (the test)",
                 "B": "B  no eviction (control)",
                 "C": "C  concatenated (cascade shape)"}[k]
        print("  %-34s %2d/%d  (%.0f%%)" % (label, n, len(rows), 100 * n / len(rows)))
    print()
    if summary.get("B", 0) < len(rows):
        print("  !! the no-eviction control is not passing: the task itself is unreliable,")
        print("     so A cannot be interpreted. Investigate that first.")
    elif summary.get("A", 0) == len(rows):
        print("  ==> CROSS-BLOCK SYNTHESIS WORKS. The query tier joins facts from two")
        print("      surviving blocks with the middle evicted, at 10/10. Neither block holds")
        print("      the answer alone, so this is a genuine relational join.")
    elif summary.get("C", 0) > summary.get("A", 0):
        print("  ==> THE BLOCK LAYOUT IS THE LIMIT. The same content answers correctly as one")
        print("      causal sequence but not as blocks, so block-diagonal masking starves the")
        print("      join. Real limitation of the tiered design, not a bug.")
    else:
        print("  ==> the join fails in both layouts, so this is a task-difficulty result")
        print("      rather than a block-layout result.")

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(dict(
                model=args.model, trials=args.trials, spam_lines=args.spam_lines,
                summary=summary,
                rows=[{"trial": r["trial"], "want": r["want"],
                       **{k: ({"ok": r[k]["ok"], "answer": r[k]["answer"][:60]}
                              if isinstance(r.get(k), dict) else None)
                          for k in arms}}
                      for r in rows])) + "\n")
        print("\nwrote", args.json_out)
    return 0 if summary.get("A", 0) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
