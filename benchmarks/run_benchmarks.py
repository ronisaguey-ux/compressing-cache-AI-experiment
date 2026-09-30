#!/usr/bin/env python3
"""Bob's benchmark matrix (2026-09-30) — head-to-head, three configurations.

    python benchmarks/run_benchmarks.py --models qwen2.5-7b,mistral-7b \
        --tests ruler,babilong,synthetic_agent,two_needle

WHAT IS COMPARED, per test and per model:

    vanilla   full causal attention over EVERY block, no pruning. This is standard linear
              serving: the whole prompt must be re-prefilled and its whole KV resident.
    runtime   the block-causal table with the distractor block EVICTED, query as its own
              tier. Surviving blocks keep their ORIGINAL absolute positions (Finding 23:
              preserving position ids removes the positional error; re-indexing does not).

METRICS (Bob's three):
    exact_match   answer contains the expected fact
    ttft_ms       time from starting the forward to first-token logits
                  - vanilla: the ENTIRE prompt must be prefilled (what standard serving pays
                    on every turn)
                  - runtime: only the query tier is forwarded (blocks are already cached).
                    The one-time block ingest is reported separately as `ingest_ms` so the
                    runtime number is never read as free.
    kv_mb         bytes resident in the KV cache at decode time
                  - vanilla: every block
                  - runtime: global + surviving blocks + query (the evicted block is gone)

★ WHY THIS IS NOT A 16k / 4k BLOCK RUN. Bob's spec asks for 2,000-4,000 token distractor
blocks and 16k-32k contexts. A 32k prefill on a 7B at 8-bit on this CPU is roughly 25-40
minutes per sample, and the two models cannot be resident at once (15.7 GB box, watchdog
pauses at 85% RAM). One RULER cell would therefore be a full day and a full matrix weeks.
The distractor is scaled to what the box completes, every row records the MEASURED token
counts, and no number here is labelled as a 16k result.

★ THREE CONSTRAINTS THAT CHANGE WHAT CAN HONESTLY BE REPORTED:
  1. meta-llama/Llama-3.1-8B is GATED (GatedRepoError 403 — the account has not accepted the
     license). Mistral-7B-v0.3 substitutes: Apache-2.0, ungated, dense causal + RoPE + GQA.
     Reported as Mistral, never as Llama.
  2. SWE-bench Mini needs a real repo, a real failing pytest and a multi-turn patch loop. Not
     built — see tasks.SWE_MINI_STATUS. A synthetic stand-in would report a patch-resolution
     rate that means nothing.
  3. Block-diagonal attention does NOT preserve bit-identity against full-causal, and cannot:
     the operator shape differs so the matmul sums in a different order. Measured in Finding
     28 as 3.05e-05 (block) vs 1.69e+01 (full causal). `torch.equal` across the two
     architectures is therefore not an assertable claim. What IS assertable, and is asserted
     here, is that EVICTION does not disturb the survivors — that is exact, and it is
     `drift_evict`.
"""
import sys, os, json, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS
from tiered_cache import build_table, concat_caches, slice_cache, cache_len, generate
import tasks as TASKS


MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",  # gated: GatedRepoError 403
}

# 8-bit for anything 7B and up. Set BEFORE the Harness is constructed -- the loader reads
# this env at init, so setting it after would silently load a 14 GB fp32 model.
NEEDS_8BIT = ("7b", "8b")


def cache_bytes(c):
    """Bytes resident in a DynamicCache. 0 for an empty/None cache."""
    if c is None or len(c.layers) == 0:
        return 0
    n = 0
    for li in range(len(c.layers)):
        k = c.layers[li].keys
        v = c.layers[li].values
        n += k.numel() * k.element_size() + v.numel() * v.element_size()
    return n


def ttft_decode(h, cache, ids, pos0, max_new):
    """Forward `ids` (at absolute positions pos0..) then greedily decode.

    Returns (text, ttft_seconds, kv_bytes). `ttft` is measured identically in both arms:
    entry to the forward -> first-token logits available. That is the comparable figure; the
    difference is HOW MANY tokens each arm has to push through that forward.
    """
    dev = h.model.device
    ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
    pos = torch.arange(pos0, pos0 + len(ids), dtype=torch.long, device=dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        if cache is None:
            out = h.model(input_ids=ids_t, use_cache=True,
                          position_ids=pos.unsqueeze(0))
        else:
            out = h.model(input_ids=ids_t, past_key_values=cache, use_cache=True,
                          position_ids=pos.unsqueeze(0), cache_position=pos)
    ttft = time.perf_counter() - t0

    c = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]
    p = pos0 + len(ids)
    for _ in range(max_new - 1):
        with torch.no_grad():
            out = h.model(input_ids=nxt, past_key_values=c, use_cache=True,
                          cache_position=torch.tensor([p], device=dev))
        c = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt)
        p += 1
        if int(nxt.item()) == h.tok.eos_token_id:
            break

    text = h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
    return text, ttft, cache_bytes(c)


def full_prompt_ids(h, global_text, block_texts, query):
    """One ordinary causal sequence: global + every block + the query as a proper turn.

    ★ THE QUERY MUST BE A TURN, not a bare string. Without the `<|im_start|>user ... assistant`
    wrapper the model is left mid-document and continues the log instead of answering -- which
    is what the first version of the two-needle test did in EVERY arm, controls included.
    """
    body = "\n".join(block_texts)
    text = (global_text + "<build_log>\n" + body + "\n</build_log><|im_end|>\n"
            + T.build_question(query))
    return h.tok(text, add_special_tokens=False)["input_ids"]


def run_vanilla(h, task, global_text, max_new):
    """Full causal attention over every block. Standard serving."""
    blocks = task["blocks"]
    ids = full_prompt_ids(h, global_text, [b[1] for b in blocks], task["query"])
    text, ttft, kv = ttft_decode(h, None, ids, 0, max_new)
    return dict(answer=text, ttft_s=ttft, kv_bytes=kv, prompt_tokens=len(ids),
                ingest_s=None, ok=bool(task["check"](text)))


def run_runtime(h, task, global_text, max_new):
    """Block-causal table with the distractor block evicted; query is its own tier."""
    blocks = task["blocks"]
    chunk_texts = [b[1] for b in blocks]
    evict_flags = [b[2] for b in blocks]

    t_ingest = time.perf_counter()
    g_ids, g_cache, blk_caches, pos, G, total = build_table(
        h, global_text, chunk_texts, verbose=False)
    kept = [c for c, ev in zip(blk_caches, evict_flags) if not ev]
    dropped = [c for c, ev in zip(blk_caches, evict_flags) if ev]
    ingest = time.perf_counter() - t_ingest

    # Assembly WITH the eviction -- this is the runtime.
    tbl = concat_caches([g_cache] + kept)

    # Assembly WITHOUT the eviction, for the isolation assertion. Surviving blocks are the
    # SAME tensors in both, so the difference must be exactly zero: evicting a block must not
    # perturb the survivors. That is the claim the whole block design rests on.
    tbl_full = concat_caches([g_cache] + [c for c in blk_caches if c is not None])
    drift = _survivor_drift(tbl, tbl_full, G, pos, evict_flags)

    qids = h.tok(T.build_question(task["query"]), add_special_tokens=False)["input_ids"]
    # Original absolute positions for the query -- the "static" arm of Bob's Test 1, and the
    # choice Finding 23 established is correct.
    text, ttft, kv = ttft_decode(h, tbl, qids, total, max_new)

    return dict(answer=text, ttft_s=ttft, kv_bytes=kv, prompt_tokens=len(qids),
                ingest_s=ingest, ok=bool(task["check"](text)),
                evicted_blocks=sum(1 for e in evict_flags if e),
                kv_evicted_bytes=cache_bytes(concat_caches(dropped)) if dropped else 0,
                drift_evict=drift)


def _survivor_drift(tbl_evict, tbl_full, G, pos, evict_flags):
    """Max abs difference between surviving-block keys in the two assemblies.

    Must be 0.0: a survivor is the same object whether or not some OTHER block was dropped.
    Computed rather than asserted by construction, because "they are the same object" is the
    kind of claim that quietly stops being true when an assembly path is rewritten.
    """
    off_e, off_f = G, G
    worst = 0.0
    for (lo, hi), ev in zip(pos, evict_flags):
        n = hi - lo
        if n <= 0:
            continue
        if not ev:
            for li in range(len(tbl_evict.layers)):
                ke = tbl_evict.layers[li].keys[:, :, off_e:off_e + n, :]
                kf = tbl_full.layers[li].keys[:, :, off_f:off_f + n, :]
                if ke.numel():
                    d = (ke.float() - kf.float()).abs().max().item()
                    worst = max(worst, d)
            off_e += n
        off_f += n
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-7b,mistral-7b",
                    help="comma-separated short names from: " + ", ".join(MODELS))
    ap.add_argument("--tests", default="ruler,babilong,synthetic_agent,two_needle",
                    help="subset of: ruler,babilong,synthetic_agent,two_needle,swe_mini")
    ap.add_argument("--spam-lines", type=int, default=140)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--outdir", default=os.path.join(_HERE, "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wanted = [t.strip() for t in args.tests.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 100)
    print("BOB BENCHMARK MATRIX 2026-09-30")
    print("  models=%s" % ", ".join(models))
    print("  tests =%s" % ", ".join(wanted))
    print("  NOTE: distractor scaled to this box; scale is recorded per row, not implied to be 16k")
    print("=" * 100, flush=True)

    rows = []
    for m in models:
        if m not in MODELS:
            print("  SKIP unknown model %r (known: %s)" % (m, ", ".join(MODELS)), flush=True)
            continue
        hf_id = MODELS[m]
        is_llama = m.startswith("llama")
        if is_llama:
            print("\n  SKIP %s -- GATED (GatedRepoError 403). Accept the license at "
                  "huggingface.co/%s, then re-run." % (m, hf_id), flush=True)
            rows.append(dict(model=m, hf_id=hf_id, status="skipped_gated"))
            continue

        need = 9000 if any(k in m for k in NEEDS_8BIT) else 3500
        T.require_memory(need, "benchmark %s" % m)
        if any(k in m for k in NEEDS_8BIT):
            os.environ["CCAI_QUANT"] = "8bit"
        else:
            os.environ.pop("CCAI_QUANT", None)

        print("\n" + "-" * 100)
        print("MODEL %s  (%s)" % (m, hf_id), flush=True)
        h = T.Harness(hf_id)
        if not hasattr(h, "model_id"):
            h.model_id = hf_id
        global_text = SYS + "<|im_start|>user\n"

        for tname in wanted:
            if tname == "swe_mini":
                print("  %-16s NOT RUNNABLE -- %s" % (tname, TASKS.SWE_MINI_REASON), flush=True)
                rows.append(dict(model=m, test=tname, status=TASKS.SWE_MINI_STATUS,
                                 reason=TASKS.SWE_MINI_REASON))
                continue
            if tname not in TASKS.BUILDERS:
                print("  %-16s SKIP unknown test" % tname, flush=True)
                continue

            task = TASKS.build(tname, args.spam_lines)
            v = run_vanilla(h, task, global_text, args.max_new)
            r = run_runtime(h, task, global_text, args.max_new)

            row = dict(model=m, hf_id=hf_id, test=tname, status="ok",
                       expect=task["expect"], note=task["note"],
                       spam_lines=args.spam_lines,
                       vanilla=v, runtime=r)
            rows.append(row)

            print("  %-16s expect=%-28s" % (tname, task["expect"]), flush=True)
            print("      vanilla  %-4s ttft=%7.1fms  kv=%6.1fMB  tok=%d" % (
                "PASS" if v["ok"] else "fail", v["ttft_s"] * 1000,
                v["kv_bytes"] / 1e6, v["prompt_tokens"]), flush=True)
            print("      runtime  %-4s ttft=%7.1fms  kv=%6.1fMB  tok=%d  (ingest %.1fs, "
                  "evicted %d blk = %.1fMB)" % (
                      "PASS" if r["ok"] else "fail", r["ttft_s"] * 1000,
                      r["kv_bytes"] / 1e6, r["prompt_tokens"], r["ingest_s"],
                      r["evicted_blocks"], r["kv_evicted_bytes"] / 1e6), flush=True)
            print("      survivor drift under eviction = %.3e (must be 0)" % r["drift_evict"],
                  flush=True)
            print("      answer: %r" % v["answer"][:90].replace("\n", " "), flush=True)
            print("           -> %r" % r["answer"][:90].replace("\n", " "), flush=True)

        del h
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    out = os.path.join(args.outdir, "benchmark_results_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(out, "w") as f:
        json.dump(dict(models=models, tests=wanted, spam_lines=args.spam_lines,
                       max_new=args.max_new, rows=rows), f, indent=2)
    print("\n" + "=" * 100)
    print("wrote %s  (%d rows)" % (out, len(rows)))
    print(_summary(rows))
    return out


def _summary(rows):
    live = [r for r in rows if r.get("status") == "ok"]
    if not live:
        return "no completed rows"
    out = ["", "  %-14s %-16s %-10s %-12s %-12s %s" % (
        "model", "test", "vanilla", "rt_ttft", "kv_saved", "drift")]
    for r in live:
        v, rt = r["vanilla"], r["runtime"]
        kvsave = (1 - rt["kv_bytes"] / v["kv_bytes"]) * 100 if v["kv_bytes"] else 0.0
        out.append("  %-14s %-16s %-10s %8.1fms  %7.1f%%  %.2e" % (
            r["model"], r["test"], "PASS" if v["ok"] else "fail",
            rt["ttft_s"] * 1000, kvsave, rt["drift_evict"]))
    return "\n".join(out)


if __name__ == "__main__":
    main()
