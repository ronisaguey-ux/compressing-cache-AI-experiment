"""Bob's TEST 1 on Modal: RoPE positional invariance under eviction.

    modal run modal/rope_offsets.py --model qwen2.5-7b

★ WHY THIS IS THE MOST INTERESTING OF BOB'S FOUR TESTS. His stated pass metric is:

    "Confirm that retaining original absolute positional coordinates preserves exact
     generation coherence WITHOUT requiring continuous re-indexing."

That is a HYPOTHESIS, and the 10-trial two-needle run (Finding 37) contradicts it:

    survivors keep ORIGINAL absolute ids (gapped)   0/10   both models
    survivors RE-INDEXED contiguously              10/10   both models

So the measurement here is not a formality — it is deciding which of two contradictory
statements is true. Everything below is arranged so the two are separated cleanly.

THREE ASSEMBLIES OF THE SAME SURVIVING TOKENS, same blocks, same KV, same query:
    full      the original sequence, nothing evicted            (reference)
    static    survivors KEEP original absolute position ids     (Bob's hypothesis)
    compact   survivors RE-INDEXED contiguously                 (Finding 37's fix)

★ WHAT IS MEASURED, and it is deliberately two different things because they can disagree:
  1. COHERENCE — does the model answer the question. This is the pass metric that matters.
  2. DRIFT — max abs difference in surviving-block keys vs the full run. This is the
     mechanism. Finding 23 established that preserving positions removes the POSITIONAL
     error but not the CONTENT error, because the survivors attended to the evicted tokens.

★ AN HONEST PREDICTION so the result cannot be read as a post-hoc story. Finding 37 predicts
`static` FAILS and `compact` PASSES. If `static` passes here, Finding 37's mechanism is wrong
and the gap explanation needs revisiting. Stated before the run.
"""
import modal, os

app = modal.App("ccai-rope")
GPU = os.environ.get("CCAI_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    # ungated mirror of meta-llama/Llama-3.1-8B (the original is 403 on file
    # access both locally and on Modal); verified same architecture.
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=3600, memory=32768)
def run_rope(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600,
             max_new: int = 40):
    import sys, random, re, gc, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, "/root/ccai/src")
    import tiered_cache as TC

    dev = "cuda"
    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=dev,
                                             attn_implementation="sdpa")
    m.eval()

    class H:
        pass
    H.tok = tok
    H.model = m
    H.model_id = mid

    use_chatml = "qwen" in model
    SYS = "You are a build assistant. Always answer with the exact value requested."

    SPAM = ["gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
            "src/core/pool.h:%d:12: warning: unused parameter 'flags'",
            "ninja: build stopped: subcommand failed at phase 0x%02x"]
    def spam(n):
        return "\n".join((SPAM[i % len(SPAM)] % (i % 256)) if "%" in SPAM[i % len(SPAM)]
                         else SPAM[i % len(SPAM)] for i in range(n))

    def greedy(cache, first, pos0, max_new):
        nxt = first.argmax(-1, keepdim=True); gen = [nxt]; c = cache; p = pos0
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = m(input_ids=nxt, past_key_values=c, use_cache=True,
                      cache_position=torch.tensor([p], device=dev))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True); gen.append(nxt); p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        return tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)

    def seq(ids, chunk=2048):
        c = None; last = None
        for lo in range(0, len(ids), chunk):
            part = ids[lo:lo + chunk]
            pos = torch.arange(lo, lo + len(part), dtype=torch.long, device=dev)
            with torch.no_grad():
                o = m(input_ids=torch.tensor([part], dtype=torch.long, device=dev),
                      past_key_values=c, use_cache=True, position_ids=pos.unsqueeze(0),
                      cache_position=pos)
            c = o.past_key_values; last = o.logits[:, -1, :]
        return greedy(c, last, len(ids), max_new), c

    rng = random.Random(20260930)
    rows = []
    for t in range(trials):
        port = rng.randint(1024, 65535)
        secret = "EXAMPLE_KEY_" + "".join(rng.choice("abcdef0123456789") for _ in range(12))
        want = "%d?auth=%s" % (port, secret)
        b1 = ('TARGET_PORT = "%d"\nFurther topology follows.' % port)
        b2 = spam(spam_lines)
        b3 = ('AUTH_SECRET = "%s"\nRotate credentials on schedule.' % secret)
        Q = ("Synthesize <TARGET_PORT>?auth=<AUTH_SECRET> as one string. "
             "Reply with only that string.")

        def wrap(body):
            if use_chatml:
                return ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                        "<build_log>\n" + body + "\n</build_log>\n" + Q +
                        "<|im_end|>\n<|im_start|>assistant\n")
            return ("<s>[INST] " + SYS + "\n<build_log>\n" + body + "\n</build_log>\n"
                    + Q + " [/INST]")

        def ok(a):
            return want in re.sub(r"\s+", "", a)

        # ---------- FULL: nothing evicted, one causal sequence (reference)
        full_txt = wrap(b1 + "\n" + b2 + "\n" + b3)
        full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
        full_ans, full_cache = seq(full_ids)
        full_ok = ok(full_ans)

        # ---------- STATIC: survivors keep ORIGINAL absolute position ids (Bob's hypothesis)
        gtext = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                 if use_chatml else "<s>[INST] " + SYS + "\n")
        g_ids, g_cache, blk, pos, G, total = TC.build_table(H, gtext, [b1, b2, b3],
                                                            verbose=False)
        qturn = ("<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n"
                 if use_chatml else Q + " [/INST]")
        qids = tok(qturn, add_special_tokens=False)["input_ids"]

        static_tbl = TC.concat_caches([g_cache, blk[0], blk[2]])   # ORIGINAL ids -> gap
        ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
        p = torch.arange(total, total + len(qids), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = m(input_ids=ids_t, past_key_values=static_tbl, use_cache=True,
                  position_ids=p.unsqueeze(0), cache_position=p)
        static_ans = greedy(o.past_key_values, o.logits[:, -1, :], total + len(qids), max_new)

        # ---------- COMPACT: survivors RE-INDEXED contiguously (Finding 37's fix)
        # Rebuilt as the survivors occupying a contiguous span, then the query at its true end.
        n1 = TC.cache_len(blk[0]); n3 = TC.cache_len(blk[2])
        b3_ids = tok(b3, add_special_tokens=False)["input_ids"]
        clone = TC.slice_cache(g_cache, 0, G)
        p_adj = list(range(G + n1, G + n1 + len(b3_ids)))
        with torch.no_grad():
            o = m(input_ids=torch.tensor([b3_ids], dtype=torch.long, device=dev),
                  past_key_values=clone, use_cache=True,
                  position_ids=torch.tensor([p_adj], dtype=torch.long, device=dev),
                  cache_position=torch.tensor(p_adj, dtype=torch.long, device=dev))
        blk3_adj = TC.slice_cache(o.past_key_values, G, G + len(b3_ids))
        compact_tbl = TC.assemble_contiguous(g_cache, [blk[0], blk3_adj])
        qp = TC.cache_len(compact_tbl)
        p2 = torch.arange(qp, qp + len(qids), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = m(input_ids=ids_t, past_key_values=compact_tbl, use_cache=True,
                  position_ids=p2.unsqueeze(0), cache_position=p2)
        compact_ans = greedy(o.past_key_values, o.logits[:, -1, :], qp + len(qids), max_new)

        # ---------- DRIFT: surviving-block keys vs the FULL run's, at the same content.
        # Both static and compact hold the same tensors, so both are compared at their own
        # slice offsets against the surviving rows of the full cache.
        def drift(cache_tbl, tbl_start, n, full_cache, full_start):
            worst = 0.0
            for li in range(min(len(cache_tbl.layers), len(full_cache.layers))):
                a = cache_tbl.layers[li].keys[:, :, tbl_start:tbl_start + n, :]
                b = full_cache.layers[li].keys[:, :, full_start:full_start + n, :]
                if a.numel():
                    worst = max(worst, (a.float() - b.float()).abs().max().item())
            return worst

        static_drift = drift(static_tbl, G, n1, full_cache, G)
        compact_drift = drift(compact_tbl, G, n1, full_cache, G)

        row = dict(trial=t + 1, want=want,
                   full=full_ok, static=ok(static_ans), compact=ok(compact_ans),
                   static_drift=static_drift, compact_drift=compact_drift,
                   static_ans=static_ans[:50], compact_ans=compact_ans[:50])
        rows.append(row)
        print("  %2d full=%-4s static=%-4s compact=%-4s  static_drift=%.2e compact_drift=%.2e"
              % (t + 1, "PASS" if full_ok else "fail", "PASS" if ok(static_ans) else "fail",
                 "PASS" if ok(compact_ans) else "fail", static_drift, compact_drift),
              flush=True)
        del full_cache, static_tbl, compact_tbl, g_cache
        gc.collect()
        torch.cuda.empty_cache()

    n = len(rows)
    rates = {k: sum(1 for r in rows if r[k]) / n for k in ("full", "static", "compact")}
    return dict(model=model, trials=n, rates=rates,
                mean_static_drift=sum(r["static_drift"] for r in rows) / n,
                mean_compact_drift=sum(r["compact_drift"] for r in rows) / n,
                rows=rows)


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600,
         max_new: int = 40):
    import json, os, time
    ms = [x.strip() for x in model.split(",") if x.strip()]
    allr = []
    for mm in ms:
        print("\n===== TEST 1 RoPE positional invariance: %s (%d trials) =====" % (mm, trials),
              flush=True)
        try:
            r = run_rope.remote(mm, trials, spam_lines, max_new)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            allr.append({"model": mm, "status": "failed", "error": str(e)[:300]})
            continue
        allr.append(r)
        p = r["rates"]
        print("\n  %-22s full %3.0f%%   STATIC(orig ids) %3.0f%%   COMPACT(re-indexed) %3.0f%%"
              % (mm, p["full"] * 100, p["static"] * 100, p["compact"] * 100), flush=True)
        print("  mean key drift vs full:  static %.3e   compact %.3e"
              % (r["mean_static_drift"], r["mean_compact_drift"]), flush=True)
        print("  Bob's pass metric = STATIC. Finding 37 predicts COMPACT is the one that holds.")

    outdir = "/tmp/opencode/ccai/benchmarks/results"
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "rope_offsets_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(allr, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
