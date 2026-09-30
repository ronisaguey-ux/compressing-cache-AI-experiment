"""Bob's Solution 1 test: does a forced SEQUENTIAL SCRATCHPAD close the state-update failure?

    modal run modal/scratchpad.py --model qwen2.5-7b --trials 10

THE RESIDUAL THIS TARGETS. The cross-block JOIN is fixed (Finding 37: re-index survivors
contiguously -> 10/10 both models). The task that still fails is a state UPDATE: the key moves
kitchen -> shed -> attic across two surviving blocks and the query must report the LAST move. The
runtime answers "garden shed" — block 1's state — so it reads the earlier fact and never applies
the later one.

BOB'S HYPOTHESIS. Force the model to write an explicit trace instead of answering in one forward
pass: "Trace each movement sequentially: 1. First, ...". Each generated token appends to the active
KV, so the trace tokens themselves re-link the isolated blocks — the join happens in DECODE rather
than in the query's single forward pass.

★ THIS IS A GENUINE MECHANISM, NOT A STYLE CHANGE, AND IT IS TESTABLE AS ONE. If it works, the
trace tokens must be doing the re-linking, so the test compares:

    direct   query -> answer in one forward                        (the current behaviour)
    trace    query -> "Step 1:" ... then the final answer          (Bob's Solution 1)

★ AND A THIRD ARM, because a result with two arms cannot distinguish "the trace helped" from "the
trace forced a second forward at all":
    trace-null   a trace with the SAME token count but no movement content ("Step 1: ok. Step 2:
                 ok."). If this matches `trace`, the win is the extra decode steps; if it matches
                 `direct`, the win is the trace's CONTENT.

All arms run on the SAME contiguity fix, so this isolates the decode strategy and nothing else.

★ WHAT WOULD FALSIFY BOB'S IDEA: trace == direct. What would confirm it: trace passes where direct
fails AND trace-null does not.
"""
import modal, os

app = modal.App("ccai-scratchpad")
GPU = os.environ.get("CCAI_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
    .add_local_file("/tmp/opencode/ccai/modal/cost.py", "/root/cost.py")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    # ungated mirror of meta-llama/Llama-3.1-8B (the original is 403 on file
    # access both locally and on Modal); verified same architecture.
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=5400, memory=32768)
def run_pad(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600,
            max_new: int = 96):
    import sys, random, re, gc, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, "/root/ccai/src")
    sys.path.insert(0, "/root")
    import tiered_cache as TC
    from cost import track, ROWS

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
    SYS = ("You are a deterministic state tracker. Answer only from verified positional events.")

    SPAM = ["gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
            "src/core/pool.h:%d:12: warning: unused parameter 'flags'",
            "ninja: build stopped: subcommand failed at phase 0x%02x",
            "ccache: cache miss for src/vm/stack.o (stats reset)"]

    def spam(n):
        return "\n".join((SPAM[i % len(SPAM)] % (i % 256)) if "%" in SPAM[i % len(SPAM)]
                         else SPAM[i % len(SPAM)] for i in range(n))

    PLACES = ["kitchen drawer", "garden shed", "attic", "basement", "garage", "cellar",
              "hallway closet", "study", "pantry", "loft"]

    def wrap(body, q):
        if use_chatml:
            return ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n<build_log>\n"
                    + body + "\n</build_log>\n" + q + "<|im_end|>\n<|im_start|>assistant\n")
        return ("<s>[INST] " + SYS + " <build_log>\n" + body + "\n</build_log>\n" + q + " [/INST]")

    def decode_from(cache, first, pos0, max_new, stop_at=None):
        """Greedy decode. `stop_at` lets the scratchpad arm cut at a sentinel phrase."""
        nxt = first.argmax(-1, keepdim=True); gen = [nxt]; c = cache; p = pos0
        text = tok.decode(nxt[0], skip_special_tokens=True)
        for _ in range(max_new - 1):
            if stop_at and stop_at.lower() in text.lower():
                break
            with torch.no_grad():
                o = m(input_ids=nxt, past_key_values=c, use_cache=True,
                      cache_position=torch.tensor([p], device=dev))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt); p += 1
            text = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
            if int(nxt.item()) == tok.eos_token_id:
                break
        return text, c, p

    rng = random.Random(20260930)
    rows = []
    # ★ INSTRUMENTED: the first version imported track() but never used it, so this run
    # cost real GPU-seconds that the ledger never saw. Wrap the work, not the import.
    _t = track("scratchpad", "%s|trials=%d" % (model, trials))
    _t.__enter__()
    _t.note(gpu=os.environ.get("CCAI_GPU", "A10G"), trials=trials)
    for t in range(trials):
        # three moves, three DIFFERENT places, answer is the last one
        picks = rng.sample(PLACES, 3)
        p1, p2, p3 = picks
        who1 = rng.choice(["Richard", "Marcus", "Elena"])
        who2 = rng.choice(["Jessica", "Priya", "Tomas"])
        b1 = ("Yesterday, the blue key was placed in the %s. %s took the blue key from the %s "
              "and placed it in the %s." % (p1, who1, p1, p2))
        b2 = spam(spam_lines)
        b3 = ("This morning, %s went to the %s, picked up the blue key, and brought it to the "
              "%s." % (who2, p2, p3))
        want = p3

        Q_DIRECT = "Where is the blue key now? Reply with only the location name."
        Q_TRACE = ("Where is the blue key now? Trace each movement sequentially, then state the "
                   "final location:\n1. First,")
        Q_NULL = ("Where is the blue key now? Then state the final location. Begin with:\n"
                  "Step 1: noted.\nStep 2: noted.\nFinal location:")

        def ok(a):
            return want.lower() in a.lower()

        # ---------------- one shared ingest, three decode strategies
        gtext = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                 if use_chatml else "<s>[INST] " + SYS + "\n")
        g_ids, g_cache, blk, pos, G, total = TC.build_table(H, gtext, [b1, b2, b3],
                                                            verbose=False)
        # Finding 37 fix: survivors re-indexed contiguously, query at the assembled length
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
        tbl = TC.assemble_contiguous(g_cache, [blk[0], blk3_adj])
        base = TC.cache_len(tbl)

        def ask(q):
            qq = ("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"
                  if use_chatml else q + " [/INST]")
            qids = tok(qq, add_special_tokens=False)["input_ids"]
            ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
            p = torch.arange(base, base + len(qids), dtype=torch.long, device=dev)
            with torch.no_grad():
                o = m(input_ids=ids_t, past_key_values=tbl, use_cache=True,
                      position_ids=p.unsqueeze(0), cache_position=p)
            return o.past_key_values, o.logits[:, -1, :], base + len(qids)

        c_d, l_d, p_d = ask(Q_DIRECT)
        a_direct, _, _ = decode_from(c_d, l_d, p_d, 24)
        del c_d

        c_t, l_t, p_t = ask(Q_TRACE)
        a_trace, _, _ = decode_from(c_t, l_t, p_t, max_new)
        del c_t

        c_n, l_n, p_n = ask(Q_NULL)
        a_null, _, _ = decode_from(c_n, l_n, p_n, max_new)
        del c_n

        # score the FINAL stated location, which is the last place mentioned
        def final_place(ans):
            hits = [pl for pl in PLACES if pl.lower() in ans.lower()]
            return hits[-1] if hits else None

        row = dict(trial=t + 1, want=want, p1=p1, p2=p2, p3=p3,
                   direct=ok(a_direct), trace=ok(a_trace), null=ok(a_null),
                   direct_ans=a_direct[:40], trace_ans=a_trace[:70].replace("\n", " "),
                   null_ans=a_null[:60].replace("\n", " "))
        rows.append(row)
        print("  %2d want=%-14s direct=%-4s(%s) trace=%-4s null=%-4s | %r" % (
            t + 1, want[:14], "PASS" if row["direct"] else "fail",
            str(final_place(a_direct))[:14], "PASS" if row["trace"] else "fail",
            "PASS" if row["null"] else "fail", a_trace[:38].replace("\n", " ")), flush=True)
        del tbl, g_cache
        gc.collect(); torch.cuda.empty_cache()

    _t.__exit__(None, None, None)
    n = len(rows)
    rates = {k: sum(1 for r in rows if r[k]) / n for k in ("direct", "trace", "null")}
    return dict(model=model, trials=n, spam_lines=spam_lines, rates=rates, rows=rows,
                cost=list(ROWS))


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600, max_new: int = 96):
    import json, os, time
    ms = [x.strip() for x in model.split(",") if x.strip()]
    out = []
    ledger = "/tmp/opencode/ccai/benchmarks/results/cost_ledger.jsonl"
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    for mm in ms:
        print("\n===== SCRATCHPAD (Bob's Solution 1): %s, %d trials =====" % (mm, trials),
              flush=True)
        try:
            r = run_pad.remote(mm, trials, spam_lines, max_new)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            out.append({"model": mm, "status": "failed", "error": str(e)[:300]})
            continue
        out.append(r)
        for row in r.get("cost", []):
            row["cell"] = mm
            open(ledger, "a").write(json.dumps(row) + "\n")
        p = r["rates"]
        print("\n  %-22s direct %3.0f%%   trace %3.0f%%   null-trace %3.0f%%" % (
            mm, p["direct"] * 100, p["trace"] * 100, p["null"] * 100), flush=True)
        print("  Bob's idea confirmed if trace > direct AND trace > null.")

    p = os.path.join("/tmp/opencode/ccai/benchmarks/results",
                     "scratchpad_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
