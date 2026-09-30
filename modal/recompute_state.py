"""Does the RECOMPUTE path close the cross-block state UPDATE that the block table cannot?

    modal run modal/recompute_state.py --model qwen2.5-7b --trials 10

THE OPEN DEFECT. Two things were separated this session:

  JOIN          combine a fact from block 1 with a fact from block 3 (port:secret)
                SOLVED by contiguity — 10/10, both architectures (Finding 37)
  UPDATE        three sequential moves across two surviving blocks; report the LAST
                NOT solved — direct decode is 30%, scratchpad is no better (Finding 40)

The remaining candidate is the recompute path: instead of assembling independent block caches,
forward the SURVIVORS AS ONE CONTIGUOUS SEQUENCE over the global anchor. Findings 18/21 reached
this from a drift direction and probe_gap's arm 5 passed with it.

THREE ARMS, same model, same randomised task, same final question:
  block      block table + the Finding 37 contiguity fix      (the shipped runtime)
  recompute  survivors re-prefilled as ONE contiguous span     (the candidate fix)
  full       every block in one causal sequence, distractor    (upper bound / is it answerable)

★ THE COST DIFFERENCE IS THE WHOLE POINT AND IS MEASURED, NOT ARGUED. The recompute path pays a
prefill of the surviving text on every eviction; the block table pays nothing. So if `recompute`
closes the gap, the result is "correctness costs a re-prefill" — the same trade Findings 18 and 21
already priced at 34-37% KV saved for a re-prefill. Reporting it as a free win would be wrong.

★ THE TASK IS RANDOMISED so no answer is memorisable: three distinct places sampled from ten, two
different actors per trial.
"""
import modal, os

app = modal.App("ccai-recompute")
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
def run_recompute(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600,
                  max_new: int = 40):
    import sys, random, time, gc, torch
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
    SYS = "You are a deterministic state tracker. Answer only from verified positional events."

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
        return "<s>[INST] " + SYS + " <build_log>\n" + body + "\n</build_log>\n" + q + " [/INST]"

    def decode(cache, first, pos0, max_new):
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

    def prefill(ids, chunk=2048):
        c = None; last = None
        for lo in range(0, len(ids), chunk):
            part = ids[lo:lo + chunk]
            pos = torch.arange(lo, lo + len(part), dtype=torch.long, device=dev)
            with torch.no_grad():
                o = m(input_ids=torch.tensor([part], dtype=torch.long, device=dev),
                      past_key_values=c, use_cache=True, position_ids=pos.unsqueeze(0),
                      cache_position=pos)
            c = o.past_key_values; last = o.logits[:, -1, :]
        return c, last

    def kvb(c):
        if c is None or len(c.layers) == 0:
            return 0
        n = 0
        for li in range(len(c.layers)):
            k, v = c.layers[li].keys, c.layers[li].values
            n += k.numel() * k.element_size() + v.numel() * v.element_size()
        return n

    rng = random.Random(20260930)
    rows = []
    with track("recompute_state", "%s|trials=%d" % (model, trials)) as _t:
        _t.note(gpu=os.environ.get("CCAI_GPU", "A10G"), trials=trials)
        for t in range(trials):
            p1, p2, p3 = rng.sample(PLACES, 3)
            who1 = rng.choice(["Richard", "Marcus", "Elena"])
            who2 = rng.choice(["Jessica", "Priya", "Tomas"])
            b1 = ("Yesterday, the blue key was placed in the %s. %s took the blue key from the "
                  "%s and placed it in the %s." % (p1, who1, p1, p2))
            b2 = spam(spam_lines)
            b3 = ("This morning, %s went to the %s, picked up the blue key, and brought it to "
                  "the %s." % (who2, p2, p3))
            want = p3
            Q = "Where is the blue key now? Reply with only the location name."

            def ok(a):
                return want.lower() in a.lower()

            row = dict(trial=t + 1, want=want, p1=p1, p2=p2, p3=p3)

            # ---- arm FULL: every block in one causal sequence (upper bound)
            ids = tok(wrap(b1 + "\n" + b2 + "\n" + b3, Q), add_special_tokens=False)["input_ids"]
            c, last = prefill(ids)
            a = decode(c, last, len(ids), max_new)
            row["full"] = ok(a); row["full_ans"] = a[:40]
            row["full_tokens"] = len(ids); row["full_kv"] = kvb(c)
            del c; gc.collect(); torch.cuda.empty_cache()

            # ---- shared ingest for the two block arms
            gtext = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                     if use_chatml else "<s>[INST] " + SYS + "\n")
            g_ids, g_cache, blk, pos, G, total = TC.build_table(H, gtext, [b1, b2, b3],
                                                                verbose=False)
            qturn = ("<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n"
                     if use_chatml else Q + " [/INST]")
            qids = tok(qturn, add_special_tokens=False)["input_ids"]

            # ---- arm BLOCK: contiguity fix (the shipped runtime)
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
            ts = time.perf_counter()
            ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
            pp = torch.arange(base, base + len(qids), dtype=torch.long, device=dev)
            with torch.no_grad():
                o = m(input_ids=ids_t, past_key_values=tbl, use_cache=True,
                      position_ids=pp.unsqueeze(0), cache_position=pp)
            block_ms = (time.perf_counter() - ts) * 1000
            ab = decode(o.past_key_values, o.logits[:, -1, :], base + len(qids), max_new)
            row["block"] = ok(ab); row["block_ans"] = ab[:40]
            row["block_ms"] = block_ms; row["block_kv"] = kvb(tbl)
            row["block_prefill_tokens"] = len(qids)
            del tbl, o; gc.collect(); torch.cuda.empty_cache()

            # ---- arm RECOMPUTE: survivors forwarded as ONE contiguous span over the global cache
            ts = time.perf_counter()
            surv_ids = tok(b1 + "\n" + b3 + "\n" + qturn, add_special_tokens=False)["input_ids"]
            psv = list(range(G, G + len(surv_ids)))
            clone2 = TC.slice_cache(g_cache, 0, G)
            with torch.no_grad():
                o = m(input_ids=torch.tensor([surv_ids], dtype=torch.long, device=dev),
                      past_key_values=clone2, use_cache=True,
                      position_ids=torch.tensor([psv], dtype=torch.long, device=dev),
                      cache_position=torch.tensor(psv, dtype=torch.long, device=dev))
            rec_ms = (time.perf_counter() - ts) * 1000
            ar = decode(o.past_key_values, o.logits[:, -1, :], G + len(surv_ids), max_new)
            row["recompute"] = ok(ar); row["recompute_ans"] = ar[:40]
            row["recompute_ms"] = rec_ms; row["recompute_kv"] = kvb(o.past_key_values)
            row["recompute_prefill_tokens"] = len(surv_ids)
            del o, clone2, g_cache, blk; gc.collect(); torch.cuda.empty_cache()

            rows.append(row)
            print("  %2d want=%-14s full=%-4s block=%-4s(%s) recompute=%-4s(%s) | "
                  "block %.0fms/%dtok  recompute %.0fms/%dtok" % (
                      t + 1, want[:14], "PASS" if row["full"] else "fail",
                      "PASS" if row["block"] else "fail", str(row["block_ans"])[:12],
                      "PASS" if row["recompute"] else "fail", str(row["recompute_ans"])[:12],
                      block_ms, len(qids), rec_ms, len(surv_ids)), flush=True)

        n = len(rows)
        rates = {k: sum(1 for r in rows if r[k]) / n for k in ("full", "block", "recompute")}
        cost = dict(
            block_ms=sum(r["block_ms"] for r in rows) / n,
            recompute_ms=sum(r["recompute_ms"] for r in rows) / n,
            block_kv=sum(r["block_kv"] for r in rows) / n,
            recompute_kv=sum(r["recompute_kv"] for r in rows) / n,
            block_tokens=sum(r["block_prefill_tokens"] for r in rows) / n,
            recompute_tokens=sum(r["recompute_prefill_tokens"] for r in rows) / n,
        )
        return dict(model=model, trials=n, spam_lines=spam_lines, rates=rates, cost_summary=cost,
                    rows=rows, cost=list(ROWS))


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", trials: int = 10, spam_lines: int = 600, max_new: int = 40):
    import json, os, time
    ms = [x.strip() for x in model.split(",") if x.strip()]
    out = []
    ledger = "/tmp/opencode/ccai/benchmarks/results/cost_ledger.jsonl"
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    for mm in ms:
        print("\n===== RECOMPUTE vs BLOCK vs FULL: %s, %d trials =====" % (mm, trials), flush=True)
        try:
            r = run_recompute.remote(mm, trials, spam_lines, max_new)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            out.append({"model": mm, "status": "failed", "error": str(e)[:300]})
            continue
        out.append(r)
        for row in r.get("cost", []):
            row["cell"] = mm
            open(ledger, "a").write(json.dumps(row) + "\n")
        p, cs = r["rates"], r["cost_summary"]
        print("\n  %-22s full %3.0f%%   block %3.0f%%   recompute %3.0f%%" % (
            mm, p["full"] * 100, p["block"] * 100, p["recompute"] * 100))
        print("  %-22s block %.1fms / %.0f tok / %.1fMB   recompute %.1fms / %.0f tok / %.1fMB"
              % ("", cs["block_ms"], cs["block_tokens"], cs["block_kv"] / 1e6,
                 cs["recompute_ms"], cs["recompute_tokens"], cs["recompute_kv"] / 1e6))
        print("  recompute closes the gap if recompute > block, at the stated cost.")

    p = os.path.join("/tmp/opencode/ccai/benchmarks/results",
                     "recompute_state_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
