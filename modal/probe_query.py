"""Is the block-runtime failure ARCHITECTURAL or an implementation bug in my decoder?

The bench run showed the runtime arm failing 4/4 on Qwen2.5-7B while the same surviving content
in one causal sequence passes 4/4. Drift is 0.0 throughout, so the blocks themselves are intact.

Two candidate explanations, and they need opposite responses:

  (a) ARCHITECTURAL. Separate forwards genuinely cannot support the cross-block join, so the
      block table is not a viable serving shape for this class of question.
  (b) MY DECODER. `prefill_and_decode` passes position_ids AND cache_position explicitly, and
      places the query at `total` (the length including the EVICTED block). The repo's verified
      `tiered_cache.generate()` does the same, but the two have never been compared on one
      assembled table.

This isolates the variable: ONE table, ONE model, two decode paths, same question. If
`generate()` answers correctly and `prefill_and_decode` does not, the finding is about my code
and nothing else. If both fail identically, the finding is about the architecture.

Usage:  modal run modal/probe_query.py
"""
import modal

app = modal.App("ccai-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=1800)
def probe():
    import sys, os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, "/root/ccai/src")
    import tiered_cache as TC

    dev = "cuda"
    mid = "Qwen/Qwen2.5-7B"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16,
                                                 device_map=dev)
    model.eval()

    class H:
        pass
    h = H(); h.model = model; h.tok = tok; h.model_id = mid

    SPAM = ["gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
            "src/core/pool.h:%d:12: warning: unused parameter 'flags'",
            "ninja: build stopped: subcommand failed at phase 0x%02x"]
    spam = "\n".join((SPAM[i % 3] % (i % 256)) if "%" in SPAM[i % 3] else SPAM[i % 3]
                     for i in range(140))

    SYS = ("<|im_start|>system\nYou are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
           "Always answer with the exact value requested, quoting it verbatim.\n<|im_end|>\n")
    global_text = SYS + "<|im_start|>user\n"
    b1 = ("Yesterday, the blue key was placed in the kitchen drawer. Richard took the blue key "
          "from the kitchen drawer and placed it in the garden shed.")
    b2 = spam
    b3 = ("This morning, Jessica went to the garden shed, picked up the blue key, and brought "
          "it to the attic.")
    Q = ("Where is the blue key now? Give only the final location name.")
    q_turn = ("<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n")
    qids = tok(q_turn, add_special_tokens=False)["input_ids"]

    g_ids, g_cache, blk, pos, G, total = TC.build_table(h, global_text, [b1, b2, b3],
                                                        verbose=False)
    kept = [blk[0], blk[2]]
    tbl = TC.concat_caches([g_cache] + kept)
    print("G=%d total=%d cache_len(tbl)=%d  (evicted block is %d tok)" % (
        G, total, TC.cache_len(tbl), TC.cache_len(blk[1])), flush=True)

    # ---- path A: the repo's own verified generate(), unchanged
    a = TC.generate(h, TC.slice_cache(tbl, 0, TC.cache_len(tbl)), qids, total, max_new=24)
    print("A  tiered_cache.generate (qpos=total)      -> %r" % a[:120], flush=True)

    # ---- path B: same table, but the query positioned contiguously after the survivors
    b = TC.generate(h, TC.slice_cache(tbl, 0, TC.cache_len(tbl)), qids,
                    TC.cache_len(tbl), max_new=24)
    print("B  tiered_cache.generate (qpos=cache_len)  -> %r" % b[:120], flush=True)

    # ---- path C: my bench decoder, qpos=total
    def mine(cache, ids, pos0, max_new=24):
        ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
        p = torch.arange(pos0, pos0 + len(ids), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = model(input_ids=ids_t, past_key_values=cache, use_cache=True,
                      position_ids=p.unsqueeze(0), cache_position=p)
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt]; q = pos0 + len(ids)
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = model(input_ids=nxt, past_key_values=c, use_cache=True,
                          cache_position=torch.tensor([q], device=dev))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt); q += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        return tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
    c = mine(TC.slice_cache(tbl, 0, TC.cache_len(tbl)), qids, total, 24)
    print("C  bench prefill_and_decode (qpos=total)   -> %r" % c[:120], flush=True)

    # ---- control: the surviving blocks as ONE causal sequence, no blocks, no eviction
    seq = global_text + b1 + "\n" + b3 + q_turn
    ids = tok(seq, add_special_tokens=False)["input_ids"]
    d = mine(None, ids, 0, 24)
    print("D  control, one causal sequence            -> %r" % d[:120], flush=True)

    # ---- control 2: full context INCLUDING the distractor, one causal sequence
    seq2 = global_text + b1 + "\n" + b2 + "\n" + b3 + q_turn
    ids2 = tok(seq2, add_special_tokens=False)["input_ids"]
    e = mine(None, ids2, 0, 24)
    print("E  control, full causal + distractor       -> %r" % e[:120], flush=True)

    return dict(A=a, B=b, C=c, D=d, E=e, G=G, total=total,
                cache_len=TC.cache_len(tbl), block_toks=[TC.cache_len(x) for x in blk])


@app.local_entrypoint()
def main():
    r = probe.remote()
    print("\n" + "=" * 80)
    expect = "attic"
    for k in ("A", "B", "C", "D", "E"):
        v = r[k]
        print("  %s  %-4s  %r" % (k, "PASS" if expect in v.lower() else "fail", v[:90]))
    print("=" * 80)
    print("  G=%d total=%d cache_len=%d block_toks=%s" % (
        r["G"], r["total"], r["cache_len"], r["block_toks"]))
