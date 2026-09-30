"""Bob's Test 4, runnable half: multi-turn serving cost, block runtime vs naive recompute.

    modal run modal/multiturn.py --model qwen2.5-7b --turns 8 --evictions 4

8 turns, 4 evictions. Two arms, same model, same task, same final question:

  vanilla   STANDARD SERVING. Every turn re-prefills the whole current context from token 0.
            An eviction or an appended turn invalidates nothing that was cached, because
            nothing was cached. Bob's "100% cache miss on edits".
  runtime   THE BLOCK TABLE. Blocks ingested ONCE and retained; an eviction drops that block's
            KV; each turn forwards only the new turn text plus the query.

★ WHAT THIS DOES *NOT* CLAIM. Finding 35 established that the block runtime cannot perform a
cross-block JOIN, so the runtime's answer here is expected to be wrong and is recorded rather
than hidden. The claim under test in THIS script is narrower and separate: what the runtime
costs in prefill tokens, wall clock and resident KV compared with re-prefilling everything,
and whether it can still answer the LATE-turn fact it directly witnessed.

★ THE CONTROL ARM IS THE POINT. Without it a "runtime is 90% cheaper" headline is worthless,
because a broken runtime is also cheap. The control carries the identical context in one causal
sequence so accuracy and cost are read side by side.
"""
import modal, os

app = modal.App("ccai-multiturn")
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

    # ★ meta-llama/Llama-3.1-8B is GATED (403 on file access, on this box AND on Modal,
    # even though the API metadata endpoint returns 200). NousResearch hosts an UNGATED
    # mirror of the same checkpoint -- verified model_type=llama, hidden_size=4096,
    # 32 heads, intermediate 14336, 128k ctx, gated:False. Using the mirror means the
    # cross-architecture claim can be tested on real Llama instead of only reporting the
    # blocker, and it is the same architecture either way.
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
}


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=3600, memory=32768)
def run_multi(model: str = "qwen2.5-7b", turns: int = 8, evictions: int = 4,
              max_new: int = 40):
    import sys, time, torch
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
    h = H(); h.model = m; h.tok = tok; h.model_id = mid

    SYS = ("<|im_start|>system\nYou are a systems agent. Maintain only the active "
           "configuration state.\n<|im_end|>\n")
    gtext = SYS + "<|im_start|>user\n"

    def turn(i):
        if i % 2 == 0 and i > 0:
            return ("Turn %d\nCommand: ip addr add 192.168.1.%d/24 dev eth0\n"
                    "Output: RTNETLINK answers: Operation not permitted. Failed with error "
                    "code 1." % (i, 50 + i))
        return ("Turn %d\nCommand: sudo ip addr add 192.168.1.%d/24 dev eth0\n"
                "Output: Success. Interface eth0 bound to 192.168.1.%d/24." % (i, 50 + i, 50 + i))

    blocks = [("Task Spec", "Objective: Configure static IP for interface eth0. "
                            "Subnet: 192.168.1.0/24. Failed attempts are not active state.")]
    for i in range(1, turns):
        blocks.append(("Turn %d" % i, turn(i)))
    evict_idx = [i for i in range(1, len(blocks)) if i % 2 == 0][:evictions]

    Q = ("What is the most recent IP address successfully assigned to eth0? "
         "Reply with only the address.")
    Q_LAST = blocks[-1][1]           # the most recent successful bind
    assert "192.168.1." in Q_LAST

    # ★ THE CHECK MUST TARGET A FACT THAT SURVIVED. Evicted turns are the FAILED attempts;
    # the newest successful bind is in the final block and is never evicted, so both arms are
    # answering about content they both retain. That isolates cost from the F35 join failure.
    import re
    def ok(a):
        nums = re.findall(r"192\.168\.\d+\.\d+", a)
        return bool(nums) and nums[-1] == "192.168.1.%d" % (50 + (turns - 1))

    def prefill_chunked(ids, chunk=2048):
        c = None; last = None
        for lo in range(0, len(ids), chunk):
            part = ids[lo:lo + chunk]
            pos = torch.arange(lo, lo + len(part), dtype=torch.long, device=dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                o = m(input_ids=torch.tensor([part], dtype=torch.long, device=dev),
                      past_key_values=c, use_cache=True, position_ids=pos.unsqueeze(0),
                      cache_position=pos)
            c = o.past_key_values; last = o.logits[:, -1, :]
            if c is None:
                break
        return c, last, time.perf_counter() - t0

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

    def kvb(c):
        if c is None or len(c.layers) == 0:
            return 0
        n = 0
        for li in range(len(c.layers)):
            k, v = c.layers[li].keys, c.layers[li].values
            n += k.numel() * k.element_size() + v.numel() * v.element_size()
        return n

    qturn = "<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n"
    live = [i for i in range(len(blocks)) if i not in evict_idx]

    # ---------------- vanilla: re-prefill everything, every turn
    v_rows = []
    for t in range(len(blocks)):
        ctx = [blocks[i][1] for i in live if i <= t]
        ids = tok(gtext + "\n".join(ctx) + qturn, add_special_tokens=False)["input_ids"]
        c, last, dt = prefill_chunked(ids)
        ans = decode(c, last, len(ids), max_new)
        v_rows.append(dict(turn=t, tokens=len(ids), wall_s=dt, kv=kvb(c), ans=ans))
        del c, last
        torch.cuda.empty_cache()
    v_final = v_rows[-1]

    # ---------------- runtime: ingest once, reuse
    t0 = time.perf_counter()
    g_ids, g_cache, blk, pos, G, total = TC.build_table(
        h, gtext, [b[1] for b in blocks], verbose=False)
    ingest_s = time.perf_counter() - t0
    r_rows = []
    for t in range(len(blocks)):
        kept = [c for i, c in enumerate(blk) if c is not None and i not in evict_idx and i <= t]
        tbl = TC.concat_caches([g_cache] + kept)
        newtext = ("<|im_start|>user\n" + blocks[t][1] + "\n" + Q +
                   "<|im_end|>\n<|im_start|>assistant\n")
        qids = tok(newtext, add_special_tokens=False)["input_ids"]
        base = TC.cache_len(tbl)
        ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
        p = torch.arange(base, base + len(qids), dtype=torch.long, device=dev)
        ts = time.perf_counter()
        with torch.no_grad():
            o = m(input_ids=ids_t, past_key_values=tbl, use_cache=True,
                  position_ids=p.unsqueeze(0), cache_position=p)
        dt = time.perf_counter() - ts
        ans = decode(o.past_key_values, o.logits[:, -1, :], base + len(qids), max_new)
        r_rows.append(dict(turn=t, tokens=len(qids), wall_s=dt, kv=kvb(tbl),
                           cached=base, ans=ans))
        del tbl
        torch.cuda.empty_cache()
    r_final = r_rows[-1]

    # ---------------- control: the survivors in ONE causal sequence, full context
    ctx = [blocks[i][1] for i in live]
    ids = tok(gtext + "\n".join(ctx) + qturn, add_special_tokens=False)["input_ids"]
    c, last, dt = prefill_chunked(ids)
    ctrl_ans = decode(c, last, len(ids), max_new)
    ctrl = dict(tokens=len(ids), wall_s=dt, kv=kvb(c), ans=ctrl_ans, ok=ok(ctrl_ans))

    v_tok = sum(r["tokens"] for r in v_rows); r_tok = sum(r["tokens"] for r in r_rows)
    v_wall = sum(r["wall_s"] for r in v_rows); r_wall = sum(r["wall_s"] for r in r_rows)
    return dict(model=model, turns=turns, evicted=evict_idx,
                expect="192.168.1.%d" % (50 + turns - 1),
                vanilla=dict(rows=v_rows, total_tokens=v_tok, wall_s=v_wall, final=v_final,
                             ok=ok(v_final["ans"])),
                runtime=dict(rows=r_rows, total_tokens=r_tok, wall_s=r_wall, final=r_final,
                             ingest_s=ingest_s, ok=ok(r_final["ans"])),
                control=ctrl)


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", turns: int = 8, evictions: int = 4, max_new: int = 40):
    import json, os, time
    ms = [x.strip() for x in model.split(",") if x.strip()]
    out = []
    for m in ms:
        print("\n===== %s =====" % m, flush=True)
        try:
            r = run_multi.remote(m, turns, evictions, max_new)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            out.append({"model": m, "status": "failed", "error": str(e)[:300]})
            continue
        out.append(r)
        v, rt, c = r["vanilla"], r["runtime"], r["control"]
        print("  expect=%s  evicted=%s" % (r["expect"], r["evicted"]), flush=True)
        print("  %-10s %-10s %-10s %-12s %s" % ("arm", "answer", "tokens", "wall", "kv_final"))
        print("  %-10s %-10s %-10d %8.2fs    %.1fMB" % (
            "control", "PASS" if c["ok"] else "fail", c["tokens"], c["wall_s"],
            c["kv"] / 1e6), flush=True)
        print("  %-10s %-10s %-10d %8.2fs    %.1fMB" % (
            "vanilla", "PASS" if v["ok"] else "fail", v["total_tokens"], v["wall_s"],
            v["final"]["kv"] / 1e6), flush=True)
        print("  %-10s %-10s %-10d %8.2fs    %.1fMB  (ingest %.1fs)" % (
            "runtime", "PASS" if rt["ok"] else "fail", rt["total_tokens"], rt["wall_s"],
            rt["final"]["kv"] / 1e6, rt["ingest_s"]), flush=True)
        if v["total_tokens"]:
            print("  prefill tokens: %.1f%% of vanilla   wall: %.1f%% of vanilla" % (
                rt["total_tokens"] / v["total_tokens"] * 100,
                rt["wall_s"] / v["wall_s"] * 100 if v["wall_s"] else 0), flush=True)
        print("  ans C=%r" % c["ans"][:70].replace("\n", " "), flush=True)
        print("  ans V=%r" % v["final"]["ans"][:70].replace("\n", " "), flush=True)
        print("  ans R=%r" % rt["final"]["ans"][:70].replace("\n", " "), flush=True)

    outdir = "/tmp/opencode/ccai/benchmarks/results"
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "multiturn_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
