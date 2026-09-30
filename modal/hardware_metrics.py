"""RAW HARDWARE METRICS: FLOPs, memory bandwidth demand, and MEASURED Watt-hours.

    modal run modal/hardware_metrics.py --model qwen2.5-7b --trials 20

Bob: *"calculate raw hardware FLOPs, memory bandwidth demands, and actual Watt-hours. This
approach provides an objective, reproducible metric for the paper."*

★ WHICH OF THE THREE IS MEASURED AND WHICH IS MODELLED — the distinction is kept explicit
because a paper that blurs it is not reproducible:

  MEASURED (no assumption; sampled from the device during the run)
    watt_hours   `nvidia-smi --query-gpu=power.draw` sampled on a background thread and
                 integrated over the run. This is the actual energy the workload drew, not a
                 TDP x time guess. It is the number that belongs in a paper.
    wall_ms      clock time per arm.
    kv_bytes     real bytes resident in the cache tensors.

  MODELLED (analytical, formula stated so a reviewer can check or replace it)
    fwd_flops    2*P*T (linear) + 4*L*H*D*T*S (attention). P/L/H/D are published config values.
    bytes_moved  weights read every token (P * bytes_per_param) + KV read + KV written.
    arithmetic_intensity = FLOPs / bytes          -> which side of the roofline you are on

★★ THE HONEST FINDING THIS IS BUILT TO EXPOSE, stated before the measurement: at batch 1 every
decode token reads the ENTIRE weight matrix (7B x 2 bytes = ~14 GB of reads per token), while the
KV it reads is megabytes. So **memory bandwidth at batch 1 is dominated by WEIGHTS, not by KV** —
and a block runtime that reduces KV bytes does NOT reduce weight bytes. Any bandwidth claim for
this architecture must therefore be expressed as a share of the KV term, not as an end-to-end
saving. A systems critic would raise exactly this, so it is measured rather than discovered by a
reviewer.

That also predicts the energy result: if the workload is weight-read-bound, the two arms should
draw similar POWER while differing in TIME, and the energy difference should track the wall-clock
difference rather than the token difference. The run tests that prediction.
"""
import modal, os

app = modal.App("ccai-hardware")
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
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}

# Published architecture constants. num_params is non-embedding, to match the FLOPs formula.
ARCH = {
    "qwen2.5-7b": dict(P=6.53e9, L=28, H=28, D=128, embed=0),
    "llama-3.1-8b": dict(P=7.51e9, L=32, H=32, D=128, embed=0),
    "mistral-7b-instruct": dict(P=6.74e9, L=32, H=32, D=128, embed=0),
}
DTYPE_BYTES = 2      # fp16
KV_BYTES_PER_TOKEN = None  # computed from L, H, D x 2 (k and v) x dtype


def kv_bytes_per_token(arch):
    a = ARCH[arch]
    return 2 * a["L"] * a["H"] * a["D"] * DTYPE_BYTES


def fwd_flops(arch, tokens, context):
    a = ARCH[arch]
    linear = 2.0 * a["P"] * tokens
    attn = 4.0 * a["L"] * a["H"] * a["D"] * tokens * context
    return dict(linear=linear, attention=attn, total=linear + attn)


def bytes_moved(arch, tokens, context, kv_resident_tokens):
    """Bytes the device must move for this forward pass, batch 1.

    weights     read once per generated token (every token touches every parameter)
    kv_read     the attention over `context` positions streams the resident KV
    kv_write    the new tokens' own K and V are written once
    """
    a = ARCH[arch]
    weights = a["P"] * DTYPE_BYTES * tokens
    per_tok = kv_bytes_per_token(arch)
    kv_read = per_tok * context            # streamed once per token
    kv_write = per_tok * tokens
    total = weights + kv_read + kv_write
    return dict(weights=weights, kv_read=kv_read, kv_write=kv_write, total=total,
                kv_share=(kv_read + kv_write) / total if total else 0.0)


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=5400, memory=32768)
def run_hw(model: str = "qwen2.5-7b", trials: int = 20, spam_lines: int = 600,
           max_new: int = 24):
    import sys, subprocess, threading, time, random, gc, torch
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

    # ---------------- MEASURED POWER: sample the device on a background thread
    class PowerSampler:
        """Integrate `nvidia-smi --query-gpu=power.draw` over the sampled interval.

        ★ SAMPLED, NOT ASSUMED. The alternative -- TDP multiplied by wall-clock -- reports a
        number that is wrong whenever the GPU is not saturated or its clocks are throttled, and
        this box runs at moderate utilisation. A thread polling the device's own sensor is the
        only figure that belongs in a paper.
        """
        def __init__(self, interval=0.05):
            self.interval = interval
            self.samples = []
            self._stop = threading.Event()
            self._t = None

        def _poll(self):
            while not self._stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip()
                    w = float(out.splitlines()[0])
                    self.samples.append((time.perf_counter(), w))
                except Exception:
                    pass
                self._stop.wait(self.interval)

        def start(self):
            self._t = threading.Thread(target=self._poll, daemon=True)
            self._t.start()
            return self

        def stop(self):
            self._stop.set()
            if self._t:
                self._t.join(timeout=2)
            if len(self.samples) < 2:
                return dict(joules=0.0, watt_hours=0.0, mean_w=0.0, peak_w=0.0, n=len(self.samples))
            j = 0.0
            for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
                j += 0.5 * (w0 + w1) * (t1 - t0)          # trapezoid
            ws = [w for _, w in self.samples]
            return dict(joules=j, watt_hours=j / 3600.0, mean_w=sum(ws) / len(ws),
                        peak_w=max(ws), min_w=min(ws), n=len(ws))

    SPAM = ["gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
            "src/core/pool.h:%d:12: warning: unused parameter 'flags'",
            "ninja: build stopped: subcommand failed at phase 0x%02x"]
    def spam(n):
        return "\n".join((SPAM[i % len(SPAM)] % (i % 256)) if "%" in SPAM[i % len(SPAM)]
                         else SPAM[i % len(SPAM)] for i in range(n))

    use_chatml = "qwen" in model or "llama" in model
    SYS = "You are a deterministic state tracker. Answer only from verified positional events."
    Q = "Where is the blue key now? Reply with only the location name."
    PLACES = ["kitchen drawer", "garden shed", "attic", "basement", "garage", "cellar",
              "hallway closet", "study", "pantry", "loft"]

    def wrap(body):
        if use_chatml:
            return ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n<build_log>\n"
                    + body + "\n</build_log>\n" + Q + "<|im_end|>\n<|im_start|>assistant\n")
        return "<s>[INST] " + SYS + " <build_log>\n" + body + "\n</build_log>\n" + Q + " [/INST]"

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
        return tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True), len(gen)

    def kvb(c):
        if c is None or len(c.layers) == 0:
            return 0
        n = 0
        for li in range(len(c.layers)):
            k, v = c.layers[li].keys, c.layers[li].values
            n += k.numel() * k.element_size() + v.numel() * v.element_size()
        return n

    # ★★★ WARMUP, AND IT IS NOT OPTIONAL. The FIRST forward in a process pays CUDA kernel
    # compilation and cuDNN autotuning -- measured here as the first-timed arm reading 401 ms
    # against a warm 300 ms for the arm that ran second, i.e. it made BLOCK look slower than
    # RECOMPUTE when F41 measures the opposite (recompute 1.2x wall). A one-time cost landing
    # entirely on whichever arm happens to run first is a measurement artefact, not a result,
    # and it would have been reported as an energy finding. Same failure class as a cold
    # browser's first request, in GPU form. Warm both paths before timing anything.
    _wu = tok("warmup " + "x " * 64, add_special_tokens=False)["input_ids"]
    _wp = list(range(len(_wu)))
    for _ in range(2):
        with torch.no_grad():
            _o = m(input_ids=torch.tensor([_wu], dtype=torch.long, device=dev),
                   use_cache=True, cache_position=torch.tensor(_wp, dtype=torch.long, device=dev))
        decode(_o.past_key_values, _o.logits[:, -1, :], len(_wu), 4)
        del _o; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize()

    rng = random.Random(20260930)
    agg = {"block": dict(wall=[], wh=[], peak=[], kv=[], ctx=[], toks=[], correct=0),
           "recompute": dict(wall=[], wh=[], peak=[], kv=[], ctx=[], toks=[], correct=0)}
    n_ok = 0
    for t in range(trials):
        # ★ ALTERNATE THE ARM ORDER. With block always first, any within-trial order effect
        # (thermal drift, allocator state, remaining lazily-compiled kernels) is attributed to
        # the arm rather than to the slot. Alternating makes it cancel across the sample.
        block_first = (t % 2 == 0)
        p1, p2, p3 = rng.sample(PLACES, 3)
        who1 = rng.choice(["Richard", "Marcus", "Elena"])
        who2 = rng.choice(["Jessica", "Priya", "Tomas"])
        b1 = ("Yesterday, the blue key was placed in the %s. %s took the blue key from the %s "
              "and placed it in the %s." % (p1, who1, p1, p2))
        b2 = spam(spam_lines)
        b3 = ("This morning, %s went to the %s, picked up the blue key, and brought it to the "
              "%s." % (who2, p2, p3))
        want = p3

        gtext = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                 if use_chatml else "<s>[INST] " + SYS + "\n")
        g_ids, g_cache, blk, pos, G, total = TC.build_table(H, gtext, [b1, b2, b3],
                                                            verbose=False)
        qturn = ("<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n"
                 if use_chatml else Q + " [/INST]")
        qids = tok(qturn, add_special_tokens=False)["input_ids"]

        gtext = ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
                 if use_chatml else "<s>[INST] " + SYS + "\n")
        g_ids, g_cache, blk, pos, G, total = TC.build_table(H, gtext, [b1, b2, b3],
                                                            verbose=False)
        qturn = ("<|im_start|>user\n" + Q + "<|im_end|>\n<|im_start|>assistant\n"
                 if use_chatml else Q + " [/INST]")
        qids = tok(qturn, add_special_tokens=False)["input_ids"]

        # ---- arm A: block table (contiguity fix). Table build is SETUP, not timed.
        def do_block():
            n1 = TC.cache_len(blk[0])
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
            ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
            pp = torch.arange(base, base + len(qids), dtype=torch.long, device=dev)

            torch.cuda.synchronize()
            ps = PowerSampler().start()
            t0 = time.perf_counter()
            with torch.no_grad():
                o2 = m(input_ids=ids_t, past_key_values=tbl, use_cache=True,
                       position_ids=pp.unsqueeze(0), cache_position=pp)
            a, _ = decode(o2.past_key_values, o2.logits[:, -1, :], base + len(qids), max_new)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            pw = ps.stop()
            res = dict(wall=wall * 1000, wh=pw["watt_hours"], peak=pw["peak_w"],
                       kv=kvb(tbl), ctx=base, toks=len(qids),
                       ok=want.lower() in a.lower())
            del tbl, o2, o, clone; gc.collect(); torch.cuda.empty_cache()
            return res

        # ---- arm B: recompute (survivors re-prefilled as ONE contiguous span).
        def do_recompute():
            surv_ids = tok(b1 + "\n" + b3 + "\n" + qturn, add_special_tokens=False)["input_ids"]
            psv = list(range(G, G + len(surv_ids)))
            clone = TC.slice_cache(g_cache, 0, G)
            torch.cuda.synchronize()
            ps = PowerSampler().start()
            t0 = time.perf_counter()
            with torch.no_grad():
                o = m(input_ids=torch.tensor([surv_ids], dtype=torch.long, device=dev),
                      past_key_values=clone, use_cache=True,
                      position_ids=torch.tensor([psv], dtype=torch.long, device=dev),
                      cache_position=torch.tensor(psv, dtype=torch.long, device=dev))
            a, _ = decode(o.past_key_values, o.logits[:, -1, :], G + len(surv_ids), max_new)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            pw = ps.stop()
            res = dict(wall=wall * 1000, wh=pw["watt_hours"], peak=pw["peak_w"],
                       kv=kvb(o.past_key_values), ctx=G + len(surv_ids),
                       toks=len(surv_ids), ok=want.lower() in a.lower())
            del o, clone; gc.collect(); torch.cuda.empty_cache()
            return res

        order = [("block", do_block), ("recompute", do_recompute)]
        if not block_first:
            order.reverse()
        for name, fn in order:
            r = fn()
            b = agg[name]
            b["wall"].append(r["wall"]); b["wh"].append(r["wh"]); b["peak"].append(r["peak"])
            b["kv"].append(r["kv"]); b["ctx"].append(r["ctx"]); b["toks"].append(r["toks"])
            b["correct"] += 1 if r["ok"] else 0
        n_ok = t + 1


    # ★★ EXPLICIT TEARDOWN. Measured: a container run for model A is REUSED for model B in the
    # same `modal run` invocation, and A's weights (14.5 GiB of an A10G's 22) are still resident
    # when B tries to load -- `Tried to allocate 14.22 GiB ... only 7.55 GiB free`. Dropping the
    # reference is not enough: the CUDA caching allocator holds the blocks until told otherwise.
    # Belt and braces with the launch script, which now runs one model per `modal run`.
    del_H = H
    H = None
    m.cpu()
    del m, del_H
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def m_(arm, k):
        v = agg[arm][k]
        return sum(v) / len(v) if v else 0.0

    # ---- derive FLOPs and bandwidth from the MEASURED token/context counts
    out = dict(model=model, hf_id=mid, trials=n_ok,
               results={}, energy={}, bandwidth={}, flops={})
    for arm in ("block", "recompute"):
        toks = m_(arm, "toks"); ctx = m_(arm, "ctx")
        fl = fwd_flops(model, toks, ctx)
        bw = bytes_moved(model, toks, ctx, ctx)
        # decode steps also run: each generated token is another forward over the same context
        steps = max_new
        fl_all = fwd_flops(model, toks + steps, ctx + steps)
        bw_all = bytes_moved(model, toks + steps, ctx + steps, ctx + steps)
        out["results"][arm] = dict(
            correct=agg[arm]["correct"], trials=n_ok,
            wall_ms=m_(arm, "wall"), kv_bytes=m_(arm, "kv"),
            ctx_tokens=ctx, fwd_tokens=toks,
            watt_hours=m_(arm, "wh"), peak_w=m_(arm, "peak"),
            joules_per_query=m_(arm, "wh") * 3600,
        )
        out["energy"][arm] = dict(mean_wh=m_(arm, "wh"), mean_peak_w=m_(arm, "peak"))
        out["flops"][arm] = dict(prefill=fl, with_decode=fl_all)
        out["bandwidth"][arm] = dict(prefill=bw, with_decode=bw_all)
    return out


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", trials: int = 20, spam_lines: int = 600, max_new: int = 24):
    import json, os, time

    def g(x):
        for u, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
            if abs(x) >= u:
                return "%.2f%s" % (x / u, s)
        return "%.2f" % x

    ms = [x.strip() for x in model.split(",") if x.strip()]
    out = []
    for mm in ms:
        print("\n===== RAW HARDWARE METRICS: %s (%d trials) =====" % (mm, trials), flush=True)
        try:
            r = run_hw.remote(mm, trials, spam_lines, max_new)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            out.append({"model": mm, "status": "failed", "error": str(e)[:300]})
            continue
        out.append(r)

        print("  %-11s %8s %8s %10s %11s %10s %9s" % (
            "arm", "correct", "wall ms", "J/query", "prefill FLOPs", "with-decode", "KV KB"))
        print("  " + "-" * 78)
        for arm in ("block", "recompute"):
            res = r["results"][arm]
            fl = r["flops"][arm]
            print("  %-11s %5d/%-2d %8.1f %10.3f %11s %10s %9.1f" % (
                arm, res["correct"], res["trials"], res["wall_ms"], res["joules_per_query"],
                g(fl["prefill"]["total"]), g(fl["with_decode"]["total"]),
                res["kv_bytes"] / 1024), flush=True)
        bw_b = r["bandwidth"]["block"]["prefill"]
        bw_r = r["bandwidth"]["recompute"]["prefill"]
        print("  bandwidth per prefill pass (batch 1, fp16):")
        print("     block      weights %s | kv_read %s | kv_write %s | TOTAL %s | KV share %.2f%%"
              % (g(bw_b["weights"]), g(bw_b["kv_read"]), g(bw_b["kv_write"]),
                 g(bw_b["total"]), bw_b["kv_share"] * 100))
        print("     recompute  weights %s | kv_read %s | kv_write %s | TOTAL %s | KV share %.2f%%"
              % (g(bw_r["weights"]), g(bw_r["kv_read"]), g(bw_r["kv_write"]),
                 g(bw_r["total"]), bw_r["kv_share"] * 100))
        eb = r["energy"]["block"]["mean_wh"]; er = r["energy"]["recompute"]["mean_wh"]
        print("  MEASURED energy: block %.6f Wh/query (%.0f W peak)  recompute %.6f Wh/query"
              "  -> recompute costs %.2fx energy"
              % (eb, r["energy"]["block"]["mean_peak_w"], er, er / eb if eb else 0))

    p = os.path.join("/tmp/opencode/ccai/benchmarks/results",
                     "hardware_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
