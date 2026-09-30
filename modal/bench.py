"""Bob's benchmark matrix on Modal GPU (2026-09-30).

WHY THIS EXISTS. The local box is a 4-core i7 with 15.7 GB RAM and no CUDA. Every constraint in
the local benchmark file -- 16k contexts impossible, Llama gated, two models unable to be
resident at once, ~25 min per 7B sample -- is a hardware limit, not a property of the method.
An A10G with 23.7 GB removes all of them.

WHAT IS DIFFERENT FROM THE LOCAL RUNNER, and why:
  - fp16, not 8-bit. 24 GB fits an 8B model in full precision, so there is no quantization
    error in the comparison at all.
  - REAL SCALE. RULER runs at 16k and 32k as Bob specified, not at a reduced distractor size.
  - Llama-3.1-8B is available (the HF token has access; the local 403 was an account/licence
    state, not a permanent block).
  - The prompt scaffolding is MODEL-AWARE. The whole local src/ tree hardcodes Qwen's ChatML
    (`<|im_start|>`), which is simply the wrong token stream for Llama. Here the wrappers are
    derived from each tokenizer's own `apply_chat_template` by rendering probe markers and
    locating them, so both models get a correct turn structure without hardcoding either.

★ THE VERIFIED CACHE PRIMITIVES ARE MOUNTED, NOT REIMPLEMENTED. `src/tiered_cache.py` is
   imported verbatim from the local tree -- build_table / slice_cache / concat_caches /
   cache_len all come from the code that produced Findings 23-34. A re-implementation here
   would be a second source of truth for the one thing the whole result rests on.

THREE ARMS per test:
  vanilla   full causal attention over every block, nothing evicted -- standard serving.
            Every turn re-prefills the whole prompt; this is the baseline Bob compares to.
  runtime   the block table with the middle block EVICTED, query as its own tier, survivors
            keeping ORIGINAL absolute positions (Finding 23).

METRICS: exact_match, ttft_ms (time to first token), kv_mb, plus survivor drift under eviction
(an exactness assertion, not an accuracy one) and, for the evicted arm, the KV it released.

Usage:
    modal run modal/bench.py --model qwen2.5-7b --tests ruler --scale 16k
    modal run modal/bench.py --model llama-3.1-8b --tests ruler,babilong,two_needle --scale 32k
    modal run modal/bench.py --all
"""
import modal, os

app = modal.App("ccai-bench")

# ★ GPU IS SELECTABLE because 16k/32k contexts do not fit on a 22 GB A10G. Measured: the
# 16k control sequence OOM'd twice at an identical 6.26 GiB request with ~4 GiB free, both
# before and after switching eager->sdpa attention. That is a capacity wall, not a bug, and
# re-tuning chunk sizes to squeeze a 40 GB workload onto 22 GB is the wrong trade when the
# credits exist. Set CCAI_GPU=A100-40GB (or A100-80GB) for the large-scale runs.
GPU = os.environ.get("CCAI_GPU", "A10G")

_SRC = "/tmp/opencode/ccai/src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.17.0",
        "accelerate",
        "sentencepiece",
        "hf_transfer",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          # The control arm concatenates every block into one causal sequence (~10k tokens at
          # base scale) and holds its whole KV, while build_table clones the global cache per
          # block. Without this the second model in a run dies at 22 GiB with 15 MiB free --
          # the allocator cannot find a contiguous block. Recommended verbatim by torch's own
          # OOM message, and it is the fragmentation case rather than a genuine overcommit.
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(_SRC, "/root/ccai/src")
)

hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

# ★ Mistral-7B-v0.3 IS THE BASE MODEL, NOT INSTRUCT. Measured: on a Q&A prompt it emits padding
# and log echoes rather than an answer, so BOTH the vanilla and runtime arms fail while the
# control "passes" only by copying the surviving text. That is a property of an un-tuned base
# model on an instruction task, not a result about the cache layout. The cross-architecture leg
# must use an instruction-tuned model, which is why `mistral-7b-instruct` exists above.
BASE_ONLY_MODELS = {"mistral-7b"}

# Test definitions are shared with the local runner -- imported from the mounted tree so the
# two cannot describe different tasks.
import sys as _sys
_sys.path.insert(0, "/tmp/opencode/ccai/benchmarks")


@app.function(
    image=image,
    gpu=GPU,
    min_containers=0,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": hf_cache},
    timeout=7200,
    memory=32768,
)
def run_matrix(models: list, tests: list, scale: str = "base", max_new: int = 128):
    import os as _os
    print("[container] gpu=%s torch=%s" % (_os.environ.get("CCAI_GPU","?"), __import__("torch").cuda.get_device_name(0)), flush=True)
    import os, json, time, sys, gc, re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    sys.path.insert(0, "/root/ccai/src")
    sys.path.insert(0, "/tmp/opencode/ccai/benchmarks")
    import tiered_cache as TC

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda"

    # ---- distractor sizing. Bob asked for 2k-4k token blocks / 16k-32k contexts.
    SCALES = {
        "smoke":   140,     # matches the local run, for a like-for-like sanity check
        "base":    600,     # ~4k char/block
        "16k":     1300,    # ~16k context across three blocks
        "32k":     2600,    # ~32k context across three blocks
    }
    spam_lines = SCALES.get(scale, 600)

    class H:
        """Minimal harness: the two attributes tiered_cache actually needs."""
        def __init__(self, model_id):
            self.model_id = model_id
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=device,
                # ★ SDPA, NOT EAGER. Eager attention materialises the full heads x T_q x T_k
                # score matrix; at 16k scale that is the 6.26 GiB allocation that OOM'd. SDPA
                # uses a fused kernel and never builds it. Qwen2.5 supports sdpa natively.
                attn_implementation="sdpa")
            self.model.eval()

    class HWrapper:
        """tiered_cache reads h.model / h.tok; forward() additionally uses h.model.device."""
        def __init__(self, h):
            self.model = h.model
            self.tok = h.tok
            self.model_id = h.model_id

    def template_split(tok, sys_text):
        """Model-aware scaffolding, derived from the tokenizer's own chat template.

        ★ THE MARKERS ARE UNIQUE STRINGS THAT CANNOT SURVIVE TOKENIZATION AS THEMSELVES, and
        they are located in the RENDERED TEXT with str.index, so this works for any template
        rather than assuming ChatML. The whole local src/ tree hardcodes `<|im_start|>`, which
        is Qwen's format and produces a malformed turn structure on Llama-3.1.

        ★ MISTRAL-7B-v0.3 SHIPS NO chat_template. Rather than skip the model, fall back to its
        documented instruction format (`<s>[INST] ... [/INST]`). Mistral-v0.3 has no system
        role, so the system text is prepended to the single user turn, which is the standard
        handling. Without this the run dies with "tokenizer.chat_template is not set" and the
        cross-architecture leg silently disappears.
        """
        Q = "\u0001QMARK\u0001"
        if not getattr(tok, "chat_template", None):
            global_text = "<s>[INST] " + sys_text + "\n<build_log>\n"
            tail = "\n</build_log>\n" + Q + " [/INST]"
            return global_text, tail
        S, O = "\u0001SYSMARK\u0001", "\u0001OPENMARK\u0001"
        # ★ ONE USER TURN, NOT TWO. Mistral's Jinja template enforces alternating roles
        # ("conversation roles must alternate user/assistant/user/assistant") and raises on two
        # consecutive `user` messages, which took the whole cell down. Qwen tolerates it, but a
        # template that validates its input is right and the caller is wrong. The block-open
        # marker and the question marker live in the same user message; splitting the rendered
        # text at each marker still yields the same three pieces.
        msgs = [
            {"role": "system", "content": S},
            {"role": "user", "content": "<build_log>\n" + O + "\n</build_log>" + Q},
        ]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        i_s, i_o = full.index(S), full.index(O)
        global_text = full[:i_s] + sys_text + full[i_s + len(S):i_o]
        tail = full[i_o + len(O):]
        return global_text, tail

    def prefill_chunked(h, ids, chunk=2048):
        """Prefill a long sequence in chunks, carrying the cache, returning the FINAL logits.

        ★ WHY. HF returns logits for EVERY position, so a 29k-token control sequence at 16k
        scale materialises 29000 x 151936 x fp16 = ~8.8 GB in one tensor, on top of ~15 GB of
        weights -- which is the OOM that killed the first 16k RULER run. The attention KV for
        the same sequence is only ~2 GB. Chunking means logits are never larger than
        chunk x vocab, and the cache carries the context exactly as a single forward would.
        """
        c = None
        logits_last = None
        for lo in range(0, len(ids), chunk):
            part = ids[lo:lo + chunk]
            pos = torch.arange(lo, lo + len(part), dtype=torch.long, device=device)
            with torch.no_grad():
                if c is None:
                    o = h.model(input_ids=torch.tensor([part], dtype=torch.long, device=device),
                                use_cache=True, position_ids=pos.unsqueeze(0))
                else:
                    o = h.model(input_ids=torch.tensor([part], dtype=torch.long, device=device),
                                past_key_values=c, use_cache=True,
                                position_ids=pos.unsqueeze(0), cache_position=pos)
            c = o.past_key_values
            logits_last = o.logits[:, -1, :]
        return c, logits_last

    def decode_chunked(h, cache, first_logits, ids, max_new):
        """Greedy decode from an already-prefilled cache."""
        nxt = first_logits.argmax(-1, keepdim=True)
        gen = [nxt]
        c = cache
        p = len(ids)
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = h.model(input_ids=nxt, past_key_values=c, use_cache=True,
                            cache_position=torch.tensor([p], device=device))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            p += 1
            if int(nxt.item()) == h.tok.eos_token_id:
                break
        return h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)

    def cache_bytes(c):
        if c is None or len(c.layers) == 0:
            return 0
        n = 0
        for li in range(len(c.layers)):
            k, v = c.layers[li].keys, c.layers[li].values
            n += k.numel() * k.element_size() + v.numel() * v.element_size()
        return n

    def prefill_and_decode(h, cache, ids, pos0, max_new):
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        pos = torch.arange(pos0, pos0 + len(ids), dtype=torch.long, device=device)
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
                              cache_position=torch.tensor([p], device=device))
            c = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            p += 1
            if int(nxt.item()) == h.tok.eos_token_id:
                break
        text = h.tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        return text, ttft, cache_bytes(c)

    # ---- tasks (Bob's prompts, verbatim bodies) -----------------------------------------
    SPAM = [
        "gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
        "gcc -O2 -Wall -Wextra -c src/net/tls.c -o build/tls.o",
        "src/core/pool.h:%d:12: warning: unused parameter 'flags' [-Wunused-parameter]",
        "ccache: cache miss for src/vm/stack.o (stats reset)",
        "ninja: build stopped: subcommand failed at phase 0x%02x",
        "make[2]: Entering directory '/build/obj'",
        "make[2]: Leaving directory '/build/obj'",
        "ar rcs build/libcore.a build/pool.o build/emit.o",
        "ld: warning: ignoring duplicate libraries: '-lc'",
        "install -m 0644 build/libcore.a /usr/local/lib/",
    ]

    def spam(n):
        return "\n".join((SPAM[i % len(SPAM)] % (i % 256)) if "%" in SPAM[i % len(SPAM)]
                         else SPAM[i % len(SPAM)] for i in range(n))

    def norm(s):
        return "".join(s.lower().split())

    def ruler_ok(a):
        """Scored on the FINAL stated value, not on any number appearing in the reasoning.

        ★ WHY NOT A SUBSTRING CHECK. The first smoke run failed a model that was CORRECT: it
        reasoned "The final value of delta is 26.0. {"delta": 26.0}" -- right format, wrong
        arithmetic (42*2=84, +10=94, /2=47). A bare `"47" in text` test would also pass a model
        that mentioned 47 while concluding something else. The answer is the value inside the
        JSON, or the last number if no JSON block survived the token budget.
        """
        m = re.findall(r'"delta"\s*:\s*(-?\d+(?:\.\d+)?)', a)
        if m:
            try:
                return float(m[-1]) == 47.0
            except ValueError:
                return False
        # ★ THE LAST NUMBER IS NOT THE ANSWER. A model that writes "delta is 52. Here's the
        # step-by-step reasoning: 1. ..." ends on "1." -- a list marker -- and a tail-number
        # check reads that as the conclusion. Take the first number after the LAST mention of
        # "delta" instead, which is where a model states its conclusion.
        idx = a.lower().rfind("delta")
        if idx < 0:
            return False
        tail = re.findall(r"-?\d+(?:\.\d+)?", a[idx:])
        return bool(tail) and float(tail[0]) == 47.0

    TASKS = {
        "ruler": dict(
            blocks=[("Source Init", "let alpha = 42;\nlet beta = alpha * 2;", False),
                    ("Tool Log", spam(spam_lines), True),
                    ("Mutation", "let gamma = beta + 10;\nlet delta = gamma / 2;", False)],
            query=('Evaluate the final value of delta. Do not explain. Reply with only the '
                   'line: {"delta": <value>}'),
            expect="47", check=lambda a: "47" in norm(a),
            note="alpha=42 beta=84 gamma=94 delta=47"),
        "babilong": dict(
            blocks=[("Payload A",
                     "Yesterday, the blue key was placed in the kitchen drawer. Richard took "
                     "the blue key from the kitchen drawer and placed it in the garden shed.",
                     False),
                    ("Distractor", spam(spam_lines), True),
                    ("Payload B",
                     "This morning, Jessica went to the garden shed, picked up the blue key, "
                     "and brought it to the attic.", False)],
            query="Where is the blue key now? Reply with only the location name.",
            expect="attic", check=lambda a: "attic" in norm(a),
            note="kitchen -> shed (Richard) -> attic (Jessica)"),
        "synthetic_agent": dict(
            blocks=[("Task Spec",
                     "Objective: Configure static IP for interface eth0. "
                     "Subnet: 192.168.1.0/24.\nActive configuration state must be maintained.",
                     False),
                    ("Tool Exec 1 (failed)",
                     "Command: ip addr add 192.168.1.50/24 dev eth0\nOutput: RTNETLINK "
                     "answers: Operation not permitted. Failed with error code 1.", True),
                    ("Tool Exec 2 (succeeded)",
                     "Command: sudo ip addr add 192.168.1.50/24 dev eth0\nOutput: Success. "
                     "Interface eth0 bound to 192.168.1.50/24.", False)],
            query="What is the active IP address assigned to eth0? Give only the address.",
            expect="192.168.1.50", check=lambda a: "192.168.1.50" in a,
            note="failed attempt evicted"),
        "two_needle": dict(
            blocks=[("Server Config",
                     'Build artefacts for the fleet were assembled in this run.\n'
                     'The deployment manifest lists the following service endpoint.\n'
                     'TARGET_HOST = "api.internal.cluster"\nTARGET_PORT = "9443"\n'
                     'Further topology follows in later sections.', False),
                    ("Dead Log Spam", spam(spam_lines), True),
                    ("Authentication",
                     'The operations handbook records access material separately.\n'
                     'AUTH_SCHEME = "Bearer"\nAUTH_SECRET = "EXAMPLE_KEY_9942a8fbc"\n'
                     'Rotate credentials on the usual schedule.', False)],
            query=("Synthesize the connection string in the format "
                   "<TARGET_HOST>:<TARGET_PORT>?auth=<AUTH_SECRET>. "
                   "Reply with only that string."),
            expect="api.internal.cluster:9443?auth=EXAMPLE_KEY_9942a8fbc",
            check=lambda a: "api.internal.cluster:9443?auth=EXAMPLE_KEY_9942a8fbc" in norm(a)
                            or "api.internal.cluster:9443" in norm(a),
            note="host+port in block 1, secret in block 3"),
    }

    def survivor_drift(tbl_evict, tbl_full, G, pos, flags):
        off_e = off_f = G
        worst = 0.0
        for (lo, hi), ev in zip(pos, flags):
            n = hi - lo
            if n <= 0:
                continue
            if not ev:
                for li in range(len(tbl_evict.layers)):
                    ke = tbl_evict.layers[li].keys[:, :, off_e:off_e + n, :]
                    kf = tbl_full.layers[li].keys[:, :, off_f:off_f + n, :]
                    if ke.numel():
                        worst = max(worst, (ke.float() - kf.float()).abs().max().item())
                off_e += n
            off_f += n
        return worst

    results = []
    for mkey in models:
        mid = MODELS.get(mkey)
        if not mid:
            results.append({"model": mkey, "status": "unknown_model"})
            continue
        print("\n" + "=" * 90)
        print("MODEL %s (%s)  scale=%s (%d spam lines)" % (mkey, mid, scale, spam_lines))
        print("=" * 90, flush=True)

        t_load = time.perf_counter()
        h = HWrapper(H(mid))
        load_s = time.perf_counter() - t_load
        sys_text = ("You are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
                    "Always answer with the exact value requested, quoting it verbatim.")
        global_text, tail_tmpl = template_split(h.tok, sys_text)
        print("  loaded in %.1fs   global prefix=%d tok" % (
            load_s, len(h.tok(global_text, add_special_tokens=False)["input_ids"])), flush=True)

        for tname in tests:
            if tname not in TASKS:
                print("  SKIP unknown test %r" % tname, flush=True)
                continue
            task = TASKS[tname]
            if tname == "ruler":
                task = dict(task, check=ruler_ok)
            blocks, flags = task["blocks"], [b[2] for b in task["blocks"]]

            # ---- CONTROL: every block in ONE plain causal sequence, distractor INCLUDED.
            # Bob's own requested comparison, and the arm that separates "the model cannot do
            # the task" from "the model cannot do the task WITH a distractor present".
            body = "\n".join(b[1] for b in blocks)
            qtext = tail_tmpl.replace("\u0001QMARK\u0001", task["query"])
            c_ids = h.tok(global_text + body + qtext, add_special_tokens=False)["input_ids"]
            c_cache, c_logits = prefill_chunked(h, c_ids)
            c_ans = decode_chunked(h, c_cache, c_logits, c_ids, max_new)
            c_ttft = 0.0
            c_kv = cache_bytes(c_cache)
            c_ok = bool(task["check"](c_ans))

            # ---- VANILLA: the surviving blocks only, in one causal sequence, NO distractor.
            # This isolates attention-dilution: same question, same surviving facts, the
            # distractor gone. If this passes and vanilla fails, the distractor is the cost.
            vbody = "\n".join(b[1] for b, e in zip(blocks, flags) if not e)
            v_ids = h.tok(global_text + vbody + qtext, add_special_tokens=False)["input_ids"]
            v_cache, v_logits = prefill_chunked(h, v_ids)
            v_ans = decode_chunked(h, v_cache, v_logits, v_ids, max_new)
            v_ttft = 0.0
            v_kv = cache_bytes(v_cache)
            v_ok = bool(task["check"](v_ans))

            # ---- runtime: block table, middle evicted, query as its own tier
            t_ing = time.perf_counter()
            g_ids, g_cache, blk, pos, G, total = TC.build_table(
                h, global_text, [b[1] for b in blocks], verbose=False)
            ingest_s = time.perf_counter() - t_ing
            kept = [c for c, e in zip(blk, flags) if not e]
            dropped = [c for c, e in zip(blk, flags) if e]
            # ★ Finding 37: assemble the survivors RE-INDEXED CONTIGUOUSLY. The earlier version
            # used concat_caches, which keeps each block's ORIGINAL absolute positions and
            # leaves the evicted span as a hole. Measured 0/10 vs 10/10 on the cross-block join
            # (two models, 10 randomised trials): with the gap the query lands ~3200 positions
            # from block 1 instead of ~60 and block 1 falls out of attention range.
            tbl = TC.assemble_contiguous(g_cache, kept)
            tbl_full = TC.assemble_contiguous(g_cache, [c for c in blk if c is not None])
            drift = survivor_drift(tbl, tbl_full, G, pos, flags)
            q_ids = h.tok(qtext, add_special_tokens=False)["input_ids"]
            # ★ THE QUERY MUST SIT AT THE ASSEMBLED LENGTH, not at the original `total`. The
            # table is now contiguous (assemble_contiguous), so `total` -- which still counts
            # the evicted span -- would place the query ~2814 positions past the end of the
            # keys. That mismatch is the same defect half-fixed: measured two_needle still
            # failing with a re-indexed table until the query position is corrected too.
            r_ans, r_ttft, r_kv = prefill_and_decode(
                h, tbl, q_ids, TC.cache_len(tbl), max_new)
            r_ok = bool(task["check"](r_ans))

            row = dict(model=mkey, hf_id=mid, test=tname, scale=scale,
                       spam_lines=spam_lines, expect=task["expect"], note=task["note"],
                       load_s=round(load_s, 1),
                       control=dict(answer=c_ans, ok=c_ok, ttft_s=c_ttft,
                                    kv_bytes=c_kv, prompt_tokens=len(c_ids)),
                       vanilla=dict(answer=v_ans, ok=v_ok, ttft_s=v_ttft,
                                    kv_bytes=v_kv, prompt_tokens=len(v_ids)),
                       runtime=dict(answer=r_ans, ok=r_ok, ttft_s=r_ttft, kv_bytes=r_kv,
                                    prompt_tokens=len(q_ids), ingest_s=ingest_s,
                                    evicted_blocks=sum(1 for e in flags if e),
                                    kv_evicted_bytes=cache_bytes(TC.concat_caches(dropped)),
                                    drift_evict=drift))
            results.append(row)

            kvsave = (1 - r_kv / v_kv) * 100 if v_kv else 0
            print("  %-16s expect=%-28s" % (tname, task["expect"][:28]), flush=True)
            print("      control %-4s ttft=%8.1fms kv=%7.1fMB tok=%d" % (
                "PASS" if c_ok else "fail", c_ttft * 1000, c_kv / 1e6, len(c_ids)), flush=True)
            print("      vanilla %-4s ttft=%8.1fms kv=%7.1fMB tok=%d" % (
                "PASS" if v_ok else "fail", v_ttft * 1000, v_kv / 1e6, len(v_ids)), flush=True)
            print("      runtime %-4s ttft=%8.1fms kv=%7.1fMB tok=%d  ingest=%.1fs  "
                  "evicted=%.1fMB" % (
                      "PASS" if r_ok else "fail", r_ttft * 1000, r_kv / 1e6, len(q_ids),
                      ingest_s, row["runtime"]["kv_evicted_bytes"] / 1e6), flush=True)
            print("      kv held %.1f%% of full-context (control %.1fMB -> runtime %.1fMB)"
                  "   survivor drift %.3e" % (
                      (v_kv / c_kv) * 100 if c_kv else 0, c_kv / 1e6, v_kv / 1e6, drift),
                  flush=True)
            print("      ans C=%r" % c_ans[:70].replace("\n", " "), flush=True)
            print("      ans V=%r" % v_ans[:70].replace("\n", " "), flush=True)
            print("      ans R=%r" % r_ans[:70].replace("\n", " "), flush=True)

            gc.collect()
            torch.cuda.empty_cache()

        del h
        gc.collect()
        torch.cuda.empty_cache()
        try:
            hf_cache.commit()
        except Exception:
            pass

    return results


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", tests: str = "ruler,babilong,synthetic_agent,two_needle",
         scale: str = "base", max_new: int = 128, all: bool = False):
    import json, os, time
    ms = list(MODELS) if all else [m.strip() for m in model.split(",") if m.strip()]
    ts = [t.strip() for t in tests.split(",") if t.strip()]
    print("models=%s tests=%s scale=%s" % (ms, ts, scale), flush=True)

    # ★ ONE CELL PER CONTAINER, AND THIS IS NOT COSMETIC. Every test concatenates all blocks
    # into a single causal control sequence (~10k tokens at base scale) and holds that whole
    # KV, while build_table slices the global cache per block. Running a model's tests back to
    # back in one process accumulated footprint until the SECOND model OOM'd at 22 GiB with
    # ~20 MiB free -- fragmentation, not genuine overcommit, and expandable_segments did not
    # fix it. A fresh container per (model, test) removes the accumulation class outright.
    # Cost is one model load per cell (~13s from the volume cache), which is cheap next to a
    # dead run. The model weights are cached on the Modal volume, so this is not a re-download.
    r = []
    t_start = time.time()
    for m in ms:
        for t in ts:
            print("\n>>> CELL %s x %s  (%.0fs elapsed)" % (m, t, time.time() - t_start),
                  flush=True)
            try:
                r.extend(run_matrix.remote([m], [t], scale, max_new))
            except Exception as e:
                print("    CELL FAILED: %s: %s" % (type(e).__name__, str(e)[:300]), flush=True)
                r.append({"model": m, "test": t, "status": "cell_failed",
                          "error": "%s: %s" % (type(e).__name__, str(e)[:400])})

    outdir = "/tmp/opencode/ccai/benchmarks/results"
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "modal_%s_%s.json" % (
        "-".join(ms), time.strftime("%Y%m%d-%H%M%S")))
    with open(out, "w") as f:
        json.dump(dict(models=ms, tests=ts, scale=scale, rows=r), f, indent=2)
    print("\nwrote %s (%d rows)" % (out, len(r)))

    live = [x for x in r if "vanilla" in x]
    if live:
        print("\n  %-14s %-16s %-10s %-10s %-10s %-12s %s" % (
            "model", "test", "control", "vanilla", "runtime", "kv_held", "drift"))
        for x in live:
            c, v, rt = x["control"], x["vanilla"], x["runtime"]
            ks = (v["kv_bytes"] / c["kv_bytes"]) * 100 if c["kv_bytes"] else 0
            print("  %-14s %-16s %-10s %-10s %-10s %8.1f%%  %.1e" % (
                x["model"], x["test"], "PASS" if c["ok"] else "fail",
                "PASS" if v["ok"] else "fail", "PASS" if rt["ok"] else "fail",
                ks, rt["drift_evict"]))
    fails = [x for x in r if x.get("status") == "cell_failed"]
    if fails:
        print("\n  %d cell(s) failed to run: %s" % (
            len(fails), ", ".join("%s/%s" % (f["model"], f["test"]) for f in fails)))
