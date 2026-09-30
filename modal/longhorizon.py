"""LONG-HORIZON TOOL CALLING — the workload this runtime actually exists for.

    modal run modal/longhorizon.py --model qwen2.5-7b --arm linear --turns 40
    modal run modal/longhorizon.py --model qwen2.5-7b --arm runtime --turns 40

★ WHAT MAKES THIS DIFFERENT FROM THE OTHER BENCHMARKS.

  ruler/babilong/two_needle  are single-question probes: one query against a pre-built context.
                             They measure whether a LAYOUT can retrieve a fact. They cannot measure
                             a long horizon, because there is no horizon -- the context is handed
                             over intact and one answer is graded.

  SWE-bench                  is long, but its horizon is bounded by the model's ability to solve
                             the task at all. A 7B that never writes a patch produces a flat line
                             for both arms, so the layout comparison is masked by capability.

  THIS                       is many turns of tool calls with an outcome that is checked EVERY turn,
                             on a task a 7B can actually do. The model's competence is held roughly
                             constant so the LAYOUT is what varies. That is the controlled
                             experiment the other two cannot be.

★ THE TASK IS A PLANTED-STATE DEBUG, AND IT IS DESIGNED TO PUNISH BAD EVICTION.

A small program is described by a sequence of tool outputs. Early turns establish FACTS that later
turns MODIFY, and a question at the end can only be answered by the most recent value of each.
Turn 1 sets a config value; turn 12 changes it. An arm that evicts turn 1 is fine; an arm that
evicts turn 12 has lost the answer. So the task distinguishes "evicted the stale copy" (correct and
cheap) from "evicted the live copy" (wrong), which is exactly the property under test and is
invisible to a benchmark that only evicts irrelevant filler.

★ EVERY TURN IS GRADED, NOT JUST THE LAST. A long-horizon run that gets turn 38 right and turns
2-37 wrong is not a success. Per-turn accuracy is the curve that shows WHERE a layout starts to
lose the thread.
"""
import json, os, modal

app = modal.App("ccai-longhorizon")
GPU = os.environ.get("CCAI_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",   # ungated mirror; meta-llama is 403
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}

SYSTEM = """You are an operations agent. Each turn you are given one tool result.
Read it, then answer the QUESTION with only the requested value, on one line.

Rules:
  - A later tool result SUPERSEDES an earlier one for the same setting. Always report the CURRENT
    value, never a historical one.
  - If a setting was never set, answer exactly: UNKNOWN
  - Answer with the value only. No explanation, no units, no punctuation.
"""


def build_turns(n, rng, spam_lines=40):
    """Generate a long tool log where early facts are SUPERSEDED later.

    Returns (turns, questions) where questions[i] is the value that must be reported at step i --
    so every turn is checkable, and a stale-eviction or a lost-update shows up as a specific miss
    rather than a single wrong final answer.

    The supersession is what makes this a state-update task: the answer at step i is the LATEST
    value, so an arm must carry forward the newest write for each key and discard the rest.
    """
    keys = ["DB_POOL", "CACHE_TTL", "MAX_RETRY", "QUEUE_DEPTH", "TIMEOUT_MS",
            "LOG_LEVEL", "BATCH_SIZE", "SHARD_COUNT"]
    live = {}                 # key -> current value, i.e. what a correct agent must track
    turns, questions = [], []

    for i in range(n):
        k = keys[i % len(keys)]
        if k in live and rng.random() < 0.55:
            # a MUTATION: the same key gets a new value. The old one is now stale and must not
            # be reported. This is the turn an over-eager evictor gets wrong.
            old = live[k]
            new = "v%d-%d" % (rng.randint(100, 999), i)
            body = ("[tool] config_write\n  set %s = %s\n  (was %s)" % (k, new, old))
            live[k] = new
        else:
            val = "v%d-%d" % (rng.randint(100, 999), i)
            body = "[tool] config_write\n  set %s = %s" % (k, val)
            live[k] = val
        # filler so the horizon is real and eviction has something to evict
        filler = "\n".join("  [log] worker-%d heartbeat ok" % j for j in range(spam_lines))
        turns.append(body + "\n" + filler)
        # the question at this turn is a RANDOM live key -- so the answer depends on state built
        # across many earlier turns, not just the one immediately before
        qk = rng.choice(list(live.keys()))
        questions.append((qk, live[qk]))
    return turns, questions


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=7200, memory=32768)
def run_lh(model: str = "qwen2.5-7b", arm: str = "linear", turns: int = 40,
           trials: int = 3, spam_lines: int = 40, keep_blocks: int = 6):
    import gc, random, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map="cuda",
                                               attn_implementation="sdpa")
    mdl.eval()
    use_chatml = "qwen" in model or "llama" in model

    def msg(role, content):
        if use_chatml:
            return "<|im_start|>%s\n%s<|im_end|>\n" % (role, content)
        return ("<s>[INST] " if role == "user" else "") + content + (
            " [/INST]" if role == "user" else "</s>")

    def complete(prompt, max_new=16):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            o = mdl(input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"),
                    use_cache=True)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen, c, p = [nxt], o.past_key_values, len(ids)
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = mdl(input_ids=nxt, past_key_values=c, use_cache=True,
                        cache_position=torch.tensor([p], device="cuda"))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        txt = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True).strip()
        kv = sum(c.layers[i].keys.numel() * c.layers[i].keys.element_size()
                 + c.layers[i].values.numel() * c.layers[i].values.element_size()
                 for i in range(len(c.layers)))
        del o, c
        gc.collect(); torch.cuda.empty_cache()
        return txt, dict(ttft_ms=ttft, kv_bytes=kv, prompt_tokens=len(ids),
                         wall_ms=(time.perf_counter() - t0) * 1000)

    results = []
    for tr in range(trials):
        rng = random.Random(90210 + tr)
        turn_bodies, questions = build_turns(turns, rng, spam_lines)
        # ★ THE TRANSCRIPT IS THE THING UNDER TEST.
        #   linear   every tool result is appended and the whole thing is re-sent. KV grows without
        #            bound and TTFT grows with it -- what a normal agent loop does.
        #   runtime  only the last `keep_blocks` tool results are kept, with the SYSTEM message and
        #            the task preamble retained. Superseded results are dropped, which is the
        #            eviction this work is about. Findings 37/41 say the survivors must be
        #            re-indexed contiguously and the query must be its own tier; the assembled
        #            prompt here is a contiguous sequence by construction, so those two fixes are
        #            applied by the way this arm builds context.
        history = []
        per_turn = []
        for i, (body, (qk, want)) in enumerate(zip(turn_bodies, questions)):
            q = "QUESTION: what is the current value of %s?" % qk
            if arm == "linear":
                keep = history + [body]
            else:
                keep = history[-(keep_blocks - 1):] + [body]
            prompt = msg("system", SYSTEM) + "".join(
                msg("user", b + "\n") + msg("assistant", "ok\n") for b in keep[:-1])
            prompt += msg("user", keep[-1] + "\n" + q)
            if not use_chatml:
                prompt += " "
            txt, m = complete(prompt)
            ok = want.lower() in txt.lower().replace(" ", "")
            # ★ THE TASK PREAMBLE IS ALWAYS RETAINED, even in the runtime arm -- a real agent
            # never evicts its own instructions, and dropping them would test something other
            # than the memory policy.
            history.append(body)
            if arm != "linear":
                history = history[-(keep_blocks - 1):]
            per_turn.append(dict(turn=i, key=qk, want=want, got=txt[:40], ok=ok,
                                 ttft_ms=m["ttft_ms"], kv_bytes=m["kv_bytes"],
                                 prompt_tokens=m["prompt_tokens"]))
        acc = sum(1 for t in per_turn if t["ok"]) / len(per_turn)
        # ★ UNKNOWN IS NOT FREE: a model that answers UNKNOWN everywhere would score 0 here, and
        # one that guesses the CURRENT value of a never-set key is impossible by construction
        # (every key is set before it is asked). So a correct answer means it tracked state.
        results.append(dict(
            trial=tr, accuracy=acc,
            first_half=sum(1 for t in per_turn[:len(per_turn) // 2] if t["ok"])
            / max(1, len(per_turn) // 2),
            second_half=sum(1 for t in per_turn[len(per_turn) // 2:] if t["ok"])
            / max(1, len(per_turn) - len(per_turn) // 2),
            ttft_first=per_turn[0]["ttft_ms"], ttft_last=per_turn[-1]["ttft_ms"],
            kv_first=per_turn[0]["kv_bytes"], kv_last=per_turn[-1]["kv_bytes"],
            kv_peak=max(t["kv_bytes"] for t in per_turn),
            prompt_first=per_turn[0]["prompt_tokens"], prompt_last=per_turn[-1]["prompt_tokens"],
            per_turn=per_turn))
    return dict(model=model, arm=arm, turns=turns, trials=trials, gpu=GPU, results=results)


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", arm: str = "linear", turns: int = 40, trials: int = 3,
         spam_lines: int = 40):
    import time
    r = run_lh.remote(model, arm, turns, trials, spam_lines)
    p = "/tmp/opencode/ccai/benchmarks/results/longhorizon_%s_%s_%s.json" % (
        model.replace("/", "-"), arm, time.strftime("%Y%m%d-%H%M%S"))
    json.dump(r, open(p, "w"), indent=2)
    res = r["results"]
    print("\n%s  arm=%-8s turns=%d  trials=%d" % (model, arm, turns, len(res)))
    print("  accuracy       %.1f%%" % (100 * sum(x["accuracy"] for x in res) / len(res)))
    print("  first half     %.1f%%" % (100 * sum(x["first_half"] for x in res) / len(res)))
    print("  second half    %.1f%%" % (100 * sum(x["second_half"] for x in res) / len(res)))
    print("  TTFT  first -> last   %.0f -> %.0f ms" % (
        sum(x["ttft_first"] for x in res) / len(res), sum(x["ttft_last"] for x in res) / len(res)))
    print("  KV    first -> last   %.1f -> %.1f MB  (peak %.1f)" % (
        sum(x["kv_first"] for x in res) / len(res) / 1e6,
        sum(x["kv_last"] for x in res) / len(res) / 1e6,
        sum(x["kv_peak"] for x in res) / len(res) / 1e6))
    print("  prompt tok first -> last  %d -> %d" % (
        sum(x["prompt_first"] for x in res) // len(res),
        sum(x["prompt_last"] for x in res) // len(res)))
    print("wrote %s" % p)
