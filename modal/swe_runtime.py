"""SWE-bench Lite on Modal: linear prefill vs the block runtime, with TTFT and peak KV logged.

    modal run modal/swe_runtime.py --model qwen2.5-7b --arm gold
    modal run modal/swe_runtime.py --model qwen2.5-7b --arm linear
    modal run modal/swe_runtime.py --model qwen2.5-7b --arm runtime

★ THE PROBLEM THIS SOLVES, and why the harness is shared with the local grader.

A SWE-bench instance has a slow environment (a real repo, a real test suite) and a fast model loop.
They belong in different places: the repo and its venv stay on this box where they are already
verified non-vacuous, and the MODEL runs on a GPU. So the model is a service and the harness drives
it turn by turn -- the same shape as the webchat gateway, for the same reason.

★ THE TWO ARMS DIFFER IN EXACTLY ONE THING: how the transcript is held between turns.

  linear   the whole conversation re-prefilled every turn; the KV cache grows without bound.
           This is what a normal agent loop does, and it is the thing the paper is comparing to.
  runtime  the transcript assembled from a TERNARY BLOCK INFRASTRUCTURE -- each turn is a block,
           superseded turns are evicted, survivors are re-indexed contiguously (Finding 37) and
           the query is its own tier (Finding 29).

Everything else -- model, prompt, tools, temperature, turn budget -- is identical, so a difference
in the table is attributable to the layout.

★ WHAT IS LOGGED PER TURN, because these are the numbers Bob asked for:
  ttft_ms        time to first token: the latency a user feels, and the number a prefix cache moves
  prefill_tokens tokens the arm re-processed this turn (the cost of the layout, directly)
  kv_bytes       RESIDENT KV for that turn -- context that has to fit in VRAM
  wall_ms        whole turn
  tool / ok      what the model did, so a failure is legible rather than just a zero
"""
import json, os, modal

app = modal.App("ccai-swe-runtime")
GPU = os.environ.get("CCAI_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece", "fastapi",
                 "uvicorn")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # ungated mirror: meta-llama/Llama-3.1-8B is 403 on file access from this box and from Modal
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=7200, memory=32768,
              allow_concurrent_inputs=8)
@modal.asgi_app()
def serve():
    """A tiny OpenAI-shaped completion endpoint, because the harness lives on the other machine.

    ★ IT RETURNS THE PROMPT-TOKEN COUNT WITH EVERY REPLY. The arm needs to know how many tokens it
    actually re-processed -- that IS the cost measurement -- and reconstructing it caller-side from
    the text would be an estimate pretending to be a measurement.
    """
    import torch, time
    from fastapi import FastAPI
    from pydantic import BaseModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = os.environ.get("SWE_MODEL_ID", MODELS["qwen2.5-7b"])
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map="cuda",
                                               attn_implementation="sdpa")
    mdl.eval()
    api = FastAPI()

    class Req(BaseModel):
        prompt: str
        max_new: int = 320
        temperature: float = 0.0

    @api.get("/health")
    def health():
        return {"ok": True, "model": mid, "gpu": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}

    @api.post("/complete")
    def complete(r: Req):
        ids = tok(r.prompt, add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            o = mdl(input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"),
                    use_cache=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0          # MEASURED: whole prefill + first logits
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt]
        cache = o.past_key_values
        pos = len(ids)
        for _ in range(r.max_new - 1):
            with torch.no_grad():
                o = mdl(input_ids=nxt, past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([pos], device="cuda"))
            cache = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            pos += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        text = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        kv = 0
        for li in range(len(cache.layers)):
            k, v = cache.layers[li].keys, cache.layers[li].values
            kv += k.numel() * k.element_size() + v.numel() * v.element_size()
        return {"text": text, "prompt_tokens": len(ids), "gen_tokens": len(gen),
                "ttft_ms": ttft * 1000, "kv_bytes": kv,
                "wall_ms": (time.perf_counter() - t0) * 1000}

    return api


if __name__ == "__main__":
    print(__doc__)
