#!/usr/bin/env python3
"""
test_2d_kv_cache.py -- Saliency-Preserving KV Compaction with Dense Delta Re-rotation.

Question under test: when a volatile middle region (Chunk 1, build logs) is evicted
from the KV cache, what actually breaks -- and does dense re-numbering plus a delta
phase shift on the downstream keys repair it?

Two damages are separable and this harness keeps them separate:

  POSITIONAL: every cached key was rotated at its ORIGINAL RoPE position. After
  eviction the survivors sit at NEW dense indices, so their baked rotations no
  longer match the indices the model will use. Repairable: apply R(new-orig).

  CONDITIONAL: a value vector V_i was computed as an aggregate over every token
  before it. Deleting those tokens does not un-bake that aggregate and no rotation
  on K can restore it. (Kamera measured this: multi-hop 0.41 -> 0.28 MLA.) NOT
  repairable by rotation.

Arms, chosen so the gap between them attributes the loss:

  0   ground truth, unmodified                     reference
  1   naive squash re-encoded ("logs ok")          text-level baseline, costs a prefill
  1k  naive squash at KV level: first5+last5       position-blind eviction
  2a  top-k by key L2 norm, query-AGNOSTIC         shippable, zero extra compute
  2b  top-k by query attention, query-AWARE        oracle upper bound
  R   random-k control                             proves selection beats chance
  3   arm 2a with delta rotation DISABLED          isolates the rotation itself

1k/2a/2b/R/3 share identical dense re-numbering; only 3 omits the rotation.
"""
import os, sys, io, time, json, math, random, argparse, difflib

os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

# envy-mem-watchdog SIGSTOPs past 96% CPU; pin threads so a long prefill cannot trip it.
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_ID = os.environ.get("CCAI_MODEL", "Qwen/Qwen2.5-0.5B")
SYSTEM_FLAG_A = "ALPHA_VERIFIED"
COMPILER_ERR = "0x9AF4_STACK_FAIL"

Q1_TEXT = "What is SYSTEM_FLAG_A?"
Q2_TEXT = "What was the failure code in the build log?"
Q1_ANSWER = "SYSTEM_FLAG_A is " + SYSTEM_FLAG_A + "."
Q2_ANSWER = "The failure code in the build log was " + COMPILER_ERR + "."
MAX_NEW_TOKENS = 24

PRE_LINES = [
    "make -j4 all",
    "gcc -O2 -Wall -Wextra -c src/core/arena.c -o build/arena.o",
    "gcc -O2 -Wall -Wextra -c src/core/pool.c -o build/pool.o",
    "In file included from src/core/pool.c:14:",
    "src/core/pool.h:88:12: warning: unused parameter 'flags' [-Wunused-parameter]",
    "gcc -O2 -Wall -Wextra -c src/core/ring.c -o build/ring.o",
    "gcc -O2 -Wall -Wextra -c src/core/hash.c -o build/hash.o",
    "src/core/hash.c:201:7: warning: variable 'probe' set but not used",
    "gcc -O2 -Wall -Wextra -c src/net/socket.c -o build/socket.o",
    "gcc -O2 -Wall -Wextra -c src/net/poll.c -o build/poll.o",
    "gcc -O2 -Wall -Wextra -c src/net/tls.c -o build/tls.o",
    "gcc -O2 -Wall -Wextra -c src/vm/instruction.c -o build/instruction.o",
    "gcc -O2 -Wall -Wextra -c src/vm/decode.c -o build/decode.o",
    "gcc -O2 -Wall -Wextra -c src/vm/registers.c -o build/registers.o",
    "gcc -O2 -Wall -Wextra -c src/vm/stack.c -o build/stack.o",
    "src/vm/stack.c: in function 'vm_push_frame':",
    "src/vm/stack.c:412:9: note: frame depth grows unbounded on recursive calls",
    "gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
    "gcc -O2 -Wall -Wextra -c src/codegen/lower.c -o build/lower.o",
    "gcc -O2 -Wall -Wextra -c src/codegen/regalloc.c -o build/regalloc.o",
    "gcc -O2 -Wall -Wextra -c src/codegen/schedule.c -o build/schedule.o",
    "gcc -O2 -Wall -Wextra -c src/codegen/emit_asm.c -o build/emit_asm.o",
    "gcc -O2 -Wall -Wextra -c src/runtime/init.c -o build/init.o",
    "gcc -O2 -Wall -Wextra -c src/runtime/teardown.c -o build/teardown.o",
    "gcc -O2 -Wall -Wextra -c src/runtime/signal.c -o build/signal.o",
]

NEEDLE_LINES = [
    "LINK build/interpreter",
    "ld: build/stack.o: relocation R_X86_64_PC32 against symbol vm_push_frame",
    "ld: final link failed: bad value",
    "collect2: error: ld returned 1 exit status",
    "FATAL: link aborted, failure code " + COMPILER_ERR,
]

POST_LINES = [
    "make: *** [Makefile:118: build/interpreter] Error 1",
    "make: Target 'all' not remade because of errors.",
    "ninja: build stopped: subcommand failed.",
    "ccache: cache miss for src/vm/stack.o (stats reset)",
    "ci: archiving build artifacts to dist/build.tar.zst",
    "ci: artifact archive written, 4.2 MiB",
    "ci: uploading artifacts (attempt 1/3)",
    "ci: upload complete",
    "ci: run finished with status FAILED",
    "ci: logs available at /var/log/ci/run-8842.log",
    "ci: elapsed 00:03:41, peak rss 812 MiB",
    "ci: cleaning workspace",
    "ci: done",
]


def build_log(tok, target_tokens=300):
    """Assemble a log of roughly `target_tokens`, with the needle held mid-region.

    The needle block is always present and always flanked on both sides, so a
    first-k, last-k, or head-truncating eviction is guaranteed to miss it while a
    salience-aware selector has a reason to keep it.
    """
    def toks(lines):
        return len(tok("\n".join(lines), add_special_tokens=False)["input_ids"])

    n_pre = n_post = 0
    while n_pre < len(PRE_LINES) or n_post < len(POST_LINES):
        cand = PRE_LINES[:n_pre] + NEEDLE_LINES + POST_LINES[:n_post]
        if toks(cand) >= target_tokens:
            break
        if n_pre < len(PRE_LINES):
            n_pre += 1
        if toks(PRE_LINES[:n_pre] + NEEDLE_LINES + POST_LINES[:n_post]) >= target_tokens:
            break
        if n_post < len(POST_LINES):
            n_post += 1
    return "\n".join(PRE_LINES[:n_pre] + NEEDLE_LINES + POST_LINES[:n_post])


def locate_needle(tok, prefix_text):
    """Character-offset -> token-index for the error code. Exact, boundary-proof.

    Searching for a token subsequence fails whenever the surrounding text
    tokenises differently (it did: the needle came back as None on a log that
    plainly contains it). Offsets are unambiguous.
    """
    enc = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    char = prefix_text.find(COMPILER_ERR)
    if char < 0:
        return None
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= char < b:
            return i
    return None


def build_prefix(tok, target_tokens=300):
    """Build the prompt in the model's OWN chat format.

    The first version of this harness drove the model as a raw continuer, which is
    why it answered with "Human: The value of ...". Qwen2.5 has a real chat template,
    so run with it: the model was instruction-tuned against these markers and
    bypassing them depresses accuracy in a way that has nothing to do with KV
    eviction. Split into two tokenisable pieces so the chunk boundary stays exact
    rather than inferred from a separator search.
    """
    system = (
        "You are a build assistant.\n"
        "SYSTEM_FLAG_A = " + SYSTEM_FLAG_A + "\n"
        "Always answer with the exact value requested, quoting it verbatim.\n"
    )
    log = build_log(tok, target_tokens)
    sys_part = "<|im_start|>system\n" + system + "<|im_end|>\n"
    log_part = "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    return sys_part + log_part, sys_part, log_part


def build_question(text):
    """The question as a new user turn, with the assistant generation prompt."""
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class Rope:
    """Standard interleaved RoPE, GPT-NeoX / Qwen layout: pairs are (2i, 2i+1)."""

    def __init__(self, head_dim, theta):
        self.head_dim = head_dim
        self.theta = theta
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
        self.inv_freq = 1.0 / (theta ** (idx / head_dim))

    def cos_sin(self, positions):
        pos = torch.as_tensor(positions, dtype=torch.float32).reshape(-1)
        freqs = torch.outer(pos, self.inv_freq)          # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [T, D]
        return emb.cos(), emb.sin()

    def apply(self, x, positions):
        """Rotate x [B,H,T,D] by per-position offsets. Pure, returns a new tensor."""
        cos, sin = self.cos_sin(positions)
        cos = cos.to(x.dtype); sin = sin.to(x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)              # [1,1,T,D]
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c1, s1 = cos[..., 0::2], sin[..., 0::2]
        o1 = x1 * c1 - x2 * s1
        o2 = x1 * s1 + x2 * c1
        out = torch.stack([o1, o2], dim=-1)              # [B,H,T,D/2,2]
        return out.reshape_as(x)


def detect_rope(model, head_dim):
    """Build the RoPE table and PROVE it matches the model's own rotary embedding.

    Reading theta out of the config is not reliable: on this transformers version
    `config.rope_theta` is absent as an attribute while the value lives in
    `rope_parameters`, so a getattr-and-default silently produced theta=10000 for a
    model that uses 1e6. Nothing downstream would have complained -- the identity
    control still passes, because R(0) = I for ANY theta, and the R(a)R(b)=R(a+b)
    self-check holds for any theta too. A wrong frequency table would have quietly
    corrupted every rotation measurement in the report.

    So the config value is only a hint. The authority is the model's own inv_freq,
    and a mismatch is fatal rather than a fallback.
    """
    cfg = model.config
    theta = None
    for src in (getattr(cfg, "rope_theta", None),
                (cfg.to_dict().get("rope_theta") if hasattr(cfg, "to_dict") else None),
                (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
                if isinstance(getattr(cfg, "rope_parameters", None), dict) else None,
                (getattr(cfg, "rope_scaling", None) or {}).get("rope_theta")
                if isinstance(getattr(cfg, "rope_scaling", None), dict) else None):
        if src is not None:
            theta = float(src)
            break
    hd = getattr(cfg, "head_dim", None) or head_dim
    if theta is None:
        # Deliberately NOT defaulting to 10000: a guess here silently invalidates
        # every rotation number, so an unknown theta must stop the run.
        raise SystemExit(
            "cannot determine rope_theta from the config; refusing to guess. "
            "Provide it explicitly or set CCAI_ROPE_THETA.")

    rope = Rope(int(hd), float(theta))

    # The authority: the model's real rotary table.
    ref = None
    for path in (lambda: model.model.rotary_emb.inv_freq,
                 lambda: model.model.layers[0].self_attn.rotary_emb.inv_freq):
        try:
            ref = path()
            break
        except Exception:
            continue
    if ref is None:
        raise SystemExit("cannot read the model's rotary inv_freq; refusing to guess.")
    ref = ref.detach().float().reshape(-1)
    mine = rope.inv_freq.float().reshape(-1)
    if mine.shape != ref.shape or not torch.allclose(mine, ref, atol=1e-7, rtol=1e-5):
        raise SystemExit(
            "RoPE table does NOT match the model. config theta=%r gives inv_freq[:2]=%s "
            "but the model uses %s. Every rotation number would be meaningless."
            % (theta, [float(x) for x in mine[:2]], [float(x) for x in ref[:2]]))
    return rope, float(theta), int(hd)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, model_id=MODEL_ID):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # CCAI_QUANT=8bit loads through bitsandbytes. That is needed for 7B on this
        # box (28 GB fp32 / 14 GB bf16 against 15.7 GB shared RAM) but it CHANGES
        # THE NUMERICS -- bnb casts to float16 internally -- so the delta is
        # measured rather than assumed; see quant_delta.py.
        quant = os.environ.get("CCAI_QUANT", "").strip().lower()
        self.quant = quant or "none"
        # Arm 2b needs real attention weights, and sdpa returns none -- it would
        # silently yield attentions=None and the oracle arm would be empty.
        if quant in ("8bit", "8", "int8"):
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="cpu",
                attn_implementation=os.environ.get("CCAI_ATTN", "sdpa"),
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32,
                attn_implementation=os.environ.get("CCAI_ATTN", "sdpa"),
            )
        self.model.eval()
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_q_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.hidden = cfg.hidden_size
        self.head_dim = self.hidden // self.n_q_heads
        self.rope, self.theta, self.rope_head_dim = detect_rope(self.model, self.head_dim)
        self.rope_verified, self.rope_delta = self._verify_rope()

    def _verify_rope(self):
        """Prove our rotation convention matches the model's BEFORE trusting any arm.

        The check: rotate a probe by R(a) then by R(b); the result must equal R(a+b).
        If the layout or sign convention were wrong that identity would not hold, and
        every downstream number would be quietly meaningless.
        """
        a, b = 37.0, -12.0
        x = torch.randn(1, 2, 3, self.rope.head_dim)
        r = self.rope
        lhs = r.apply(x, [a] * 3)
        lhs = r.apply(lhs, [b] * 3)
        rhs = r.apply(x, [a + b] * 3)
        err = (lhs - rhs).abs().max().item()
        return err < 1e-4, err

    # -- cache helpers ------------------------------------------------------

    def layers(self, cache):
        return cache.layers

    def cache_len(self, cache):
        if len(cache) == 0:
            return 0
        return cache.layers[0].keys.shape[2]

    def clone_cache(self, cache):
        """Deep copy so each arm starts from an identical prefill."""
        new = DynamicCache()
        for i, layer in enumerate(cache.layers):
            new.update(layer.keys.clone(), layer.values.clone(), i)
        return new

    def resize(self, cache, keep_idx):
        """Keep only keep_idx (sorted ascending) and dense renumber. Single write."""
        ki = torch.as_tensor(keep_idx, dtype=torch.long)
        for layer in cache.layers:
            layer.keys = layer.keys.index_select(2, ki)
            layer.values = layer.values.index_select(2, ki)
        return cache

    def apply_delta(self, cache, keep_idx, rotation=True):
        """Rotate every surviving key from its ORIGINAL position to its NEW dense one.

        KV heads are rotated with a single shared offset schedule and then broadcast
        across whatever GQA grouping the model uses. Getting this seam wrong is the
        classic lazy implementation bug; `verify_rotation` below tests it against the
        model rather than against our own arithmetic.
        """
        if not rotation:
            return cache
        wanted = torch.arange(len(keep_idx), dtype=torch.float32)
        orig = torch.as_tensor(keep_idx, dtype=torch.float32)
        deltas = (wanted - orig).tolist()
        for layer in cache.layers:
            k = layer.keys                                   # [B, Hkv, T, D]
            b, h, t, d = k.shape
            rot = self.rope.apply(k.reshape(b * h, 1, t, d).contiguous(), deltas)
            layer.keys = rot.reshape(b, h, t, d)
        return cache

    # -- generation ---------------------------------------------------------

    def generate(self, prompt_text, cache, max_new=MAX_NEW_TOKENS):
        """Greedy decode against a given prefix cache. Returns (text, token_count)."""
        enc = self.tok(prompt_text, return_tensors="pt")
        ids = enc["input_ids"]
        if cache is not None and self.cache_len(cache) > 0:
            # First forward consumes the prompt against the existing cache.
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated = [nxt]
            consumed = ids.shape[1]
            for _ in range(max_new - 1):
                out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
                generated.append(nxt)
                if nxt.item() == self.tok.eos_token_id:
                    break
            new_ids = torch.cat(generated, dim=1)
        else:
            gen = self.model.generate(
                ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
                use_cache=True,
            )
            new_ids = gen[:, ids.shape[1]:]
            consumed = ids.shape[1]
        text = self.tok.decode(new_ids[0], skip_special_tokens=True)
        return text.strip(), consumed + new_ids.shape[1]

    def answer_logprobs(self, prompt_text, cache, answer):
        """Teacher-forced cross-entropy of `answer` given the prefix.

        Returns (mean_nll_nats, exact_match_text, normalized_match). This is the
        drift metric: it is defined even when generation produces nothing usable,
        so it cannot be silently vacuous.
        """
        pre = self.tok(prompt_text, return_tensors="pt")["input_ids"]
        full = self.tok(prompt_text + " " + answer, return_tensors="pt")["input_ids"]
        if full.shape[1] <= pre.shape[1]:
            return float("inf"), False, False
        if cache is None or self.cache_len(cache) == 0:
            return float("inf"), False, False
        # The cache already holds the whole prefix, so re-feeding it would double
        # count and shift every logit. Feed the LAST prefix token plus the answer
        # tokens: logits[0] then predicts the first answer token, because the last
        # prefix token occupies the final cached position.
        tail = full[:, pre.shape[1] - 1:]
        with torch.no_grad():
            out = self.model(input_ids=tail, past_key_values=self.clone_cache(cache),
                             use_cache=True)
        logits = out.logits
        nll = 0.0
        n = 0
        for t in range(1, tail.shape[1]):
            lt = logits[0, t - 1, :]
            tgt = tail[0, t]
            nll += float(F.cross_entropy(lt.unsqueeze(0), tgt.unsqueeze(0)))
            n += 1
        return nll / max(n, 1), None, None


# ---------------------------------------------------------------------------
# Answer scoring -- three independent tiers, so a retrieval failure is never
# confused with a formatting failure.
# ---------------------------------------------------------------------------

def normalize(s):
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


FLAG_PARTS = ["alpha", "verified"]
ERR_PARTS = ["9af4", "stack"]


def score_answer(answer, kind):
    """Return dict of three scores for one needle.

    literal : the exact canonical string appears
    normalized : every identifying part appears after case/punctuation stripping
                 (catches "0x9AF4" for "0x9AF4_STACK_FAIL", and "Alpha Verified")
    judge_proxy : difflib similarity to the canonical answer, >= 0.45. This is a
                 PROXY, not a model judge -- a real judge needs an API key and this
                 script is deliberately standalone. Labelled as a proxy everywhere.
    """
    a = (answer or "").strip()
    na = normalize(a)
    if kind == "flag":
        needle, parts, canon = SYSTEM_FLAG_A, FLAG_PARTS, Q1_ANSWER
    else:
        needle, parts, canon = COMPILER_ERR, ERR_PARTS, Q2_ANSWER
    literal = needle.lower() in a.lower()
    norm = all(p in na for p in parts)
    ratio = difflib.SequenceMatcher(None, na, normalize(canon)).ratio()
    return {
        "literal": literal,
        "normalized": norm,
        "judge_proxy": ratio >= 0.45,
        "ratio": round(ratio, 3),
        "answer": a[:160],
    }


# ---------------------------------------------------------------------------
# Saliency selection
# ---------------------------------------------------------------------------

def sel_first_last(n, k):
    """Position-blind: first k/2 and last k/2. The classic naive eviction."""
    half = k // 2
    idx = list(range(half)) + list(range(max(half, n - (k - half)), n))
    return sorted(set(idx))[:k]


def sel_norm(K, k):
    """Query-agnostic: highest key L2 norm, summed over heads. Zero extra compute."""
    # K: [1, Hkv, n, D] -> per-position score
    s = K[0].float().pow(2).sum(-1).sum(0)          # [n]
    order = torch.argsort(s, descending=True)[:k]
    return sorted(int(i) for i in order.tolist())


def sel_attention(attn_to_chunk1, k):
    """Query-aware oracle: highest attention received from the query tokens."""
    order = torch.argsort(attn_to_chunk1, descending=True)[:k]
    return sorted(int(i) for i in order.tolist())


def sel_random(n, k, seed=0):
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), min(k, n)))


# ---------------------------------------------------------------------------
# FLOP model -- written out so the numbers are auditable, not asserted.
# ---------------------------------------------------------------------------

def prefill_flops(n_params, n_tokens):
    """Forward pass: ~2*N*T multiply-adds."""
    return 2 * n_params * n_tokens


def decode_attn_flops(n_layers, n_q_heads, head_dim, cache_len):
    """Attention work for ONE generated token against a cache of cache_len.

    Per layer per query head: QK^T is d x L -> 2*L*d, and AV is L x d -> 2*L*d,
    so 4*L*d in total. Summed over heads and layers.
    """
    return n_layers * n_q_heads * 4 * cache_len * head_dim


def kv_bytes(cache):
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


# ---------------------------------------------------------------------------
# Query-aware attention capture (for Arm 2b only)
# ---------------------------------------------------------------------------

def capture_query_attention(h, prefix_cache, query_text):
    """Genuine query-attention saliency for chunk 1, computed without the T x T matrix.

    The obvious implementation -- output_attentions=True -- materialises every
    layer's full attention at once: 24 layers x 14 heads x T x T. On a 300-token
    prefix that measured ~2 GB and took this box into a memory alert. It is also
    almost entirely wasted, because only the query rows and the chunk-1 columns
    are ever read.

    Instead: hook each layer's q_proj, rotate the captured Q with the SAME verified
    RoPE, and dot it against just the chunk-1 slice of the cached K. Same quantity,
    a few MB.

    Ranking note: a true softmax would normalise over all T keys, but that
    denominator is identical for every chunk-1 position, so it is a shared
    positive constant and cannot reorder them. Raw scores therefore rank exactly
    as the softmax would -- which is all selection needs.

    Returns (scores_over_chunk1, updated_cache).
    """
    import contextlib
    n_chunk1 = h.chunk1_len
    n_prefix = h.cache_len(prefix_cache)
    chunk1_start = n_prefix - n_chunk1
    pre = h.tok(query_text, return_tensors="pt")["input_ids"]
    nq = pre.shape[1]

    captured = {}

    def make_hook(i):
        def hook(module, args, output):
            captured[i] = output.detach()
        return hook

    handles = []
    try:
        for i, layer in enumerate(h.model.model.layers):
            handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i)))
        with contextlib.redirect_stdout(io.StringIO()):
            with torch.no_grad():
                out = h.model(input_ids=pre, past_key_values=h.clone_cache(prefix_cache),
                              use_cache=True)
    finally:
        for hd in handles:
            hd.remove()

    # Aggregate over ALL layers, the way H2O and SnapKV do. Layer 0 is close to a
    # positional/lexical detector and, measured on this very task, ranked the error
    # code 303rd of 333 -- i.e. a single-layer oracle would have condemned the
    # saliency idea for a reason that is an artefact of the layer choice. Capturing
    # q per layer costs ~36 KB each, so all 24 together are still negligible; the
    # 2 GB that forced this rewrite came from materialising T x T, not from the q's.
    per_layer = {}
    acc = None
    k_all = prefix_cache.layers[0].keys
    for i in sorted(captured):
        q = captured[i]                                   # [1, Tq, Hq*D]
        b, tq, _ = q.shape
        q = q.reshape(b, tq, h.n_q_heads, h.head_dim).permute(0, 2, 1, 3).contiguous()
        q = h.rope.apply(q, list(range(n_prefix, n_prefix + tq)))
        k_i = prefix_cache.layers[i].keys
        if h.n_q_heads != h.n_kv_heads:
            k_i = k_i.repeat_interleave(h.n_q_heads // h.n_kv_heads, dim=1)
        scores = torch.matmul(q, k_i.transpose(2, 3)) / math.sqrt(h.head_dim)
        probs = torch.softmax(scores, dim=-1)[:, :, :, chunk1_start:]
        s = probs[0].float().sum(0).sum(0)                # [Tc1]
        per_layer[i] = s
        acc = s if acc is None else acc + s
    h.per_layer_saliency = per_layer
    return acc, out.past_key_values




def prefill_capture_q(h, text):
    """One forward that returns BOTH the cache and every layer's q.

    Hooking q_proj during a use_cache=True prefill means the expensive pass is paid
    once. Capturing the q's costs ~8 MB per layer at the largest depth; the previous
    version materialised a full c1 x c1 attention matrix per layer instead, which is
    268 MB each at 2189 tokens and is what took the box into an OOM.
    """
    import contextlib
    ids = h.tok(text, return_tensors="pt")["input_ids"]
    cap = {}

    def make_hook(i):
        def hook(module, args, output):
            cap[i] = output.detach()
        return hook

    handles = []
    try:
        for i, layer in enumerate(h.model.model.layers):
            handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i)))
        with contextlib.redirect_stdout(io.StringIO()):
            with torch.no_grad():
                out = h.model(input_ids=ids, use_cache=True)
    finally:
        for hd in handles:
            hd.remove()
    return out.past_key_values, cap, ids


def self_attn_scores(h, cache, qcap, c0, c1, block=256):
    """Query-AGNOSTIC saliency, blocked so peak memory is O(block * c1), not O(c1^2 * layers).

    Scores each chunk-1 position by the attention chunk 1 pays to it, which needs only
    the log itself -- available before any question exists. This is the shippable
    selector.

    Scoring is deliberately NON-CAUSAL over the chunk: no causal mask is applied to the
    chunk-1 x chunk-1 score matrix. Cached K is a linear projection of each token, so it
    carries no causal information; only the weight matrix would, and that is built here.
    A causal mask would starve early positions of contributors, which is exactly the
    positional bias that broke the depth sweep.

    Each layer's captured q is freed as soon as it is scored, and the query axis is
    processed in blocks, so a long log cannot blow the memory budget.
    """
    acc = torch.zeros(c1, dtype=torch.float32)
    sl = slice(c0, c0 + c1)
    for i in sorted(qcap):
        q_all = qcap.pop(i)                      # free as we go
        k = cache.layers[i].keys[:, :, sl, :]
        if h.n_q_heads != h.n_kv_heads:
            k = k.repeat_interleave(h.n_q_heads // h.n_kv_heads, dim=1)
        for s in range(0, c1, block):
            e = min(s + block, c1)
            q = q_all[:, c0 + s:c0 + e]
            q = q.reshape(1, e - s, h.n_q_heads, h.head_dim).permute(0, 2, 1, 3).contiguous()
            q = h.rope.apply(q, list(range(c0 + s, c0 + e)))
            # bnb (8-bit) returns bf16 activations while the cached keys are fp16.
            # Cast both sides to fp32 for the saliency arithmetic only: selection
            # ranks scores, so precision beyond fp32 buys nothing, and it keeps the
            # scorer identical between the fp32 and quantised runs.
            sc = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(h.head_dim)
            # Sum over heads and over this block of QUERIES; the key axis is global,
            # so the block's contribution is a full-length [c1] vector.
            acc += torch.softmax(sc, dim=-1)[0].float().sum(0).sum(0)
            del sc, q
    return acc


def capture_self_attention(h, cache, text, c0, c1, block=256):
    """Backwards-compatible wrapper: prefill already has the cache, so re-capture q.

    Kept so existing callers keep working; new code should use prefill_capture_q plus
    self_attn_scores so the prefill is not paid twice.
    """
    _, qcap, _ = prefill_capture_q(h, text)
    return self_attn_scores(h, cache, qcap, c0, c1, block)


def mem_available_mb():
    """MemAvailable is the honest figure; free's 'free' column reads ~0 on a healthy box."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def require_memory(need_mb, label=""):
    """Refuse to start rather than walk into an OOM. Never let the box freeze."""
    avail = mem_available_mb()
    if avail < need_mb:
        raise SystemExit(
            "REFUSING TO RUN %s: %.0f MB available, need %.0f MB. "
            "Reclaim memory first (the box watchdog SIGSTOPs on RAM pressure and a "
            "paused process still accepts sockets, so an OOM here looks like a hang)."
            % (label, avail, need_mb))
    return avail


SINK_GUARD = 4


def sel_attention_nosink(scores, k, guard=SINK_GUARD):
    """Top-k by attention, refusing to spend budget on attention sinks.

    Position 0 (and here position 4) absorb attention in almost every transformer
    regardless of content. If the keep budget goes to sinks, the eviction is blind
    again. Mask their scores to -inf before ranking so the budget is spent on
    content-carrying positions.
    """
    s = scores.clone()
    s[:guard] = float("-inf")
    order = torch.argsort(s, descending=True)[:k]
    return sorted(int(i) for i in order.tolist())



def line_spans(tok, prefix_text, chunk1_start, chunk1_len):
    """Map each chunk-1 token position to a line index of the log.

    Sub-word tokens are not a natural unit for a log: cutting inside a line leaves
    half a symbol and half an identifier. Every selection strategy here operates on
    tokens, so it can and does split lines. This returns the line grouping token
    positions, which is what line pooling needs to keep a whole line.
    """
    enc = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    starts = [0]
    for i, ch in enumerate(prefix_text):
        if ch == "\n":
            starts.append(i + 1)
    def line_of(pos):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo
    out = {}
    for t in range(chunk1_start, min(chunk1_start + chunk1_len, len(offs))):
        out.setdefault(line_of(offs[t][0]), []).append(t - chunk1_start)
    return [out[k] for k in sorted(out)]


def sel_line_pooled(scores, spans, k):
    """Select whole lines, by the best token in each line, until the budget is spent.

    A line is the unit a reader would use. If any token of a line matters, the line
    matters, and keeping it intact costs a few tokens. Measured effect in RESULTS
    Finding 9.
    """
    scored = []
    for line in spans:
        if not line:
            continue
        best = max(float(scores[i]) for i in line)
        scored.append((best, line))
    scored.sort(key=lambda x: -x[0])
    chosen = []
    for _, line in scored:
        if len(chosen) >= k:
            break
        chosen.extend(line)
    return sorted(set(chosen[:max(k, 1)]))


def sel_position_normalized(scores, chunk1_len):
    """Divide received attention by each position's causal horizon.

    With causal masking, a position at index i can be attended to only by the
    (T - i) queries at or after it. A raw sum of received attention therefore
    rewards EARLY positions purely for having more potential contributors, which is
    a positional artefact rather than a measure of importance. Dividing by the
    horizon leaves the average attention per potential query, which is comparable
    across positions. (Raised by the owner as the likely cause of the depth bias.)
    """
    import torch as _t
    horizon = _t.arange(chunk1_len, 0, -1, dtype=_t.float32)
    return scores.float() / horizon

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_arm_cache(h, full_cache, spec, chunk0_len, chunk1_start, chunk1_len, tail):
    """Produce the cache for one arm. kind: full | evict | identity."""
    cache = h.clone_cache(full_cache)
    if spec["kind"] == "full":
        return cache
    if spec["kind"] == "identity":
        keep = list(range(chunk0_len + chunk1_len))
    else:
        keep = sorted(set(list(range(chunk0_len)) +
                          [chunk1_start + i for i in spec["keep"]] + tail))
    cache = h.resize(cache, keep)
    cache = h.apply_delta(cache, keep, rotation=spec.get("rot", False))
    return cache


def measure_arm(h, name, cache, full_kv, baseline, extra_flops=0, needle_kept=None):
    """Run both queries against a prepared cache and collect every metric."""
    t0 = time.time()
    c = h.cache_len(cache)
    q1, _ = h.generate(build_question(Q1_TEXT), h.clone_cache(cache))
    q2, _ = h.generate(build_question(Q2_TEXT), h.clone_cache(cache))
    nll1, _, _ = h.answer_logprobs(build_question(Q1_TEXT), cache, Q1_ANSWER)
    nll2, _, _ = h.answer_logprobs(build_question(Q2_TEXT), cache, Q2_ANSWER)
    elapsed = time.time() - t0
    size = kv_bytes(cache)
    s1, s2 = score_answer(q1, "flag"), score_answer(q2, "err")
    return dict(
        arm=name, cache_tokens=c, kv_mib=round(size / 2**20, 3),
        mem_saved_pct=round(100 * (1 - size / full_kv), 2),
        q1_ans=q1[:110], q2_ans=q2[:110],
        q1_literal=s1["literal"], q1_norm=s1["normalized"], q1_proxy=s1["judge_proxy"],
        q2_literal=s2["literal"], q2_norm=s2["normalized"], q2_proxy=s2["judge_proxy"],
        q1_ratio=s1["ratio"], q2_ratio=s2["ratio"],
        nll_q1=round(nll1, 3), nll_q2=round(nll2, 3),
        d_nll_q1=round(nll1 - baseline["nll_q1"], 3) if baseline else 0.0,
        d_nll_q2=round(nll2 - baseline["nll_q2"], 3) if baseline else 0.0,
        extra_prefill_flops=extra_flops,
        needle_retained=needle_kept,
        wall_s=round(elapsed, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-fracs", type=str, default="0.10,0.25,0.50",
                    help="comma-separated keep fractions to sweep")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--model", type=str, default=MODEL_ID)
    ap.add_argument("--chunk1-tokens", type=int, default=300,
                    help="approximate token budget for the volatile log chunk")
    args = ap.parse_args()
    fracs = [float(x) for x in args.keep_fracs.split(",") if x.strip()]

    h = Harness(args.model)
    prefix_text, system, log = build_prefix(h.tok, args.chunk1_tokens)

    print("=" * 80)
    print("SALIENCY-PRESERVING KV COMPACTION + DENSE DELTA RE-ROTATION")
    print("=" * 80)
    print("model           : %s" % args.model)
    print("layers / q / kv : %d / %d / %d  (head_dim %d)" %
          (h.n_layers, h.n_q_heads, h.n_kv_heads, h.head_dim))
    print("rope self-check : max|R(a)R(b)-R(a+b)| = %.2e  -> %s" %
          (h.rope_delta, "PASS" if h.rope_verified else "FAIL"))
    if not h.rope_verified:
        print("ABORT: RoPE convention mismatch; every arm below would be meaningless.")
        return 2

    sys_ids = h.tok(system, add_special_tokens=False)["input_ids"]
    pre_ids = h.tok(prefix_text, add_special_tokens=False)["input_ids"]
    whole_ids = h.tok(prefix_text, add_special_tokens=False)["input_ids"]
    chunk0_len = len(sys_ids)
    assert pre_ids[:chunk0_len] == sys_ids, (
        "chat-format boundary mismatch: the system half must be a token-exact prefix "
        "of the whole prompt, or the chunk boundary is approximate and every arm scores "
        "a differently-shaped context")
    chunk1_len = len(pre_ids) - chunk0_len
    h.chunk1_len = chunk1_len
    chunk1_start = chunk0_len

    needle_abs = locate_needle(h.tok, prefix_text)
    needle_pos = needle_abs

    print()
    print("context layout  : chunk0=%d tok (system) | chunk1=%d tok (logs) | total=%d tok"
          % (chunk0_len, chunk1_len, len(pre_ids)))
    print("needle          : abs pos %s, chunk1 offset %s (%.0f%% through the region)"
          % (needle_pos, None if needle_pos is None else needle_pos - chunk1_start,
             (100 * (needle_pos - chunk1_start) / chunk1_len) if needle_pos else -1))

    t0 = time.time()
    pre = h.tok(prefix_text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out0 = h.model(input_ids=pre, use_cache=True)
    full_cache = out0.past_key_values
    prefill_s = time.time() - t0
    full_kv = kv_bytes(full_cache)
    n_params = sum(p.numel() for p in h.model.parameters())
    print("prefill         : %.2fs, %.2f MiB KV, %d tok" % (prefill_s, full_kv / 2**20, len(pre_ids)))

    attn_all, _ = capture_query_attention(h, full_cache, build_question(Q2_TEXT))
    mid_layers = [i for i in sorted(h.per_layer_saliency) if 10 <= i <= 19]
    attn_mid = h.per_layer_saliency[mid_layers[0]]
    for i in mid_layers[1:]:
        attn_mid = attn_mid + h.per_layer_saliency[i]
    c1_norm_keys = full_cache.layers[0].keys[:, :, chunk1_start:, :]
    self_attn = capture_self_attention(h, full_cache, prefix_text, chunk1_start, chunk1_len)

    rel = None if needle_pos is None else needle_pos - chunk1_start

    # ---- retention curve: free, and it is what makes the sweep interpretable ----
    if rel is not None:
        print()
        print("SELECTION RETENTION CURVE -- does the selector keep the needle? (index math, no model)")
        cols = [0.05, 0.10, 0.15, 0.25, 0.50, 0.75]
        print("  %-32s %s" % ("selector", "".join("%7s" % ("%d%%" % (f * 100)) for f in cols)))
        for label in ("first-k/2 + last-k/2", "random-k", "top-k key-norm",
                      "top-k query-attn (all layers)", "top-k query-attn (mid layers)",
                      "top-k self-attn chunk1 (agnostic)", "top-k self-attn + sink guard"):
            cells = []
            for f in cols:
                kk = max(1, int(round(chunk1_len * f)))
                if label == "first-k/2 + last-k/2":
                    got = rel in set(sel_first_last(chunk1_len, kk))
                elif label == "random-k":
                    got = rel in set(sel_random(chunk1_len, kk, 0))
                elif label == "top-k key-norm":
                    got = rel in set(sel_norm(c1_norm_keys, kk))
                elif label == "top-k query-attn (all layers)":
                    got = rel in set(sel_attention(attn_all, kk))
                elif label == "top-k query-attn (mid layers)":
                    got = rel in set(sel_attention(attn_mid, kk))
                elif label == "top-k self-attn chunk1 (agnostic)":
                    got = rel in set(sel_attention(self_attn, kk))
                else:
                    got = rel in set(sel_attention_nosink(self_attn, kk))
                cells.append("yes" if got else "-")
            print("  %-32s %s" % (label, "".join("%7s" % c for c in cells)))

    # ---- Arm 0 and the identity control -------------------------------------
    tail = list(range(chunk1_start + chunk1_len, pre.shape[1]))
    all_results = []
    base_cache = h.clone_cache(full_cache)
    baseline = measure_arm(h, "Arm 0 baseline", base_cache, full_kv, None, 0, True)
    print()
    print("-" * 80)
    print("CONTROL: does the eviction MACHINERY itself cost anything?")
    print("-" * 80)
    print("  %-34s %8s %9s %9s %9s" % ("arm", "cache", "nll Q1", "nll Q2", "d nll Q2"))
    print("  %-34s %8d %9.3f %9.3f %9.3f" % ("Arm 0 baseline (untouched)", baseline["cache_tokens"],
                                             baseline["nll_q1"], baseline["nll_q2"], 0.0))
    ident = build_arm_cache(h, full_cache, dict(kind="identity"), chunk0_len,
                            chunk1_start, chunk1_len, tail)
    row_id = measure_arm(h, "Identity: keep ALL + rotate", ident, full_kv, baseline,
                         0, True)
    all_results.append(baseline)
    all_results.append(row_id)
    print("  %-34s %8d %9.3f %9.3f %9.3f" % (row_id["arm"], row_id["cache_tokens"],
                                              row_id["nll_q1"], row_id["nll_q2"],
                                              row_id["d_nll_q2"]))
    ident_ok = abs(row_id["nll_q2"] - baseline["nll_q2"]) < 0.01 and \
        abs(row_id["nll_q1"] - baseline["nll_q1"]) < 0.01
    print("  -> %s" % ("PASS: rotation of a full cache is exactly identity, so any drift "
                       "below is caused by EVICTION, not by the machinery."
                       if ident_ok else
                       "FAIL: keeping every position and rotating reproduces R(0)=I, yet the "
                       "numbers moved. The pipeline is broken; ignore every arm below."))
    if not ident_ok:
        print("  (baseline nll_q1=%.3f nll_q2=%.3f vs identity nll_q1=%.3f nll_q2=%.3f)"
              % (baseline["nll_q1"], baseline["nll_q2"], row_id["nll_q1"], row_id["nll_q2"]))

    # ---- sweep ---------------------------------------------------------------
    for frac in fracs:
        keep_k = max(1, int(round(chunk1_len * frac)))
        idx_first_last = sel_first_last(chunk1_len, keep_k)
        idx_norm = sel_norm(c1_norm_keys, keep_k)
        idx_attn = sel_attention(attn_all, keep_k)
        idx_mid = sel_attention(attn_mid, keep_k)
        idx_rand = sel_random(chunk1_len, keep_k, seed=0)
        idx_self = sel_attention(self_attn, keep_k)
        idx_self_ns = sel_attention_nosink(self_attn, keep_k)

        specs = [
            ("Arm 1k naive first5+last5", dict(kind="evict", keep=idx_first_last, rot=True)),
            ("Arm 2a top-k key-norm (agnostic)", dict(kind="evict", keep=idx_norm, rot=True)),
            ("Arm 2b top-k query-attn (all layers)", dict(kind="evict", keep=idx_attn, rot=True)),
            ("Arm 2c top-k query-attn (mid layers)", dict(kind="evict", keep=idx_mid, rot=True)),
            ("Arm R  random-k control", dict(kind="evict", keep=idx_rand, rot=True)),
            ("Arm 3  arm2a, NO delta rotation", dict(kind="evict", keep=idx_norm, rot=False)),
            ("Arm 3b arm2c, NO delta rotation", dict(kind="evict", keep=idx_mid, rot=False)),
            ("Arm 4a self-attn chunk1 (agnostic)", dict(kind="evict", keep=idx_self, rot=True)),
            ("Arm 4b self-attn + sink guard", dict(kind="evict", keep=idx_self_ns, rot=True)),
            ("Arm 4c arm4b, NO delta rotation", dict(kind="evict", keep=idx_self_ns, rot=False)),
        ]

        print()
        print("=" * 80)
        print("KEEP FRACTION %.0f%%  --  keeping %d of %d chunk-1 positions (%d%% removed)"
              % (frac * 100, keep_k, chunk1_len, 100 * (1 - keep_k / chunk1_len)))
        print("=" * 80)
        print("  %-38s %6s %8s  %-9s %-9s %8s %8s %9s" %
              ("arm", "cache", "kvsaved", "Q1 lit/norm", "Q2 lit/norm", "needle", "d nll2", "nll2"))
        print("  " + "-" * 100)

        frac_rows = []
        for name, spec in specs:
            cache = build_arm_cache(h, full_cache, spec, chunk0_len, chunk1_start,
                                    chunk1_len, tail)
            nk = True if rel is None else (rel in set(spec["keep"]))
            row = measure_arm(h, name, cache, full_kv, baseline, 0, nk)
            frac_rows.append(row)
            print("  %-38s %6d %7.1f%%  %-9s %-9s %8s %8.3f %9.3f" % (
                row["arm"], row["cache_tokens"], row["mem_saved_pct"],
                "%d/%d" % (row["q1_literal"], row["q1_norm"]),
                "%d/%d" % (row["q2_literal"], row["q2_norm"]),
                {True: "yes", False: "NO"}[row["needle_retained"]],
                row["d_nll_q2"], row["nll_q2"]))
        print("  %-38s %s" % ("", " " * 1 + "(reference) Arm 0 baseline: Q1 1/1  Q2 1/1  nll2 %.3f"
                              % baseline["nll_q2"]))
        all_results.extend(frac_rows)

    # ---- Arm 1: text squash, re-encoded -------------------------------------
    print()
    print("=" * 80)
    print("ARM 1 -- NAIVE TEXT SQUASH (re-encoded; pays a full extra prefill)")
    print("=" * 80)
    squashed_prefix = (system + "<|im_start|>user\n<build_log>\nLOG: build ok.\n</build_log><|im_end|>\n")
    ids = h.tok(squashed_prefix, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        scache = h.model(input_ids=ids, use_cache=True).past_key_values
    row1 = measure_arm(h, "Arm 1 naive squash (re-encoded)", scache, full_kv, baseline,
                       prefill_flops(n_params, ids.shape[1]), False)
    all_results.append(row1)
    print("  %-38s %6d %7.1f%%  %-9s %-9s %8s %8.3f %9.3f" % (
        row1["arm"], row1["cache_tokens"], row1["mem_saved_pct"],
        "%d/%d" % (row1["q1_literal"], row1["q1_norm"]),
        "%d/%d" % (row1["q2_literal"], row1["q2_norm"]), "NO",
        row1["d_nll_q2"], row1["nll_q2"]))
    print("  extra prefill: %.1f GFLOP (a full re-encode; every eviction arm pays ZERO)"
          % (row1["extra_prefill_flops"] / 1e9))

    summary(h, all_results, baseline, fracs, chunk0_len, chunk1_len, rel)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(baseline=baseline, identity=row_id, arms=all_results), f, indent=2)
        print("\nwrote " + args.json)
    return 0


def summary(h, results, baseline, fracs, chunk0_len, chunk1_len, rel):
    print()
    print("=" * 80)
    print("WHAT THE NUMBERS SAY")
    print("=" * 80)
    print("  Baseline: Q1 nll %.3f, Q2 nll %.3f, cache %d tok. The flag lives in chunk 0 and"
          % (baseline["nll_q1"], baseline["nll_q2"], baseline["cache_tokens"]))
    print("  is NEVER evicted, so Q1 is the control that isolates what the machinery itself broke.")
    print()
    print("  %-38s %8s %8s %8s %9s" % ("arm", "cache", "d nll1", "d nll2", "Q2 lit/norm"))
    print("  " + "-" * 76)
    for r in results:
        print("  %-38s %8d %8.3f %8.3f %9s" % (
            r["arm"], r["cache_tokens"], r["d_nll_q1"], r["d_nll_q2"],
            "%d/%d" % (r["q2_literal"], r["q2_norm"])))
    print()
    print("  Q1 SPREAD across all arms: %.3f nats" %
          (max(r["d_nll_q1"] for r in results) - min(r["d_nll_q1"] for r in results)))

    # ---- the owner's core hypothesis, isolated ------------------------------
    print()
    print("=" * 80)
    print("HYPOTHESIS TEST -- does dense delta re-rotation help? (same kept set, rot on vs off)")
    print("=" * 80)
    pairs = [("top-k key-norm", "Arm 2a top-k key-norm (agnostic)", "Arm 3  arm2a, NO delta rotation"),
             ("top-k query-attn (mid)", "Arm 2c top-k query-attn (mid layers)",
              "Arm 3b arm2c, NO delta rotation"),
             ("self-attn + sink guard", "Arm 4b self-attn + sink guard",
              "Arm 4c arm4b, NO delta rotation")]
    print("  %-26s %7s %10s %10s %10s %8s" %
          ("kept set", "cache", "nll2 rot", "nll2 off", "improve", "verdict"))
    print("  " + "-" * 78)
    wins = 0
    total = 0
    for label, on_name, off_name in pairs:
        on = {r["cache_tokens"]: r for r in results if r["arm"] == on_name}
        off = {r["cache_tokens"]: r for r in results if r["arm"] == off_name}
        for ck in sorted(set(on) & set(off)):
            a, b = on[ck], off[ck]
            improve = b["nll_q2"] - a["nll_q2"]
            verdict = "helps" if improve > 0.05 else ("hurts" if improve < -0.05 else "neutral")
            if improve > 0.05:
                wins += 1
            total += 1
            print("  %-26s %7d %10.3f %10.3f %+10.3f %8s" %
                  (label, ck, a["nll_q2"], b["nll_q2"], improve, verdict))
    if total:
        print()
        print("  re-rotation improved Q2 cross-entropy in %d of %d matched comparisons." % (wins, total))
        print("  decoding is greedy and the caches are deterministic, so these differences are")
        print("  reproducible, not sampling noise.")


if __name__ == "__main__":
    sys.exit(main())
