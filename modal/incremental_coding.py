"""INCREMENTAL SPEC CODING — a long-horizon coding benchmark that can actually discriminate.

    modal run modal/incremental_coding.py --model qwen2.5-coder-32b --arm linear  --features 16
    modal run modal/incremental_coding.py --model qwen2.5-coder-32b --arm runtime --features 16

★ WHY THIS EXISTS AND SWE-BENCH DOES NOT DO THE JOB.

SWE-bench Lite measured 0/10 on every model and both arms. That is a real result but it is a
CONSTANT: when every arm sits on the floor, resolution cannot separate the layouts and the
comparison disappears. A benchmark that cannot discriminate is not evidence about a cache policy,
however respectable its provenance.

This is built against that failure directly. Three properties, each chosen because its absence
makes a benchmark unable to discriminate:

  1. SOLVABLE PER STEP, UNFORGIVING IN TOTAL. Each turn asks for one small, unambiguous function —
     trivially within reach of any of these models. The difficulty is not any single turn; it is
     that the module must still contain ALL the earlier functions when the suite runs at the end.
     A model that cannot manage a growing context does not fail to write code, it fails to KEEP it.

  2. THE FAILURE IS ATTRIBUTABLE. The suite has one test per feature, so a lost requirement is a
     specific named failure, not a lower score. `test_feature_07` failing means feature 7 was
     overwritten or forgotten, which is a statement about context retention rather than about
     coding ability. That is the measurement this whole project is about.

  3. NO SHORTCUT THAT BYPASSES MEMORY. Re-reading the file is allowed and realistic — but the file
     only contains what the model already wrote, so re-reading cannot recover a function that was
     overwritten. The only way to pass all N tests is to have built the module additively across
     the turns, which requires carrying the earlier turns forward.

★ THE ARM DIFFERENCE, and it is deliberately the ONLY one: `linear` keeps every turn's tool output
in the transcript; `runtime` keeps the system prompt plus the last K turns. Everything else —
model, prompt, tools, turn budget, temperature — is identical, so a difference in the table is
attributable to the context policy.

★ WHAT IS LOGGED, because "did it win" is not one number: per-feature pass/fail, the turn at which
each feature was written, the context length per turn, TTFT, and peak resident KV.
"""
import json, os, modal

app = modal.App("ccai-incremental-coding")

# Context budget for the kept transcript, applied IDENTICALLY to both arms.
# 24000 leaves ~8k of the 32k window for the system prompt, the new instruction
# and the reply, so the comparison is about the context POLICY, not about which
# arm happened to overflow the model first.
MAX_PROMPT_TOKENS = int(os.environ.get("CCAI_MAX_PROMPT_TOKENS", "12000"))

# ★ THE RUNTIME'S OWN CONTEXT BUDGET -- deliberately SMALL, because a bounded context is the whole
# claim being tested. Linear is capped only by the model window (MAX_PROMPT_TOKENS); the runtime is
# capped here, so the two arms have genuinely different working sets rather than both converging on
# whatever fits in the larger number. Sized to hold the anchored turn-1 contract plus the last few
# appends.
RUNTIME_TOKENS = int(os.environ.get("CCAI_RUNTIME_TOKENS", "2048"))
GPU = os.environ.get("CCAI_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "bitsandbytes")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          # ★★ WITHOUT THIS THE RUN IS INVISIBLE. Python block-buffers stdout when it is not a tty,
          # so every per-turn print sat in an 8 KB buffer and `modal app logs` showed nothing past
          # "Loading weights 100%" -- while the container was happily generating. Only tqdm leaked
          # through, because it writes to stderr (unbuffered). A 35-minute run with a frozen log is
          # indistinguishable from a wedged one, which is exactly the false alarm this prevents.
          "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    # ★ GEMMA 4 IS THE COMPETITION MODEL (Kaggle "Gemma 4 Developer Agent Paper Track",
    # closes 2026-11-12). Apache 2.0 and UNGATED -- unlike every other Gemma (all `gated=manual`,
    # HTTP 403 on file access from this account). Verified by reading config.json, because a 200 on
    # the metadata endpoint does NOT mean access (that cost us a cycle on Llama-3.1).
    #   gemma-4-12B-it      23.9 GB bf16 -> ~7 GB at 4-bit; fits A100-40GB comfortably
    #   gemma-4-E4B-it      16.0 GB bf16 -> runs on a cheap L4
    #   gemma-4-26B-A4B-it  51.6 GB bf16 -> MoE, 3.8B active; needs A100-80GB at bf16
    "gemma-4-12b": "google/gemma-4-12B-it",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
    "gemma-4-26b-a4b": "google/gemma-4-26B-A4B-it",
    # prior benchmark models, kept so old runs stay reproducible
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    # tiny, for zero-GPU logic tests (resume/checkpoint) on CPU
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

# ── THE FEATURE SET ──────────────────────────────────────────────────────────────────────────
# Each entry is (spec_text, test_body). The specs are short and unambiguous on purpose: the
# difficulty must come from the HORIZON, not from any one instruction, or the benchmark measures
# prompt comprehension instead of context retention.
#
# ★ SOME VALUES ARE DELIBERATELY NON-OBVIOUS. A spec saying "returns 42" can be guessed; these
# carry specific constants and orderings that can only be satisfied by having read the spec. That
# matters because a model with no memory of turn 3 must FAIL feature 3 rather than guess it.
FEATURES = [
    ("`fn_00_k` returns the integer 17", "assert m.fn_00_k() == 17"),
    ("`fn_01_name` returns the string \"quokka\"", "assert m.fn_01_name() == \"quokka\""),
    ("`fn_02_double(n)` returns n * 2", "assert m.fn_02_double(21) == 42"),
    ("`fn_03_add(a, b)` returns the sum of a and b", "assert m.fn_03_add(19, 23) == 42"),
    ("`fn_04_rev(s)` returns s reversed", "assert m.fn_04_rev(\"abc\") == \"cba\""),
    ("`fn_05_len(xs)` returns the number of items in xs", "assert m.fn_05_len([1,2,3,4]) == 4"),
    ("`fn_06_up(s)` returns s upper-cased", "assert m.fn_06_up(\"ab\") == \"AB\""),
    ("`fn_07_max3(a,b,c)` returns the largest of three", "assert m.fn_07_max3(3,9,4) == 9"),
    ("`fn_08_evens(xs)` returns only the even numbers, in order",
     "assert m.fn_08_evens([1,2,3,4,5,6]) == [2,4,6]"),
    ("`fn_09_wordcount(s)` returns the number of whitespace-separated words",
     "assert m.fn_09_wordcount(\"a b c d\") == 4"),
    ("`fn_10_sign(n)` returns -1 for negative, 0 for zero, 1 for positive",
     "assert (m.fn_10_sign(-5), m.fn_10_sign(0), m.fn_10_sign(5)) == (-1, 0, 1)"),
    ("`fn_11_clamp(n, lo, hi)` clamps n into [lo, hi]", "assert m.fn_11_clamp(15, 0, 10) == 10"),
    ("`fn_12_fact(n)` returns n factorial (n! ), and fn_12_fact(0) is 1",
     "assert (m.fn_12_fact(0), m.fn_12_fact(5)) == (1, 120)"),
    ("`fn_13_unique(xs)` returns the distinct items preserving first-seen order",
     "assert m.fn_13_unique([3,1,3,2,1]) == [3,1,2]"),
    ("`fn_14_strip_vowels(s)` removes a,e,i,o,u (lowercase only)",
     "assert m.fn_14_strip_vowels(\"banana\") == \"bnn\""),
    ("`fn_15_base_to_dec(s, base)` parses s as a number in the given base",
     "assert m.fn_15_base_to_dec(\"ff\", 16) == 255"),
    ("`fn_16_chunk(xs, n)` splits xs into consecutive lists of length n, last may be shorter",
     "assert m.fn_16_chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]"),
    ("`fn_17_title(s)` capitalises the first letter of every whitespace-separated word",
     "assert m.fn_17_title(\"the quick fox\") == \"The Quick Fox\""),
    ("`fn_18_rle(s)` returns run-length encoding as a list of [char, count] pairs",
     "assert m.fn_18_rle(\"aabbb\") == [[\"a\",2],[\"b\",3]]"),
    ("`fn_19_dot(a, b)` returns the dot product of two equal-length lists",
     "assert m.fn_19_dot([1,2,3],[4,5,6]) == 32"),
    ("`fn_20_is_pal(s)` returns True iff s is a palindrome, ignoring case",
     "assert (m.fn_20_is_pal(\"Racecar\"), m.fn_20_is_pal(\"ab\")) == (True, False)"),
    ("`fn_21_flatten(nested)` returns one flat list from a list containing lists",
     "assert m.fn_21_flatten([[1,2],[3],[4,5]]) == [1,2,3,4,5]"),
    ("`fn_22_gcd(a, b)` returns the greatest common divisor",
     "assert m.fn_22_gcd(48, 18) == 6"),
    ("`fn_23_caesar(s, k)` shifts each lowercase letter by k, wrapping past z",
     "assert m.fn_23_caesar(\"xyz\", 3) == \"abc\""),
    ("`fn_24_zip2(a, b)` returns a list of [a[i], b[i]] pairs, stopping at the shorter input",
     "assert m.fn_24_zip2([1,2,3],[4,5]) == [[1,4],[2,5]]"),
    ("`fn_25_count_char(s, c)` returns how many times character c occurs in s",
     "assert m.fn_25_count_char(\"banana\", \"a\") == 3"),
    ("`fn_26_median(xs)` returns the middle value for an odd-length sorted-by-you list",
     "assert m.fn_26_median([5,1,3]) == 3"),
    ("`fn_27_swap_case(s)` swaps lower to upper and upper to lower",
     "assert m.fn_27_swap_case(\"AbC\") == \"aBc\""),
    ("`fn_28_intersect(a, b)` returns items present in both lists, without duplicates",
     "assert sorted(m.fn_28_intersect([1,2,3],[2,3,4])) == [2,3]"),
    ("`fn_29_roman(n)` converts 1..20 to a roman numeral string",
     "assert m.fn_29_roman(14) == \"XIV\""),
    ("`fn_30_bin_count(n)` returns the number of 1 bits in n's binary form",
     "assert m.fn_30_bin_count(13) == 3"),
    ("`fn_31_rotate(xs, k)` rotates xs left by k positions",
     "assert m.fn_31_rotate([1,2,3,4], 1) == [2,3,4,1]"),
    ("`fn_32_all_same(xs)` returns True iff every item is equal",
     "assert (m.fn_32_all_same([7,7]), m.fn_32_all_same([7,8])) == (True, False)"),
    ("`fn_33_ord_sum(s)` sums the unicode code points of the characters in s",
     "assert m.fn_33_ord_sum(\"AB\") == 131"),
    ("`fn_34_apply_n(f, x, n)` applies f to x exactly n times",
     "assert m.fn_34_apply_n(lambda v: v + 3, 1, 4) == 13"),
    ("`fn_35_trim_dup(xs)` removes CONSECUTIVE duplicates only",
     "assert m.fn_35_trim_dup([1,1,2,2,2,3,1]) == [1,2,3,1]"),
    ("`fn_36_is_sorted(xs)` returns True iff xs is non-decreasing",
     "assert (m.fn_36_is_sorted([1,2,2]), m.fn_36_is_sorted([2,1])) == (True, False)"),
    ("`fn_37_split_pairs(s)` splits a string into consecutive 2-char chunks",
     "assert m.fn_37_split_pairs(\"abcde\") == [\"ab\",\"cd\",\"e\"]"),
    ("`fn_38_scale(xs, f)` returns every item of xs multiplied by f",
     "assert m.fn_38_scale([1,2,3], 3) == [3,6,9]"),
    ("`fn_39_second_max(xs)` returns the second largest DISTINCT value",
     "assert m.fn_39_second_max([5,1,5,3]) == 3"),
]

# ── EXTENDING THE SET TO 120 ─────────────────────────────────────────────────────────────────
# ★ WHY GENERATE RATHER THAN HAND-WRITE. The 40-feature set saturated: the whole module still fit
# in context, so BOTH arms passed (40/41) and the benchmark could not discriminate -- the same
# failure it was built to avoid. The difficulty is meant to come from HORIZON, so what is needed is
# more turns, not harder individual turns.
#
# ★ THE GENERATED ONES ARE DELIBERATELY DISSIMILAR from each other in their constant and their
# shape: a model that lost the spec for feature 73 cannot infer it from feature 74. Every test
# carries a specific constant, so it cannot be guessed.
_TEMPLATES = [
    ("addk", "`fn_%02d_addk(xs, k)` returns every item of xs increased by %d"),
    ("mulk", "`fn_%02d_mulk(xs, k)` returns every item of xs multiplied by %d"),
    ("firstk", "`fn_%02d_firstk(xs, k)` returns the first %d items of xs"),
    ("lastk", "`fn_%02d_lastk(xs, k)` returns the last %d items of xs"),
    ("sum_sq", "`fn_%02d_sum_sq(xs)` returns the sum of the squares of xs"),
    ("count", "`fn_%02d_count(s, ch)` returns how many times the character `%s` occurs in s"),
    ("repeat", "`fn_%02d_repeat(s, n)` returns s repeated n times, joined with `%s`"),
    ("zero_pad", "`fn_%02d_zero_pad(n, w)` returns n as a string left-padded to width w with zeros"),
    ("minmax", "`fn_%02d_minmax(xs)` returns a tuple (minimum, maximum) of xs"),
    ("join", "`fn_%02d_join(xs, sep)` joins xs with the separator `%s`"),
    ("absmax", "`fn_%02d_absmax(xs)` returns the item of xs with the largest absolute value"),
    ("step", "`fn_%02d_step(xs, start, k)` returns the items of xs from index start, taking every kth item"),
]


def _one_feature(i):
    """Deterministically build feature `i` as (spec, test_body). No RNG: reruns are identical."""
    kind, tmpl = _TEMPLATES[i % len(_TEMPLATES)]
    k = (i % 7) + 2
    base = ((i * 13) % 40) + 1
    if kind in ("addk", "mulk"):
        spec = tmpl % (i, k)
        if kind == "addk":
            xs = [base, base + 1]
            want = [v + k for v in xs]
        else:
            xs = [base, base + 1]
            want = [v * k for v in xs]
        body = "assert m.fn_%02d_%s([%s], %d) == %s" % (
            i, kind, ", ".join(map(str, xs)), k, want)
    elif kind in ("firstk", "lastk"):
        spec = tmpl % (i, k)
        xs = [base, base + 1, base + 2, base + 3]
        want = xs[:k] if kind == "firstk" else xs[-k:]
        body = "assert m.fn_%02d_%s([%s], %d) == %s" % (
            i, kind, ", ".join(map(str, xs)), k, want)
    elif kind == "sum_sq":
        spec = tmpl % i
        xs = [base, base + 1, base + 2]
        body = "assert m.fn_%02d_sum_sq([%s]) == %d" % (i, ", ".join(map(str, xs)),
                                                        sum(v * v for v in xs))
    elif kind == "count":
        ch = "abcdefg"[i % 7]
        s = ch * ((i % 3) + 1) + "xyz"
        spec = tmpl % (i, ch)
        body = "assert m.fn_%02d_count(%r, %r) == %d" % (i, s, ch, s.count(ch))
    elif kind == "repeat":
        sep = ["-", ".", "_"][i % 3]
        n = (i % 3) + 2
        spec = tmpl % (i, sep)
        body = "assert m.fn_%02d_repeat('ab', %d) == %r" % (i, n, sep.join(["ab"] * n))
    elif kind == "zero_pad":
        spec = tmpl % i
        n = (i % 90) + 5
        body = "assert m.fn_%02d_zero_pad(%d, 4) == %r" % (i, n, str(n).zfill(4))
    elif kind == "minmax":
        spec = tmpl % i
        xs = [base, base - 1, base + 2]
        body = "assert m.fn_%02d_minmax([%s]) == %s" % (i, ", ".join(map(str, xs)),
                                                       (min(xs), max(xs)))
    elif kind == "join":
        sep = ["-", "|", ":"][i % 3]
        spec = tmpl % (i, sep)
        body = "assert m.fn_%02d_join(['x', 'yz'], %r) == %r" % (i, sep, sep.join(["x", "yz"]))
    elif kind == "absmax":
        spec = tmpl % i
        xs = [base, -base - 5, base + 1]
        # ★ `max`, NOT `min`. The spec says "largest absolute value", and the first version of this
        # test asserted `min(xs, key=abs)` -- which failed a CORRECT reference implementation on 7 of
        # the 120 features (every absmax one). A grader that rejects the right answer would have been
        # read as a model failure. Caught by generating a reference solution and grading it before
        # spending any GPU time.
        body = "assert m.fn_%02d_absmax([%s]) == %d" % (i, ", ".join(map(str, xs)),
                                                        max(xs, key=abs))
    else:  # step -- start and k are BOTH parameters, so the spec and the test cannot disagree
        off = i % 3
        kk = (i % 2) + 2
        spec = tmpl % i
        xs = [base + j for j in range(6)]
        body = "assert m.fn_%02d_step([%s], %d, %d) == %s" % (
            i, ", ".join(map(str, xs)), off, kk, xs[off::kk])
    return spec, body


FEATURES_ALL = list(FEATURES) + [_one_feature(i) for i in range(len(FEATURES), 200)]

# ── THE TURN-1 CONTRACT ──────────────────────────────────────────────────────────────────────
# ★★ THIS IS THE DISCRIMINATOR, AND IT IS STATED EXACTLY ONCE.
#
# A long context is only a problem when something in it still matters. The failure tested here does
# not need a context overflow: an instruction given at the START, never repeated, that must still be
# in force at turn 120 with ~120 turns of tool output between the model and it.
#
# The runtime anchors turn 1 (the spec) and evicts the disposable middle; `linear` carries all of it
# and has to find the contract in the noise. If the regular version loses it, that is an ACCURACY
# difference rather than a cost difference -- which is what makes this worth running.
CONTRACT = (
    "STANDING CONTRACT (set now, in force for the rest of this session, and it will NOT be "
    "repeated): solution.py must ALSO define a module-level dict named REGISTRY. Every function you "
    "implement, in this turn and in every later turn, must be registered in it: the key is the "
    "function's name as a string, the value is the function object. REGISTRY must always contain "
    "every function defined so far. "
    # ★★ THE NONCE IS THE CHECK THAT CANNOT BE RE-DERIVED, and it is here because the REGISTRY rule
    # alone turned out to be SELF-SUSTAINING. A model that writes `REGISTRY[...] = ...` in every
    # append leaves that pattern in its recent turns, so a policy keeping only the last K turns can
    # still infer the rule without ever seeing turn 1 -- the check would pass for BOTH arms and
    # discriminate nothing (the same "both arms converge" failure the shared ceiling caused).
    #
    # A value that appears ONLY here, and is required ONLY in the FINAL append, cannot be inferred
    # from recent turns: nothing else in the transcript ever contains it. So it fails exactly when
    # turn 1 has been evicted, which is the property being measured.
    "FINALLY: when you have implemented the last feature you are asked for, append one more line to "
    "solution.py, as its last line, reading exactly: CONTRACT_STAMP = \"QX7-4420-BRAVO\"  "
    "(that exact value; it is given here and nowhere else, and it must appear exactly once)."
)
CONTRACT_NONCE = "QX7-4420-BRAVO"

# ★ THE FINAL TURN IS THE INTERACTION TEST, AND IT IS WHAT MAKES THIS A CONTEXT RETENTION TASK
# RATHER THAN N INDEPENDENT TASKS. It requires functions written many turns earlier to still exist
# and to compose. A model that silently overwrote fn_02 or fn_14 while adding later features passes
# their individual tests by luck of having rewrote them, but cannot satisfy this unless the module
# genuinely accumulated — so it converts "kept a growing module" from a per-feature property into
# one explicit measurement.
INTERACTION_SPEC = ("`fn_99_pipeline(s)` must strip the vowels from s, then upper-case the result, "
                    "then split it into consecutive 2-char chunks. It must reuse "
                    "fn_14_strip_vowels, fn_06_up and fn_37_split_pairs by CALLING them, not by "
                    "reimplementing them.")
INTERACTION_TEST = ("assert m.fn_99_pipeline(\"banana split\") == "
                    "m.fn_37_split_pairs(m.fn_06_up(m.fn_14_strip_vowels(\"banana split\")))")

SYSTEM = """You are a Python engineer building one module, `solution.py`, across many turns.

You are given ONE instruction per turn. Apply it.

Reply with EXACTLY ONE json object and nothing else:

  {"tool": "read_file"}
  {"tool": "write_file", "content": "<FULL new contents of solution.py>"}
  {"tool": "append", "content": "<code to ADD to the end of solution.py>"}
  {"tool": "done"}

Rules:
  - You may only touch solution.py. There is no internet and no other files.
  - ★ THE MODULE IS ACCUMULATIVE. Every instruction you were given is STILL REQUIRED.
  - ★ WHEN AN INSTRUCTION ONLY ADDS A NEW FUNCTION, USE `append`. Do not resend the file.
    Resending a long module wastes your reply budget and is how a reply gets cut off. `append`
    adds to the end and keeps everything already there.
  - Use `write_file` only when you must CHANGE something already in the file.
  - Omit nothing. A missing earlier function is a failure.
  - No explanations. One json object per reply.
"""


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=14400, memory=40960)
def run_inc(model: str = "qwen2.5-coder-32b", arm: str = "linear", features: int = 16,
            keep_turns: int = 4, max_new: int = 3072):
    # ★★ TWO HARD CEILINGS, AND BOTH WERE HIT. Modal kills a function at `timeout`, and the first
    # version used 7200 (2h) with max_new=3072 -- so a 200-turn run on a 32B died at exactly 2h
    # with `FunctionTimeoutError` and THREW AWAY every turn it had completed. The launcher retries
    # three times, so that would have burned ~6 GPU-hours producing nothing.
    #
    # The fix is not only a bigger ceiling:
    #   1. `timeout=14400` (4h) gives real headroom.
    #   2. `max_new=3072` is a CEILING, not a target: the loop stops at EOS, and a one-function
    #      reply is ~150 tokens. But it must be big enough for the WORST case, which is a full-file
    #      rewrite of a 200-function module (~3000 tokens). Cutting it to 768 to save time is what
    #      truncated turn 17 and started the cascade -- the ceiling has to accommodate the largest
    #      legitimate reply, and only then does the EOS stop keep the average cheap.
    #   3. A WALL-CLOCK GUARD inside the loop that stops *before* the platform does and grades
    #      whatever completed. A partial result with "turns_completed=N" is evidence; a timeout
    #      exception is nothing. This is the difference between a job that can be interrupted and
    #      one that can only be wasted.
    TIME_BUDGET_S = 12000          # 3h20m, comfortably inside the 4h platform kill
    import ast as _ast, gc, json as _json, os as _os, re, subprocess as sp, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    # ★ DEVICE AND QUANTISATION ARE CONFIGURABLE so the same file runs on a GPU host, on a smaller
    # GPU, or on CPU for a zero-cost logic test. Bits are the caller's choice and the trade-off is
    # NOT "more bits = cheaper": the 32B is ~19 GB at 4-bit and ~34 GB at 8-bit, so 8-bit needs a
    # LARGER GPU and buys strictly less runway for the same wall-clock. 4-bit is the cheap option.
    _dev = _os.environ.get("CCAI_DEVICE", "cuda")
    _quant = _os.environ.get("CCAI_QUANT", "auto")      # auto | 4 | 8 | none
    if _quant == "auto":
        _quant = "4" if "32b" in model.lower() else "none"
    if _quant in ("4", "8"):
        from transformers import BitsAndBytesConfig
        if _quant == "4":
            _qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True)
        else:
            _qc = BitsAndBytesConfig(load_in_8bit=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=_qc, device_map=_dev,
            attn_implementation="sdpa").eval()
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16,
            device_map=(_dev if _dev != "cpu" else None),
            attn_implementation="sdpa").eval()
        if _dev == "cpu":
            mdl = mdl.to("cpu")

    WORK = _os.environ.get("CCAI_WORK", "/work/inc")
    _os.makedirs(WORK, exist_ok=True)
    solution = _os.path.join(WORK, "solution.py")

    def complete(prompt, mnew):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        # ★★ num_logits_to_keep=1 IS THE FIX FOR THE T4 OOM, AND IT KEEPS THE 12B VIABLE.
        #
        # MEASURED failure: `logits = logits / final_logit_softcapping` -> "Tried to allocate
        # 2.62 GiB" on a 15.6 GB T4 whose weights only take 7.7 GB. The model was computing logits
        # for EVERY position in the prompt: 6000 tokens x 262144 vocab x fp32 = ~6 GB, purely to
        # throw away all but the last row. Greedy decoding needs one row.
        #
        # This is not a workaround for a small GPU -- computing all-positions logits during a
        # prefill is wasted work on ANY card, and it is what made a 7.7 GB model fail on a 15.6 GB
        # device. Passed on the prefill only; the decode step already has seq=1.
        # ★★ CHUNKED PREFILL. A single forward over the whole prompt materialises the attention
        # workspace for every position at once, which is what OOM'd a 15.6 GB T4 holding a 7.7 GB
        # model (measured: `logits = logits / final_logit_softcapping` -> "Tried to allocate
        # 2.62 GiB"). Feeding the prompt in PREFILL_CHUNK-token pieces carries the KV cache across
        # chunks, so peak activation memory is bounded by the chunk rather than the whole prompt --
        # the same technique the Modal harness needed for 16k contexts.
        PREFILL_CHUNK = int(_os.environ.get("CCAI_PREFILL_CHUNK", "512"))
        from transformers.cache_utils import DynamicCache as _DynCache
        _cache = _DynCache()
        _pos = 0
        with torch.no_grad():
            _n = len(ids)
            for _s in range(0, _n, PREFILL_CHUNK):
                _ch = ids[_s:_s + PREFILL_CHUNK]
                o = mdl(input_ids=torch.tensor([_ch], dtype=torch.long, device=_dev),
                        past_key_values=_cache, use_cache=True,
                        cache_position=torch.arange(_pos, _pos + len(_ch), device=_dev),
                        num_logits_to_keep=1)
                _cache = o.past_key_values
                _pos += len(_ch)
        if _dev == "cuda":
            torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen, c, p = [nxt], o.past_key_values, len(ids)
        for _ in range(mnew - 1):
            with torch.no_grad():
                o = mdl(input_ids=nxt, past_key_values=c, use_cache=True,
                        cache_position=torch.tensor([p], device=_dev),
                        num_logits_to_keep=1)
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt); p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        txt = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        kv = sum(c.layers[i].keys.numel() * c.layers[i].keys.element_size()
                 + c.layers[i].values.numel() * c.layers[i].values.element_size()
                 for i in range(len(c.layers)))
        del o, c
        gc.collect(); torch.cuda.empty_cache()
        return txt, dict(ttft_ms=ttft, kv_bytes=kv, prompt_tokens=len(ids),
                         wall_ms=(time.perf_counter() - t0) * 1000)

    def parse(t):
        t = t.strip()
        if "```" in t:
            for part in t.split("```"):
                pp = part[4:].strip() if part.strip().startswith("json") else part.strip()
                if pp.startswith("{"):
                    t = pp; break
        i = t.find("{")
        if i < 0:
            # ★ A CODING MODEL'S DOMINANT PRIOR IS A FENCED CODE BLOCK, NOT JSON. Measured:
            # the linear arm emitted no JSON object for SIXTEEN consecutive turns and wrote
            # exactly once, so its final file was 29 chars and it scored 1/17. Reporting that
            # as a context-retention result would have been a harness bug dressed as a finding.
            # The reply is unambiguous in intent -- a ```python fence in a turn whose only job is
            # to write one file -- so it is accepted as a write_file rather than rejected.
            m = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
            if m and m.group(1).strip():
                return {"tool": "write_file", "content": m.group(1)}, None
            return None, "no JSON object and no code fence"
        body = t[i:]
        try:
            d, _ = _json.JSONDecoder().raw_decode(body)
        except Exception as e:
            # ★ PYTHON-LITERAL FALLBACK — this is the fix that mattered, and the defect it
            # exposes is the whole reason the linear arm scored 1/17.
            #
            # Measured raw reply from the 32B coder:
            #     {'tool': 'write_file', 'content': 'def fn_00_k(): ...'}
            # Single quotes. That is a Python repr, not JSON, and json.loads rejects it with
            # "Expecting property name enclosed in double quotes". `ast.literal_eval` parses
            # exactly that grammar, and it is SAFE: it evaluates literals only and cannot
            # execute a call, so a hostile reply gains nothing over json.loads.
            #
            # The model was doing the work correctly the entire time -- the raw reply shows it
            # accumulating fn_00 AND fn_01 across turns -- and the parser was discarding it.
            # **A parser that rejects valid input is indistinguishable from a model that cannot
            # produce valid output**, which is the same blind spot recorded for the harness's
            # flat-args and raw-newline bugs. Both scored zero for the harness, not the model.
            try:
                lit = _ast.literal_eval(body)
                if isinstance(lit, dict) and lit.get("tool") in (
                        "read_file", "write_file", "append", "done"):
                    return lit, None
            except Exception:
                pass
            try:
                d, _ = _json.JSONDecoder().raw_decode("".join(out))
            except Exception:
                fm = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
                if fm and fm.group(1).strip():
                    return {"tool": "write_file", "content": fm.group(1)}, None
                return None, "invalid JSON: %s" % e
        if not isinstance(d, dict) or d.get("tool") not in ("read_file", "write_file", "append", "done"):
            return None, "bad or unknown tool"
        return d, None

    turns_log, per_feature_written = [], {}
    use = [f for f in FEATURES_ALL[:max(1, min(features, len(FEATURES_ALL)))]]
    specs = [f[0] for f in use] + [INTERACTION_SPEC]

    # ★ THE MODEL'S OWN WINDOW IS THE BUDGET, with headroom for the system prompt, the new
    # instruction and the reply. Qwen2.5-Coder-32B is 32k; leaving ~8k for output and the system
    # prompt keeps the comparison about the CONTEXT POLICY rather than about who hit the limit
    # first. Applied identically to both arms, so it cannot favour either.
    # ★ COUNT ONCE PER BLOCK, NOT ONCE PER CANDIDATE WINDOW. The first version called the
    # tokenizer on the WHOLE transcript to discover it was too big, then popped one block and
    # tokenized the whole thing again -- O(n^2) per turn, and it emitted a real
    # "sequence longer than the model maximum (33516 > 32768)" warning on every turn even though
    # the prompt actually sent was correctly capped. One tokenization per block, then a greedy walk
    # from the end, removes both costs.
    # ★★ THE PROMPT FORMAT IS MODEL-SPECIFIC AND THIS WAS HARDCODED TO QWEN.
    #
    # The benchmark rendered every turn as ChatML (`<|im_start|>user\n…<|im_end|>\n`), which is
    # Qwen's format. Gemma 4 renders `<bos><|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought
    # <channel|>` — it has NO `<|im_start|>` token at all, so a ChatML prompt reaches it as literal
    # text and the instruction boundaries are invisible. Feeding one model another's chat format
    # does not error; it just quietly degrades everything, which would have made the Gemma numbers
    # meaningless. Verified by rendering a real message through each tokenizer before wiring this.
    #
    # Both have a chat_template, so use it. Per-message rendering (BOS stripped) keeps the block
    # structure the eviction policies need, while staying faithful to what the model was trained on.
    _HAS_TPL = bool(getattr(tok, "chat_template", None))
    _BOS = getattr(tok, "bos_token", None) or ""

    def _render_block(role, content):
        if not _HAS_TPL:
            return "<|im_start|>%s\n%s<|im_end|>\n" % (role, content)
        txt = tok.apply_chat_template([{"role": role, "content": content}],
                                      tokenize=False, add_generation_prompt=False)
        if _BOS and txt.startswith(_BOS):
            txt = txt[len(_BOS):]
        return txt

    def _render_prompt(blocks, instr):
        msgs = [{"role": "system", "content": SYSTEM}]
        for r, c in blocks:
            # transcript blocks carry role "user"/"assistant"; a pre-formatted legacy block keeps
            # its own role marker, so map it back rather than guessing.
            msgs.append({"role": r, "content": c})
        msgs.append({"role": "user", "content": instr})
        if _HAS_TPL:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                + "".join("<|im_start|>%s\n%s<|im_end|>\n" % (r, c) for r, c in blocks)
                + "<|im_start|>user\n" + instr + "<|im_end|>\n<|im_start|>assistant\n")

    def _counts(blocks):
        return [len(tok(_render_block(r, c), add_special_tokens=False)["input_ids"])
                for r, c in blocks]

    _SYS_TOKENS = len(tok(_render_block("system", SYSTEM),
                          add_special_tokens=False)["input_ids"])

    transcript = []
    _fail = 0
    _t_start = time.time()
    _stopped_early = None

    # ★★★ CHECKPOINT / RESUME. A GPU run that dies must not cost the whole budget.
    #
    # WHY THIS EXISTS: the first 200-turn run was killed by the platform at turn 62/71 and NOTHING
    # was recoverable -- the only thing persisted was a one-line progress marker, so 71 turns of
    # real model work and every cent spent on them were thrown away. The model's own output (the
    # transcript and solution.py) lived in container-local storage that is destroyed with it.
    # MEASURED on the Modal volume afterwards: `prog_linear.txt` = "turn 62/201 ..." and nothing
    # else. There was no checkpoint to resume from, which is why the run had to be paid for twice.
    #
    # What is saved each turn: the transcript (the model's context), solution.py as it stands, the
    # per-feature attribution, and the metrics log. Resuming requires ALL of them -- restoring the
    # transcript without the file (or the reverse) would silently grade a different run than the
    # one that was interrupted.
    _CKPT = _os.environ.get("CCAI_CKPT_DIR", "/cache")
    _ckpt_path = _os.path.join(_CKPT, "ckpt_%s_%s_f%d.json" % (
        model.replace("/", "-"), arm, features))

    def _save_ckpt(turn_ix):
        try:
            _os.makedirs(_CKPT, exist_ok=True)
            _json.dump({
                "turn_ix": turn_ix, "model": model, "arm": arm, "features": features,
                "transcript": transcript, "turns_log": turns_log,
                "per_feature_written": per_feature_written,
                "solution": (open(solution).read() if _os.path.exists(solution) else ""),
            }, open(_ckpt_path, "w"))
        except Exception as _e:
            print("[%s] checkpoint write failed (%s) -- continuing" % (arm, _e), flush=True)

    _start_i = 0
    if _os.environ.get("CCAI_RESUME", "1") == "1" and _os.path.exists(_ckpt_path):
        try:
            _ck = _json.load(open(_ckpt_path))
            # Only resume a checkpoint that describes THIS experiment. A stale file from a different
            # feature count or model would silently grade the wrong run.
            if (_ck.get("model") == model and _ck.get("arm") == arm
                    and _ck.get("features") == features):
                transcript = [tuple(b) for b in _ck["transcript"]]
                turns_log = list(_ck.get("turns_log") or [])
                per_feature_written = dict(_ck.get("per_feature_written") or {})
                _start_i = int(_ck.get("turn_ix", 0)) + 1
                if _ck.get("solution"):
                    open(solution, "w").write(_ck["solution"])
                print("[%s] RESUMED from checkpoint at turn %d/%d (transcript=%d blocks)"
                      % (arm, _start_i, len(specs), len(transcript)), flush=True)
        except Exception as _e:
            print("[%s] checkpoint unreadable (%s) -- starting fresh" % (arm, _e), flush=True)

    for i, spec in enumerate(specs):
        if i < _start_i:
            continue                                    # already done in an earlier attempt
        # ★ STOP BEFORE THE PLATFORM DOES. A partial run that grades is evidence; a run killed by
        # the platform is an exception with nothing in it. Whatever completed is still a valid
        # measurement of the turns that ran.
        if time.time() - _t_start > TIME_BUDGET_S:
            _stopped_early = i
            print("[%s] TIME BUDGET reached at turn %d/%d -- grading what completed"
                  % (arm, i, len(specs)), flush=True)
            break
        instr = ("INSTRUCTION %d of %d: implement %s\n"
                 "Write the FULL solution.py including every function from every previous "
                 "instruction." % (i + 1, len(specs), spec))
        if i == 0:
            # ★ THE CONTRACT IS STATED EXACTLY ONCE, IN TURN 1, and never repeated. That is the
            # whole point: at turn 120 it is ~120 turns away from the model, and a policy that
            # carries everything has to find it while a policy that anchors it does not.
            instr = CONTRACT + "\n\n" + instr
        if arm == "linear":
            # ★ FULL TRANSCRIPT, TRUNCATED ONLY WHEN THE MODEL'S WINDOW FORCES IT -- oldest first.
            # This is what "linear prefill" MEANS and the benchmark was WRONG about it until now:
            # it kept only transcript[-20:], which is a WINDOW, so "linear" was already evicting
            # and the two arms never really differed.
            #
            # ★ WHY FRONT-TRUNCATION IS THE HONEST FAILURE MODE, not a rigged one: a fixed context
            # window means SOMETHING must go, and dropping the oldest is what every naive policy
            # does. The contract sits at the oldest position precisely because a task brief
            # normally does. `runtime` is the one that keeps it deliberately.
            keep = list(transcript)
            costs = _counts(keep)
            budget = MAX_PROMPT_TOKENS - _SYS_TOKENS
            total = sum(costs)
            while keep and total > budget:
                total -= costs.pop(0)
                keep.pop(0)                                # drop oldest
        else:
            # ★★ THE RUNTIME IS BOUNDED BY TOKENS, NOT BY A BLOCK COUNT -- and this was WRONG in
            # the first version in a way that made the two arms indistinguishable.
            #
            # The first version kept `transcript[:1] + transcript[-8:]` and only shrank that when it
            # exceeded MAX_PROMPT_TOKENS (12k). Measured consequence: it grew to ~9.9k while linear
            # reached ~11.2k, so at turn 62 the two arms had the SAME latency (4180 vs 4308 ms) and
            # the experiment could not discriminate at all. **The ceiling was setting the working
            # set, not the policy** -- both arms converged on "whatever fits in 12k".
            #
            # The runtime's entire claim is a context that stays SMALL and FLAT. So it gets its own
            # modest budget (RUNTIME_TOKENS, default 2k) and is capped there regardless of how much
            # room the model has. That is the eviction being tested: anchor + the most recent turns
            # that fit, and the middle is dropped on every turn once the module outgrows the budget.
            keep = transcript[:1] + transcript[-(keep_turns * 2):]
            costs = _counts(keep)
            budget = RUNTIME_TOKENS - _SYS_TOKENS
            total = sum(costs)
            # index 1 onwards is droppable; index 0 is the anchored contract and is never popped.
            while len(keep) > 1 and total > budget:
                total -= costs.pop(1)
                keep.pop(1)                                # drop the oldest MIDDLE turn
        prompt = _render_prompt(keep, instr)
        # ★ DUMP THE EXACT FIRST PROMPT ONCE. A prompt rendered for the wrong model family does not
        # error -- it reaches the model as literal text with invisible turn boundaries, and every
        # number after that is about the wrong thing. This is the only way to prove what was sent.
        if i == 0 and _os.environ.get("CCAI_DEBUG_PROMPT") == "1":
            try:
                print("[%s] FIRST PROMPT (%d chars, %d tok) starts: %r"
                      % (arm, len(prompt),
                         len(tok(prompt, add_special_tokens=False)["input_ids"]), prompt[:150]),
                      flush=True)
            except Exception:
                pass
        txt, m = complete(prompt, max_new)
        out_text = txt
        call, err = parse(txt)
        obs = ""
        if err:
            _fail += 1
            obs = "could not parse your reply (%s). Reply with exactly one json object." % err
            if _fail >= 3:
                obs += '\nReply with EXACTLY: {"tool": "write_file", "content": "..."}'
        else:
            _fail = 0
            if call["tool"] in ("write_file", "append"):
                content = str(call.get("content", ""))
                if call["tool"] == "write_file":
                    open(solution, "w").write(content)
                else:
                    # `append` keeps what is already on disk, so the model does not have to
                    # re-emit a long module. It is equivalent to write_file only when the model
                    # is purely adding, which the prompt states.
                    with open(solution, "a") as f:
                        f.write(("\n" if _os.path.exists(solution) else "") + content)
                # ★ RECORD WHICH TURN DEFINED EACH FUNCTION, so a later failure can be traced to
                # the point it was lost rather than only reported as absent. Read the WHOLE file:
                # on an append the file is the source of truth, not the reply.
                full = open(solution).read()
                for fi, (fspec, _t) in enumerate(use):
                    fx = "fn_%02d_" % fi
                    if "def %s" % fx in full:
                        per_feature_written.setdefault(fi, i)
                if "def fn_99_" in full:
                    per_feature_written.setdefault(99, i)
                obs = "%s solution.py now (%d chars)" % (
                    "wrote" if call["tool"] == "write_file" else "appended to", len(full))
            elif call["tool"] == "read_file":
                obs = open(solution).read() if _os.path.exists(solution) else "(empty)"
            else:
                obs = "acknowledged"
        turns_log.append(dict(turn=i, spec=spec[:70], tool=(call or {}).get("tool"),
                              parse_error=err, raw_reply=txt[:400],
                              ttft_ms=m["ttft_ms"], kv_bytes=m["kv_bytes"],
                              prompt_tokens=m["prompt_tokens"], wall_ms=m["wall_ms"]))
        # ★★ PROGRESS MUST BE OBSERVABLE FROM OUTSIDE THE CONTAINER. The script printed nothing
        # until the very end, so a 1.5-hour run looked identical to a wedged one and the only way
        # to answer "is it progressing?" was to guess. /cache is a mounted Modal volume, so a line
        # written here is readable from this box with `modal volume get` WHILE the run is live.
        # A multi-hour job with no progress signal is a job nobody can supervise.
        try:
            with open("/cache/prog_%s.txt" % arm, "w") as _pf:
                _pf.write("turn %d/%d  kept_blocks=%d  prompt_tokens=%d  ttft_ms=%.0f\n"
                          % (i + 1, len(specs), len(keep), m["prompt_tokens"], m["ttft_ms"]))
            if (i + 1) % 20 == 0:
                hf_cache.commit()                      # make it visible without waiting for exit
        except Exception:
            pass                                       # progress reporting must never fail a run
        print("[%s] turn %d/%d kept=%d prompt=%d tok ttft=%.0fms"
              % (arm, i + 1, len(specs), len(keep), m["prompt_tokens"], m["ttft_ms"]),
              flush=True)
        # ★★★ THE TRANSCRIPT MUST RECORD WHAT THE MODEL ACTUALLY SAID, AND MUST DELIVER THE
        # CORRECTION. Both were wrong, and together they turned 4 failures into 184.
        #
        # MEASURED, on the first 200-turn run: turn 17's reply overran max_new and was truncated, so
        # it failed to parse. Turns 17-20 failed the same way. Then from TURN 21 TO TURN 199 -- 179
        # consecutive turns -- the model replied with the single word `error`, and the final file
        # was 1732 chars. Two causes, both the harness lying to the model:
        #
        #   1. On failure this recorded the assistant turn as the literal string "error". The model
        #      reads its own history, saw `assistant: error` repeated, and concluded that `error`
        #      was the expected reply. It was imitating the transcript.
        #   2. `obs` -- the correction ("could not parse your reply ... reply with exactly one json
        #      object") -- was COMPUTED AND NEVER SENT. `grep -n obs` showed it assigned five times
        #      and read zero. So the model was never told it had failed, let alone what shape was
        #      wanted, and had no way to recover.
        #
        # ⇒ Record the real reply (bounded), then send the correction as a user turn. A harness that
        # silently substitutes its own text for the model's is not measuring the model.
        if err:
            transcript.append(("user", instr))
            transcript.append(("assistant", str(txt)[:1200]))
            transcript.append(("user", obs))
        else:
            transcript.append(("user", instr))
            transcript.append(("assistant", str(call)[:4000]))
        _save_ckpt(i)                   # ★ persist the turn so a dead run is resumable, not lost
    # ── GRADE: one test per feature, plus the interaction, run in a fresh interpreter ─────────
    # ★ GRADE ONLY WHAT WAS ATTEMPTED. A run stopped early by the time budget has NOT attempted the
    # remaining features, so grading all 200 would report never-tried work as failures and make a
    # partial run look like a catastrophic result. `n_features_done` is the number of FEATURE turns
    # completed; the interaction ran only if every feature turn did.
    n_features_done = (len(use) if _stopped_early is None
                       else max(0, min(_stopped_early, len(use))))
    ran_interaction = _stopped_early is None
    lines = ["import sys", "sys.path.insert(0, %r)" % WORK, "import solution as m", "RESULTS = {}"]
    for fi, (_s, tbody) in enumerate(use[:n_features_done]):
        lines.append("try:")
        lines.append("    %s" % tbody)
        lines.append("    RESULTS['feature_%02d'] = True" % fi)
        lines.append("except Exception:")
        lines.append("    RESULTS['feature_%02d'] = False" % fi)
    if ran_interaction:
        lines.append("try:")
        lines.append("    %s" % INTERACTION_TEST)
        lines.append("    RESULTS['interaction_99'] = True")
        lines.append("except Exception:")
        lines.append("    RESULTS['interaction_99'] = False")
    # ★ THE CONTRACT CHECK, graded separately from the feature tests. A model can pass every
    # feature and still fail this, because the contract was stated once at turn 1 and never
    # repeated -- so this isolates RETENTION OF AN EARLY INSTRUCTION from coding ability.
    lines.append("try:")
    lines.append("    names = set(m.REGISTRY) if hasattr(m, 'REGISTRY') else set()")
    lines.append("    N = %d" % n_features_done)
    lines.append("    ok = hasattr(m, 'REGISTRY') and all(")
    lines.append("        any(str(n).startswith('fn_%02d_' % i) for n in names) "
                 "for i in range(N))")
    lines.append("    RESULTS['contract'] = bool(ok and len(names) >= N)")
    lines.append("except Exception:")
    lines.append("    RESULTS['contract'] = False")
    # ★★ THE NONCE CHECK -- the one that CANNOT be re-derived from recent turns. The value is stated
    # once in turn 1 and required only in the final append, so it is recoverable exactly when turn 1
    # survived. Purely additive: if a model wrote the stamp early it is present for BOTH arms and
    # this check simply does not discriminate; it can never make a run look WORSE than it is.
    lines.append("try:")
    lines.append("    _src = open(%r).read()" % solution)
    lines.append("    _n = _src.count(%r)" % CONTRACT_NONCE)
    lines.append("    RESULTS['nonce'] = bool(_n == 1)")
    lines.append("except Exception:")
    lines.append("    RESULTS['nonce'] = False")
    lines.append("import json; print(json.dumps(RESULTS))")
    grader = _os.path.join(WORK, "grade.py")
    open(grader, "w").write("\n".join(lines))
    r = sp.run(["python3", grader], capture_output=True, text=True, timeout=300)
    try:
        results = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        # ★★ A GRADER THAT CRASHES MUST NOT LOOK LIKE A MODEL THAT FAILED EVERYTHING.
        # MEASURED: when solution.py was never written, `import solution` raised
        # ModuleNotFoundError, the grader printed a traceback instead of JSON, this branch
        # swallowed it, and the run reported `PASSED 0/0` with an EMPTY per_feature dict --
        # identical to "ran fine, nothing passed". The run's stdout said `PASSED` and the
        # launcher greps for that word, so a completely broken harness would have been read
        # as a finished experiment. Surface the reason and mark the run as harness-failed.
        results = {}
        _grader_error = (r.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        print("[%s] GRADER FAILED -- results are NOT a model measurement: %s"
              % (arm, _grader_error[0]), flush=True)
        # A missing solution file is the common cause and deserves to be named explicitly.
        if not _os.path.exists(solution):
            print("[%s] cause: solution.py was never written (no valid tool call was ever parsed)"
                  % arm, flush=True)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    ttfts = [t["ttft_ms"] for t in turns_log] or [0]
    kvs = [t["kv_bytes"] for t in turns_log] or [0]
    pts = [t["prompt_tokens"] for t in turns_log] or [0]
    _harness_failed = not results
    return dict(model=model, arm=arm, harness_failed=_harness_failed, features=len(use),
                features_completed=n_features_done,
                stopped_early=_stopped_early,
                seconds=round(time.time() - _t_start, 1),
                gpu=GPU,
                passed=passed, total=total,
                per_feature=results, written=per_feature_written,
                ttft_first=ttfts[0], ttft_last=ttfts[-1],
                ttft_mean=sum(ttfts) / len(ttfts),
                kv_peak=max(kvs), kv_last=kvs[-1],
                prompt_first=pts[0], prompt_last=pts[-1],
                solution_chars=len(open(solution).read()) if _os.path.exists(solution) else 0,
                turns=turns_log,
                failed=[k for k, v in results.items() if not v])


@app.local_entrypoint()
def main(model: str = "qwen2.5-coder-32b", arm: str = "linear", features: int = 16,
         keep_turns: int = 4):
    import time
    r = run_inc.remote(model, arm, features, keep_turns)
    p = "/tmp/opencode/ccai/benchmarks/results/incremental_%s_%s_%s.json" % (
        model.replace("/", "-"), arm, time.strftime("%Y%m%d-%H%M%S"))
    json.dump(r, open(p, "w"), indent=2)
    print("\n%s  arm=%s  features=%d" % (r["model"], r["arm"], r["features"]))
    if r.get("stopped_early") is not None:
        print("  ⚠️ PARTIAL RUN: stopped at turn %s after %.0f s (time budget) -- "
              "grading only the %d features attempted"
              % (r["stopped_early"], r.get("seconds", 0), r.get("features_completed", 0)))
    print("  PASSED %d/%d  (%.0f%%)" % (r["passed"], r["total"],
                                        100.0 * r["passed"] / max(1, r["total"])))
    print("  failed: %s" % (", ".join(r["failed"]) if r["failed"] else "none"))
    print("  TTFT  %6.0f -> %6.0f ms     prompt %5d -> %5d tok" % (
        r["ttft_first"], r["ttft_last"], r["prompt_first"], r["prompt_last"]))
    print("  KV    peak %6.1f MB   solution.py %d chars" % (r["kv_peak"] / 1e6, r["solution_chars"]))
    print("wrote %s" % p)
