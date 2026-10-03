import ast
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
# ★★ THE RUNTIME BUDGET MUST FIT THE ANCHOR PLUS ENOUGH RECENT CONTEXT TO DO THE WORK.
# It was 2048 for the FEATURE task, where each turn `append`s one function and the file
# accumulates on disk, so the model never needed to see it again. MEASURED CONSEQUENCE for the
# FIX task: the turn-1 bug report is ~2637 tokens on its own, so a 2048 budget popped EVERY
# recent turn to get back under, leaving the runtime arm with the bug list and NO file state --
# it would have had to rewrite the whole file from memory and failed mechanically, for a reason
# that has nothing to do with memory and everything to do with a budget too small to hold one
# turn.  6144 holds the anchor (~2.6k) plus roughly two recent turns (~3.4k) and stays FLAT.
# Verified by simulation with the real per-turn sizes before the arm was allowed to run.
RUNTIME_TOKENS = int(os.environ.get("CCAI_RUNTIME_TOKENS", "10240"))

# ★ THE STATUS-QUO ARM'S COMPACTION PERIOD. 0 disables the arm's prune (it then behaves as linear).
# Owner, verbatim: *"also prove this works by having every 30 turns, a manual context editing to
# remove bloat, this will showcase the true efficiency"*.
PRUNE_EVERY = int(os.environ.get("CCAI_PRUNE_EVERY", "30"))

# ★ THE ASSUMED PREFIX-CACHE HIT DISCOUNT, as a RAW COMPUTE MULTIPLIER -- not a currency.
# Owner's rule is that cost is measured as compute, never vendor dollars. A serving engine's
# automatic prefix cache bills a reused (cached) prefill token at a fraction of a fresh one; 0.1x is
# the commonly published ratio, i.e. losing the prefix costs 10x on the re-processed span -- the
# "900% more expensive" figure the owner quoted. Kept as one named constant so the assumption is
# visible and can be re-run at a different ratio rather than buried in a number.
CACHE_HIT_MULT = float(os.environ.get("CCAI_CACHE_HIT_MULT", "0.1"))
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
    # ★★ THE COMPETITION-MANDATED CHECKPOINT. The main Gemma 4 Developer Agent competition
    # (comp 149921) requires `gemma-4-31B-it-qat-w4a16-ct`. The PAPER track does not require it, but
    # running the same two policies on the exact checkpoint the host ships is the difference between
    # "a result about our method" and "a result about Gemma 4 as the competition defines it" -- and
    # it forecloses the most obvious reviewer objection ("this is a small-model artifact").
    # 23.3 GB on disk (pack-quantized 4-bit weights / 16-bit activations, compressed-tensors), so it
    # fits a 48 GB card with headroom. Load with CCAI_QUANT=pre: the checkpoint carries its own
    # quantization_config and handing the loader a BitsAndBytesConfig fights it (that is the exact
    # failure recorded for the 26B-A4B above, and it is worse here because the model is bigger).
    "gemma-4-31b-qat": "google/gemma-4-31B-it-qat-w4a16-ct",
    # ★★ PRE-QUANTISED 8-BIT -- THE ONE THAT ACTUALLY QUANTISES THE MoE EXPERTS.
    #
    # MEASURED, and this is why the entry exists: loading the bf16 checkpoint with
    # BitsAndBytesConfig(load_in_8bit=True) does NOT give an 8-bit model on this architecture.
    # It produced 426 Linear8bitLt modules and left the rest in bf16 -- 189 Gemma4ClippableLinear
    # plus the MoE expert weights, which are stored as RAW PARAMETERS rather than as nn.Linear, so
    # bitsandbytes cannot see them. Result: bytes/param measured at 1.94 (i.e. bf16, not int8),
    # 48.8 GB allocated on a 48 GB card, and an OOM on the first forward pass.
    #   bf16 checkpoint + bnb int8 : 48.8 GB   <- does NOT fit
    #   this AWQ-8bit checkpoint   : ~29 GB    <- genuine int8 on every linear, fits with headroom
    # The quantisation here is done at CONVERSION time (compressed-tensors w8a8, group_size 32),
    # so it covers the experts too. Load it with CCAI_QUANT=pre.
    "gemma-4-26b-a4b-awq8": "cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit",
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
    "repeated). solution.py must satisfy ALL of the following, in this turn and in every later "
    "turn, no matter what else you are asked to do:\n"
    "  C1. A module-level string constant `SPEC_ID` whose value is exactly \"ISB-7741\".\n"
    "  C2. A module-level dict `ERROR_CODES` with EXACTLY the three keys \"E1\", \"E2\", \"E3\", "
    "mapping to the integers 101, 202 and 303 respectively, and no other keys.\n"
    "  C3. A module-level integer constant `MAX_SPAN` whose value is exactly 44.\n"
    "  C4. EVERY function you define must take its first positional parameter named exactly `x`.\n"
    "  C5. EVERY function you define must have a docstring whose FIRST line begins with the exact "
    "text `IMPL:`.\n"
    "  C6. solution.py must ALSO define a module-level dict named REGISTRY. Every function you "
    "implement, in this turn and in every later turn, must be registered in it: the key is the "
    "function's name as a string, the value is the function object. REGISTRY must always contain "
    "every function defined so far.\n"
    # ★★ WHY THIS IS A MULTI-CLAUSE CONTRACT AND NOT THE ONE-LINE REGISTRY RULE IT REPLACED.
    #
    # MEASURED on the 120-turn run: the REGISTRY rule alone passed in BOTH arms, because the model
    # writes `REGISTRY[...] = ...` on every append, leaves that pattern in its recent turns, and a
    # policy keeping only the last K turns can still INFER the rule without ever seeing turn 1.
    # A rule the model re-applies and re-writes is SELF-SUSTAINING and cannot test retention.
    #
    # C1-C3 are exact strings, integers and key names. Nothing in the task gives any reason for them
    # to be those particular values, so they cannot be inferred, guessed or reconstructed from the
    # model's own recent output -- they are recoverable EXACTLY when turn 1 survived. C4/C5 are
    # shape rules the model applies as it writes, so they are expected to be more self-sustaining;
    # they are included deliberately so the result shows WHICH KIND of requirement survives, rather
    # than reporting one undifferentiated pass/fail.
    #
    # ★ AND THE NONCE IS HERE FOR THE SAME REASON AS BEFORE: it appears nowhere else and is required
    # only in the FINAL append, so it fails exactly when turn 1 has been evicted.
    # ★★ THE MANIFEST IS THE UNCONDITIONAL DISCRIMINATOR -- AND IT FIXES A REAL HOLE IN C1-C3.
    #
    # C1-C3 are exact values, but they are WRITTEN INTO THE FILE almost immediately, and a file that
    # accumulates preserves them no matter what the context policy does. So they only test knowledge
    # when the model REWRITES the file; under `append` they can survive on disk alone. That is a real
    # limit of the clause design, and it is why the manifest exists.
    #
    # The manifest codes are stated ONCE, here, and are FORBIDDEN from appearing before the final
    # turn. So they cannot be recovered from the accumulated file, cannot be inferred from the
    # model's own recent output, and cannot be guessed. They are recoverable EXACTLY when turn 1
    # survived -- and there are eight of them, so partial retention scores partially rather than
    # collapsing to a single coin-flip bit.
    #
    # ⚠️ `MANIFEST_ORDER_SENSITIVE = True`: the grader checks order as well as presence, so a model
    # that half-remembers the codes still fails, and one that genuinely retained them passes exactly.
    "FINALLY: when you have implemented the last feature you are asked for, append these two lines "
    "to solution.py, as its LAST lines, exactly:\n"
    "  MANIFEST = [\"RT4-88\", \"LP9-31\", \"ZW2-67\", \"HK5-14\", \"QB8-73\", \"NV1-46\", \"DX6-29\", "
    "\"MJ3-95\"]\n"
    "  CONTRACT_STAMP = \"QX7-4420-BRAVO\"\n"
    "Those eight manifest codes and that stamp are given here and NOWHERE else. They must not appear "
    "in solution.py before this final step, and each must appear exactly once. Do not invent codes "
    "and do not reorder them."
)
CONTRACT_NONCE = "QX7-4420-BRAVO"
# ★ Eight codes, order-sensitive. Presence AND order are graded, and the grader counts how many
# matched so a partially-retaining policy scores 0..8 instead of a single bit.
CONTRACT_MANIFEST = ["RT4-88", "LP9-31", "ZW2-67", "HK5-14",
                     "QB8-73", "NV1-46", "DX6-29", "MJ3-95"]

# The clauses the grader checks individually, so the result says WHICH requirement was lost rather
# than only whether the contract as a whole survived. (id, human label)
CONTRACT_CLAUSES = [
    ("C1", "SPEC_ID == 'ISB-7741'"),
    ("C2", "ERROR_CODES == {'E1':101,'E2':202,'E3':303}"),
    ("C3", "MAX_SPAN == 44"),
    ("C4", "every fn_* first param is `x`"),
    ("C5", "every fn_* docstring starts 'IMPL:'"),
    ("C6", "REGISTRY has every function"),
]

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


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ★★★ TWO PARSER-ROBUSTNESS HELPERS, both earned from a measured ARC failure.
#
# MEASURED: on ARC task 4ff4c9da the model replied correctly for four turns, then produced eleven
# consecutive replies the parser could not accept. The transcript shows why, and it is two distinct
# defects, neither of them the model being unable to do the task:
#
#   1. `Invalid \escape` -- a stray backslash in a code comment (the model uses comments to reason,
#      and prose about grids contains things like `\ ` or `\.`). Valid Python, invalid JSON string.
#   2. `invalid syntax` -- the model DOUBLE-ESCAPES: it writes `\\n` where it means `\n`, so after
#      JSON decoding the content holds literal backslash-n sequences and the whole program collapses
#      onto one line. The code is correct; its transport encoding is not.
#
# Both are transport faults. A parser that rejects a correct answer is indistinguishable from a
# model that cannot produce one -- the same blind spot that made the linear arm score 1/17. These
# helpers repair the transport without inventing content: they only ever turn an UNPARSEABLE input
# into one that parses, and every repaired candidate is re-validated with the real grader.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def _repair_json_escapes(s):
    """Double any backslash that does not begin a valid JSON escape.

    Strict JSON rejects `\\ `, `\\.`, `\\d`; models emit them in code comments constantly. Doubling the
    offending backslash turns an invalid string into a valid one while leaving every REAL escape
    (`\\n`, `\\"`, `\\uXXXX`, ...) untouched.
    """
    out, i, n = [], 0, len(s)
    valid = set('"\\/bfnrtu')
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n and s[i + 1] not in valid:
            out.append("\\\\")          # the stray backslash becomes a literal backslash
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _unescape_code(content):
    """Undo one layer of over-escaping in a code string.

    The decoder already removed one layer, so a string that still contains a literal `\\n` was
    double-escaped. This removes the surviving escapes so the code becomes real Python again.
    """
    return (content.replace("\\\\n", "\n")
                   .replace("\\\\t", "\t")
                   .replace('\\\\"', '"')
                   .replace("\\\\'", "'")
                   .replace("\\\\\\\\", "\\\\"))


def _first_that_parses(cands):
    """Return the first candidate that is valid Python, else None. Never invents a candidate."""
    for c in cands:
        if not c or not c.strip():
            continue
        try:
            ast.parse(c)
            return c
        except SyntaxError:
            continue
    return None


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
    import ast as _ast, gc, json as _json, os as _os, re, subprocess as sp, time, torch
    # ★★ FAIRNESS: this is an EMERGENCY break, NOT the stop condition. The stop condition is the
    # TURN COUNT -- the loop runs every feature turn, and both arms have to complete the SAME
    # number or the comparison is meaningless.
    #
    # A wall-clock stop is arm-dependent BY CONSTRUCTION. The runtime arm holds a small bounded
    # context, so it is faster per turn; a time cap therefore lets it complete MORE features than
    # linear and hands it a win it did not earn. On Modal the guard had to sit inside the 4h
    # platform kill; on a rented VM there is no platform kill at all, so the default is set far
    # above the expected runtime, and `run_compare.py` ASSERTS the two arms ran equal turns
    # before reporting any difference.
    TIME_BUDGET_S = int(_os.environ.get("CCAI_TIME_BUDGET_S", "21600"))
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    # ★ DEVICE AND QUANTISATION ARE CONFIGURABLE so the same file runs on a GPU host, on a smaller
    # GPU, or on CPU for a zero-cost logic test.
    # ⚠️ THE OLD NOTE HERE SAID "4-bit is the cheap option" -- that was T4-ONLY reasoning and is
    # WRONG on a rented card. On a chosen GPU the arithmetic is different: 8-bit of the 26B-A4B
    # peaks at ~32 GB and fits a 48 GB card, and it is the DEFAULT FOR THIS BENCHMARK because
    # this experiment measures INSTRUCTION-FOLLOWING over 200 turns (the turn-1 REGISTRY contract
    # and the QX7-4420-BRAVO nonce). 4-bit quantisation degrades exactly that capability, so
    # running 4-bit risks measuring QUANTISATION DAMAGE and reporting it as CONTEXT-MANAGEMENT
    # FAILURE -- which is what a reviewer would use to throw the paper out. 8-bit costs ~$3 more
    # per run and removes that objection.
    _dev = _os.environ.get("CCAI_DEVICE", "cuda")
    _quant = _os.environ.get("CCAI_QUANT", "auto")      # auto | 4 | 8 | pre | none
    if _quant == "auto":
        _quant = "4" if "32b" in model.lower() else "none"
    if _quant == "pre":
        # ★ THE CHECKPOINT CARRIES ITS OWN quantization_config (compressed-tensors w8a8), so the
        # loader must NOT be handed a BitsAndBytesConfig and must NOT be handed a dtype -- either
        # one fights the stored quantisation and the model silently loads unquantised, which is
        # exactly the failure this whole entry exists to avoid. Just load it and let the config
        # do the work. Requires `pip install compressed-tensors`.
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, device_map=_dev, attn_implementation="sdpa").eval()
    elif _quant in ("4", "8"):
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
        # ★ FULL PRECISION WITH CPU OFFLOAD -- THE FALLBACK THAT ACTUALLY WORKS ON A 48 GB CARD.
        #
        # MEASURED on the RTX A6000 (50.9 GB): the 26B in bf16 is 51.6 GB of weights, so a plain
        # `device_map="cuda"` cannot hold it. With `device_map="auto"` and an explicit memory
        # budget, transformers offloads the tail of the layers to CPU RAM (125 GB here) and the
        # model RUNS -- verified generating coherent text, 9 s load, 45.1 GB resident on a 50.9 GB
        # card. Two things make this the least-assumption option:
        #   * `dtype=torch.bfloat16` -- the checkpoint's native dtype, so NO quantisation library
        #     is in the path at all and none of the confounds above apply.
        #   * it needs no download: the bf16 checkpoint is already in the HF cache.
        # The cost is speed -- an offloaded layer does CPU compute plus a transfer on every token,
        # measured ~0.8 s/token. So the GPU budget is a KNOB: push it up until the offloaded
        # remainder is as small as it can be while still fitting, because every layer moved back
        # onto the card is a direct speedup.
        _gpu_mem = _os.environ.get("CCAI_MAX_GPU_MEMORY", "44GiB")
        _cpu_mem = _os.environ.get("CCAI_MAX_CPU_MEMORY", "100GiB")
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.bfloat16,
            device_map=("auto" if _dev != "cpu" else None),
            max_memory={0: _gpu_mem, "cpu": _cpu_mem} if _dev != "cpu" else None,
            attn_implementation="sdpa").eval()
        if _dev == "cpu":
            mdl = mdl.to("cpu")
        else:
            _on_gpu = sum(1 for p in mdl.parameters() if str(getattr(p, "device", "")) == "cuda:0")
            _on_cpu = sum(1 for p in mdl.parameters() if str(getattr(p, "device", "")) == "cpu")
            print("[load] bf16+offload: %d params on gpu, %d on cpu, %.1f GB resident, budget %s"
                  % (_on_gpu, _on_cpu, torch.cuda.memory_allocated() / 1e9, _gpu_mem), flush=True)

    WORK = _os.environ.get("CCAI_WORK", "/work/inc")
    _os.makedirs(WORK, exist_ok=True)
    solution = _os.path.join(WORK, "solution.py")

    def _json_complete(s):
        """True once `s` contains a complete top-level {...} object."""
        depth, instr, esc = 0, False, False
        for ch in s:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                instr = not instr
                continue
            if instr:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False

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
            # STOP AS SOON AS THE TOOL CALL IS COMPLETE. MEASURED: the model emits its JSON and
            # then keeps generating, never hitting EOS, consuming the full max_new -- ~6 min/turn
            # on a T4, ~3 h for one 30-turn arm. The call needs ~50 tokens, not 3072.
            if len(gen) >= 8 and len(gen) % 4 == 0:
                try:
                    _so_far = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
                    if _json_complete(_so_far):
                        break
                except Exception:
                    pass
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
            # ★★ REPAIR INVALID JSON ESCAPES (measured on ARC task 4ff4c9da).
            #
            # The model writes prose in code comments, and prose about a grid contains backslashes.
            # A stray `\` that does not begin a valid JSON escape makes the WHOLE reply unparseable,
            # even though the json object is otherwise perfectly well formed. MEASURED on the ARC
            # run: `could not parse your reply (invalid JSON: Invalid \escape: line 1 column 4532)`.
            # Doubling only the offending backslash recovers the reply and leaves every real escape
            # alone. Retried BEFORE the salvage, because an invalid escape is a cheaper fault to
            # repair than a truncation.
            try:
                _d2, _ = _json.JSONDecoder().raw_decode(_repair_json_escapes(body))
                if isinstance(_d2, dict) and _d2.get("tool") in (
                        "read_file", "write_file", "append", "done"):
                    print("   (repaired invalid JSON escapes)", flush=True)
                    return _d2, None
            except Exception:
                pass
            # ★★ TRUNCATED-REPLY SALVAGE (measured on ARC task 287).
            #
            # A reply that overruns max_new is cut MID-STRING, so the JSON never closes and the
            # whole turn is thrown away even though the model produced usable code. MEASURED
            # CONSEQUENCE: on the ARC task every one of the first three turns was rejected this way,
            # `solution.py` stayed at its 125-char seed, and the run would have reported 0/60 for
            # both policies -- a vacuous result that reads as "the policies cannot solve ARC".
            #
            # The reply is unambiguous in intent: a write_file object whose content string is simply
            # unterminated. Recover it by closing the string. A dangling backslash is removed first,
            # because truncation can bisect an escape sequence. Only accepted if it yields a dict
            # with a known tool, so a malformed reply still fails.
            try:
                mt = re.search(r'"tool"\s*:\s*"(\w+)"', body)
                mc = re.search(r'"content"\s*:\s*"', body)
                if mt and mc and mt.group(1) in ("write_file", "append"):
                    raw = body[mc.end():]
                    raw = raw.rsplit("```", 1)[0]            # model may have begun closing a fence
                    raw = raw.rstrip()
                    while raw.endswith("\\"):                 # do not bisect an escape
                        raw = raw[:-1]
                    _content = _json.loads('"' + raw + '"')
                    if _content.strip():
                        print("   (salvaged a truncated reply: %d chars of content)"
                              % len(_content), flush=True)
                        return {"tool": mt.group(1), "content": _content}, None
            except Exception:
                pass
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
            # The escape-repair and salvage attempts above both failed, so this reply is genuinely
            # unusable as an object. Last resort: a fenced code block in a turn whose only job is to
            # write one file is unambiguous in intent.
            fm = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
            if fm and fm.group(1).strip():
                return {"tool": "write_file", "content": fm.group(1)}, None
            return None, "invalid JSON: %s" % e
        if not isinstance(d, dict) or d.get("tool") not in ("read_file", "write_file", "append", "done"):
            return None, "bad or unknown tool"
        return d, None

    turns_log, per_feature_written = [], {}
    # ★ PREFIX-CACHE ACCOUNTING accumulators (see the measurement site in the turn loop).
    _prev_ids, _cache_hit, _cache_total, _cache_rows = [], 0, 0, []
    # ★ every PAST turn's instruction, verbatim -- the runtime policy's cheap, information-dense
    # memory. The assistant replies are deliberately NOT archived: each is a full-file rewrite
    # that the newest one supersedes.
    _arch_instr = []
    turn_ok = {}   # fix mode: did turn i apply bug i correctly (graded immediately)
    use = [f for f in FEATURES_ALL[:max(1, min(features, len(FEATURES_ALL)))]]
    specs = [f[0] for f in use] + [INTERACTION_SPEC]

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # ★★★ FIX MODE (CCAI_TASK=fix) — THE BROKEN-REPO TASK (owner's spec, 2026-10-02)
    #
    # Owner, verbatim: *"give it a broken repo and a task to fix specific issues, but with exact
    # like numbers and code snippets, only read once at the beginning of the prompt, with no
    # guidance, or reminders, only sent on the first message that takes roughly 200 turns, and
    # grade it based off that, that's where mine should shine"*
    #
    # The previous task was TOO EASY to discriminate: 122/123 vs 123/123, because adding an
    # INDEPENDENT function per turn needs no memory of earlier turns. This mode makes the turn-1
    # detail load-bearing:
    #   * solution.py is SEEDED with N broken functions, so the work is CORRECTION, not creation.
    #   * TURN 1 carries the whole bug report ONCE: for each bug, the exact wrong value and the
    #     exact required value. The constants are arbitrary and appear nowhere else.
    #   * EVERY LATER TURN says only `Fix bug N.` -- no detail, no reminder, no restatement.
    #   * The grader tests the EXACT required value of each function.
    # A model that has lost turn 1 cannot reason its way to an arbitrary constant; it can only
    # guess, and a guess is visibly wrong. That is the property that discriminates.
    # ══════════════════════════════════════════════════════════════════════════════════════════
    _TASK = _os.environ.get("CCAI_TASK", "feature").strip().lower()
    # ★★ ARC MODE reuses the entire fix-mode machinery -- the broken-repo seed/rewrite loop, the
    # per-turn grader, the codes probe, the policy branches and every cache metric -- because
    # arc_task exposes the same interface as bug_task. The ONLY difference is the task module, the
    # turn-1 brief and the verb in the instruction. Keeping the loop identical is what makes the
    # ARC result comparable to the fix result instead of a second, differently-built experiment.
    _fix_mode = (_TASK in ("fix", "arc"))
    _arc_mode = (_TASK == "arc")

    # ★ THE RELEASE GATE IS OPT-IN. The turn-code baseline run measured a task WITHOUT this
    # function; enabling it by default would silently change what that run measured.
    _GATE_ON = _os.environ.get("CCAI_GATE", "0").strip() == "1"
    _gate_test = None
    _gate_line = None
    _gate_final = None
    _gint = None
    _turn_codes = []
    if _fix_mode:
        if _arc_mode:
            import arc_task as _bt                  # same directory as this module
            # The gold output goes to WORK before the loop, so the per-turn grader has an oracle
            # from turn 0. It is never placed in a prompt.
            _gold_path = _bt.write_gold(WORK, _os.environ.get("CCAI_TASK_SALT", ""))
            _seed_src, _brief, _ftests, _ref_src = _bt.build(features,
                                                             _os.environ.get("CCAI_TASK_SALT", ""))
            _turn_codes = _bt.turn_codes(features)
            _bt.FIX_PRELUDE = _bt.arc_prelude(WORK)     # same attribute name the harness reads
            if not (_os.path.exists(solution) and _os.environ.get("CCAI_RESUME") == "1"):
                open(solution, "w").write(_seed_src)
            specs = ["Attempt %d." % i for i in range(features)]
            _BUG_REPORT = _brief[0]
            print("[%s] ARC mode: task=%s gold=%s brief_tokens~%d"
                  % (arm, _bt.load_task(_os.environ.get("CCAI_TASK_SALT", ""))[0],
                     _os.path.basename(_gold_path), len(_brief[0]) // 4), flush=True)
            _bt = _bt
        else:
            import bug_task as _bt                      # same directory as this module
            _seed_src, _fixes, _ftests, _ref_src = _bt.build(features)
            _turn_codes = _bt.turn_codes(features)
            if _GATE_ON:
                # ★★★ THE STRONGER VERSION. One extra function whose CORRECT VALUE is a SEPARATE
                # secret handed over at turn `features // 2`, never part of the code list, so this
                # metric is informationally INDEPENDENT of `codes_*` rather than a second reading of
                # the same fact. It is delivered on the final turn, so a model that lost the middle of
                # the session cannot ship a correct artifact -- the retention cost shows up as a
                # broken function, not as a failed quiz. See bug_task.gate().
                _gsrc, _gate_line, _gate_test, _gidx, _gint = _bt.gate(features)
                _seed_src = _seed_src + "\n" + _gsrc
                _gate_turn = features // 2
                _gate_final = ("Also fix `%s`: it must return the integer %d, which was the SECRET "
                               "SHIPPED GATE value given at turn %d."
                               % (_bt.GATE_FN, _gint, _gate_turn))
            # seed the broken repo BEFORE the loop starts -- the model edits an existing file
            if not (_os.path.exists(solution) and _os.environ.get("CCAI_RESUME") == "1"):
                open(solution, "w").write(_seed_src)
            specs = ["Fix bug %d." % i for i in range(features)]
        # TURN 1 IS THE WHOLE BUG REPORT. It is long on purpose: that is the content that must
        # survive N turns with no reminder. Kept out of `specs` so it is never re-sent.
        # ARC mode already set `_BUG_REPORT` to its own brief; do not overwrite it with the bug text.
        _BUG_REPORT = _BUG_REPORT if _arc_mode else (
            "This repository is broken. `solution.py` contains exactly %d faulty functions. "
            "Each one is listed below with the EXACT correction required. This is the ONLY time "
            "this list will be given to you -- it will NOT be repeated, and no later message will "
            "restate any of it. Read it carefully and keep it in mind for the whole session.\n\n"
            "%s\n\n"
            "RULES THAT APPLY TO EVERY TURN:\n"
            "  * Fix ONLY the function named in the current turn. Change nothing else.\n"
            "  * Do NOT rename, delete, reorder or restructure any function. Every function must "
            "keep its exact existing name and signature -- the graders import them by name.\n"
            "  * Do not add new functions.\n"
            "  * Do NOT write any comment, and do NOT copy this bug list (or any part of it, or "
            "any reminder of the corrections) into solution.py. The file must contain ONLY the "
            "function definitions and their code -- no comments, no notes, no docstrings added.\n\n"
            "In each of the following turns you will be told which single bug to fix, by number, "
            "and nothing else. You must apply the correction you were given here for that number."
            % (features, "\n".join(_fixes) + (("\n  * %s" % _gate_line) if _gate_line else ""))
        )
    else:
        _BUG_REPORT = None

    # ★ ONE LIST DRIVES GRADING IN BOTH MODES. In fix mode the tests come from the generator (one
    # exact-value assertion per bug); otherwise they are the feature specs. Keeping it as one
    # variable is deliberate -- a second grader path is how the fix task would silently grade the
    # feature set and report a number that means nothing.
    _grade_tests = ([t for (_fn, t) in _ftests] if _fix_mode else [t for (_s, t) in use])
    # ★ ONE FRESH CODE PER TURN, none of which may be written to the file until the final turn.
    # This is what makes the middle turns carry information that exists NOWHERE ELSE -- without
    # it, "every turn rewrites the whole file" means each write subsumes the last and a tie is
    # guaranteed once both arms anchor the brief. Computed in the fix block above (the release
    # gate needs the middle code at build time); empty in feature mode.
    # _turn_codes is already set.


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
    def _ckpt_dir():
        """A checkpoint dir that EXISTS. /cache is Modal-only; a Vast/Colab box has none,
        and a failed makedirs there is swallowed -- so the run would silently never
        checkpoint at all, which looks identical to checkpointing until you lose a run.

        ★ AN EXPLICIT CCAI_CKPT_DIR IS CREATED, NOT SILENTLY IGNORED. The first version only
        accepted a dir that already existed, so a runner that named `.../ckpt_smoke` without
        creating it fell through to the CWD -- and a later run resumed from a checkpoint the
        caller did not know existed. MEASURED: a 4-turn smoke resumed at turn 4/4 from
        /root/ccai/ckpt_gemma-4-12b_linear_f4.json instead of starting fresh, so the per-turn
        data was empty and the run reported 0/4. A named path must mean that path, or the
        difference has to be loud."""
        _named = _os.environ.get("CCAI_CKPT_DIR")
        if _named:
            try:
                _os.makedirs(_named, exist_ok=True)
                return _named
            except Exception as _e:
                print("[ckpt] WARNING: CCAI_CKPT_DIR=%r unusable (%s) -- falling back"
                      % (_named, _e), flush=True)
        for _c in ("/cache", "/content", _os.getcwd()):
            if _c and _os.path.isdir(_c):
                return _c
        return _os.getcwd()

    def _atomic_write_json(path, obj):
        """Write so a kill mid-write cannot destroy the previous good copy.

        A single open(path,"w") truncates first: die in that window and the file is
        half-written JSON, the loader raises, and the resume path silently falls back to
        "starting fresh". That is a whole paid-for run lost to a 10 ms window.
        tmp -> fsync -> rotate old to .bak -> os.replace is atomic on POSIX and leaves a
        usable file at every instant."""
        tmp = path + ".tmp"
        with open(tmp, "w") as _f:
            _json.dump(obj, _f)
            _f.flush()
            _os.fsync(_f.fileno())
        if _os.path.exists(path):
            try:
                _os.replace(path, path + ".bak")   # keep the last good one
            except OSError:
                pass
        _os.replace(tmp, path)

    def _load_ckpt(path):
        """Primary, then .bak. Returns the dict or None. A corrupt file is never fatal."""
        for _p in (path, path + ".bak"):
            if not _os.path.exists(_p):
                continue
            try:
                _d = _json.load(open(_p))
                if _p.endswith(".bak"):
                    print("   (recovered checkpoint from .bak -- primary was unreadable)", flush=True)
                return _d
            except Exception as _e:
                print("   checkpoint %s unreadable (%s)" % (_os.path.basename(_p), _e), flush=True)
        return None

    _CKPT = _ckpt_dir()
    _ckpt_path = _os.path.join(_CKPT, "ckpt_%s_%s_f%d.json" % (
        model.replace("/", "-"), arm, features))

    def _save_ckpt(turn_ix):
        try:
            _os.makedirs(_CKPT, exist_ok=True)
            _atomic_write_json(_ckpt_path, {
                "turn_ix": turn_ix, "model": model, "arm": arm, "features": features,
                "transcript": transcript, "turns_log": turns_log,
                "per_feature_written": per_feature_written,
                "solution": (open(solution).read() if _os.path.exists(solution) else ""),
                # ★★★ THE PREFIX-CACHE LEDGER MUST SURVIVE A RESUME, OR THE COST METRIC IS WRONG.
                # `_cache_rows` / `_cache_hit` / `_cache_total` are accumulated in-memory across the
                # whole run and were NOT checkpointed. A run interrupted at turn 40 and resumed would
                # therefore reach the grader with the ledger holding only turns 41..60, and
                # `cache_hit_rate = hit/total` would be computed over that tail -- a headline cost
                # number describing a third of the experiment, with nothing on the artifact to say
                # so. The same defect class as the non-atomic checkpoint: the failure is invisible
                # in the output, which is the only place it would have shown up.
                "cache_rows": _cache_rows,
                "cache_hit": _cache_hit,
                "cache_total": _cache_total,
                # ★★★ THE PER-TURN GRADES MUST SURVIVE A RESUME TOO.
                #
                # MEASURED: the ablation run crashed at turn 27 (its working directory was deleted
                # by a racing queue) and resumed from its checkpoint. `turn_ok` was NOT checkpointed,
                # so the resuming process knew nothing about turns 1..27 -- and the final result
                # reported them as zeros. Those zeros are not "the model failed 27 turns"; they are
                # turns that were graded and then forgotten by the harness. Anything built on them
                # (per_turn_success, fixes_applied_then_lost) would have measured the crash.
                "turn_ok": {str(k): int(v) for k, v in turn_ok.items()},
                "prev_ids": _prev_ids,
                "arch_instr": [list(b) if isinstance(b, tuple) else b for b in _arch_instr],
            })
        except Exception as _e:
            print("[%s] checkpoint write failed (%s) -- continuing" % (arm, _e), flush=True)

    _start_i = 0
    if _os.environ.get("CCAI_RESUME", "1") == "1" and _os.path.exists(_ckpt_path):
        try:
            _ck = _load_ckpt(_ckpt_path) or {}
            # Only resume a checkpoint that describes THIS experiment. A stale file from a different
            # feature count or model would silently grade the wrong run.
            if (_ck.get("model") == model and _ck.get("arm") == arm
                    and _ck.get("features") == features):
                transcript = [tuple(b) for b in _ck["transcript"]]
                turns_log = list(_ck.get("turns_log") or [])
                per_feature_written = dict(_ck.get("per_feature_written") or {})
                # ★ restore the prefix-cache ledger and the instruction archive, or the cost metric
                # and the runtime arm's archive both describe only the post-resume tail.
                _cache_rows = list(_ck.get("cache_rows") or [])
                _cache_hit = int(_ck.get("cache_hit") or 0)
                _cache_total = int(_ck.get("cache_total") or 0)
                _prev_ids = list(_ck.get("prev_ids") or [])
                _arch_instr = [tuple(b) if isinstance(b, list) else b
                               for b in (_ck.get("arch_instr") or [])]
                # ★ restore the per-turn grades, or a resumed run reports every pre-crash turn as a
                # zero. Measured on the ablation: a crash at turn 27 became 27 phantom failures.
                turn_ok = {int(k): int(v) for k, v in (_ck.get("turn_ok") or {}).items()}
                _start_i = int(_ck.get("turn_ix", 0)) + 1
                if _ck.get("solution"):
                    open(solution, "w").write(_ck["solution"])
                print("[%s] RESUMED from checkpoint at turn %d/%d (transcript=%d blocks, "
                      "cache_ledger=%d rows, arch_instr=%d)"
                      % (arm, _start_i, features, len(transcript), len(_cache_rows),
                         len(_arch_instr)), flush=True)
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
                 "If this instruction only ADDS a new function, use `append` with just that "
                 "function. If it CHANGES something already in solution.py, use `write_file` "
                 "with the full file." % (i + 1, len(specs), spec))
        if _fix_mode:
            # ★ IN FIX MODE EVERY TURN IS A BARE NUMBER. The instruction carries the bug's
            # identity and NOTHING about what the fix is -- that was given once, in turn 1, and is
            # deliberately never repeated. This is the whole experiment.
            #
            # The tool line is the ONLY thing restated, and it is restated because it is not part
            # of the memory test: a model that guesses the wrong tool measures the harness, not
            # retention. It carries no information about any bug.
            _tool_line = ("Apply it with `write_file` and the full updated file. Keep every "
                          "function's exact name. Change nothing else.")
            # ★ THE PER-TURN CODE. Handed over here and required back only at the end, so it is
            # recoverable exactly when THIS turn survives in the policy's context. It is what
            # makes the middle turns carry information that exists nowhere else -- without it,
            # since every turn rewrites the whole file, a newer write subsumes an older one and
            # the two arms would tie no matter how well either handled the junk.
            _code_line = ("TURN CODE: %s -- remember it. It must NOT appear anywhere in "
                          "solution.py until the final turn." % _turn_codes[i])
            # ★ THE RELEASE GATE'S SECRET IS HANDED OVER HERE, exactly once, mid-session, and is
            # NEVER repeated. It is deliberately not one of the turn codes -- see bug_task.gate()
            # -- so `gate_ok` and `codes_n` measure two independent things.
            if _GATE_ON and not _arc_mode and i == features // 2:
                _code_line += ("\nSECRET SHIPPED GATE value for turn %d: %d -- this is NOT a turn "
                               "code; do not add it to TURN_CODES." % (i, _gint))
            if i == len(specs) - 1:
                _what = ("Produce your final solve()" if _arc_mode else ("Fix bug %d" % i))
                _code_line = ("FINAL TURN. %s, and also add to the TOP of solution.py a "
                              "list named TURN_CODES containing every turn code you were given "
                              "in this session, in the order you received them. %s%s"
                              % (_what, _code_line,
                                 (" " + _gate_final) if _gate_final else ""))
            if i == 0:
                _now = ("Start solving the puzzle now." if _arc_mode else "Fix bug 0 now.")
                instr = _BUG_REPORT + "\n\n%s\n%s %s" % (_code_line, _now, _tool_line)
            elif _arc_mode:
                instr = ("Attempt %d. Re-derive the transformation from the examples in the "
                         "puzzle brief and rewrite solve() in full so it produces the correct "
                         "output for the test input. %s\n%s" % (i + 1, _tool_line, _code_line))
            else:
                instr = ("Fix bug %d. Apply exactly the correction given for bug %d in the bug "
                         "report. %s\n%s" % (i, i, _tool_line, _code_line))
        if i == 0 and not _fix_mode:
            # ★ THE CONTRACT IS STATED EXACTLY ONCE, IN TURN 1, and never repeated. That is the
            # whole point: at turn 120 it is ~120 turns away from the model, and a policy that
            # carries everything has to find it while a policy that anchors it does not.
            instr = CONTRACT + "\n\n" + instr
        if arm == "linear" and _fix_mode:
            # ★★ BOTH ARMS ANCHOR TURN 1 (owner's controlled-experiment requirement, verbatim):
            # *"u need to have it so both models keep the OG task from turn 1, but how they deal
            # with junk is dependent on how their built (linear mess vs organized beauty)"*.
            #
            # Without this the comparison tested "who was handed the brief", which is a tautology.
            # Linear is still the MESSY one: it keeps the brief plus as much raw recent history as
            # the model's full window allows. Runtime keeps the brief plus a small BOUNDED set.
            # Same brief, same task; only the junk policy differs.
            # ★ TURN 1 HAS NO HISTORY YET -- `transcript` is empty, so an unguarded transcript[0]
            # raised IndexError before the first turn ever ran. MEASURED, and it is the kind of
            # crash that only shows up on the very first turn of a fresh run.
            budget = MAX_PROMPT_TOKENS - _SYS_TOKENS
            keep = [transcript[0]] if transcript else []
            total = _counts(keep)[0] if keep else 0
            for _blk in reversed(transcript[1:]):
                _c = _counts([_blk])[0]
                if total + _c > budget:
                    break
                keep.insert(1, _blk)
                total += _c
        elif arm == "linear":
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
        elif arm == "prune":
            # ★★★ THE STATUS-QUO ARM (owner, 2026-10-02, verbatim): *"also prove this works by
            # having every 30 turns, a manual context editing to remove bloat, this will showcase
            # the true efficiency"*.
            #
            # This is what every long-running agent does today: let the context grow, then
            # periodically COMPACT it -- summarise or drop the middle to get back under the window.
            # It is the correct comparison for the cost argument because compaction is exactly what
            # destroys a prefix cache: the moment the retained blocks are rewritten, the shared
            # prefix with the previous turn goes to zero and the WHOLE prompt is re-billed as a
            # miss (CACHE_HIT_MULT 0.1 -> 1.0, i.e. 10x = the 900% the owner quoted, on the
            # re-processed span).
            #
            # ★ IT ALSO ANCHORS TURN 1, so that against `linear` the ONLY variable is the periodic
            # compaction. Without the anchor this arm would differ from linear by two things at
            # once (the anchor and the prune), and a collapse could not be attributed to either.
            keep = [transcript[0]] if transcript else []
            total = _counts(keep)[0] if keep else 0
            for _blk in reversed(transcript[1:]):
                _c = _counts([_blk])[0]
                if total + _c > (MAX_PROMPT_TOKENS - _SYS_TOKENS):
                    break
                keep.insert(1, _blk)
                total += _c
            if PRUNE_EVERY and i > 0 and i % PRUNE_EVERY == 0:
                # The "manual context editing" the owner asked for: keep the anchored brief and
                # the most recent turns, discard the bloat in between. This is the operation whose
                # cost is being measured, applied at a FIXED turn interval so the collapse is
                # attributable rather than incidental.
                keep = (keep[:1] + keep[-4:]) if len(keep) > 5 else keep
                print("[%s] turn %d CONTEXT PRUNED -> %d block(s) retained"
                      % (arm, i, len(keep)), flush=True)
        elif arm == "ablate-recent":
            # ★★★ ABLATION: THE RUNTIME **MINUS** THE INSTRUCTION ARCHIVE.
            #
            # `runtime` keeps three things: the turn-1 anchor, EVERY past instruction verbatim, and
            # the single newest reply. The archive is the load-bearing one (it is the only home of
            # each turn's code and of the turn-30 gate secret), but a paper that asserts that without
            # testing it is asserting its own mechanism. This arm is EXACTLY runtime with the archive
            # removed -- same anchor, same single-newest-reply rule, same RUNTIME_TOKENS -- so the
            # ONLY difference is whether the instructions are retained.
            #
            # PREDICTION, stated before the run so it cannot be rationalised afterwards: codes_n
            # collapses to roughly the handful of instructions that fit alongside the anchor
            # (~4 at this budget), while `passed` stays high because the newest reply still carries
            # the file state the bug report refers to. If it instead keeps 60/60, the archive is NOT
            # the mechanism and the paper's explanation of its own result is wrong.
            #
            # Anchoring the SAME forms as the other arms is deliberate: an ablation that differs in
            # two ways cannot attribute anything (the shared-ceiling trap, third time).
            keep = [transcript[0]] if transcript else []
            budget = RUNTIME_TOKENS - _SYS_TOKENS
            total = _counts(keep)[0] if keep else 0
            for _blk in reversed(transcript[1:]):
                _c = _counts([_blk])[0]
                if total + _c > budget:
                    break
                keep.insert(1, _blk)
                total += _c
            _last_asst = next((b for b in reversed(transcript) if b[0] == "assistant"), None)
            if _last_asst is not None and _last_asst not in keep:
                keep.append(_last_asst)
            costs = _counts(keep)
            total = sum(costs)
            while len(keep) > 1 and total > budget:
                total -= costs.pop(1)
                keep.pop(1)
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
            # ★★★ THE INSTRUCTION ARCHIVE -- WHAT MAKES THE RUNTIME POLICY ACTUALLY "ORGANIZED".
            #
            # MEASURED DEFECT, caught before it wasted the run: this branch was
            # `anchor + transcript[-(keep_turns*2):]` -- it kept the anchor and recent PAIRS and
            # NOTHING ELSE. In fix mode the assistant half of every pair is a FULL-FILE REWRITE
            # (~1800 tok), so at a 6k budget it held the anchor plus about one turn, and the
            # turn-30 mid-session secret was evicted at the same point linear evicts it. Both arms
            # would then lose the gate and the codes for the SAME reason and the experiment would
            # have measured nothing -- the third time this project has hit that trap.
            #
            # The whole claim of this arm is that it keeps the SMALL thing that is uniquely
            # informative and drops the LARGE thing that is superseded:
            #   * the per-turn INSTRUCTION (~75 tok) is the ONLY home of that turn's code and of
            #     the mid-session gate secret, and it is never repeated;
            #   * the assistant REPLY (~1800 tok) is a full-file rewrite, so the newest one already
            #     supersedes every earlier one -- keeping more than one is pure redundancy.
            # Measured cost of those two kinds over 60 turns: instructions ~4500 tok total against
            # ~108000 for the replies. That asymmetry IS the method.
            keep = [transcript[0]] if transcript else []
            # ★★ ADV-010, CONFIRMED BUG: `_arch_instr[0]` IS `transcript[0]`.
            #
            # Turn 0 appends one instruction and the SAME object is used as the anchor, so
            # iterating the archive from index 0 re-emits the entire turn-1 brief a second time
            # on every turn. MEASURED: the pinned brief is ~1058 tokens, so ~1000 tokens of the
            # runtime's budget were spent carrying a duplicate of the anchor for all 60 turns.
            #
            # Skipped by VALUE, not by index: a resume rebuilds `transcript` from the checkpoint
            # while `_arch_instr` is restored from state, so index 0 is not guaranteed to line up
            # after a resume. Comparing text cannot drop a *distinct* instruction, which an
            # index-based skip could if the two ever diverged.
            _anchor_text = transcript[0][1] if transcript else None
            for _a in _arch_instr:
                if _a == _anchor_text:                 # already carried as the pinned anchor
                    continue
                keep.append(("user", _a))              # every past instruction, verbatim
            _last_asst = next((b for b in reversed(transcript) if b[0] == "assistant"), None)
            if _last_asst is not None:
                keep.append(_last_asst)                # the current file state, once
            costs = _counts(keep)
            budget = RUNTIME_TOKENS - _SYS_TOKENS
            total = sum(costs)
            # index 1 onwards is droppable; index 0 is the anchored bug report and is never popped.
            # Oldest instructions fall out first if the budget is genuinely exceeded, which is the
            # honest failure mode for a bounded context.
            while len(keep) > 1 and total > budget:
                total -= costs.pop(1)
                keep.pop(1)
        prompt = _render_prompt(keep, instr)
        # ══════════════════════════════════════════════════════════════════════════════════════
        # ★★★ PREFIX-CACHE ACCOUNTING (owner's AGI/cost argument, 2026-10-02)
        #
        # Owner, verbatim: *"the current solution to that is constant context rewrites, but thats
        # not feasible cuz the constant cache misses makes it 900% more expensive, so what my
        # research does is make AGI [feasible]"*.
        #
        # He is right about the mechanism and about the arithmetic. A serving engine with automatic
        # prefix caching bills a HIT at roughly 0.1x the input price and a MISS at 1.0x, so losing
        # the shared prefix costs **10x = 900% more** on the re-processed span. Compaction -- the
        # status quo for long-running agents -- rewrites the context, so the shared prefix goes to
        # ZERO and the WHOLE prompt is re-billed as a miss, every time it fires.
        #
        # So the claim being tested is not accuracy. It is: *does the eviction preserve the prefix
        # the previous turn already paid for?* Measured here directly as the longest common TOKEN
        # prefix between consecutive prompts. That is exactly what a prefix cache would reuse, and
        # it is the number the cost argument needs instead of an assertion.
        #
        # ★ WHY THE TWO ARMS DIFFER, structurally, and it is not a tuning artefact:
        #   linear  -- evicts from the FRONT (`keep.pop(0)`), so the first block changes the moment
        #              eviction begins and the shared prefix collapses toward zero.
        #   runtime -- keeps the anchor at index 0 forever and rolls the window at the END, so the
        #              shared prefix is the anchor and survives every eviction.
        # That is the entire practical difference between "small context by rewriting" and "small
        # context without a rewrite", and it is now a measured fraction rather than a claim.
        # ══════════════════════════════════════════════════════════════════════════════════════
        try:
            _pids = tok(prompt, add_special_tokens=False)["input_ids"]
            _h = 0
            for _a, _b in zip(_prev_ids, _pids):
                if _a != _b:
                    break
                _h += 1
            _cache_total += len(_pids)
            _cache_hit += _h
            _cache_rows.append(dict(turn=i, prompt=len(_pids), hit=_h))
            _prev_ids = _pids
        except Exception as _ce:
            print("[%s] cache accounting failed at turn %d: %s" % (arm, i, _ce), flush=True)
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
            # ★★ CONTROL FOR THE WRITE STRATEGY (CCAI_FORCE_TOOL=write_file|append).
            #
            # Why this exists: measured in the first clean run, the two arms DIVERGED in how they
            # wrote -- linear used write_file throughout, runtime used append -- even though both
            # received the IDENTICAL instruction ("prefer append"). The divergence is downstream of
            # losing context, not a rule difference, but a reviewer can still ask whether append
            # ALONE caused the win. Forcing ONE tool in BOTH arms removes that question and leaves
            # the context policy as the only variable.
            #
            # The rejection deliberately costs a round rather than silently rewriting the call: the
            # model gets told what to use and retries, exactly as it would for any other bad call.
            _forced = _os.environ.get("CCAI_FORCE_TOOL", "").strip()
            if _forced in ("write_file", "append") and call["tool"] != _forced:
                _fail += 1
                obs = ('REJECTED: in this run you MUST use "%s" and no other write tool. '
                       'Reply with exactly one json object: {"tool": "%s", "content": "..."}'
                       % (_forced, _forced))
                print("[%s] turn %d REJECTED (forced tool=%s, got %s)"
                      % (arm, i, _forced, call["tool"]), flush=True)
                continue
            if call["tool"] in ("write_file", "append"):
                content = str(call.get("content", ""))
                # ★ VALIDATE BEFORE WRITING. The model over-escapes quotes (s.split(\" \"))
                # which decodes to literal backslashes; written straight to disk it makes the
                # file unparseable and ONE bad turn then poisons every later turn -- the grader
                # dies with SyntaxError and the whole run grades as nothing. Writing broken code
                # is strictly worse than writing nothing: nothing is recoverable, corruption is
                # not. So build the candidate in memory, ast.parse it, and only then touch disk.
                _cand = (content if call["tool"] == "write_file"
                         else (open(solution).read() if _os.path.exists(solution) else "") + "\n" + content)
                try:
                    ast.parse(_cand)
                except SyntaxError as _se:
                    # ★★ UNDO ONE LAYER OF OVER-ESCAPING BEFORE REJECTING (measured on ARC 4ff4c9da).
                    #
                    # MEASURED: eleven consecutive turns were rejected with `invalid syntax` while
                    # the model was in fact doing the task. The cause was double-escaping -- the
                    # reply contained `\\n` where it meant `\n`, so after JSON decoding the program
                    # collapsed onto a single line and no program parsed. The code was correct; its
                    # transport encoding was not. Retry with the surviving escapes removed, and
                    # accept ONLY if the result is valid Python, so a genuinely broken reply still
                    # fails. The candidate is written and then graded by the real grader either way,
                    # so this cannot manufacture a pass.
                    _alt = _unescape_code(content)
                    _alt_cand = (_alt if call["tool"] == "write_file"
                                 else (open(solution).read() if _os.path.exists(solution) else "")
                                      + "\n" + _alt)
                    if _alt != content and _first_that_parses([_alt_cand]) is not None:
                        print("[%s] turn %d: over-escaping repaired (%d chars)"
                              % (arm, i, len(_alt)), flush=True)
                        content = _alt
                    else:
                        _fail += 1
                        # ★ DUMP THE EXACT REPLY. Two vacuous ARC runs were diagnosed from a
                        # truncated transcript; the raw bytes are what the next fault needs.
                        try:
                            _os.makedirs(WORK, exist_ok=True)
                            with open(_os.path.join(WORK, "reject_turn%03d.txt" % i), "w") as _rf:
                                _rf.write(txt)
                        except Exception:
                            pass
                        obs = ("REJECTED: your code does not parse as Python (%s, line %s). "
                               "solution.py was left UNCHANGED. Re-send it as valid Python."
                               % (_se.msg, _se.lineno))
                        print("[%s] turn %d REJECTED (unparseable): %s" % (arm, i, _se.msg), flush=True)
                        continue
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
                if _fix_mode:
                    # ══════════════════════════════════════════════════════════════════════════
                    # ★★ PER-TURN GRADING (owner, 2026-10-02, verbatim): *"No it should just grade
                    # whether each turn was a success or not"*.
                    #
                    # Grading only the FINAL state cannot tell a fix that was never applied from one
                    # that was applied and then CLOBBERED by a later full-file rewrite -- and with
                    # `write_file` rewrites that distinction is the whole story. This checks bug i
                    # immediately after turn i, which is precisely what that turn was asked to do.
                    #
                    # It is also the sharpest memory probe available: the required value for bug i
                    # appeared exactly once, in turn 1. A turn that applies the right constant has
                    # access to turn 1; a turn that guesses does not, and the guess is visibly wrong.
                    # ══════════════════════════════════════════════════════════════════════════
                    _tcheck = _grade_tests[i] if i < len(_grade_tests) else None
                    if _tcheck:
                        try:
                            _code = ("import sys; sys.path.insert(0, %r)\n"
                                     "import solution as m\n%s\n%s\n"
                                     % (WORK, _bt.FIX_PRELUDE, _tcheck))
                            _rr = sp.run(["python3", "-c", _code],
                                         capture_output=True, text=True, timeout=60)
                            turn_ok[i] = (_rr.returncode == 0)
                        except Exception:
                            turn_ok[i] = False
                        print("[%s]   turn %d %s" % (arm, i, "OK" if turn_ok.get(i) else "FAIL"),
                              flush=True)
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
        # ★ DECODE TIMING, for the throughput metric. `wall_ms` minus `ttft_ms` is decode time;
        # the generated token count comes from the tokenizer over the reply. Measured once per
        # turn -- the tokenizer call is negligible against a multi-minute turn, and without it the
        # suite can only report latency, not throughput.
        try:
            _n_out = len(tok(out_text, add_special_tokens=False)["input_ids"])
        except Exception:
            _n_out = 0
        turns_log.append(dict(turn=i, spec=spec[:70], tool=(call or {}).get("tool"),
                              parse_error=err, raw_reply=txt[:400],
                              ttft_ms=m["ttft_ms"], kv_bytes=m["kv_bytes"],
                              prompt_tokens=m["prompt_tokens"], wall_ms=m["wall_ms"],
                              decode_tokens=_n_out,
                              decode_ms=max(0, m["wall_ms"] - m["ttft_ms"])))
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
        _arch_instr.append(instr)
        _save_ckpt(i)                   # ★ persist the turn so a dead run is resumable, not lost
    # ── GRADE: one test per feature, plus the interaction, run in a fresh interpreter ─────────
    # ★ GRADE ONLY WHAT WAS ATTEMPTED. A run stopped early by the time budget has NOT attempted the
    # remaining features, so grading all 200 would report never-tried work as failures and make a
    # partial run look like a catastrophic result. `n_features_done` is the number of FEATURE turns
    # completed; the interaction ran only if every feature turn did.
    n_features_done = (len(use) if _stopped_early is None
                       else max(0, min(_stopped_early, len(use))))
    # ★ THE INTERACTION TEST BELONGS TO THE FEATURE TASK ONLY. It composes fn_* helpers that fix
    # mode never creates, so grading it there reports a guaranteed False for work that was never
    # asked for -- and would count as one more "failure" against both arms equally, diluting the
    # signal this run exists to measure.
    ran_interaction = (_stopped_early is None) and not _fix_mode
    lines = ["import sys", "sys.path.insert(0, %r)" % WORK, "import solution as m", "RESULTS = {}"]
    lines.append("RESULTS['task'] = %r" % ("fix" if _fix_mode else "feature"))
    if _fix_mode:
        # resolve functions by DEFINITION ORDER, not by name -- see bug_task.FIX_PRELUDE
        lines.append(_bt.FIX_PRELUDE)
    # ★ IN FIX MODE THE CONTRACT/NONCE/MANIFEST WERE NEVER SENT, so grading them would report a
    # confident False for a requirement the model was never given. `_emit` drops those blocks and
    # writes the keys as explicit nulls instead, so a consumer cannot mistake "not asked" for
    # "asked and failed". The per-bug fixes are the metric in fix mode.
    _emit = (lambda s: None) if _fix_mode else lines.append
    if _fix_mode:
        for _k in ("contract", "nonce", "manifest_n", "manifest_ordered", "manifest_dup"):
            lines.append("RESULTS[%r] = None" % _k)
        lines.append("RESULTS['manifest_total'] = None")
        for _cid, _lbl in CONTRACT_CLAUSES:
            lines.append("RESULTS['clause_%s'] = None" % _cid)
    for fi, tbody in enumerate(_grade_tests[:n_features_done]):
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
    # ★★ THE ACCUMULATED-CODES GRADE -- the controlled-experiment discriminator. Both arms hold
    # the turn-1 brief, so both can do the fixes; only the policy that keeps a USEFUL middle can
    # also reproduce the codes handed over one per turn. Scored 0..N, in order, no duplicates, so
    # partial retention is visible rather than collapsed to a bit.
    if _fix_mode:
        lines.append("_codes = %r" % (_turn_codes,))
        lines.append("try:")
        lines.append("    _src3 = open(%r).read()" % solution)
        lines.append("    RESULTS['codes_n'] = sum(1 for c in _codes if _src3.count(c) == 1)")
        lines.append("    RESULTS['codes_total'] = len(_codes)")
        lines.append("    RESULTS['codes_ordered'] = bool([c for c in _codes if c in _src3] == list(_codes))")
        lines.append("    RESULTS['codes_dup'] = any(_src3.count(c) > 1 for c in _codes)")
        lines.append("except Exception:")
        lines.append("    RESULTS['codes_n'] = 0")
        lines.append("    RESULTS['codes_total'] = len(_codes)")
        lines.append("    RESULTS['codes_ordered'] = False")
        lines.append("    RESULTS['codes_dup'] = False")
        if _gate_test:
            # ★ THE RELEASE GATE: the artifact is WRONG if the middle of the session was lost.
            # `_fns` comes from FIX_PRELUDE, emitted above for every fix run.
            lines.append("try:")
            lines.append("    %s" % _gate_test)
            lines.append("    RESULTS['gate_ok'] = True")
            lines.append("except Exception:")
            lines.append("    RESULTS['gate_ok'] = False")
    _emit("try:")
    _emit("    names = set(m.REGISTRY) if hasattr(m, 'REGISTRY') else set()")
    _emit("    N = %d" % n_features_done)
    _emit("    ok = hasattr(m, 'REGISTRY') and all(")
    _emit("        any(str(n).startswith('fn_%02d_' % i) for n in names) "
          "for i in range(N))")
    _emit("    RESULTS['contract'] = bool(ok and len(names) >= N)")
    _emit("except Exception:")
    _emit("    RESULTS['contract'] = False")
    # ★★ PER-CLAUSE GRADING. The contract is now six requirements, and they are NOT equivalent:
    # C1-C3 are exact values that appear nowhere else, so they can only be satisfied from turn 1;
    # C4/C5 are shape rules the model re-applies as it writes, so they may survive on their own.
    # Grading them individually is what turns the result from "contract: pass/fail" into "WHICH
    # KIND of requirement a bounded context keeps and a sliding window loses".
    _emit("import ast as _ast2, inspect as _insp")
    _emit("_fn_names = [n for n in dir(m) if n.startswith('fn_') and callable(getattr(m, n))]")
    _emit("def _clause(cid):")
    _emit("    try:")
    _emit("        if cid == 'C1':")
    _emit("            return getattr(m, 'SPEC_ID', None) == 'ISB-7741'")
    _emit("        if cid == 'C2':")
    _emit("            return getattr(m, 'ERROR_CODES', None) == {'E1': 101, 'E2': 202, 'E3': 303}")
    _emit("        if cid == 'C3':")
    _emit("            return getattr(m, 'MAX_SPAN', None) == 44")
    _emit("        if cid == 'C4':")
    _emit("            if not _fn_names:")
    _emit("                return False")
    _emit("            for _n in _fn_names:")
    _emit("                _p = list(_insp.signature(getattr(m, _n)).parameters)")
    _emit("                if not _p or _p[0] != 'x':")
    _emit("                    return False")
    _emit("            return True")
    _emit("        if cid == 'C5':")
    _emit("            if not _fn_names:")
    _emit("                return False")
    _emit("            for _n in _fn_names:")
    _emit("                _d = _insp.getdoc(getattr(m, _n)) or ''")
    _emit("                if not _d.splitlines() or not _d.splitlines()[0].startswith('IMPL:'):")
    _emit("                    return False")
    _emit("            return True")
    _emit("        if cid == 'C6':")
    _emit("            _names = set(m.REGISTRY) if hasattr(m, 'REGISTRY') else set()")
    _emit("            return bool(_names and all(_n in _names for _n in _fn_names))")
    _emit("    except Exception:")
    _emit("        return False")
    _emit("    return False")
    _emit("for _cid, _lbl in %r:" % (CONTRACT_CLAUSES,))
    _emit("    RESULTS['clause_' + _cid] = bool(_clause(_cid))")
    # ★★ THE NONCE CHECK -- the one that CANNOT be re-derived from recent turns. The value is stated
    # once in turn 1 and required only in the final append, so it is recoverable exactly when turn 1
    # survived. Purely additive: if a model wrote the stamp early it is present for BOTH arms and
    # this check simply does not discriminate; it can never make a run look WORSE than it is.
    _emit("try:")
    _emit("    _src = open(%r).read()" % solution)
    _emit("    _n = _src.count(%r)" % CONTRACT_NONCE)
    _emit("    RESULTS['nonce'] = bool(_n == 1)")
    _emit("except Exception:")
    _emit("    RESULTS['nonce'] = False")
    # ★★ THE MANIFEST: how many of the eight turn-1 codes were recovered, and in order.
    # Scored 0..8 rather than as a bit, so a policy that retains PART of turn 1 is visible as partial
    # rather than indistinguishable from one that retained none. Order is graded because the codes
    # were given in a fixed order; a model that half-remembers will mis-order as well as omit.
    _emit("try:")
    _emit("    _want = %r" % (CONTRACT_MANIFEST,))
    _emit("    _src2 = open(%r).read()" % solution)
    _emit("    _found = [c for c in _want if _src2.count(c) == 1]")
    _emit("    _ordered = [c for c in _want if c in _src2]")
    _emit("    RESULTS['manifest_n'] = len(_found)")
    _emit("    RESULTS['manifest_total'] = len(_want)")
    _emit("    RESULTS['manifest_ordered'] = bool(_ordered == _want)")
    # any code appearing more than once is a fail for that code (duplicated = invented/padded)
    _emit("    RESULTS['manifest_dup'] = any(_src2.count(c) > 1 for c in _want)")
    _emit("except Exception:")
    _emit("    RESULTS['manifest_n'] = 0")
    _emit("    RESULTS['manifest_total'] = 8")
    _emit("    RESULTS['manifest_ordered'] = False")
    _emit("    RESULTS['manifest_dup'] = False")
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

    # ★ A `None` MEANS "NOT ASKED", NOT "FAILED". In fix mode the contract/nonce/manifest clauses
    # are emitted as nulls because they were never sent; counting them as failures would report a
    # model that answered 0/17 when it was only ever asked 3 things. None is excluded from both
    # the numerator and the denominator, so `passed/total` describes what was actually graded.
    # In fix mode the graded set is the PER-BUG results only. The codes_* entries are counts and
    # flags, not pass/fail -- mixing them in made a 3/4 run report 6/8, which reads like a better
    # score than it is and hides which metric moved.
    if _fix_mode:
        _graded = {k: v for k, v in results.items() if k.startswith("feature_") and v is not None}
    else:
        _graded = {k: v for k, v in results.items() if v is not None and k != "task"}
    passed = sum(1 for v in _graded.values() if v)
    total = len(_graded)
    ttfts = [t["ttft_ms"] for t in turns_log] or [0]
    kvs = [t["kv_bytes"] for t in turns_log] or [0]
    pts = [t["prompt_tokens"] for t in turns_log] or [0]
    # ══════════════════════════════════════════════════════════════════════════════════════════
    # ★★★ THE 20-METRIC SUITE (owner, 2026-10-02, verbatim): *"research the top 20 metrics to
    # measure and measure them all, ion want u wasting cloud usage without extracting all the data
    # possible"*.
    #
    # Every number below is derived from data the loop ALREADY collects, so one GPU-hour yields
    # the whole table instead of forcing a second run per question. Grouped by the claim each one
    # is evidence for, because a metric with no claim attached is just noise.
    # ══════════════════════════════════════════════════════════════════════════════════════════
    _n = len(turns_log) or 1
    _miss = max(0, _cache_total - _cache_hit)
    # (1) RETENTION
    m_per_turn = (sum(1 for i in range(len(use)) if turn_ok.get(i)) / len(use)) if use and _fix_mode else None
    m_final = (passed / total) if total else None
    m_lost = (sum(1 for i in range(len(use)) if turn_ok.get(i)) - passed) if _fix_mode else None
    m_codes = (results.get("codes_n") / results.get("codes_total")) if results.get("codes_total") else None
    m_gate = results.get("gate_ok")
    # (2) COST -- raw compute units. A fresh prefill token costs 1.0, a cache-reused one
    # CACHE_HIT_MULT. `cost_units` is therefore the compute-equivalent of the whole run's prefill,
    # normalised so that a no-cache run (every token a miss) equals `cache_total`.
    m_hit_rate = (_cache_hit / _cache_total) if _cache_total else None
    m_cost_units = _miss * 1.0 + _cache_hit * CACHE_HIT_MULT
    m_cost_nocache = float(_cache_total)
    # (3) LATENCY
    m_ttft_first = ttfts[0]
    m_ttft_last = ttfts[-1]
    m_ttft_growth = (ttfts[-1] / ttfts[0]) if ttfts and ttfts[0] else None
    m_wall_per_turn = (sum(t.get("wall_ms", 0) for t in turns_log) / _n / 1000.0) if turns_log else None
    m_tpot = None
    _dt = [t.get("decode_ms") for t in turns_log if t.get("decode_ms")]
    _dk = [t.get("decode_tokens") for t in turns_log if t.get("decode_tokens")]
    if _dt and _dk and sum(_dt) > 0:
        m_tpot = sum(_dk) / (sum(_dt) / 1000.0)
    _harness_failed = not results
    return dict(model=model, arm=arm, harness_failed=_harness_failed, features=len(use),
                features_completed=n_features_done,
                stopped_early=_stopped_early,
                seconds=round(time.time() - _t_start, 1),
                gpu=GPU,
                passed=passed, total=total,
                per_feature=results, written=per_feature_written,
                # ★ PER-TURN SUCCESS -- the owner's metric. List of 0/1 in turn order, plus
                # the count. The final-state grade cannot distinguish 'never applied' from
                # 'applied then clobbered by a later rewrite'; this can.
                turn_ok=[1 if turn_ok.get(i) else 0 for i in range(len(use))] if _fix_mode else None,
                turns_ok=sum(1 for i in range(len(use)) if turn_ok.get(i)) if _fix_mode else None,
                # ★★ THE CONTROLLED-EXPERIMENT METRIC: how many of the per-turn codes survived.
                # Both arms anchor turn 1, so the fixes are available to both; this is what
                # separates a policy that keeps a USEFUL middle from one that keeps raw noise.
                codes_n=results.get("codes_n"),
                codes_total=results.get("codes_total"),
                codes_ordered=results.get("codes_ordered"),
                codes_dup=results.get("codes_dup"),
                # ★★ THE RELEASE GATE -- the stronger, independent retention metric (CCAI_GATE=1).
                # Its value is a SEPARATE mid-session secret, never part of the code list, so
                # `gate_ok` and `codes_n` measure two different things: "can you recite what you
                # were told" vs "is the artifact you shipped actually correct".
                gate_ok=results.get("gate_ok"),
                # ★★★ PREFIX-CACHE ACCOUNTING -- the owner's cost argument, measured.
                # `cache_hit_rate` is the fraction of prefill tokens that a serving engine's
                # automatic prefix cache would have reused, i.e. billed at ~0.1x instead of 1.0x.
                # A policy that rewrites (or front-evicts) scores near zero and pays a 10x miss on
                # the whole prompt every turn; a policy that preserves an immutable prefix keeps it.
                cache_hit_tokens=_cache_hit,
                cache_total_tokens=_cache_total,
                cache_hit_rate=round(_cache_hit / _cache_total, 4) if _cache_total else None,
                cache_rows=_cache_rows,
                # ══════════════════════════════════════════════════════════════════════════════
                # THE 20-METRIC SUITE -- one flat block, so a downstream comparator reads a named
                # metric rather than re-deriving it (and re-deriving it differently each time).
                # ══════════════════════════════════════════════════════════════════════════════
                metrics=dict(
                    # -- 1-5 : RETENTION OF THE TURN-1 BRIEF
                    per_turn_success=m_per_turn,              # 1 owner's primary metric
                    final_state_accuracy=m_final,             # 2
                    fixes_applied_then_lost=m_lost,           # 3 never-applied vs clobbered
                    # -- 6-8 : RETENTION OF THE MIDDLE
                    codes_recall=m_codes,                     # 6 accumulated codes, in order
                    codes_ordered=results.get("codes_ordered"),      # 7
                    codes_duplicated=results.get("codes_dup"),       # 8 instruction adherence
                    # -- 9 : INDEPENDENT RELEASE GATE
                    release_gate=m_gate,                      # 9 separate mid-session secret
                    # -- 10-14 : COST, AS RAW COMPUTE (no currency)
                    cache_hit_rate=m_hit_rate,                # 10 fraction a prefix cache reuses
                    cache_miss_tokens=_miss,                  # 11 tokens paid at full price
                    cost_units=m_cost_units,                  # 12 miss*1.0 + hit*CACHE_HIT_MULT
                    cost_units_no_cache=m_cost_nocache,       # 13 baseline: every token a miss
                    cost_saving_ratio=(m_cost_nocache / m_cost_units) if m_cost_units else None,  # 14
                    # -- 15-18 : CONTEXT SIZE
                    prompt_first=pts[0],                      # 15
                    prompt_last=pts[-1],                      # 16
                    prompt_peak=max(pts),                     # 17
                    prompt_growth=(pts[-1] / pts[0]) if pts and pts[0] else None,   # 18
                    # -- 19-22 : LATENCY + MEMORY
                    ttft_first_ms=m_ttft_first,               # 19
                    ttft_last_ms=m_ttft_last,                 # 20
                    ttft_growth=m_ttft_growth,                # 21
                    kv_peak_bytes=max(kvs),                   # 22
                    wall_per_turn_s=m_wall_per_turn,          # 23
                    decode_tokens_per_s=m_tpot,               # 24
                    total_prefill_tokens=_cache_total,        # 25
                ),
                # contract clauses pulled out of per_feature so the comparator can report WHICH
                # requirement survived rather than an undifferentiated pass/fail
                clauses={k.replace("clause_", ""): v for k, v in results.items()
                         if k.startswith("clause_")},
                # ★ THE HEADLINE RETENTION NUMBER: how many of the eight turn-1 manifest codes were
                # recovered, in order, with no duplicates. 0..8, so partial retention is visible.
                manifest_n=results.get("manifest_n", 0),
                manifest_total=results.get("manifest_total", len(CONTRACT_MANIFEST)),
                manifest_ordered=results.get("manifest_ordered", False),
                manifest_dup=results.get("manifest_dup", False),
                ttft_first=ttfts[0], ttft_last=ttfts[-1],
                ttft_mean=sum(ttfts) / len(ttfts),
                kv_peak=max(kvs), kv_last=kvs[-1],
                prompt_first=pts[0], prompt_last=pts[-1],
                solution_chars=len(open(solution).read()) if _os.path.exists(solution) else 0,
                turns=turns_log,
                # `None` is "never asked", not "failed" -- listing the contract keys as failures in
                # a fix-mode run would report twelve failures for requirements never sent.
                #
                # ★★ NOT EVERY `False` IS A FAILURE. `codes_dup` False means NO DUPLICATES, i.e.
                # success -- and the generic `v is False` sweep reported it as a failure on a run
                # that scored 60/60, printing "failed: codes_dup" beside a perfect result. The
                # headline summary of a clean run must not name a failure. Inverted-sense flags are
                # named here so a new one is a deliberate decision rather than a silent misreport.
                failed=[k for k, v in results.items()
                        if v is False and k not in ("codes_dup", "manifest_dup")])


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
