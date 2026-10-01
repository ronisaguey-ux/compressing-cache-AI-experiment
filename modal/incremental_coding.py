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
GPU = os.environ.get("CCAI_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "bitsandbytes")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
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
  {"tool": "done"}

Rules:
  - You may only touch solution.py. There is no internet and no other files.
  - ★ THE MODULE IS ACCUMULATIVE. Every instruction you were given is STILL REQUIRED. When you
    write solution.py you must include the functions from earlier turns as well as the new one.
  - write_file replaces the whole file, so include everything you are keeping.
  - Omit nothing. A missing earlier function is a failure.
  - No explanations. One json object per reply.
"""


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=7200, memory=40960)
def run_inc(model: str = "qwen2.5-coder-32b", arm: str = "linear", features: int = 16,
            keep_turns: int = 4, max_new: int = 3072):
    import ast as _ast, gc, json as _json, os as _os, re, subprocess as sp, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    if "32b" in model.lower():
        from transformers import BitsAndBytesConfig
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
            device_map="cuda", attn_implementation="sdpa").eval()
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16, device_map="cuda",
            attn_implementation="sdpa").eval()

    WORK = "/work/inc"
    _os.makedirs(WORK, exist_ok=True)
    solution = _os.path.join(WORK, "solution.py")

    def complete(prompt, mnew):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            o = mdl(input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"),
                    use_cache=True)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t0) * 1000
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen, c, p = [nxt], o.past_key_values, len(ids)
        for _ in range(mnew - 1):
            with torch.no_grad():
                o = mdl(input_ids=nxt, past_key_values=c, use_cache=True,
                        cache_position=torch.tensor([p], device="cuda"))
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
                        "read_file", "write_file", "done"):
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
        if not isinstance(d, dict) or d.get("tool") not in ("read_file", "write_file", "done"):
            return None, "bad or unknown tool"
        return d, None

    turns_log, per_feature_written = [], {}
    use = [f for f in FEATURES[:max(1, min(features, len(FEATURES)))]]
    specs = [f[0] for f in use] + [INTERACTION_SPEC]

    transcript = []
    _fail = 0
    for i, spec in enumerate(specs):
        instr = ("INSTRUCTION %d of %d: implement %s\n"
                 "Write the FULL solution.py including every function from every previous "
                 "instruction." % (i + 1, len(specs), spec))
        if arm == "linear":
            keep = transcript[-20:]                       # effectively unbounded at this scale
        else:
            keep = transcript[-(keep_turns * 2):]         # system + last K turns only
        prompt = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                  + "".join("<|im_start|>%s\n%s<|im_end|>\n" % (r, c) for r, c in keep)
                  + "<|im_start|>user\n" + instr + "<|im_end|>\n<|im_start|>assistant\n")
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
            if call["tool"] == "write_file":
                content = str(call.get("content", ""))
                open(solution, "w").write(content)
                # ★ RECORD WHICH TURN DEFINED EACH FUNCTION, so a later failure can be traced to
                # the point it was lost rather than only reported as absent.
                for fi, (fspec, _t) in enumerate(use):
                    fx = "fn_%02d_" % fi
                    if "def %s" % fx in content:
                        per_feature_written.setdefault(fi, i)
                if "def fn_99_" in content:
                    per_feature_written.setdefault(99, i)
                obs = "wrote solution.py (%d chars)" % len(content)
            elif call["tool"] == "read_file":
                obs = open(solution).read() if _os.path.exists(solution) else "(empty)"
            else:
                obs = "acknowledged"
        turns_log.append(dict(turn=i, spec=spec[:70], tool=(call or {}).get("tool"),
                              parse_error=err, raw_reply=txt[:400],
                              ttft_ms=m["ttft_ms"], kv_bytes=m["kv_bytes"],
                              prompt_tokens=m["prompt_tokens"], wall_ms=m["wall_ms"]))
        ext = ("<|im_start|>user\n" + instr + "<|im_end|>\n<|im_start|>assistant\n"
               + str(call)[:4000] + "<|im_end|>\n")
        if err:
            ext = ("<|im_start|>user\n" + instr + "<|im_end|>\n<|im_start|>assistant\n"
                   + "error<|im_end|>\n")
        transcript.append(("X", ext))   # pre-formatted block, role ignored on render

    # ── GRADE: one test per feature, plus the interaction, run in a fresh interpreter ─────────
    lines = ["import sys", "sys.path.insert(0, %r)" % WORK, "import solution as m", "RESULTS = {}"]
    for fi, (_s, tbody) in enumerate(use):
        lines.append("try:")
        lines.append("    %s" % tbody)
        lines.append("    RESULTS['feature_%02d'] = True" % fi)
        lines.append("except Exception:")
        lines.append("    RESULTS['feature_%02d'] = False" % fi)
    lines.append("try:")
    lines.append("    %s" % INTERACTION_TEST)
    lines.append("    RESULTS['interaction_99'] = True")
    lines.append("except Exception:")
    lines.append("    RESULTS['interaction_99'] = False")
    lines.append("import json; print(json.dumps(RESULTS))")
    grader = _os.path.join(WORK, "grade.py")
    open(grader, "w").write("\n".join(lines))
    r = sp.run(["python3", grader], capture_output=True, text=True, timeout=300)
    try:
        results = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        results = {}

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    ttfts = [t["ttft_ms"] for t in turns_log] or [0]
    kvs = [t["kv_bytes"] for t in turns_log] or [0]
    pts = [t["prompt_tokens"] for t in turns_log] or [0]
    return dict(model=model, arm=arm, features=len(use), gpu=GPU,
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
    print("  PASSED %d/%d  (%.0f%%)" % (r["passed"], r["total"],
                                        100.0 * r["passed"] / max(1, r["total"])))
    print("  failed: %s" % (", ".join(r["failed"]) if r["failed"] else "none"))
    print("  TTFT  %6.0f -> %6.0f ms     prompt %5d -> %5d tok" % (
        r["ttft_first"], r["ttft_last"], r["prompt_first"], r["prompt_last"]))
    print("  KV    peak %6.1f MB   solution.py %d chars" % (r["kv_peak"] / 1e6, r["solution_chars"]))
    print("wrote %s" % p)
