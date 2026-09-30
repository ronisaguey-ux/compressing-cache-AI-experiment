"""Bob's "Frankenstein vs Reg" test: multi-turn code repair under history eviction.

    modal run modal/frankenstein.py --model qwen2.5-7b --rounds 3

THE QUESTION BOB ASKED. When a model repairs a function across several iterations, does it
cleanly overwrite its previous mental draft, or does it hallucinate deprecated variables from
earlier attempts? And does the contiguous-RoPE assembly change that?

★ "FRANKENSTEIN" IS MADE MEASURABLE, NOT IMPRESSIONISTIC. A Frankenstein artifact is a name the
model uses that is DEFINED NOWHERE in the code it just wrote -- a ghost of an earlier draft. That
is directly detectable: exec the model's output and a leftover name raises NameError. So the
metric is the rate of outputs that fail with an undefined name, which is a real defect rather than
a stylistic judgement, and it is separated from ordinary wrong answers.

TWO ARMS, and the difference is ONLY what stays in context:
  history   every previous attempt and its error remain visible      (standard serving)
  evict     earlier FAILED attempts are dropped from context          (block runtime, evicted)

★ WHY THIS IS THE RIGHT SHAPE. Bob's Test 4 wanted a SWE-bench Mini with a real repo, a real
failing pytest and a 10-turn patch loop. That harness does not exist here and a synthetic
stand-in would produce a meaningless patch-resolution rate. What CAN be done honestly is the
mechanism his test names, on functions small enough that correctness is decided by real
execution: does the model produce a function that passes, and does discarding its earlier drafts
help or hurt. The code is genuinely exec'd and genuinely asserted, so Pass@1 and Pass@3 are real.

METRICS: Pass@1, Pass@3, and Frankenstein rate (undefined-name failures in the final answer),
plus cost per arm.
"""
import modal, os

app = modal.App("ccai-frankenstein")
GPU = os.environ.get("CCAI_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
    .add_local_file("/tmp/opencode/ccai/modal/cost.py", "/root/cost.py")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}

# ---------------------------------------------------------------------------------------------
# Tasks. Each is a BUGGY function plus an assertion suite that decides correctness by execution.
#
# ★★ THE FIRST DRAFT OF THIS SET WAS WORTHLESS AND THE MEASUREMENT SAID SO. Six trivial bugs
# (off-by-one, wrong operator, discarded accumulator) scored **pass@1 = 100% on BOTH arms**,
# which means the repair loop never ran: every task was fixed on the first attempt, so there was
# no multi-turn iteration to study and no opportunity for a Frankenstein artifact. A benchmark
# where everything passes on attempt 1 measures nothing.
#
# These tasks are chosen so a PLAUSIBLE FIRST FIX IS WRONG. Each has a trap that a model reaches
# for naturally and which only the edge cases in the suite expose:
#   - `windowed`  the obvious fix fixes the window but not the trailing-partial window
#   - `counter`   a cached-default argument makes every call share one accumulator
#   - `close_all` shallow copy leaves the mutation reaching the caller's objects
#   - `near`      exact equality on floats, where the fix must be a tolerance not a rounding
#   - `make_adders` late binding, where the intuitive closure fix is still wrong
#   - `deep_sum`  recursion that must handle a nested empty container and a non-list
# ---------------------------------------------------------------------------------------------
TASKS = [
    dict(
        name="windowed",
        buggy=("def windowed(xs, k):\n    out = []\n    for i in range(len(xs)):\n"
               "        out.append(sum(xs[i:i + k - 1]))\n    return out\n"),
        tests=("assert windowed([1, 2, 3, 4], 2) == [3, 5, 7]\n"
               "assert windowed([1, 2, 3, 4, 5], 3) == [6, 9, 12]\n"
               "assert windowed([1, 2], 5) == []\n"
               "assert windowed([], 2) == []\n"
               "assert windowed([5], 1) == [5]\n"),
        issue=("windowed(xs, k) should return the sums of every length-k sliding window. "
               "windowed([1,2,3,4], 2) returns [1,2,3,4] but should return [3,5,7]. "
               "Windows that would run past the end must be omitted, and k=1 must work."),
    ),
    dict(
        name="counter",
        # ★ THE MUTABLE-DEFAULT TRAP DOES NOT FIRE INSIDE A FACTORY. `def bump(key, step=1,
        # into={})` looks like the classic shared-state bug, but the def executes on every
        # make_counter() call, so `into={}` is re-evaluated and each counter gets its OWN dict.
        # Measured: that version PASSED the tests. Genuine sharing needs state that outlives the
        # call, so the dict is at module scope here.
        buggy=("_COUNTS = {}\n\n"
               "def make_counter():\n"
               "    def bump(key, step=1):\n"
               "        _COUNTS[key] = _COUNTS.get(key, 0) + step\n"
               "        return _COUNTS[key]\n"
               "    return bump\n"),
        tests=("a = make_counter()\nassert a('x') == 1\nassert a('x') == 2\n"
               "assert a('y', 3) == 3\nb = make_counter()\nassert b('x') == 1\n"
               "assert a('x') == 3\n"),
        issue=("make_counter() should return a callable keeping its OWN private counts. Two "
               "counters must not share state, and bump(key, step) must advance by step. "
               "Two counters currently share one store."),
    ),
    dict(
        name="close_all",
        buggy=("def close_all(items):\n    done = []\n    for it in items:\n"
               "        it['open'] = False\n        done.append(it)\n"
               "        items.remove(it)\n    return done\n"),
        tests=("a = [{'id': 1, 'open': True}, {'id': 2, 'open': True}]\n"
               "r = close_all(a)\n"
               "assert len(r) == 2\n"
               "assert [x['open'] for x in r] == [False, False]\n"
               "assert a == []\n"
               "src = [{'id': 9, 'open': True}]\n"
               "close_all(src)\n"
               "assert src == []\n"),
        issue=("close_all(items) should mark every item closed, return them all, and leave the "
               "passed list EMPTY. close_all([{...},{...}]) currently returns only one item."),
    ),
    dict(
        name="near",
        buggy=("def near(a, b):\n    return round(a, 2) == round(b, 2)\n"),
        tests=("assert near(0.1 + 0.2, 0.3)\nassert near(1.0, 1.0)\n"
               "assert not near(0.004, 0.001)\nassert not near(1.0, 1.01)\n"
               "assert near(-0.0, 0.0)\n"),
        issue=("near(a, b) should report whether two floats are equal to within a small "
               "absolute tolerance (1e-6). near(0.1 + 0.2, 0.3) must be True and "
               "near(0.004, 0.001) must be False — an absolute tolerance, not rounding."),
    ),
    dict(
        name="make_adders",
        buggy=("def make_adders(n):\n    return [lambda x: x + i for i in range(n)]\n"),
        tests=("fs = make_adders(3)\nassert fs[0](10) == 10\nassert fs[1](10) == 11\n"
               "assert fs[2](10) == 12\nassert [f(0) for f in make_adders(2)] == [0, 1]\n"),
        issue=("make_adders(3) should return three functions adding 0, 1 and 2 respectively. "
               "Every returned function currently adds the same value."),
    ),
    dict(
        name="deep_sum",
        buggy=("def deep_sum(x):\n    if isinstance(x, list):\n"
               "        return sum(x)\n    return x\n"),
        tests=("assert deep_sum([1, [2, [3, 4]], 5]) == 15\n"
               "assert deep_sum([]) == 0\n"
               "assert deep_sum([[], 1]) == 1\n"
               "assert deep_sum(7) == 7\n"
               "assert deep_sum([[1], [2, [3]]]) == 6\n"),
        issue=("deep_sum(x) should sum every number inside arbitrarily nested lists, "
               "treating an empty list as 0. deep_sum([[], 1]) currently fails."),
    ),
]

# ---------------------------------------------------------------------------------------------
# THE HARD SET. The first two task sets both scored pass@1 = 100%, which means the model fixed
# every bug on the FIRST attempt and the repair loop never ran. A Frankenstein artifact can only
# appear when an attempt FAILS and the model must reconcile with its own earlier draft, so a set
# the model one-shots cannot test the phenomenon at all.
#
# These require MULTIPLE interacting pieces, which is where partial fixes are natural and an
# earlier draft is most likely to contaminate the next one.
# ---------------------------------------------------------------------------------------------
TASKS_HARD = [
    dict(
        name="lru",
        buggy=("class LRU:\n    def __init__(self, cap):\n        self.cap = cap\n"
               "        self.d = {}\n    def get(self, k):\n        return self.d.get(k, -1)\n"
               "    def put(self, k, v):\n        self.d[k] = v\n"),
        tests=("c = LRU(2)\nc.put('a', 1)\nc.put('b', 2)\nassert c.get('a') == 1\n"
               "c.put('c', 3)\nassert c.get('b') == -1\nassert c.get('a') == 1\n"
               "assert c.get('c') == 3\nc.put('a', 9)\nassert c.get('a') == 9\n"
               "assert c.get('c') == 3\n"
               "d = LRU(1)\nd.put('x', 1)\nd.put('y', 2)\nassert d.get('x') == -1\n"
               "assert d.get('y') == 2\n"),
        issue=("Implement an LRU cache with capacity cap. get(k) returns the value or -1 and "
               "counts as a use. put(k, v) inserts and counts as a use, evicting the "
               "LEAST-RECENTLY-USED key when over capacity. It currently never evicts."),
    ),
    dict(
        name="merge_intervals",
        buggy=("def merge(iv):\n    iv = sorted(iv)\n    out = []\n    for s, e in iv:\n"
               "        if out and out[-1][1] > s:\n            out[-1] = (out[-1][0], e)\n"
               "        else:\n            out.append((s, e))\n    return out\n"),
        tests=("assert merge([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)]\n"
               "assert merge([]) == []\nassert merge([(1,4),(4,5)]) == [(1,5)]\n"
               "assert merge([(1,10),(2,3)]) == [(1,10)]\n"
               "assert merge([(5,6),(1,2)]) == [(1,2),(5,6)]\n"),
        issue=("merge(intervals) must merge all OVERLAPPING AND TOUCHING intervals and return "
               "them sorted. merge([(1,4),(4,5)]) returns [(1,4),(4,5)] but should be [(1,5)]; "
               "and an interval fully contained in a previous one must not shrink it."),
    ),
    dict(
        name="calc",
        buggy=("def calc(s):\n    return eval(s)\n"),
        tests=("assert calc('2+3*4') == 14\nassert calc('(2+3)*4') == 20\n"
               "assert calc('10-4-3') == 3\nassert calc('2*3+4*5') == 26\n"
               "assert calc('(((7)))') == 7\n"
               "assert isinstance(calc('8/2/2'), int), 'even division must give an int'\n"
               "assert calc('7/2') == 3.5\n"
               "assert type(calc('12')) is int\n"),
        issue=("Write a calculator that evaluates + - * / with parentheses, correct "
               "precedence and left-associativity, WITHOUT using eval(). 10-4-3 must be 3 "
               "(left to right). Division that comes out exact must return an INT "
               "(6/3 -> 2 not 2.0), inexact stays a float (7/2 -> 3.5)."),
    ),
    dict(
        name="tok",
        buggy=("def toks(s):\n    out = []\n    for t in s.split(' '):\n"
               "        if t:\n            out.append(t)\n    return out\n"),
        tests=("assert toks('a b  c') == ['a','b','c']\n"
               "assert toks('x=1+2') == ['x','=','1','+','2']\n"
               "assert toks('foo(bar,1)') == ['foo','(','bar',',','1',')']\n"
               "assert toks('') == []\n"
               "assert toks('12 3') == ['12','3']\n"
               "assert toks('a<=b') == ['a','<=','b']\n"),
        issue=("Tokenize into identifiers (letters/digits/underscore), numbers, and the "
               "operators + - * / = ( ) , and the two-character <= . Whitespace separates but "
               "'x=1+2' must split into five tokens; 'a<=b' must not split '<' and '='."),
    ),
]


def extract_code(text):
    """Take the fenced block if present, else the whole reply. Never guess mid-function."""
    import re
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    return text


def run_case(code, tests):
    """Execute the candidate against the assertions in a fresh namespace.

    Returns (passed, error_kind, error_detail). `error_kind` distinguishes a Frankenstein
    artifact (undefined name) from an ordinary failure, because they mean different things.
    """
    ns = {}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
    except SyntaxError as e:
        return False, "syntax", str(e)[:160]
    except Exception as e:
        return False, type(e).__name__, str(e)[:160]
    try:
        exec(compile(tests, "<tests>", "exec"), ns)
    except NameError as e:
        return False, "undefined_name", str(e)[:160]
    except AssertionError as e:
        return False, "assertion", str(e)[:160]
    except Exception as e:
        return False, type(e).__name__, str(e)[:160]
    return True, None, None


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=5400, memory=32768)
def run_repair(model: str = "qwen2.5-7b", rounds: int = 3, max_new: int = 320,
               hard: bool = False):
    import sys, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, "/root/ccai/src")
    sys.path.insert(0, "/root")
    from cost import track, ROWS

    dev = "cuda"
    mid = MODELS[model]
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=dev,
                                             attn_implementation="sdpa")
    m.eval()

    use_chatml = "qwen" in model

    def chat(system, user):
        if use_chatml:
            return ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
                    + user + "<|im_end|>\n<|im_start|>assistant\n")
        return "<s>[INST] " + system + " " + user + " [/INST]"

    def gen(prompt, max_new):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            o = m(input_ids=ids_t, use_cache=True)
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        out = [nxt]
        p = len(ids)
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = m(input_ids=nxt, past_key_values=c, use_cache=True,
                      cache_position=torch.tensor([p], device=dev))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(nxt); p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        return tok.decode(torch.cat(out, dim=1)[0], skip_special_tokens=True)

    SYS = ("You are a Python engineer. Reply with ONLY the complete corrected function in a "
           "single ```python block. No explanation.")

    task_set = TASKS_HARD if hard else TASKS
    results = []
    for arm in ("history", "evict"):
        with track("frankenstein", "%s/%s" % (model, arm)) as c:
            c.note(gpu=os.environ.get("CCAI_GPU", "A10G"))
            for task in task_set:
                attempts = []       # (code, error_kind, error_detail)
                first_pass = None
                passed_any = False
                frankenstein = False
                for r in range(rounds):
                    # ---- build the user message, and THIS is the only arm difference
                    parts = ["Fix this function.\n\n```python\n%s```\n\n%s\n"
                             % (task["buggy"], task["issue"])]
                    if arm == "history":
                        for i, (code, kind, detail) in enumerate(attempts):
                            parts.append(
                                "\nAttempt %d failed (%s):\n```python\n%s```\nError: %s\n"
                                % (i + 1, kind, code.strip()[:900], detail))
                    else:
                        # EVICT everything except the most recent failure: earlier drafts are
                        # gone, which is the block-runtime behaviour under Bob's test.
                        if attempts:
                            code, kind, detail = attempts[-1]
                            parts.append(
                                "\nThe last attempt failed (%s):\n```python\n%s```\nError: %s\n"
                                % (kind, code.strip()[:900], detail))
                    parts.append("\nReturn the corrected function.")
                    prompt = chat(SYS, "\n".join(parts))

                    reply = gen(prompt, max_new)
                    code = extract_code(reply)
                    ok, kind, detail = run_case(code, task["tests"])
                    attempts.append((code, kind, detail))
                    if ok:
                        if first_pass is None:
                            first_pass = r + 1
                        passed_any = True
                        break
                    if kind == "undefined_name":
                        frankenstein = True

                results.append(dict(
                    model=model, arm=arm, task=task["name"],
                    pass_at_1=(first_pass == 1),
                    pass_at_3=passed_any,
                    rounds_used=len(attempts),
                    frankenstein=frankenstein,
                    final_kind=attempts[-1][1],
                    final_detail=attempts[-1][2],
                ))
                print("    %-14s %-8s %-8s pass@1=%-5s pass@3=%-5s frankenstein=%s"
                      % (arm, task["name"], attempts[-1][1] or "ok",
                         first_pass == 1, passed_any, frankenstein), flush=True)
            c.note(tasks=len(task_set))
            torch.cuda.empty_cache()

    def rate(arm, key):
        rs = [r for r in results if r["arm"] == arm]
        return sum(1 for r in rs if r[key]) / len(rs) if rs else 0.0

    return dict(model=model, rounds=rounds, results=results, cost=list(ROWS),
                history=dict(pass1=rate("history", "pass_at_1"), pass3=rate("history", "pass_at_3"),
                             frank=rate("history", "frankenstein")),
                evict=dict(pass1=rate("evict", "pass_at_1"), pass3=rate("evict", "pass_at_3"),
                           frank=rate("evict", "frankenstein")))


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", rounds: int = 3, max_new: int = 320,
         hard: bool = False):
    import json, os, time, sys
    sys.path.insert(0, "/tmp/opencode/ccai/modal")
    from cost import summary
    ms = [x.strip() for x in model.split(",") if x.strip()]
    out = []
    # ★ PERSIST THE CONTAINER'S COST ROWS LOCALLY. A Modal container is discarded after the run,
    # so a ledger written inside it is lost -- measured: the host reported "no ledger" even
    # though the rows were recorded. The function returns them and they are appended here.
    import json as _json
    ledger = "/tmp/opencode/ccai/benchmarks/results/cost_ledger.jsonl"
    for mm in ms:
        print("\n===== FRANKENSTEIN vs REG: %s (%d repair rounds) =====" % (mm, rounds),
              flush=True)
        try:
            r = run_repair.remote(mm, rounds, max_new, hard)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)
            out.append({"model": mm, "status": "failed", "error": str(e)[:300]})
            continue
        out.append(r)
        for row in r.get("cost", []):
            with open(ledger, "a") as lf:
                lf.write(_json.dumps(row) + "\n")
        h, e = r["history"], r["evict"]
        print("\n  %-12s arm        pass@1   pass@3   frankenstein" % "")
        print("  %-12s history    %5.0f%%  %5.0f%%   %5.0f%%" % (
            mm, h["pass1"] * 100, h["pass3"] * 100, h["frank"] * 100))
        print("  %-12s evict      %5.0f%%  %5.0f%%   %5.0f%%" % (
            "", e["pass1"] * 100, e["pass3"] * 100, e["frank"] * 100))

    outdir = "/tmp/opencode/ccai/benchmarks/results"
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "frankenstein_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    json.dump(out, open(p, "w"), indent=2)
    print("\nwrote %s" % p)
    print()
    summary()
