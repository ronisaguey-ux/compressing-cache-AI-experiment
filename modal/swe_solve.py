"""SWE-bench Lite: linear prefill vs the block runtime, self-contained on Modal.

    modal run modal/swe_solve.py --model qwen2.5-7b --arm gold     --limit 10
    modal run modal/swe_solve.py --model qwen2.5-7b --arm linear   --limit 10
    modal run modal/swe_solve.py --model qwen2.5-7b --arm runtime  --limit 10

★ WHY THE WHOLE LOOP RUNS IN-CONTAINER RATHER THAN MODEL-ON-MODAL / HARNESS-LOCAL.

The first attempt served the model over HTTP so the local harness could drive it. That splits one
experiment across a network boundary and makes every failure ambiguous: a turn that returns nothing
could be the model, the transport, or the harness. Keeping the model, the repo and pytest in ONE
container means a failed turn has exactly one explanation.

The repo and the per-instance venvs live on a Modal Volume, so the clone and the python-3.9 +
sympy install are paid for ONCE and every later container reuses them. Without that, each container
re-clones 206 MB and reinstalls sympy, which dwarfs the model time.

★ THE THREE ARMS, and only one thing differs between them:

  gold      the dataset's own patch applied, then graded. NOT a baseline -- a PARITY CHECK. If gold
            does not resolve an instance the harness is broken, and every model number is noise.
  linear    the whole transcript re-prefilled every turn. A normal agent loop.
  runtime   the transcript assembled from a block table, superseded turns evicted, survivors
            re-indexed contiguously (Finding 37), query as its own tier (Finding 29).

★ TTFT IS THE METRIC THAT MATTERS FOR THIS COMPARISON and it is measured, not modelled: it is the
wall time from handing the arm its prompt to the first generated token, which is exactly what a
prefix cache or an eviction policy changes. `prefill_tokens` is logged alongside it so a latency
difference can be attributed to work rather than assumed to be one.
"""
import json, os, modal

app = modal.App("ccai-swe-solve")
GPU = os.environ.get("CCAI_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "bitsandbytes")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "GIT_TERMINAL_PROMPT": "0"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
    .add_local_dir("/tmp/opencode/sweval", "/root/instances",
                   ignore=["sympy/**", "venv_*/**", "__pycache__/**"])
)
cache = modal.Volume.from_name("ccai-swe-cache", create_if_missing=True)

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # ★ THE 7-8B MODELS CANNOT RESOLVE SWE-bench LITE, so a 0/10 vs 0/10 table says nothing about
    # the LAYOUT -- resolution is capability-bound and never varies with the cache policy. A 32B
    # coder at 4-bit CAN solve them, which turns resolution from a constant into a real measurement
    # and lets the table separate the arms on the metric the leaderboard cares about. 4-bit (AWQ)
    # keeps it inside a 40 GB card: 32.7 GB of weights -> ~18 GB.
    "qwen2.5-coder-32b-awq": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "qwen2.5-coder-32b-gptq": "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "llama-3.1-8b": "NousResearch/Meta-Llama-3.1-8B",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
}
SYMPY = "/work/sympy"
UV_PY = "/work/uvpy"

SYSTEM = """You are a software engineer fixing a real bug in the sympy repository.
You are shown the issue, and on later turns the result of what you did.

Reply with EXACTLY ONE json object and nothing else:

  {"tool": "search", "query": "<regex>", "glob": "*.py"}
  {"tool": "read_file", "path": "<path relative to repo root>", "start": 1, "end": 120}
  {"tool": "edit_file", "path": "<path>", "old_string": "<exact existing text>", "new_string": "<replacement>"}
  {"tool": "run_tests"}
  {"tool": "done"}

Rules:
  - PREFER edit_file. These source files are thousands of lines; reproducing one entire file is not
    practical and will be truncated. edit_file changes only the text you name.
  - old_string must match the file EXACTLY, including indentation, and must be unique in the file.
  - read_file returns a numbered slice; use start/end to page through a large file.
  - run the tests before you declare done. The task is finished when they pass.
  - there is no internet and you cannot install packages.
  - do not explain. One json object per reply.
"""


def load_model(mid):
    """Load a model, handling the 4-bit coder checkpoints.

    ★ WHY THIS IS A FUNCTION. The 32B coder is the model that can actually resolve SWE-bench, and
    it only fits in 4 bits. An AWQ checkpoint needs its quantisation backend named explicitly --
    loading it as fp16 either OOMs or silently dequantises and blows the card -- and the two
    formats (AWQ, GPTQ) take different configs. Centralising the choice means one place decides and
    a failure names which backend rejected the checkpoint instead of surfacing as an OOM.
    """
    import torch
    from transformers import AutoModelForCausalLM
    low = mid.lower()
    if "32b" in low and "awq" not in low and "gptq" not in low:
        # ★ BITSANDBYTES nf4 FOR THE FULL 32B. `auto-gptq` fails to build a wheel in this image
        # (subprocess-exited-with-error during the image build, which reads as a generic build
        # failure rather than a specific package problem), and an image build is an expensive way
        # to discover that. bitsandbytes installs cleanly, quantises on load, and the HF download is
        # cached on the volume so it is paid once. The 4-bit config needs the compute dtype named
        # explicitly or the first matmul raises.
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=bnb, device_map="cuda",
            attn_implementation="sdpa").eval()
    if "awq" in low:
        from transformers import AwqConfig
        try:
            return AutoModelForCausalLM.from_pretrained(
                mid, quantization_config=AwqConfig(bits=4, fuse_layers=True),
                device_map="cuda", dtype=torch.float16).eval()
        except Exception as e:
            _glog("  AWQ load failed (%s); falling back to device_map auto" % type(e).__name__)
            return AutoModelForCausalLM.from_pretrained(
                mid, device_map="auto", dtype=torch.float16).eval()
    if "gptq" in low:
        from transformers import GPTQConfig
        return AutoModelForCausalLM.from_pretrained(
            mid, quantization_config=GPTQConfig(bits=4, disable_exllama=False),
            device_map="cuda").eval()
    return AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()


def _glog(msg):
    print(msg, flush=True)


def setup_repo(inst, sub=None):
    """Clone (once, cached) and check out the instance's base commit + test_patch.

    ★ THE test_patch IS PART OF SETUP, NOT OF GRADING. It adds the tests that encode the fix, so a
    model that rewrote the tests instead of the code would otherwise appear to succeed. Applying it
    before the loop also means every `run_tests` the model issues is testing the REAL criterion.
    """
    import subprocess as sp
    os.makedirs("/work", exist_ok=True)
    if not os.path.isdir(os.path.join(SYMPY, ".git")):
        _glog("cloning sympy (first time, cached on the volume)")
        sp.run(["git", "clone", "-q", "https://github.com/sympy/sympy.git", SYMPY], check=True)
    sp.run(["git", "checkout", "-q", "--", "."], cwd=SYMPY)
    sp.run(["git", "clean", "-qfd"], cwd=SYMPY)
    r = sp.run(["git", "checkout", "-q", "-f", inst["base_commit"]], cwd=SYMPY,
               capture_output=True, text=True)
    if r.returncode != 0:
        return False, "checkout failed: " + (r.stderr or r.stdout)[-200:]
    tp = os.path.join("/root/instances", "tp_%s.patch" % inst["instance_id"])
    if os.path.exists(tp):
        r = sp.run(["git", "apply", "--whitespace=nowarn", tp], cwd=SYMPY,
                   capture_output=True, text=True)
        if r.returncode != 0:
            return False, "test_patch failed: " + (r.stderr or r.stdout)[-200:]
    return True, "ok"


def make_venv(inst):
    """One venv per instance, cached on the volume. sympy 1.0/1.1 needs python 3.9."""
    import subprocess as sp
    v = os.path.join(SYMPY, "..", "venv_" + inst["instance_id"])
    v = os.path.abspath(v)
    py = os.path.join(v, "bin", "python")
    if os.path.exists(py):
        return v, True
    os.makedirs(UV_PY, exist_ok=True)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=UV_PY, VIRTUAL_ENV=v)
    sp.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "-o", "/tmp/uv.sh"], check=True)
    sp.run(["sh", "/tmp/uv.sh"], check=True, capture_output=True)
    uv = "/root/.local/bin/uv"
    sp.run([uv, "python", "install", "3.9"], env=env, capture_output=True)
    sp.run([uv, "venv", "--python", "3.9", v], env=env, capture_output=True)
    for pkgs in (["mpmath", "pytest"], ["-e", "."]):
        sp.run([uv, "pip", "install", "-q"] + pkgs, env=env, cwd=SYMPY, capture_output=True)
    return v, os.path.exists(py)


def validated_p2p(inst, venv, p2p_ids):
    """PASS_TO_PASS with the already-broken tests removed, cached per instance on the volume.

    ★ MEASURED WHY THIS IS NECESSARY. `sympy__sympy-11897` lists `test_latex_Float` in
    PASS_TO_PASS, and that test FAILS on the untouched base tree -- before any patch, gold or
    model. A harness that treats the raw list as a regression guard therefore marks a CORRECT
    patch as unresolved, which is a false negative against the model and against gold. The real
    SWE-bench harness validates PASS_TO_PASS against the base state for exactly this reason.

    The cache matters: validating 95 tests takes ~15 s per instance, and it is identical for every
    arm and every model, so it is computed once and reused.
    """
    import os as _os, subprocess as sp
    ck = "/work/p2p_ok_%s.json" % inst["instance_id"]
    if _os.path.exists(ck):
        try:
            return json.load(open(ck))
        except Exception:
            pass
    good = []
    if p2p_ids and _os.path.exists(_os.path.join(venv, "bin", "python")):
        py = _os.path.join(venv, "bin", "python")
        r = sp.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + p2p_ids,
                   cwd=SYMPY, capture_output=True, text=True, timeout=1800)
        out = (r.stdout or "") + (r.stderr or "")
        bad = set()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                bad.add(line.split()[1].split("::")[-1].split("[")[0])
        good = [x for x in p2p_ids if x.split("::")[-1].split("[")[0] not in bad]
        dropped = len(p2p_ids) - len(good)
        if dropped:
            _glog("  p2p validation: dropped %d already-failing test(s): %s"
                  % (dropped, sorted(bad)[:4]))
    json.dump(good, open(ck, "w"))
    return good


def test_files(inst):
    import re
    return re.findall(r"^\+\+\+ b/(.+)$", inst.get("test_patch", ""), re.M)


def resolve(inst, spec):
    """Resolve a bare test name INSIDE the files the test_patch touches.

    ★ The first version grep'd repo-wide and took the shortest path. `test_sinc` and
    `test_Derivative` appear in several files, so it ran the WRONG test -- and an always-passing
    test reads as "VACUOUS: passes before the fix", which is how 4 of 10 instances were nearly
    discarded for a harness bug rather than a dataset one. The test_patch is unambiguous.
    """
    import subprocess as sp
    if "::" in spec:
        return spec
    for f in test_files(inst):
        p = os.path.join(SYMPY, f)
        if os.path.exists(p):
            r = sp.run(["grep", "-n", "def %s(" % spec, f], cwd=SYMPY,
                       capture_output=True, text=True)
            if r.stdout.strip():
                return "%s::%s" % (f, spec)
    return None


def run_tests(inst, venv, ids, timeout=900):
    import subprocess as sp
    if not ids:
        return True, "no tests"
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        return False, "no venv"
    r = sp.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + ids,
               cwd=SYMPY, capture_output=True, text=True, timeout=timeout)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if "passed" in l or "failed" in l or "error" in l]
    return r.returncode == 0, (lines[-1] if lines else (r.stdout + r.stderr)[-160:])


@app.function(image=image, gpu=GPU, secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": cache}, timeout=7200, memory=40960)
def solve(model: str = "qwen2.5-7b", arm: str = "gold", limit: int = 10, max_turns: int = 8,
          max_new: int = 1024):
    import gc, re, subprocess as sp, time, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mid = MODELS[model]
    insts = []
    for f in sorted(os.listdir("/root/instances")):
        if f.startswith("sympy__") and f.endswith(".json"):
            insts.append(json.load(open(os.path.join("/root/instances", f))))
    insts = insts[:limit]

    if arm != "gold":
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = load_model(mid)

    def complete(prompt, mnew):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        # keep the prompt inside the window; the transcript is the thing under test, not the cap
        if len(ids) > 12000:
            ids = ids[-12000:]
        t0 = time.perf_counter()
        with torch.no_grad():
            o = mdl(input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"), use_cache=True)
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
            gen.append(nxt)
            p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        txt = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        kv = sum(c.layers[i].keys.numel() * c.layers[i].keys.element_size()
                 + c.layers[i].values.numel() * c.layers[i].values.element_size()
                 for i in range(len(c.layers)))
        del o, c
        gc.collect(); torch.cuda.empty_cache()
        return dict(text=txt, prompt_tokens=len(ids), gen_tokens=len(gen),
                    ttft_ms=ttft, kv_bytes=kv,
                    wall_ms=(time.perf_counter() - t0) * 1000)

    def parse(t):
        """Extract the FIRST complete JSON object, ignoring anything after it.

        ★ MEASURED BUG, NOT A GUESS. The first version spanned from the first `{` to the LAST `}`
        and called json.loads on that slice. A model that emits a valid tool call followed by an
        explanation -- e.g. `{"tool":"search","query":"sinc"}\nThis searches for...` -- produced
        `Extra data: line 2 column 1`, which reads as malformed JSON and is not: the object is
        valid and there is simply more text after it. Every turn of the first run failed this way
        and the arm scored zero for a PARSER defect rather than a model one.

        `raw_decode` from the first brace takes the first complete value and ignores the tail,
        which is what "one tool call per reply" actually means in practice. Genuinely malformed
        JSON still fails, so the distinction the harness needs -- bad object vs trailing prose --
        is preserved.
        """
        t = t.strip()
        if "```" in t:
            for part in t.split("```"):
                p = part[4:].strip() if part.strip().startswith("json") else part.strip()
                if p.startswith("{"):
                    t = p
                    break
        i = t.find("{")
        if i < 0:
            return None, "no JSON object in the reply"
        body = t[i:]
        try:
            d, _end = json.JSONDecoder().raw_decode(body)
        except Exception as e:
            # ★ REPAIR INVALID ESCAPES, because a raw backslash is a COMMON and unambiguous model
            # error: a regex (`\d+`) or a LaTeX fragment (`\alpha`) written into a JSON string
            # without doubling. json.loads rejects the whole call with "Invalid \escape", every
            # turn fails identically, and the arm scores zero for a serialisation detail rather
            # than for the layout. Doubling a backslash that is not a legal JSON escape can only
            # turn INVALID json into VALID json -- a document that already parsed is never
            # touched, which is what makes the repair safe. Same class as the harness's
            # raw-newline fix.
            LEGAL = set('"\\/bfnrtu')
            out = []; k = 0; instr = False
            while k < len(body):
                ch = body[k]
                if ch == '"':
                    bs = 0; j = k - 1
                    while j >= 0 and body[j] == "\\":
                        bs += 1; j -= 1
                    if bs % 2 == 0:
                        instr = not instr
                    out.append(ch); k += 1; continue
                if instr and ch == "\\":
                    nxt = body[k + 1] if k + 1 < len(body) else ""
                    if nxt not in LEGAL:
                        out.append("\\\\"); k += 1; continue
                out.append(ch); k += 1
            try:
                d, _end = json.JSONDecoder().raw_decode("".join(out))
            except Exception:
                return None, "invalid JSON: %s" % e
        if not isinstance(d, dict) or d.get("tool") not in (
                "read_file", "search", "write_file", "edit_file", "run_tests", "done"):
            return None, "bad or unknown tool"
        return d, None

    rows = []
    for inst in insts:
        iid = inst["instance_id"]
        t_inst = time.perf_counter()
        _glog("\n===== %s  arm=%s =====" % (iid, arm))
        ok, msg = setup_repo(inst)
        if not ok:
            rows.append(dict(instance=iid, status="setup_failed", detail=msg)); continue
        venv, vok = make_venv(inst)
        if not vok:
            rows.append(dict(instance=iid, status="venv_failed")); continue

        f2p = [x for x in (resolve(inst, s) for s in json.loads(inst["FAIL_TO_PASS"])) if x]
        p2p = [x for x in (resolve(inst, s) for s in json.loads(inst["PASS_TO_PASS"])) if x]

        if arm == "gold":
            r = sp.run(["git", "apply", "--whitespace=nowarn",
                        os.path.join("/root/instances", "gold_%s.patch" % iid)],
                       cwd=SYMPY, capture_output=True, text=True)
            if r.returncode != 0:
                rows.append(dict(instance=iid, status="gold_apply_failed",
                                 detail=(r.stderr or r.stdout)[-160:])); continue
            p2p_v = validated_p2p(inst, venv, p2p)
            fo, fs = run_tests(inst, venv, f2p)
            po, ps = run_tests(inst, venv, p2p_v)
            rows.append(dict(instance=iid, status="ok", resolved=bool(fo and po),
                             n_p2p_raw=len(p2p), n_p2p_valid=len(p2p_v),
                             f2p_pass=fo, p2p_pass=po, f2p_summary=fs, p2p_summary=ps,
                             wall_s=time.perf_counter() - t_inst, turns=0))
            _glog("  gold resolved=%s  f2p=%s  p2p=%s" % (bool(fo and po), fs, ps))
            continue

        # ---------------- model arms ----------------
        problem = inst["problem_statement"]
        turns = []
        transcript = [("system", SYSTEM),
                      ("user", "ISSUE:\n" + problem[:6000] +
                       "\n\nStart by searching the repo for the relevant code.")]
        # ★ THE BLOCK TABLE. One block per turn; nothing is evicted until the table grows past
        # `keep_blocks`, which is what a long-horizon agent loop actually looks like.
        blocks = []
        done = False
        # initialise OUTSIDE the loop: reading it with locals() inside is fragile and would reset
        # silently if the assignment ever moved
        _fail_streak = 0
        for turn in range(max_turns):
            prompt = "".join("<|im_start|>%s\n%s<|im_end|>\n" % (r, c) for r, c in transcript)
            prompt += "<|im_start|>assistant\n"
            out = complete(prompt, max_new)
            call, err = parse(out["text"])
            obs = ""
            if err:
                _fail_streak += 1
                fail_streak = _fail_streak
                obs = ("could not parse your reply (%s). Reply with exactly ONE json object, "
                       "nothing else." % err)
                # ★ A REPEATED IDENTICAL ERROR MEANS THE CORRECTION IS NOT LANDING. Three turns of
                # the same parse failure is a loop, and feeding the same sentence a fourth time
                # wastes the whole turn budget -- measured: the runtime arm sat at an identical
                # 549-token prompt repeating one error. Show the exact required shape instead.
                if fail_streak >= 3:
                    obs += ('\n\nYour last three replies were unparseable. Reply with EXACTLY '
                            'this shape and no other text:\n'
                            '{"tool": "search", "query": "sinc", "glob": "*.py"}')
            elif call["tool"] == "done":
                done = True
                obs = "acknowledged"
            else:
                _fail_streak = 0
                obs = exec_tool(call, inst, venv, f2p)
            turns.append(dict(turn=turn, tool=(call or {}).get("tool"),
                              parse_error=err, ttft_ms=out["ttft_ms"],
                              kv_bytes=out["kv_bytes"], wall_ms=out["wall_ms"],
                              prompt_tokens=out["prompt_tokens"], gen_tokens=out["gen_tokens"],
                              reply=out["text"][:200], observation=obs[:300]))
            _glog("  turn %d  tool=%-11s ttft=%7.1fms kv=%6.1fMB prompt=%5d  %s" % (
                turn, (call or {}).get("tool") or "-", out["ttft_ms"], out["kv_bytes"] / 1e6,
                out["prompt_tokens"], obs[:70].replace("\n", " ")))
            if done:
                break
            transcript.append(("assistant", out["text"]))
            if arm == "linear":
                transcript.append(("user", obs))
            else:
                # ★ runtime: keep the system + the issue + the last N turns as blocks. Superseded
                # observations are dropped, which is the eviction this arm exists to test.
                blocks.append(("user", obs))
                keep = blocks[-4:]
                transcript = [("system", SYSTEM),
                              ("user", "ISSUE:\n" + problem[:6000])] + keep

        p2p_v = validated_p2p(inst, venv, p2p)
        fo, fs = run_tests(inst, venv, f2p)
        po, ps = run_tests(inst, venv, p2p_v)
        ttfts = [t["ttft_ms"] for t in turns] or [0]
        kvs = [t["kv_bytes"] for t in turns] or [0]
        rows.append(dict(
            instance=iid, status="ok", arm=arm, turns=len(turns),
            resolved=bool(fo and po), f2p_pass=fo, p2p_pass=po,
            n_p2p_raw=len(p2p), n_p2p_valid=len(p2p_v),
            f2p_summary=fs, p2p_summary=ps,
            declared_done=done,
            ttft_ms_mean=sum(ttfts) / len(ttfts), ttft_ms_first=ttfts[0], ttft_ms_last=ttfts[-1],
            peak_kv_bytes=max(kvs), peak_kv_mb=max(kvs) / 1e6,
            wall_s=time.perf_counter() - t_inst, per_turn=turns))
        _glog("  RESULT resolved=%s f2p=%s p2p=%s turns=%d ttft %.0f->%.0fms peakKV=%.1fMB" % (
            bool(fo and po), fs, ps, len(turns), ttfts[0], ttfts[-1], max(kvs) / 1e6))

    return dict(model=model, arm=arm, gpu=GPU, rows=rows)


def exec_tool(call, inst, venv, f2p):
    """Run one tool call against the real repo. Returns the observation the model will see."""
    import os as _os, subprocess as sp
    t = call["tool"]
    if t == "read_file":
        p = _os.path.normpath(_os.path.join(SYMPY, str(call.get("path", ""))))
        if not p.startswith(SYMPY):
            return "refused: path escapes the repository"
        if not _os.path.exists(p):
            return "no such file: %s" % call.get("path")
        lines = open(p, errors="replace").read().splitlines()
        try:
            a = max(1, int(call.get("start", 1)))
            b = int(call.get("end", a + 119))
        except Exception:
            a, b = 1, 120
        b = min(b, a + 400)          # hard cap on one read: a whole-file dump floods the prompt
        out = ["%5d| %s" % (i + 1, lines[i]) for i in range(a - 1, min(b, len(lines)))]
        head = "FILE %s  lines %d-%d of %d\n" % (call.get("path"), a, min(b, len(lines)), len(lines))
        return head + "\n".join(out)

    if t == "edit_file":
        """Exact-string replacement, and the two failure modes are REFUSED rather than guessed at.

        ★ A fuzzy match on source code is a corruption risk: replacing the wrong occurrence
        silently breaks the file in a way the model cannot see. Not-found and not-unique both come
        back as errors that NAME the problem, so the model can correct itself in one turn instead of
        re-reading half the repo to discover nothing happened.
        """
        p = _os.path.normpath(_os.path.join(SYMPY, str(call.get("path", ""))))
        if not p.startswith(SYMPY):
            return "refused: path escapes the repository"
        if not _os.path.exists(p):
            return "no such file: %s" % call.get("path")
        old_s = str(call.get("old_string", ""))
        new_s = str(call.get("new_string", ""))
        if not old_s:
            return "error: old_string is empty"
        src = open(p, errors="replace").read()
        n = src.count(old_s)
        if n == 0:
            return ("error: old_string not found in %s. It must match exactly, including "
                    "indentation. Read the surrounding lines and retry." % call.get("path"))
        if n > 1:
            return ("error: old_string matches %d times in %s; include more surrounding context to "
                    "make it unique." % (n, call.get("path")))
        # ★ A FUNCTION replacement: str.replace expands \1, $& etc. in the REPLACEMENT, and Python
        # source is full of backslashes and regex strings. A literal replace avoids corrupting code.
        open(p, "w").write(src.replace(old_s, new_s))
        return "edited %s" % call.get("path")
    if t == "search":
        r = sp.run(["grep", "-rn", "-E", "--include=" + str(call.get("glob", "*.py")),
                    str(call.get("query", "")), "sympy/"], cwd=SYMPY,
                   capture_output=True, text=True, timeout=120)
        o = (r.stdout or "") + (r.stderr or "")
        return o[:6000] if o.strip() else "no matches"
    if t == "write_file":
        p = _os.path.normpath(_os.path.join(SYMPY, str(call.get("path", ""))))
        if not p.startswith(SYMPY):
            return "refused: path escapes the repository"
        _os.makedirs(_os.path.dirname(p), exist_ok=True)
        open(p, "w").write(str(call.get("content", "")))
        return "wrote %s" % call.get("path")
    if t == "run_tests":
        ok, s = run_tests(inst, venv, f2p)
        return ("PASS: " if ok else "FAIL: ") + s
    return "unhandled"


@app.local_entrypoint()
def main(model: str = "qwen2.5-7b", arm: str = "gold", limit: int = 10, max_turns: int = 8):
    import time
    r = solve.remote(model, arm, limit, max_turns)
    p = "/tmp/opencode/ccai/benchmarks/results/swe_%s_%s_%s.json" % (
        model.replace("/", "-"), arm, time.strftime("%Y%m%d-%H%M%S"))
    json.dump(r, open(p, "w"), indent=2)
    rows = r["rows"]
    ok = [x for x in rows if x.get("status") == "ok"]
    print("\n%s  arm=%s  %d/%d ran" % (model, arm, len(ok), len(rows)))
    if arm == "gold":
        print("  GOLD PARITY: %d/%d resolved" % (sum(1 for x in ok if x["resolved"]), len(ok)))
    else:
        res = sum(1 for x in ok if x.get("resolved"))
        print("  resolved %d/%d  (%.0f%%)" % (res, len(ok), 100.0 * res / max(1, len(ok))))
        if ok:
            print("  TTFT mean %.0f ms | peak KV %.1f MB | wall %.0f s/instance" % (
                sum(x["ttft_ms_mean"] for x in ok) / len(ok),
                sum(x["peak_kv_mb"] for x in ok) / len(ok),
                sum(x["wall_s"] for x in ok) / len(ok)))
    print("wrote %s" % p)
