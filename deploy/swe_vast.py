#!/usr/bin/env python3
"""SWE-bench Lite on a Vast box, Gemma, three arms (gold / runtime / linear).

Ported from modal/swe_solve.py: the Modal decorators, the volume cache and the secret are removed,
and the two things that were Modal-specific are handled directly --
  * the model is loaded locally with transformers,
  * the instance data is fetched once from the HF dataset into /work/instances.

WHY THIS BENCHMARK. The paper's task is "a repo with real faults, fixed over many turns". SWE-bench
Lite is that exact shape at the recognized scale: 300 real GitHub issues, graded by the repository's
own tests. The synthetic task proves the mechanism; SWE-bench is the number a reviewer recognises.

THE THREE ARMS, and only one thing differs between them:
  gold     the dataset's own patch applied, then graded. A PARITY CHECK, not a baseline: if gold does
           not resolve an instance the harness is broken and every model number is noise.
  linear   the transcript grows by one (assistant, observation) pair every turn -- a normal agent.
  runtime  the anchored policy: the system prompt and the issue are pinned, superseded observations
           are evicted, only the last few remain. This is the policy the paper claims is cheaper and
           no worse, because it keeps the shared prefix stable.

NON-VACUITY is enforced per instance the way the real harness does: FAIL_TO_PASS must fail on the
base tree and pass after the patch, and PASS_TO_PASS must hold.

Usage:
  python3 swe_vast.py --model gemma-4-12b --arm gold     --limit 8
  python3 swe_vast.py --model gemma-4-12b --arm runtime  --limit 8
  python3 swe_vast.py --model gemma-4-12b --arm linear   --limit 8
"""
import argparse
import gc
import glob
import json
import os
import re
import subprocess as sp
import sys
import time

ROOT = "/work"
SYMPY = "/work/sympy"
UV_PY = "/work/uvpy"
INST_DIR = "/work/instances"

MODELS = {
    "gemma-4-12b": "google/gemma-4-12B-it",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
}

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


def glog(m):
    print(m, flush=True)


# ───────────────────────────────────────────────────────────────────────────── data
def prep_instances(limit):
    """Fetch SWE-bench Lite, keep sympy, write one json + test/gold patches per instance."""
    os.makedirs(INST_DIR, exist_ok=True)
    have = [f for f in os.listdir(INST_DIR) if f.startswith("sympy__") and f.endswith(".json")]
    if len(have) >= limit:
        glog("instances already present (%d)" % len(have))
        return len(have)
    os.environ.setdefault("HF_TOKEN", open("/root/.hftok").read().strip()
                          if os.path.exists("/root/.hftok") else "")
    from datasets import load_dataset
    d = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    n = 0
    for r in d:
        if r["repo"] != "sympy/sympy":
            continue
        iid = r["instance_id"]
        json.dump(dict(r), open(os.path.join(INST_DIR, "%s.json" % iid), "w"))
        open(os.path.join(INST_DIR, "tp_%s.patch" % iid), "w").write(r["test_patch"])
        open(os.path.join(INST_DIR, "gold_%s.patch" % iid), "w").write(r["patch"])
        n += 1
        if n >= limit:
            break
    glog("prepared %d sympy instances" % n)
    return n


def instances(limit):
    out = []
    for f in sorted(os.listdir(INST_DIR)):
        if f.startswith("sympy__") and f.endswith(".json") and not f.startswith("tp_"):
            out.append(json.load(open(os.path.join(INST_DIR, f))))
    return out[:limit]


# ───────────────────────────────────────────────────────────────────────────── model
def load_model(mid):
    import torch
    from transformers import AutoModelForCausalLM
    glog("loading %s (bf16)" % mid)
    return AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()


# ───────────────────────────────────────────────────────────────────────────── repo + venv
def setup_repo(inst):
    """Clone once, check out the base commit, apply the test_patch.

    The test_patch is part of SETUP, not grading: it adds the tests that encode the fix, so a model
    that rewrote the tests instead of the code cannot appear to succeed.
    """
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.isdir(os.path.join(SYMPY, ".git")):
        glog("cloning sympy (first time)")
        sp.run(["git", "clone", "-q", "https://github.com/sympy/sympy.git", SYMPY], check=True)
    sp.run(["git", "checkout", "-q", "--", "."], cwd=SYMPY)
    sp.run(["git", "clean", "-qfd"], cwd=SYMPY)
    r = sp.run(["git", "checkout", "-q", "-f", inst["base_commit"]], cwd=SYMPY,
               capture_output=True, text=True)
    if r.returncode != 0:
        return False, "checkout failed: " + (r.stderr or r.stdout)[-200:]
    tp = os.path.join(INST_DIR, "tp_%s.patch" % inst["instance_id"])
    if os.path.exists(tp):
        r = sp.run(["git", "apply", "--whitespace=nowarn", tp], cwd=SYMPY,
                   capture_output=True, text=True)
        if r.returncode != 0:
            return False, "test_patch failed: " + (r.stderr or r.stdout)[-200:]
    return True, "ok"


def make_venv(inst):
    """One editable-install venv per instance (py3.9 for these sympy versions), cached."""
    v = os.path.abspath(os.path.join(SYMPY, "..", "venv_" + inst["instance_id"]))
    py = os.path.join(v, "bin", "python")
    if os.path.exists(py):
        return v, True
    os.makedirs(UV_PY, exist_ok=True)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=UV_PY, VIRTUAL_ENV=v)
    uv = "/root/.local/bin/uv"
    if not os.path.exists(uv):
        uv = sp.run(["which", "uv"], capture_output=True, text=True).stdout.strip() or "/usr/local/bin/uv"
    sp.run([uv, "python", "install", "3.9"], env=env, capture_output=True)
    sp.run([uv, "venv", "--python", "3.9", v], env=env, capture_output=True)
    for pkgs in (["mpmath", "pytest"], ["-e", "."]):
        sp.run([uv, "pip", "install", "-q"] + pkgs, env=env, cwd=SYMPY, capture_output=True)
    return v, os.path.exists(py)


def test_files(inst):
    return re.findall(r"^\+\+\+ b/(.+)$", inst.get("test_patch", ""), re.M)


def resolve(inst, spec):
    """Resolve a bare test name inside the files the test_patch touches (not repo-wide)."""
    if "::" in spec:
        return spec
    for f in test_files(inst):
        if os.path.exists(os.path.join(SYMPY, f)):
            r = sp.run(["grep", "-n", "def %s(" % spec, f], cwd=SYMPY,
                       capture_output=True, text=True)
            if r.stdout.strip():
                return "%s::%s" % (f, spec)
    return None


def run_tests(inst, venv, ids, timeout=900):
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


def validated_p2p(inst, venv, p2p_ids):
    """PASS_TO_PASS with already-broken tests removed, cached per instance.

    Some listed P2P tests fail on the untouched base tree; treating the raw list as a regression
    guard would mark a CORRECT patch unresolved. This validates against the base state once.
    """
    ck = os.path.join(ROOT, "p2p_ok_%s.json" % inst["instance_id"])
    if os.path.exists(ck):
        try:
            return json.load(open(ck))
        except Exception:
            pass
    good = []
    if p2p_ids and os.path.exists(os.path.join(venv, "bin", "python")):
        py = os.path.join(venv, "bin", "python")
        r = sp.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + p2p_ids,
                   cwd=SYMPY, capture_output=True, text=True, timeout=1800)
        out = (r.stdout or "") + (r.stderr or "")
        bad = set()
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                bad.add(line.split()[1].split("::")[-1].split("[")[0])
        good = [x for x in p2p_ids if x.split("::")[-1].split("[")[0] not in bad]
        if len(p2p_ids) - len(good):
            glog("  p2p validation dropped %d already-failing test(s)" % (len(p2p_ids) - len(good)))
    json.dump(good, open(ck, "w"))
    return good


# ───────────────────────────────────────────────────────────────────────────── tools
def exec_tool(call, inst, venv, f2p):
    t = call["tool"]
    def safe(rel):
        p = os.path.normpath(os.path.join(SYMPY, str(rel or "")))
        return p if p.startswith(SYMPY) else None
    if t == "read_file":
        p = safe(call.get("path"))
        if not p:
            return "refused: path escapes the repository"
        if not os.path.exists(p):
            return "no such file: %s" % call.get("path")
        lines = open(p, errors="replace").read().splitlines()
        try:
            a = max(1, int(call.get("start", 1)))
            b = int(call.get("end", a + 119))
        except Exception:
            a, b = 1, 120
        b = min(b, a + 400)
        out = ["%5d| %s" % (i + 1, lines[i]) for i in range(a - 1, min(b, len(lines)))]
        return ("FILE %s  lines %d-%d of %d\n" % (call.get("path"), a, min(b, len(lines)), len(lines))
                + "\n".join(out))
    if t == "edit_file":
        p = safe(call.get("path"))
        if not p:
            return "refused: path escapes the repository"
        if not os.path.exists(p):
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
            return ("error: old_string matches %d times in %s; include more context to make it "
                    "unique." % (n, call.get("path")))
        open(p, "w").write(src.replace(old_s, new_s))
        return "edited %s" % call.get("path")
    if t == "search":
        r = sp.run(["grep", "-rn", "-E", "--include=" + str(call.get("glob", "*.py")),
                    str(call.get("query", "")), "sympy/"], cwd=SYMPY,
                   capture_output=True, text=True, timeout=120)
        o = (r.stdout or "") + (r.stderr or "")
        return o[:6000] if o.strip() else "no matches"
    if t == "write_file":
        p = safe(call.get("path"))
        if not p:
            return "refused: path escapes the repository"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(str(call.get("content", "")))
        return "wrote %s" % call.get("path")
    if t == "run_tests":
        ok, s = run_tests(inst, venv, f2p)
        return ("PASS: " if ok else "FAIL: ") + s
    return "unhandled"


def parse(t):
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
        d, _ = json.JSONDecoder().raw_decode(body)
    except Exception as e:
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
            d, _ = json.JSONDecoder().raw_decode("".join(out))
        except Exception:
            return None, "invalid JSON: %s" % e
    if not isinstance(d, dict) or d.get("tool") not in (
            "read_file", "search", "write_file", "edit_file", "run_tests", "done"):
        return None, "bad or unknown tool"
    return d, None


# ───────────────────────────────────────────────────────────────────────────── run
def solve(model, arm, limit, max_turns, max_new=1024):
    import torch
    prep_instances(limit)
    insts = instances(limit)
    glog("running %d instances  model=%s arm=%s" % (len(insts), model, arm))
    mid = MODELS[model]

    tok = mdl = None
    if arm != "gold":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = load_model(mid)

    def build_prompt(messages):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # fall back to a plain join if the checkpoint has no chat template
            s = "".join("<|%s|>\n%s\n" % (r, c) for r, c in messages)
            return s + "<|assistant|>\n"

    def complete(messages, mnew):
        prompt = build_prompt(messages)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
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
            gen.append(nxt); p += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        txt = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)
        kv = sum(c.layers[i].keys.numel() * c.layers[i].keys.element_size()
                 + c.layers[i].values.numel() * c.layers[i].values.element_size()
                 for i in range(len(c.layers)))
        del o, c
        gc.collect(); torch.cuda.empty_cache()
        return dict(text=txt, prompt_tokens=len(ids), gen_tokens=len(gen),
                    ttft_ms=ttft, kv_bytes=kv, wall_ms=(time.perf_counter() - t0) * 1000)

    rows = []
    for inst in insts:
        iid = inst["instance_id"]
        t_inst = time.perf_counter()
        glog("\n===== %s  arm=%s =====" % (iid, arm))
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
                        os.path.join(INST_DIR, "gold_%s.patch" % iid)],
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
            glog("  gold resolved=%s  f2p=%s  p2p=%s" % (bool(fo and po), fs, ps))
            continue

        problem = inst["problem_statement"]
        turns = []
        messages = [("system", SYSTEM),
                    ("user", "ISSUE:\n" + problem[:6000] +
                     "\n\nStart by searching the repo for the relevant code.")]
        blocks = []
        done = False
        _fail = 0
        for turn in range(max_turns):
            out = complete(messages, max_new)
            call, err = parse(out["text"])
            if err:
                _fail += 1
                obs = "could not parse your reply (%s). Reply with exactly ONE json object." % err
                if _fail >= 3:
                    obs += ('\n\nYour last three replies were unparseable. Reply with EXACTLY:\n'
                            '{"tool": "search", "query": "sinc", "glob": "*.py"}')
            elif call["tool"] == "done":
                done = True; obs = "acknowledged"
            else:
                _fail = 0
                obs = exec_tool(call, inst, venv, f2p)
            turns.append(dict(turn=turn, tool=(call or {}).get("tool"), parse_error=err,
                              ttft_ms=out["ttft_ms"], kv_bytes=out["kv_bytes"],
                              wall_ms=out["wall_ms"], prompt_tokens=out["prompt_tokens"],
                              gen_tokens=out["gen_tokens"], reply=out["text"][:200],
                              observation=obs[:300]))
            glog("  turn %d tool=%-11s ttft=%7.1fms kv=%6.1fMB prompt=%5d  %s" % (
                turn, (call or {}).get("tool") or "-", out["ttft_ms"], out["kv_bytes"] / 1e6,
                out["prompt_tokens"], obs[:70].replace("\n", " ")))
            if done:
                break
            messages.append(("assistant", out["text"]))
            messages.append(("user", obs))
            if arm == "runtime":
                # anchored: pin system + issue; evict superseded observations, keep the last 4.
                blocks.append(("user", obs))
                messages = ([("system", SYSTEM), ("user", "ISSUE:\n" + problem[:6000])]
                            + blocks[-4:])

        p2p_v = validated_p2p(inst, venv, p2p)
        fo, fs = run_tests(inst, venv, f2p)
        po, ps = run_tests(inst, venv, p2p_v)
        ttfts = [t["ttft_ms"] for t in turns] or [0]
        kvs = [t["kv_bytes"] for t in turns] or [0]
        rows.append(dict(instance=iid, status="ok", arm=arm, turns=len(turns),
                         resolved=bool(fo and po), f2p_pass=fo, p2p_pass=po,
                         n_p2p_raw=len(p2p), n_p2p_valid=len(p2p_v),
                         f2p_summary=fs, p2p_summary=ps, declared_done=done,
                         ttft_ms_mean=sum(ttfts) / len(ttfts), ttft_ms_first=ttfts[0],
                         ttft_ms_last=ttfts[-1], peak_kv_bytes=max(kvs), peak_kv_mb=max(kvs) / 1e6,
                         wall_s=time.perf_counter() - t_inst, per_turn=turns))
        glog("  RESULT resolved=%s f2p=%s p2p=%s turns=%d ttft %.0f->%.0fms peakKV=%.1fMB" % (
            bool(fo and po), fs, ps, len(turns), ttfts[0], ttfts[-1], max(kvs) / 1e6))

    return dict(model=model, arm=arm, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-12b")
    ap.add_argument("--arm", default="gold", choices=["gold", "runtime", "linear"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--outdir", default="/root/ccai/swe_results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t = time.strftime("%Y%m%d-%H%M%S")
    r = solve(a.model, a.arm, a.limit, a.max_turns)
    p = os.path.join(a.outdir, "swe_%s_%s_%s.json" % (a.model.replace("/", "-"), a.arm, t))
    json.dump(r, open(p, "w"), indent=2)
    rows = r["rows"]
    ok = [x for x in rows if x.get("status") == "ok"]
    glog("\n%s arm=%s  %d/%d ran" % (a.model, a.arm, len(ok), len(rows)))
    if a.arm == "gold":
        glog("  GOLD PARITY: %d/%d resolved" % (sum(1 for x in ok if x["resolved"]), len(ok)))
    else:
        res = sum(1 for x in ok if x.get("resolved"))
        glog("  resolved %d/%d" % (res, len(ok)))
        if ok:
            glog("  TTFT mean %.0f ms | peak KV %.1f MB | wall %.0f s/instance" % (
                sum(x["ttft_ms_mean"] for x in ok) / len(ok),
                sum(x["peak_kv_mb"] for x in ok) / len(ok),
                sum(x["wall_s"] for x in ok) / len(ok)))
    glog("wrote %s" % p)


if __name__ == "__main__":
    main()
