"""SWE-bench Lite evaluation: linear prefill vs the block runtime, on real repository instances.

    python3 modal/swe_bench.py --list
    python3 modal/swe_bench.py --model qwen2.5-7b --arm linear     (Modal: modal run modal/swe_bench.py ...)
    python3 modal/swe_bench.py --grade <run.json>

★ WHY THIS IS BUILT THE WAY IT IS.

A SWE-bench number is only meaningful if the task genuinely fails before the fix. Every criterion
below exists because without it the table would be empty or false:

  1. NON-VACUITY, CHECKED PER INSTANCE. For each instance the dataset's own `test_patch` is applied
     to the UNFIXED tree and the FAIL_TO_PASS test must FAIL, then the gold `patch` is applied and it
     must PASS. An instance that passes before the fix measures nothing and is dropped, loudly.
     This is the gate the earlier Frankenstein run lacked: its easy set scored 100%/100% on
     attempt 1, so the repair loop never ran and the whole experiment measured the absence of work.

  2. THE GOLD PATCH IS THE CEILING, NOT THE BASELINE. `--arm gold` runs the same loop with the
     dataset's own patch available, so if gold does not resolve the instance the harness is broken
     and the model numbers mean nothing. A SWE-bench run without a working gold arm cannot tell
     "the model failed" from "the environment is wrong".

  3. PASS_TO_PASS IS ENFORCED. Editing one function is not hard; not breaking 58 others is the task.
     A patch that fixes F2P while breaking P2P is a failure, which is what the real leaderboard does.

  4. NO INTERNET IN THE SOLVER. A model that can `pip install` its way around the problem or look
     up the fix has not resolved anything.
"""
import json, os, subprocess, sys, time, textwrap

ROOT = "/tmp/opencode/sweval"
SYMPY = os.path.join(ROOT, "sympy")


def instances():
    out = []
    for f in sorted(os.listdir(ROOT)):
        if f.startswith("sympy__") and f.endswith(".json"):
            d = json.load(open(os.path.join(ROOT, f)))
            d["_venv"] = os.path.join(ROOT, "venv_" + d["instance_id"])
            out.append(d)
    return out


def sh(args, cwd=None, timeout=900, env=None):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def reset_tree(inst):
    """base_commit + test_patch. The test_patch is REQUIRED before any run: it adds the tests that
    encode the fix, so without it a patch could pass by changing the tests instead of the code."""
    sh(["git", "checkout", "-q", "--", "."], cwd=SYMPY)
    sh(["git", "clean", "-qfd"], cwd=SYMPY)
    rc, o = sh(["git", "checkout", "-q", "-f", inst["base_commit"]], cwd=SYMPY)
    if rc != 0:
        return False, "checkout failed: " + o[-300:]
    tp = os.path.join(ROOT, "tp_%s.patch" % inst["instance_id"])
    if os.path.exists(tp):
        rc, o = sh(["git", "apply", "--whitespace=nowarn", tp], cwd=SYMPY)
        if rc != 0:
            return False, "test_patch failed: " + o[-300:]
    return True, "ok"


def patch_test_files(inst):
    """The FILES the dataset's own `test_patch` modifies -- the authoritative scope for its tests.

    ★ THIS IS THE FIX FOR A REAL BUG IN THIS FILE. SWE-bench stores FAIL_TO_PASS as `path::name`
    OR as a bare test name, and for a bare name the first version grep'd the repo for `def <name>(`
    and took the SHORTEST match. That is wrong: `test_sinc` and `test_Derivative` exist in more
    than one place, the shortest match is not the one the dataset means, and the harness then ran a
    DIFFERENT, always-passing test -- so 4 of 10 instances reported VACUOUS ("passes before the
    fix") when in fact the test was simply the wrong one. The test_patch cannot be ambiguous: it is
    the diff SWE-bench authored for this instance. Parse it.
    """
    import re
    return re.findall(r"^\+\+\+ b/(.+)$", inst.get("test_patch", ""), re.M)


def resolve_test_id(inst, spec):
    """Resolve an F2P/P2P entry to a runnable `path::name`, scoped by the test_patch's files."""
    if "::" in spec:
        return spec
    cands = []
    for f in patch_test_files(inst):
        if os.path.exists(os.path.join(SYMPY, f)):
            rc, o = sh(["grep", "-n", "def %s(" % spec, f], cwd=SYMPY, timeout=60)
            if o.strip():
                cands.append("%s::%s" % (f, spec))
    if cands:
        return cands[0]
    # fall back to a repo-wide search only when the test_patch names no file for this entry
    rc, o = sh(["grep", "-rl", "--include=*.py", "def %s(" % spec, "."], cwd=SYMPY)
    files = [l for l in o.splitlines() if l.strip().startswith("./sympy")]
    if not files:
        return None
    return "%s::%s" % (sorted(files, key=len)[0].lstrip("./"), spec)


def run_tests(inst, ids, venv):
    """Run a set of test ids in the instance's own venv. Returns (ok, summary)."""
    if not ids:
        return True, "no tests"
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        return False, "no venv"
    rc, o = sh([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + ids,
               cwd=SYMPY, timeout=1200)
    tail = [l for l in o.splitlines() if "passed" in l or "failed" in l or "error" in l]
    return rc == 0, (tail[-1] if tail else o[-200:])


def check_instance(inst, venv=None, quiet=False):
    """Apply the dataset patch -> F2P must fail. Apply gold -> F2P and P2P must pass."""
    venv = venv or inst["_venv"]
    f2p_raw = json.loads(inst["FAIL_TO_PASS"])
    p2p_raw = json.loads(inst["PASS_TO_PASS"])
    ok, msg = reset_tree(inst)
    if not ok:
        return dict(instance=inst["instance_id"], status="setup_failed", detail=msg)

    f2p = [x for x in (resolve_test_id(inst, s) for s in f2p_raw) if x]
    if not f2p:
        return dict(instance=inst["instance_id"], status="unresolved_test_id",
                    detail=str(f2p_raw[:2]))

    pre_ok, pre_sum = run_tests(inst, f2p, venv)
    if pre_ok:
        return dict(instance=inst["instance_id"], status="VACUOUS",
                    detail="F2P passes before the fix: %s" % pre_sum)

    gold = os.path.join(ROOT, "gold_%s.patch" % inst["instance_id"])
    rc, o = sh(["git", "apply", "--whitespace=nowarn", gold], cwd=SYMPY)
    if rc != 0:
        return dict(instance=inst["instance_id"], status="gold_apply_failed", detail=o[-200:])
    post_ok, post_sum = run_tests(inst, f2p, venv)
    return dict(instance=inst["instance_id"], status="ok" if post_ok else "gold_not_resolving",
                f2p=f2p, pre=pre_sum, post=post_sum,
                n_p2p=len(p2p_raw), p2p_raw=p2p_raw)


# ---------------------------------------------------------------------------------------------
# The solving loop. Two arms that differ ONLY in how the conversation history is held:
#   linear   -- the whole transcript re-prefilled each turn (standard KV cache growth)
#   runtime  -- the transcript assembled from a block table, superseded turns evicted
# ---------------------------------------------------------------------------------------------

SYSTEM = textwrap.dedent("""\
You are a software engineer fixing a real bug in the sympy repository.
You will be shown the issue and, on later turns, the output of the tests you ran.

Reply with EXACTLY ONE of these, and nothing else:

  To read a file:      {"tool": "read_file", "path": "<path relative to repo root>"}
  To search the repo:  {"tool": "search", "query": "<regex>", "glob": "*.py"}
  To write a file:     {"tool": "write_file", "path": "<path>", "content": "<full new contents>"}
  To run the tests:    {"tool": "run_tests"}
  When finished:       {"tool": "done"}

Rules:
  - write_file replaces the WHOLE file; include every line you are keeping.
  - run the tests before declaring done. A patch is finished only when they pass.
  - you cannot install packages and you have no internet.
""")

TOOLS = ("read_file", "search", "write_file", "run_tests", "done")


def parse_call(text):
    """Extract one tool call. Rejects anything that is not a single JSON object.

    ★ A parse that silently accepts a malformed call is indistinguishable from a model that
    cannot work the protocol -- that exact ambiguity hid a real bug for a whole session in the
    webchat harness. So a call that fails to parse is reported back verbatim.
    """
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.replace("json", "", 1).strip() if p.strip().startswith("json") else p.strip()
            if p.startswith("{"):
                t = p
                break
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None, "no JSON object in the reply"
    try:
        d = json.loads(t[i:j + 1])
    except Exception as e:
        return None, "invalid JSON: %s" % e
    if not isinstance(d, dict) or "tool" not in d:
        return None, "no 'tool' field"
    if d["tool"] not in TOOLS:
        return None, "unknown tool %r" % d.get("tool")
    return d, None


def exec_call(call, inst, venv, max_file=60000):
    """Execute one tool call against the real repo. Returns (observation, done)."""
    t = call["tool"]
    if t == "done":
        return "done", True
    if t == "read_file":
        p = os.path.normpath(os.path.join(SYMPY, str(call.get("path", ""))))
        if not p.startswith(SYMPY):
            return "refused: path escapes the repository", False
        if not os.path.exists(p):
            return "no such file: %s" % call.get("path"), False
        try:
            s = open(p, errors="replace").read()
        except Exception as e:
            return "read failed: %s" % e, False
        if len(s) > max_file:
            s = s[:max_file] + "\n... [truncated, %d more chars]" % (len(s) - max_file)
        return s, False
    if t == "search":
        q = str(call.get("query", ""))
        g = str(call.get("glob", "*.py"))
        rc, o = sh(["grep", "-rn", "-E", "--include=" + g, q, "sympy/"], cwd=SYMPY, timeout=120)
        return (o[:8000] if o.strip() else "no matches"), False
    if t == "write_file":
        p = os.path.normpath(os.path.join(SYMPY, str(call.get("path", ""))))
        if not p.startswith(SYMPY):
            return "refused: path escapes the repository", False
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(str(call.get("content", "")))
        except Exception as e:
            return "write failed: %s" % e, False
        return "wrote %s" % call.get("path"), False
    if t == "run_tests":
        f2p = [x for x in (resolve_test_id(inst, s) for s in json.loads(inst["FAIL_TO_PASS"])) if x]
        ok, s = run_tests(inst, f2p, venv)
        return ("PASS: " if ok else "FAIL: ") + s, False
    return "unhandled", False


def grade(inst, venv=None):
    """The real criterion: F2P all pass AND no P2P regressed."""
    venv = venv or inst["_venv"]
    f2p = [x for x in (resolve_test_id(inst, s) for s in json.loads(inst["FAIL_TO_PASS"])) if x]
    ok_f, s_f = run_tests(inst, f2p, venv)
    p2p = [x for x in (resolve_test_id(inst, s) for s in json.loads(inst["PASS_TO_PASS"])) if x]
    ok_p, s_p = run_tests(inst, p2p, venv)
    return dict(f2p_pass=ok_f, p2p_pass=ok_p, resolved=bool(ok_f and ok_p),
                f2p_summary=s_f, p2p_summary=s_p)


if __name__ == "__main__":
    if "--list" in sys.argv:
        for i in instances():
            print("%-30s venv=%s f2p=%s" % (
                i["instance_id"], "yes" if os.path.exists(i["_venv"]) else "NO",
                json.loads(i["FAIL_TO_PASS"])))
    elif "--check" in sys.argv:
        for i in instances():
            r = check_instance(i)
            print("%-30s %-22s %s" % (r["instance"], r["status"], r.get("post", r.get("detail", ""))))
    else:
        print(__doc__)
