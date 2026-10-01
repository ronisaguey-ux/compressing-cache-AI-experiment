#!/usr/bin/env python3
"""Run the incremental-coding benchmark on a plain GPU box (Lightning Studio, etc.).

The benchmark lives in `modal/incremental_coding.py` and is decorated with Modal. Rather than fork
the logic -- which would let the two copies drift and silently change what is being measured -- this
loads the *same file* with a minimal stub for the `modal` module. `app.function(...)` is a plain
decorator that returns the function unchanged, so `run_inc` below is the real, undecorated function.

Nothing about the experiment changes here: same prompts, same transcript policy, same grader.

Usage
-----
    python lightning/run_incremental.py --arm linear  --features 40
    python lightning/run_incremental.py --arm runtime --features 40
    CCAI_MAX_PROMPT_TOKENS=6000 python lightning/run_incremental.py --arm linear --features 40

Env (read by the benchmark itself)
----------------------------------
    CCAI_DEVICE             cuda | cpu                       (default cuda)
    CCAI_QUANT              auto | 4 | 8 | none              (default auto -> 4 for 32B)
    CCAI_MAX_PROMPT_TOKENS  linear arm's context ceiling     (default 12000)
    CCAI_RUNTIME_TOKENS     runtime arm's own small ceiling  (default 2048)
    CCAI_CKPT_DIR           checkpoint directory             (default /cache)
    CCAI_RESUME             1 to resume, 0 to start fresh    (default 1)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types

# ★ `__file__` IS NOT DEFINED WHEN THIS FILE IS TRANSMITTED TO A KERNEL. `colab exec -f` reads the
# file locally and sends its SOURCE to the Jupyter kernel as a cell, so the module has no path.
# NameError: name '__file__' is not defined -- which cost a cycle. Fall back to explicit locations,
# and allow an override so the benchmark can live anywhere.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.environ.get("CCAI_HOME", "/content/ccai/lightning")

_BENCH_ENV = os.environ.get("CCAI_BENCH")
BENCH = _BENCH_ENV or os.path.join(os.path.dirname(HERE), "modal", "incremental_coding.py")
if not os.path.exists(BENCH):
    for _c in ("/content/ccai/modal/incremental_coding.py",
               os.path.join(HERE, "incremental_coding.py")):
        if os.path.exists(_c):
            BENCH = _c
            break


def install_modal_stub():
    """A no-op `modal` module, installed BEFORE the benchmark is imported.

    Modal's decorators are pass-through for our purposes: @app.function(...) returns the function,
    @app.local_entrypoint() likewise. The image/volume/secret objects are only ever used as
    arguments, so they can be inert placeholders.
    """
    class _Inert:
        def __getattr__(self, _name):
            return lambda *a, **k: self

        def __call__(self, *a, **k):
            return self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        def local_entrypoint(self, *a, **k):
            def deco(fn):
                return fn
            return deco

    m = types.ModuleType("modal")
    m.App = App
    m.Image = _Inert()
    m.Volume = _Inert()
    m.Secret = _Inert()
    sys.modules["modal"] = m


def load_benchmark():
    install_modal_stub()
    spec = importlib.util.spec_from_file_location("incremental_coding", BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["incremental_coding"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    # ★ DEFAULTS COME FROM THE ENVIRONMENT so the script can be run with NO arguments.
    # `colab exec -f script.py -- --flag` is NOT supported ("Got unexpected extra argument(s)"),
    # and `colab exec` has no way to pass argv -- but it DOES have `--env KEY=VALUE`. So the
    # parameters that vary per run are read from env, and the same uploaded file works for both arms.
    ap.add_argument("--model", default=os.environ.get("CCAI_MODEL", "gemma-4-12b"))
    ap.add_argument("--arm", default=os.environ.get("CCAI_ARM", "linear"),
                    choices=["linear", "runtime"])
    ap.add_argument("--features", type=int, default=int(os.environ.get("CCAI_TURNS", "40")))
    ap.add_argument("--keep-turns", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--out", default=None)
    # ★ parse_known_args, NOT parse_args. When this file is transmitted to a Colab kernel, the
    # Jupyter launcher injects its OWN argv (measured: "-f /root/.local/share/jupyter/runtime/
    # kernel-<id>.json"), which argparse rejects with "unrecognized arguments" and the run dies
    # before it starts. We only care about our own flags; anything else in argv belongs to the
    # kernel host and must be ignored.
    a, _unknown = ap.parse_known_args()

    mod = load_benchmark()
    print("[run] model=%s arm=%s features=%d  max_prompt=%s runtime=%s device=%s quant=%s"
          % (a.model, a.arm, a.features, os.environ.get("CCAI_MAX_PROMPT_TOKENS", "12000"),
             os.environ.get("CCAI_RUNTIME_TOKENS", "2048"),
             os.environ.get("CCAI_DEVICE", "cuda"), os.environ.get("CCAI_QUANT", "auto")),
          flush=True)

    t0 = time.time()
    r = mod.run_inc(a.model, a.arm, a.features, a.keep_turns, a.max_new)
    elapsed = time.time() - t0

    out = a.out or os.path.join(os.environ.get("CCAI_OUT", HERE), "results", "incremental_%s_%s_%s.json"
                                % (a.model.replace("/", "-"), a.arm,
                                   time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r["elapsed_s"] = elapsed
    json.dump(r, open(out, "w"), indent=2)

    # ★ PRINT THE ONE LINE THAT MATTERS. The contract + nonce are the discriminator; a run that
    # reports only a feature count hides the actual measurement.
    print("\n%s  arm=%s  features=%d" % (r["model"], r["arm"], r["features"]))
    if r.get("stopped_early") is not None:
        print("  PARTIAL: stopped at turn %s after %.0f s"
              % (r["stopped_early"], r.get("seconds", 0)))
    pf = r.get("per_feature") or {}
    print("  PASSED %d/%d" % (r.get("passed", 0), r.get("total", 0)))
    print("  CONTRACT(REGISTRY)=%s   NONCE(early instruction)=%s"
          % (pf.get("contract"), pf.get("nonce")))
    print("  TTFT %6.0f -> %6.0f ms    prompt %5d -> %5d tok"
          % (r.get("ttft_first", 0), r.get("ttft_last", 0),
             r.get("prompt_first", 0), r.get("prompt_last", 0)))
    print("  elapsed %.0f s   wrote %s" % (elapsed, out))
    # ★ THE MARKER MUST NOT BE UNCONDITIONAL. An earlier version printed a bare "PASSED" every
    # time, and the launcher greps for that word -- so a run whose grader crashed (0/0, empty
    # results) still looked like a completed experiment. The marker now reports what actually
    # happened, and a harness failure is named as such.
    if r.get("harness_failed"):
        print("  HARNESS-FAILED — the grader did not run; this is NOT a model measurement")
    else:
        print("  RUN-COMPLETE arm=%s passed=%d/%d contract=%s nonce=%s"
              % (r["arm"], r.get("passed", 0), r.get("total", 0),
                 pf.get("contract"), pf.get("nonce")), flush=True)


if __name__ == "__main__":
    main()
