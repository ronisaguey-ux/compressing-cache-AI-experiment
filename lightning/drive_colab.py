#!/usr/bin/env python3
"""Drive both arms of the incremental-coding benchmark on Colab, sequentially.

Why sequential and not parallel: one T4, and two 12B models at 4-bit would not co-reside in 15.6 GB.
Why a driver instead of two `colab exec` calls: both arms must share ONE kernel, and a single
process is the only way to guarantee the second arm starts only after the first has fully released
its GPU memory.

Run:  colab exec -s ccai -f drive_colab.py
"""
import os
import subprocess
import sys
import time

ARMS = ["linear", "runtime"]
TURNS = int(os.environ.get("TURNS", "40"))
WINDOW = os.environ.get("WINDOW", "6000")     # smaller window => pressure builds sooner => cheaper run
MODEL = os.environ.get("MODEL", "gemma-4-12b")
RUNNER = "/content/ccai/lightning/run_incremental.py"
BENCH = "/content/ccai/modal/incremental_coding.py"

assert os.path.exists(RUNNER) and os.path.exists(BENCH), "benchmark files not uploaded"

for arm in ARMS:
    log = "/content/%s_%s.log" % (MODEL, arm)
    env = dict(os.environ)
    env.update({
        "HF_HOME": "/content/hf",
        "CCAI_DEVICE": "cuda",
        "CCAI_QUANT": "4",                      # 12B at 4-bit = 7.7 GB, measured, fits the T4
        "CCAI_MAX_PROMPT_TOKENS": WINDOW,
        "CCAI_RUNTIME_TOKENS": "2048",
        "CCAI_WORK": "/content/inc_%s" % arm,   # separate work dir per arm so solution.py cannot mix
        "CCAI_CKPT_DIR": "/content/ckpt",       # checkpoints survive a Colab reclaim
        "CCAI_RESUME": "1",
        "PYTHONUNBUFFERED": "1",
    })
    cmd = [sys.executable, RUNNER, "--model", MODEL, "--arm", arm,
           "--features", str(TURNS), "--max-new", "3072"]
    print("### %s/%s turns=%d window=%s %s" % (MODEL, arm, TURNS, WINDOW, time.strftime("%H:%M:%S")),
          flush=True)
    with open(log, "w") as fh:
        p = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print("### %s exit=%d" % (arm, p.returncode), flush=True)
    # read back the one line that carries the actual measurement
    try:
        for line in open(log):
            if line.startswith("  CONTRACT") or line.startswith("  RUN-COMPLETE") \
                    or line.startswith("  HARNESS-FAILED"):
                print("   " + line.rstrip(), flush=True)
    except Exception:
        pass

print("COLAB-RUN-DONE", flush=True)
