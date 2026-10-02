#!/usr/bin/env bash
# ★★ v2 — THE CONTROLLED EXPERIMENT.
# Both arms ANCHOR TURN 1 (owner's requirement), so both can do every fix. The ONLY difference is
# how each handles the accumulating history:
#   linear  — keeps the recent user+assistant PAIRS (the assistant half is a redundant full-file
#             rewrite, ~1700 tok), so the brief and old instructions fall out of the window.
#   runtime — archives the INSTRUCTIONS (~40 tok each, the only home of each turn's code) plus the
#             single most recent reply, so the prompt stays small AND keeps everything.
set -u
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok)"
N="${CCAI_FEATURES:-80}"
LOG=/root/runfix2.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== FIX-MODE v2 (both anchor turn 1)  N=$N  gemma-4-12b bf16 ==="
for ARM in linear runtime; do
  rm -rf /work/inc /root/ccai/ckpt_$ARM
  say "--- arm=$ARM starting ---"
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=$ARM CCAI_FEATURES=$N CCAI_TIME_BUDGET_S=21600 CCAI_CKPT_DIR=/root/ccai/ckpt_$ARM \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm2_$ARM.log 2>&1
  rc=$?
  say "arm=$ARM exit=$rc"
  grep -aE "PASSED|PER-TURN|GRADER FAILED" /root/arm2_$ARM.log | tail -3 | tee -a "$LOG"
done
say "=== BOTH ARMS DONE ==="
ls -la /root/ccai/results/ | tee -a "$LOG"
