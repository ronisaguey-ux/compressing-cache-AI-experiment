#!/usr/bin/env bash
# Fix-mode A/B: identical turn count on both arms, identical instruction. The only difference is
# the context policy (linear prefill vs anchored runtime).
set -u
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok)"
N="${CCAI_FEATURES:-80}"
LOG=/root/runfix.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== FIX-MODE RUN  N=$N  model=gemma-4-12b bf16 ==="
for ARM in linear runtime; do
  rm -rf /work/inc /root/ccai/ckpt_$ARM
  say "--- arm=$ARM starting ---"
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=$ARM CCAI_FEATURES=$N CCAI_TIME_BUDGET_S=18000 CCAI_CKPT_DIR=/root/ccai/ckpt_$ARM \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm_$ARM.log 2>&1
  rc=$?
  say "arm=$ARM exit=$rc"
  grep -aE "PASSED|GRADER FAILED|PARTIAL" /root/arm_$ARM.log | tail -3 | tee -a "$LOG"
  [ $rc -ne 0 ] && say "WARN: arm $ARM exited nonzero"
done
say "=== BOTH ARMS DONE ==="
ls -la /root/ccai/results/ | tee -a "$LOG"
