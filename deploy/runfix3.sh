#!/usr/bin/env bash
# ★★ v3 — THREE ARMS, ONE RUN, THE WHOLE METRIC SUITE.
#
# Owner, verbatim: *"research the top 20 metrics to measure and measure them all, ion want u wasting
# cloud usage without extracting all the data possible"* and *"also prove this works by having every
# 30 turns, a manual context editing to remove bloat, this will showcase the true efficiency"*.
#
# So one rented box does all three policies back to back and every number the harness can produce
# is written to results/, rather than paying for a second box per question.
#
# ARMS
#   runtime — anchor turn 1 + archived INSTRUCTIONS (~40 tok each) + the latest reply, bounded by
#             RUNTIME_TOKENS. Flat context.
#   linear  — anchor turn 1 + as many recent user/assistant PAIRS as fit 12k. Grows, then front-evicts.
#   prune   — identical to linear, PLUS a context compaction every PRUNE_EVERY (30) turns. This is
#             the status quo: it is what long-running agents do today, and it is the operation that
#             should destroy the prefix cache.
#
# ORDER: runtime first. It is the fastest and it is the novel arm, so if the box dies mid-run the
# headline result already exists. A partial run beats none.
set -u
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok 2>/dev/null || echo '')"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N="${CCAI_FEATURES:-60}"
LOG=/root/runfix3.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== FIX-MODE v3  3 arms  N=$N  gemma-4-12b bf16  gate=ON  prune_every=30 ==="
for ARM in runtime linear prune; do
  rm -rf /work/inc
  mkdir -p /root/ccai/ckpt_$ARM
  say "--- arm=$ARM starting ---"
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=$ARM CCAI_FEATURES=$N CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
  CCAI_TIME_BUDGET_S="${CCAI_TIME_BUDGET_S:-10800}" CCAI_CKPT_DIR=/root/ccai/ckpt_$ARM \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm3_$ARM.log 2>&1
  rc=$?
  say "arm=$ARM exit=$rc"
  grep -aE "PASSED|PER-TURN|GRADER FAILED" /root/arm3_$ARM.log | tail -2 | tee -a "$LOG"
done
say "=== ALL ARMS DONE ==="
ls -la /root/ccai/results/ | tee -a "$LOG"
