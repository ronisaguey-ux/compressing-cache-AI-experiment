#!/usr/bin/env bash
# ★★★ RE-RUN THE RUNTIME ARM WITH THE ADV-010 FIX.
#
# WHY THIS EXISTS: the reported runtime numbers were produced by code that carried the turn-1 brief
# TWICE every turn (`_arch_instr[0]` is `transcript[0]`). The fix is committed, so the committed code
# no longer reproduces the paper's numbers -- and "logs that disagree with reported numbers" is a
# named rejection trigger. Re-running is the honest fix; reverting and documenting would leave the
# defect in place.
#
# Runs AFTER the 31B (its completion marker), because the 31B is the competition checkpoint and the
# box is serial.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
LOG=/home/roni-saguey/.local/share/ccai-results/chain_runtime_fixed.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "waiting for '31B ARMS DONE'"
for i in $(seq 1 720); do
  if timeout 60 $SSH 'grep -qa "31B ARMS DONE" /root/run31b.log /root/run31b_outer.log 2>/dev/null'; then
    say "31B finished"; break
  fi
  sleep 60
done

say "--- re-running runtime with the ADV-010 fix ---"
timeout 60 $SSH 'cd /root && setsid nohup bash -lc "
  export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  rm -rf /work/inc; mkdir -p /root/ccai/ckpt_runtime_fixed
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=runtime CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 \
  CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800 CCAI_CKPT_DIR=/root/ccai/ckpt_runtime_fixed \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm3_runtime_fixed.log 2>&1
  echo \"RUNTIME_FIXED EXIT=\$?\" >> /root/chain_runtime_fixed.marker
" > /dev/null 2>&1 < /dev/null & echo launched'
say "runtime-fixed launched"
