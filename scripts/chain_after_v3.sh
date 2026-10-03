#!/usr/bin/env bash
# ★★★ CHAIN AFTER THE v3 THREE-ARM RUN.
#
# Waits for the 3-arm run to print ALL ARMS DONE, then runs, back to back on the SAME box:
#
#   1. ablate-recent  — `runtime` MINUS the instruction archive. The ablation the paper needs:
#                       it isolates WHICH part of the policy carries the result. Prediction stated
#                       in incremental_coding.py before the run: codes_n collapses to the ~4
#                       instructions that fit beside the anchor; `passed` stays high (the newest
#                       reply still carries the file state). Same task, same salt, same budget —
#                       one variable removed.
#
#   2. 31B runtime    — the COMPETITION checkpoint (google/gemma-4-31B-it-qat-w4a16-ct), so the
#                       result is not dismissible as a small-model artifact.
#   3. 31B linear     — its baseline.
#
# ORDER REASON: the ablation is on the same 12B as the headline, so it is directly comparable and
# it is quick. The 31B arms are the slow ones and go last, so a box death still leaves the
# mechanism result in hand.
#
# The 31B needs ~26GB GPU and prune holds ~29GB, hence strictly sequential.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
LOG=/home/roni-saguey/.local/share/ccai-results/chain_after_v3.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "waiting for 'ALL ARMS DONE' in /root/runfix3.log"
for i in $(seq 1 360); do
  if timeout 60 $SSH 'grep -qa "ALL ARMS DONE" /root/runfix3.log 2>/dev/null'; then
    say "3-arm run DONE"
    break
  fi
  sleep 60
done

# ── 1. the ablation ──────────────────────────────────────────────────────────────────────────────
say "--- launching ablate-recent (12B, same task/salt/budget) ---"
timeout 60 $SSH 'cd /root && setsid nohup bash -lc "
  export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  rm -rf /work/inc; mkdir -p /root/ccai/ckpt_ablate-recent
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=ablate-recent CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 \
  CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800 CCAI_CKPT_DIR=/root/ccai/ckpt_ablate-recent \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm3_ablate-recent.log 2>&1
  echo \"ABLATION EXIT=\$?\" >> /root/chain_after_v3.marker
" > /dev/null 2>&1 < /dev/null & echo ablation-launched'

# wait for the ablation marker
for i in $(seq 1 240); do
  if timeout 60 $SSH 'grep -qa "ABLATION EXIT" /root/chain_after_v3.marker 2>/dev/null'; then
    say "ablation finished"
    break
  fi
  sleep 60
done

# ── 2 & 3. the 31B competition checkpoint ────────────────────────────────────────────────────────
say "--- launching 31B (runtime then linear) ---"
timeout 60 $SSH 'cd /root && setsid nohup bash /root/ccai/run31b.sh 60 > /root/run31b_outer.log 2>&1 < /dev/null & echo 31b-launched'
say "31B launched -- chain complete"
