#!/usr/bin/env bash
# ★★★ SECOND-BOX VARIANCE: independent salts 4 and 5.
#
# The paper's stated weakness is "one seed per arm, no error bars". The old box is running salts 1..3;
# this runs salts 4 and 5 on the second box IN PARALLEL, so the paired-ratio confidence interval rests
# on five salts instead of three and finishes in half the wall-clock. Salts 4/5 do not collide with the
# old box's 1..3 -- the result filename carries the salt.
#
# (The 31B QAT run was abandoned here: compressed_tensors decompresses the whole model on first
# forward, needing packed + unpacked weights resident, which does not fit a 48 GB card even with CPU
# offload. It needs an 80 GB card, and neither paper track mandates the 31B.)
set -u
SSH="ssh -p 36882 root@ssh5.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -P 36882 -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_var_box2.log
mkdir -p "$OUT/variance"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

wait_free(){
  local i n
  for i in $(seq 1 60); do
    n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .')
    [ "${n:-1}" = "0" ] && { say "  GPU free"; return 0; }
    sleep 30
  done
  say "  GPU still busy"; return 1
}

run_arm(){
  local salt="$1" arm="$2" remote="$3" dest="$4" work="$5" i
  if timeout 45 $SSH "test -s /root/ccai/results/$remote"; then
    say "  $salt/$arm: already present"; return 0
  fi
  wait_free || { say "  $salt/$arm ABORT: GPU busy"; return 1; }
  timeout 60 $SSH "cd /root && setsid nohup bash -lc '
    export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    rm -rf $work; mkdir -p $work
    CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
    CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
    CCAI_TIME_BUDGET_S=10800 CCAI_ARM=$arm CCAI_TASK_SALT=$salt CCAI_RUN_TAG=_var_$salt \
    CCAI_CKPT_DIR=/root/ccai/ckpt_${salt}_${arm} CCAI_WORK=$work \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/log_VAR_${salt}_${arm}.log 2>&1
  ' > /dev/null 2>&1 < /dev/null & echo launched" >/dev/null 2>&1 || say "  launch ssh non-zero (advisory)"
  for i in $(seq 1 300); do
    timeout 45 $SSH "test -s /root/ccai/results/$remote" && { say "  $salt/$arm result after ${i}m"; break; }
    sleep 60
  done
  if timeout 45 $SSH "test -s /root/ccai/results/$remote"; then
    timeout 180 $SCP "root@ssh5.vast.ai:/root/ccai/results/$remote" "$OUT/$dest" >/dev/null 2>&1 \
      && say "  pulled $dest" || say "  scp FAILED $dest"
  else
    say "  $salt/$arm TIMEOUT"
  fi
}

say "=== BOX2 VARIANCE (salts 4,5) START $(date -u) ==="
# Ship the current harness so the second box runs the same code as the first.
for f in incremental_coding.py vast_entry.py bug_task.py; do
  timeout 120 $SCP "/home/roni-saguey/.local/share/ccai-repo/modal/$f" root@ssh5.vast.ai:/root/ccai/ >/dev/null 2>&1 || true
done
# stop any leftover 31B attempt
timeout 45 $SSH 'for p in $(ps -eo pid,cmd|grep -E "vast_entry"|grep -v grep|awk "{print \$1}"); do kill -9 $p; done; sleep 2' >/dev/null 2>&1

for SALT in salt4 salt5; do
  for ARM in runtime linear; do
    run_arm "$SALT" "$ARM" "incremental_gemma-4-12b_${ARM}_${SALT}_var_${SALT}.json" \
      "variance/var_${SALT}_${ARM}.json" "/work/v_${SALT}_${ARM}"
  done
done
say "=== BOX2 VARIANCE DONE $(date -u) ==="
ls -la "$OUT/variance" 2>/dev/null
