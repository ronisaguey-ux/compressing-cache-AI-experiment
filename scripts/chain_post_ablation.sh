#!/usr/bin/env bash
# ★★★ POST-ABLATION QUEUE — runs the remaining arms AFTER the ablation that is currently running.
#
# WHY A NEW FILE
# --------------
# The previous queue (chain_master.sh) launched at 07:13 and immediately ran its arms, because it was
# waiting for `DONE_ABLATE` in chain_master.marker -- a tag written only by ITS OWN runs. The real
# ablation, started hours earlier, signals completion differently (`ABLATION EXIT` in
# chain_after_v3.marker). So the queue never waited: it started a second run on the same GPU, which
# OOM'd (`torch.OutOfMemoryError: Tried to allocate 22.28 GiB ... 19.70 GiB free`), and every arm
# after it OOM'd in turn. The whole queue was wasted, and it did not disturb the real ablation only
# because that run already held the memory.
#
# THE RULES, both learned by paying for them:
#   1. Wait on the marker the ACTUAL running job writes, after deleting it so a stale one cannot pass.
#   2. Before every launch, assert the GPU is free. A marker only says a process ENDED; it does not
#      say the memory was released, and it says nothing about a DIFFERENT job holding the card.
#
# ORDER: ablation is already running. Then runtime_fixed (integrity: the committed runtime numbers
# predate the fixed code), then variance 3x2 (error bars), then 31B (kills the small-model objection).
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_post_ablation.log
mkdir -p "$OUT" "$OUT/variance" "$OUT/arc"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# Wait until the GPU has no compute process, up to `mins`. Returns 0 when free.
wait_gpu_free(){
  local mins="$1" i n
  for i in $(seq 1 "$mins"); do
    n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .' 2>/dev/null)
    if [ "${n:-1}" = "0" ]; then say "  GPU free"; return 0; fi
    sleep 30
  done
  say "  GPU still busy after ${mins}m"; return 1
}

# run_arm <tag> <wait-min> <remote-filename> <local-dest> <workdir> <env>
run_arm(){
  local tag="$1" mins="$2" remote="$3" dest="$4" work="$5" envs="$6" i rc rf
  timeout 45 $SSH 'rm -f /root/chain_post.marker' >/dev/null 2>&1
  timeout 60 $SSH "cd /root && setsid nohup bash -lc '
    export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    rm -rf $work; mkdir -p $work
    $envs CCAI_WORK=$work /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/log_${tag}.log 2>&1
    echo \"DONE_${tag} EXIT=\$?\" >> /root/chain_post.marker
  ' > /dev/null 2>&1 < /dev/null & echo launched" || { say "  LAUNCH FAILED $tag"; return 1; }

  for i in $(seq 1 "$mins"); do
    if timeout 45 $SSH "grep -qa 'DONE_${tag}' /root/chain_post.marker 2>/dev/null"; then
      say "  $tag: $(timeout 45 $SSH "grep -a 'DONE_${tag}' /root/chain_post.marker | tail -1")"
      break
    fi
    sleep 60
  done
  [ "$i" -ge "$mins" ] && say "  $tag TIMED OUT"

  rf=$(timeout 45 $SSH "ls /root/ccai/results/$remote 2>/dev/null | head -1")
  if [ -n "$rf" ]; then
    timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "root@ssh6.vast.ai:$rf" "$OUT/$dest" >/dev/null 2>&1 \
      && say "  pulled $remote -> $dest" || say "  scp FAILED $remote"
  else
    say "  NO RESULT at $remote (run may have failed)"
  fi
}

say "=== POST-ABLATION QUEUE START $(date -u) ==="
say "waiting for the RUNNING ablation to write 'ABLATION EXIT'"
for i in $(seq 1 180); do
  if timeout 45 $SSH 'grep -qa "ABLATION EXIT" /root/chain_after_v3.marker 2>/dev/null'; then
    say "ablation reported done"; break
  fi
  sleep 60
done
# The marker says the process ended; it does not say the memory came back.
wait_gpu_free 20 || say "continuing anyway, next run may OOM"

say "--- pull the ablation result ---"
rf=$(timeout 45 $SSH "ls /root/ccai/results/incremental_gemma-4-12b_ablate-recent.json 2>/dev/null")
if [ -n "$rf" ]; then
  timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null "root@ssh6.vast.ai:$rf" \
    "$OUT/fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" >/dev/null 2>&1 \
    && say "  ablation result pulled" || say "  SCp FAILED"
else
  say "  no ablation result file -- the run may have died"
fi

M12="CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800"

say "--- runtime_fixed (corrected runtime, integrity) ---"
run_arm "RUNTIMEFIX" 300 "incremental_gemma-4-12b_runtime_fixed.json" \
  "fixmode-v3/incremental_gemma-4-12b_runtime_fixed.json" "/work/rfix" \
  "$M12 CCAI_ARM=runtime CCAI_RUN_TAG=_fixed CCAI_CKPT_DIR=/root/ccai/ckpt_runtime_fixed"

say "--- variance: 3 salts x 2 arms ---"
for SALT in salt1 salt2 salt3; do
  for ARM in runtime linear; do
    run_arm "VAR_${SALT}_${ARM}" 300 "incremental_gemma-4-12b_${ARM}_${SALT}_var_${SALT}.json" \
      "variance/var_${SALT}_${ARM}.json" "/work/v_${SALT}_${ARM}" \
      "$M12 CCAI_ARM=$ARM CCAI_TASK_SALT=$SALT CCAI_RUN_TAG=_var_${SALT} CCAI_CKPT_DIR=/root/ccai/ckpt_${SALT}_${ARM}"
  done
done

say "--- 31B competition checkpoint ---"
timeout 60 $SSH 'cd /root && setsid nohup bash /root/ccai/run31b.sh 60 > /root/run31b_outer.log 2>&1 < /dev/null & echo launched'
for i in $(seq 1 900); do
  timeout 45 $SSH 'grep -qa "31B ARMS DONE" /root/run31b.log 2>/dev/null' && { say "  31B done"; break; }
  sleep 60
done
[ "$i" -ge 900 ] && say "  31B TIMED OUT"

say "=== POST-ABLATION QUEUE DONE $(date -u) ==="
ls -la "$OUT/variance" 2>/dev/null
