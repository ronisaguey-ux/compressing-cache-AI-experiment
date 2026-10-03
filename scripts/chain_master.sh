#!/usr/bin/env bash
# ★★★ MASTER RUN QUEUE — the ablation, then everything else, strictly serial.
#
# WHY THIS FILE WAS REWRITTEN
# ---------------------------
# The first version waited for the literal string "ALL ARMS DONE" in runfix3.log. That string was
# already there from the earlier prune arm, so the wait passed instantly and the queue launched ARC
# WHILE the ablation was still running. Both runs shared the default work dir /work/inc, and the new
# run's `rm -rf /work/inc` deleted the ablation's solution.py mid-turn:
#
#     FileNotFoundError: [Errno 2] No such file or directory: '/work/inc/solution.py'
#
# The ablation died at turn 27 and that result is lost. Two lessons, both now enforced below:
#   1. NEVER wait on a marker that a previous run could have written. Each run gets a UNIQUE tag and
#      a marker file that is deleted before its run starts, so a stale marker cannot satisfy the wait.
#   2. NEVER let two runs share a work dir. CCAI_WORK is set per run; the harness already honours it.
# Wiping a sibling's working directory is not a GPU-contention problem, it is a data-loss problem,
# and it cost two hours of GPU.
#
# ORDER (by deadline and value, since the box is serial):
#   1. ablation    12B  -- the mechanism result; half a run was already lost to the bug above
#   2. ARC        12B   -- the ARC paper's central claim, currently argued; ARC closes Nov 9
#   3. runtime_fixed    -- the committed runtime numbers predate the fixed code (integrity)
#   4. variance 3x2     -- error bars; trimmed from 5 salts to fit
#   5. 31B              -- pre-empts "small-model artifact"; goes last so preemption costs least
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_master.log
mkdir -p "$OUT" "$OUT/variance" "$OUT/arc"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# run ONE arm, wait for its unique completion tag, then pull its result.
#   run_arm <tag> <wait-minutes> <local-dest> <workdir> <env-string>
run_arm(){
  local tag="$1" mins="$2" dest="$3" work="$4" envs="$5" i rc
  timeout 60 $SSH "rm -f /root/chain_master.marker" >/dev/null 2>&1
  timeout 60 $SSH "cd /root && setsid nohup bash -lc '
    export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    rm -rf $work; mkdir -p $work
    $envs CCAI_WORK=$work /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/log_${tag}.log 2>&1
    echo \"DONE_${tag} EXIT=\$?\" >> /root/chain_master.marker
  ' > /dev/null 2>&1 < /dev/null & echo launched" || { say "  LAUNCH FAILED $tag"; return 1; }

  for i in $(seq 1 "$mins"); do
    if timeout 60 $SSH "grep -qa 'DONE_${tag}' /root/chain_master.marker 2>/dev/null"; then
      rc=$(timeout 60 $SSH "grep -a 'DONE_${tag}' /root/chain_master.marker | tail -1")
      say "  $tag -> $rc"
      break
    fi
    sleep 60
  done
  [ "$i" -ge "$mins" ] && say "  $tag TIMED OUT after ${mins}m"

  local rf
  rf=$(timeout 60 $SSH "ls -t /root/ccai/results/*.json 2>/dev/null | head -1")
  if [ -n "$rf" ]; then
    timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "root@ssh6.vast.ai:$rf" "$OUT/$dest" >/dev/null 2>&1 \
      && say "  pulled -> $dest" || say "  scp FAILED $rf"
  else
    say "  no result file to pull for $tag"
  fi
}

say "=== MASTER QUEUE START $(date -u) ==="
# Clear every stale marker before anything runs, so no wait can be satisfied by history.
timeout 60 $SSH 'rm -f /root/chain_master.marker /root/chain_after_v3.marker /root/chain_arc.marker'

M12="CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800"

say "--- [1/6] ablation: anchored policy minus the instruction archive ---"
run_arm "ABLATE" 300 "fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" "/work/ablate" \
  "$M12 CCAI_ARM=ablate-recent CCAI_CKPT_DIR=/root/ccai/ckpt_ablate-recent"

say "--- [2/6] ARC: preflight, then both arms ---"
timeout 120 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  scripts/run_arc.sh root@ssh6.vast.ai:/root/ccai/run_arc.sh >/dev/null 2>&1
timeout 60 $SSH 'cd /root && setsid nohup bash /root/ccai/run_arc.sh 60 287 > /root/run_arc_outer.log 2>&1 < /dev/null & echo launched'
for i in $(seq 1 600); do
  timeout 60 $SSH 'grep -qa "ARC ARMS DONE" /root/run_arc.log 2>/dev/null' && { say "  ARC arms done"; break; }
  sleep 60
done
[ "$i" -ge 600 ] && say "  ARC TIMED OUT"
timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@ssh6.vast.ai:/root/ccai/results/incremental_gemma-4-12b_runtime.json" "$OUT/arc/arc_runtime.json" >/dev/null 2>&1 \
  && say "  pulled arc_runtime" || say "  arc pull failed"

say "--- [3/6] runtime_fixed (corrected runtime) ---"
run_arm "RUNTIMEFIX" 300 "fixmode-v3/incremental_gemma-4-12b_runtime_fixed.json" "/work/rfix" \
  "$M12 CCAI_ARM=runtime CCAI_CKPT_DIR=/root/ccai/ckpt_runtime_fixed"

say "--- [4/6] variance: 3 salts x 2 arms ---"
for SALT in salt1 salt2 salt3; do
  for ARM in runtime linear; do
    run_arm "VAR_${SALT}_${ARM}" 300 "variance/var_${SALT}_${ARM}.json" "/work/v_${SALT}_${ARM}" \
      "$M12 CCAI_ARM=$ARM CCAI_TASK_SALT=$SALT CCAI_CKPT_DIR=/root/ccai/ckpt_${SALT}_${ARM}"
  done
done

say "--- [5/6] 31B competition checkpoint ---"
timeout 60 $SSH 'cd /root && setsid nohup bash /root/ccai/run31b.sh 60 > /root/run31b_outer.log 2>&1 < /dev/null & echo launched'
for i in $(seq 1 900); do
  timeout 60 $SSH 'grep -qa "31B ARMS DONE" /root/run31b.log 2>/dev/null' && { say "  31B done"; break; }
  sleep 60
done
[ "$i" -ge 900 ] && say "  31B TIMED OUT"

say "=== MASTER QUEUE DONE $(date -u) ==="
ls -la "$OUT/arc" "$OUT/variance" 2>/dev/null
