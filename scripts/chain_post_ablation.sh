#!/usr/bin/env bash
# ★★★ POST-ABLATION QUEUE — runs the remaining arms after the ablation.
#
# ORDER: runtime_fixed (integrity: the committed runtime numbers predate the fixed code), then
# variance 3x2 (error bars), then 31B (kills the small-model objection).
#
# THREE RULES, each one paid for:
#   1. NEVER launch on a busy GPU. Every OOM in this project (RUNTIMEFIX, six variance salts, arm31)
#      came from launching while another run held the card. A skipped arm is cheaper than a corrupted
#      one, so a GPU that never frees ABORTS the queue rather than forcing a launch.
#   2. Wait on the RESULT FILE, not a marker. A marker only says a shell exited -- and one marker
#      turned out to be owned by a wrapper shell that had already been cleaned up. The result file is
#      evidence the run finished AND wrote output.
#   3. An ssh non-zero exit code is ADVISORY, not proof a launch failed. A transient connection blip
#      made this script report "LAUNCH FAILED RUNTIMEFIX" while the run was in fact starting; treating
#      that as fatal is what let a later arm launch on top of it. Verify by polling for the result.
#
# The queue is IDEMPOTENT: an arm whose result already exists is pulled and skipped, so the script can
# be re-run at any point without relaunching finished work.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_post_ablation.log
mkdir -p "$OUT" "$OUT/fixmode-v3" "$OUT/variance" "$OUT/arc"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# Wait until the GPU has no compute process, up to `mins` minutes. Returns 0 when free.
wait_gpu_free(){
  local mins="$1" i n
  for i in $(seq 1 "$mins"); do
    n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .' 2>/dev/null)
    if [ "${n:-1}" = "0" ]; then say "  GPU free"; return 0; fi
    sleep 60
  done
  say "  GPU still busy after ${mins}m"; return 1
}

# Pull an exact remote result filename to a local dest. Never "the newest file" -- that pulled the
# same stale json for six different runs and produced six md5-identical "variance" files.
pull(){
  local remote="$1" dest="$2"
  timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@ssh6.vast.ai:/root/ccai/results/$remote" "$OUT/$dest" >/dev/null 2>&1 \
    && say "  pulled $remote -> $dest" || say "  scp FAILED $remote"
}

# run_arm <tag> <wait-min> <remote-filename> <local-dest> <workdir> <env>
run_arm(){
  local tag="$1" mins="$2" remote="$3" dest="$4" work="$5" envs="$6" i
  # Already finished? Then there is nothing to launch.
  if timeout 45 $SSH "test -s /root/ccai/results/$remote"; then
    say "  $tag: result already present -- skipping launch"
    pull "$remote" "$dest"; return 0
  fi
  # A run may be in flight. Wait for the card. NEVER launch on a busy GPU.
  if ! wait_gpu_free 180; then
    say "  $tag: ABORT -- GPU never freed, refusing to launch on a busy card"; return 1
  fi
  # The in-flight run may have been THIS arm, finished while we waited.
  if timeout 45 $SSH "test -s /root/ccai/results/$remote"; then
    say "  $tag: result appeared while waiting -- skipping launch"
    pull "$remote" "$dest"; return 0
  fi

  timeout 45 $SSH 'rm -f /root/chain_post.marker' >/dev/null 2>&1
  timeout 60 $SSH "cd /root && setsid nohup bash -lc '
    export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    rm -rf $work; mkdir -p $work
    $envs CCAI_WORK=$work /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/log_${tag}.log 2>&1
    echo \"DONE_${tag} EXIT=\$?\" >> /root/chain_post.marker
  ' > /dev/null 2>&1 < /dev/null & echo launched" >/dev/null 2>&1 || say "  note: launch ssh returned non-zero (advisory only -- verifying below)"

  # Verify by RESULT, not by ssh exit code.
  for i in $(seq 1 "$mins"); do
    if timeout 45 $SSH "test -s /root/ccai/results/$remote"; then
      say "  $tag: result present after ${i}m"
      pull "$remote" "$dest"; return 0
    fi
    sleep 60
  done
  say "  $tag TIMED OUT after ${mins}m -- polling once more"
  pull "$remote" "$dest"
  return 1
}

say "=== POST-ABLATION QUEUE (re)START $(date -u) ==="

M12="CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800"

say "--- runtime_fixed (corrected runtime, integrity) ---"
run_arm "RUNTIMEFIX" 240 "incremental_gemma-4-12b_runtime_fixed.json" \
  "fixmode-v3/incremental_gemma-4-12b_runtime_fixed.json" "/work/rfix" \
  "$M12 CCAI_ARM=runtime CCAI_RUN_TAG=_fixed CCAI_CKPT_DIR=/root/ccai/ckpt_runtime_fixed"

say "--- variance: 3 salts x 2 arms ---"
for SALT in salt1 salt2 salt3; do
  for ARM in runtime linear; do
    run_arm "VAR_${SALT}_${ARM}" 240 "incremental_gemma-4-12b_${ARM}_${SALT}_var_${SALT}.json" \
      "variance/var_${SALT}_${ARM}.json" "/work/v_${SALT}_${ARM}" \
      "$M12 CCAI_ARM=$ARM CCAI_TASK_SALT=$SALT CCAI_RUN_TAG=_var_${SALT} CCAI_CKPT_DIR=/root/ccai/ckpt_${SALT}_${ARM}"
  done
done

say "--- 31B competition checkpoint ---"
# The 31B arm writes its result under the model name it is given, which is `gemma-4-31b-qat`
# (README: gemma-4-31b-qat). Check both arms by glob rather than a guessed exact name.
timeout 60 $SSH 'rm -f /root/run31b.log; cd /root && setsid nohup bash /root/ccai/run31b.sh 60 > /root/run31b_outer.log 2>&1 < /dev/null & echo launched' >/dev/null 2>&1
for i in $(seq 1 900); do
  timeout 45 $SSH 'ls /root/ccai/results/incremental_gemma-4-31b*.json >/dev/null 2>&1 || grep -qa "31B ARMS DONE" /root/run31b.log 2>/dev/null' \
    && { say "  31B done"; break; }
  sleep 60
done
[ "$i" -ge 900 ] && say "  31B TIMED OUT"
# Pull whatever 31B wrote, by exact name.
for _m in gemma-4-31b-qat gemma-4-31b; do
  for _a in runtime linear; do
    _f="incremental_${_m}_${_a}.json"
    timeout 45 $SSH "test -s /root/ccai/results/$_f" 2>/dev/null && pull "$_f" "fixmode-v3/$_f"
  done
done

say "=== POST-ABLATION QUEUE DONE $(date -u) ==="
ls -la "$OUT/variance" 2>/dev/null
