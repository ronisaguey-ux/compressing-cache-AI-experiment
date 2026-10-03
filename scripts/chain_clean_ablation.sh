#!/usr/bin/env bash
# ★★★ AFTER THE MAIN QUEUE: re-run the ablation CLEAN, then regenerate the results section.
#
# WHY. The first ablation crashed at turn 27 (a racing queue deleted its working directory) and
# resumed. `turn_ok` was not checkpointed at that time, so the resuming process knew nothing about
# turns 1..27 and the result carried them as ZEROS -- per_turn_success=0.55, applied-then-lost=-27.
# Those are not measurements of the model; they are turns the harness graded and then forgot. The
# ablation is the run that tests whether the instruction archive is the mechanism, so it cannot ship
# with a harness artifact in its headline metric.
#
# The contaminated result is archived, not deleted, so the before/after is auditable.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_clean_ablation.log
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

wait_gpu_free(){
  local mins="$1" i n
  for i in $(seq 1 "$mins"); do
    n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .' 2>/dev/null)
    if [ "${n:-1}" = "0" ]; then say "  GPU free"; return 0; fi
    sleep 60
  done
  say "  GPU still busy after ${mins}m"; return 1
}

say "=== CLEAN ABLATION QUEUE START $(date -u) ==="
say "waiting for the main coordinator to finish (chain_post_ablation.log)"
for i in $(seq 1 600); do
  grep -qa "POST-ABLATION QUEUE DONE" "$OUT/chain_post_ablation.log" 2>/dev/null && { say "main queue done"; break; }
  sleep 60
done
[ "$i" -ge 600 ] && say "main queue did not report done in 600m -- continuing anyway"

wait_gpu_free 60 || { say "ABORT: GPU never freed"; exit 1; }

# Archive the contaminated result so the re-run cannot be confused with it.
RF=/root/ccai/results/incremental_gemma-4-12b_ablate-recent.json
if timeout 45 $SSH "test -s $RF"; then
  timeout 120 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null "root@ssh6.vast.ai:$RF" \
    "$OUT/fixmode-v3/CONTAMINATED_incremental_gemma-4-12b_ablate-recent.resume-artifact.json" >/dev/null 2>&1 \
    && say "  archived the contaminated ablation result" || say "  archive scp failed (continuing)"
fi

say "--- clean ablation re-run ---"
timeout 45 $SSH 'rm -f /root/chain_post.marker' >/dev/null 2>&1
timeout 60 $SSH 'rm -f /root/ccai/results/incremental_gemma-4-12b_ablate-recent.json; rm -rf /root/ccai/ckpt_ablate_clean /work/ablate_clean' >/dev/null 2>&1
timeout 60 $SSH "cd /root && setsid nohup bash -lc '
  export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  mkdir -p /work/ablate_clean
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
  CCAI_TIME_BUDGET_S=10800 CCAI_ARM=ablate-recent CCAI_CKPT_DIR=/root/ccai/ckpt_ablate_clean \
  CCAI_WORK=/work/ablate_clean /venv/main/bin/python -u /root/ccai/vast_entry.py \
  > /root/log_ABLATE_CLEAN.log 2>&1
  echo \"DONE_ABLATE_CLEAN EXIT=\$?\" >> /root/chain_post.marker
' > /dev/null 2>&1 < /dev/null & echo launched" >/dev/null 2>&1 || say "  launch ssh non-zero (advisory)"

for i in $(seq 1 300); do
  timeout 45 $SSH "test -s $RF" && { say "  clean ablation result present after ${i}m"; break; }
  sleep 60
done
if timeout 45 $SSH "test -s $RF"; then
  timeout 180 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null "root@ssh6.vast.ai:$RF" \
    "$OUT/fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" >/dev/null 2>&1 \
    && say "  pulled the clean ablation result" || say "  scp FAILED"
else
  say "  TIMEOUT: no clean ablation result"
fi

say "--- regenerate the results section ---"
cd /home/roni-saguey/.local/share/ccai-repo && python3 tools/make_results_section.py >>"$LOG" 2>&1
say "=== CLEAN ABLATION QUEUE DONE $(date -u) ==="
