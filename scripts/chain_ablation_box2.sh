#!/usr/bin/env bash
# ★★★ RUN THE CLEAN ABLATION ON THE SECOND BOX (ARCHIVE box), AFTER ITS ARC RUN FINISHES.
#
# WHY A SECOND BOX. The old box already has the variance sweep (6 runs) and then the 31B queued, which
# is ~20h serial. The ablation is the mechanism evidence and only takes ~2h, so waiting behind a 20h
# queue is waste. The ARCHIVE box frees when its ARC linear arm ends, and it already holds the model
# and the harness.
#
# The ablation re-run must be CLEAN: the first one resumed after a crash and, because turn_ok was not
# checkpointed then, reported its pre-crash turns as failures. The contaminated result is archived
# first so the before/after is auditable.
set -u
SSH="ssh -p 36882 root@ssh5.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -P 36882 -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_ablation_box2.log
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== CLEAN ABLATION ON ARCHIVE BOX START $(date -u) ==="

# 1. wait for the ARC run to finish (its linear arm writes the result last)
say "waiting for the ARC run to finish on the archive box"
for i in $(seq 1 300); do
  timeout 45 $SSH 'test -s /root/ccai/results/incremental_gemma-4-12b_linear.json && ! pgrep -f "[v]ast_entry" >/dev/null' \
    && { say "ARC run finished"; break; }
  sleep 60
done
[ "$i" -ge 300 ] && say "ARC run did not finish in 300m -- continuing anyway"

# 2. GPU must be free before launching anything
say "waiting for the GPU to free"
for i in $(seq 1 60); do
  n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .')
  [ "${n:-1}" = "0" ] && { say "GPU free"; break; }
  sleep 30
done
[ "${n:-1}" != "0" ] && { say "ABORT: GPU still busy"; exit 1; }

# 3. archive the contaminated ablation result from the old box, for the record
if [ -f "$OUT/fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" ]; then
  cp "$OUT/fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" \
     "$OUT/fixmode-v3/CONTAMINATED_incremental_gemma-4-12b_ablate-recent.resume-artifact.json" \
     && say "archived the contaminated ablation result"
fi

# 4. run the clean ablation on the archive box
say "--- clean ablation on the archive box ---"
timeout 60 $SSH 'rm -rf /work/ablate_clean /root/ccai/ckpt_ablate_clean; mkdir -p /work/ablate_clean' >/dev/null 2>&1
timeout 60 $SSH "cd /root && setsid nohup bash -lc '
  export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
  CCAI_TIME_BUDGET_S=14400 CCAI_ARM=ablate-recent CCAI_CKPT_DIR=/root/ccai/ckpt_ablate_clean \
  CCAI_WORK=/work/ablate_clean /venv/main/bin/python -u /root/ccai/vast_entry.py \
  > /root/log_ABLATE_CLEAN.log 2>&1
  echo \"DONE_ABLATE_CLEAN EXIT=\$?\" >> /root/ablate_clean.marker
' > /dev/null 2>&1 < /dev/null & echo launched" >/dev/null 2>&1 || say "launch ssh non-zero (advisory)"

RF=/root/ccai/results/incremental_gemma-4-12b_ablate-recent.json
for i in $(seq 1 300); do
  timeout 45 $SSH "test -s $RF" && { say "clean ablation result present after ${i}m"; break; }
  sleep 60
done
if timeout 45 $SSH "test -s $RF"; then
  $SCP "root@ssh5.vast.ai:$RF" "$OUT/fixmode-v3/incremental_gemma-4-12b_ablate-recent.json" >/dev/null 2>&1 \
    && say "pulled the clean ablation result" || say "scp FAILED"
else
  say "TIMEOUT: no clean ablation result"
fi
say "=== CLEAN ABLATION ON ARCHIVE BOX DONE $(date -u) ==="
