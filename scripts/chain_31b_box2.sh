#!/usr/bin/env bash
# ★★★ RUN THE 31B COMPETITION CHECKPOINT ON THE SECOND BOX.
#
# The old box has the variance sweep queued ahead of the 31B, so the 31B (the long pole, ~8h) would
# not start for hours. The archive box is idle and has the model's HF cache, so the 31B runs there in
# parallel and the old box is stopped before its own 31B launch to avoid duplicate work.
#
# The 31B is the exact checkpoint the main Gemma 4 competition mandates
# (google/gemma-4-31B-it-qat-w4a16-ct). Running the same two policies on it forecloses the "small-model
# artifact" objection.
set -u
SSH="ssh -p 36882 root@ssh5.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -P 36882 -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes"
OUT=/home/roni-saguey/.local/share/ccai-results
LOG=$OUT/chain_31b_box2.log
mkdir -p "$OUT/fixmode-v3"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== 31B ON ARCHIVE BOX START $(date -u) ==="

# GPU must be free.
for i in $(seq 1 60); do
  n=$(timeout 45 $SSH 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .')
  [ "${n:-1}" = "0" ] && { say "GPU free"; break; }
  sleep 30
done
[ "${n:-1}" != "0" ] && { say "ABORT: GPU busy"; exit 1; }

# Ship the driver and make sure the entry point is current.
for f in run31b.sh vast_entry.py incremental_coding.py bug_task.py; do
  timeout 120 $SCP "/home/roni-saguey/.local/share/ccai-repo/scripts/$f" root@ssh5.vast.ai:/root/ccai/ >/dev/null 2>&1 \
    || timeout 120 $SCP "/home/roni-saguey/.local/share/ccai-repo/deploy/$f" root@ssh5.vast.ai:/root/ccai/ >/dev/null 2>&1 \
    || timeout 120 $SCP "/home/roni-saguey/.local/share/ccai-repo/modal/$f" root@ssh5.vast.ai:/root/ccai/ >/dev/null 2>&1 \
    || say "  could not ship $f (may already exist)"
done

timeout 60 $SSH 'rm -rf /work/inc /root/ccai/ckpt31_runtime /root/ccai/ckpt31_linear; mkdir -p /work/inc' >/dev/null 2>&1
timeout 60 $SSH 'cd /root && setsid nohup bash /root/ccai/run31b.sh 60 > /root/run31b_outer.log 2>&1 < /dev/null & echo launched'

# Wait for BOTH arms' results.
for i in $(seq 1 900); do
  if timeout 45 $SSH 'test -s /root/ccai/results/incremental_gemma-4-31b-qat_runtime.json && test -s /root/ccai/results/incremental_gemma-4-31b-qat_linear.json'; then
    say "both 31B results present after ${i}m"; break
  fi
  sleep 60
done

for a in runtime linear; do
  RF=/root/ccai/results/incremental_gemma-4-31b-qat_$a.json
  if timeout 45 $SSH "test -s $RF"; then
    timeout 180 $SCP "root@ssh5.vast.ai:$RF" "$OUT/fixmode-v3/incremental_gemma-4-31b-qat_$a.json" >/dev/null 2>&1 \
      && say "pulled 31B $a" || say "scp FAILED 31B $a"
  else
    say "no 31B $a result"
  fi
done

say "--- 31B summary ---"
grep -aE "PASSED|ARMS DONE" /root/run31b.log 2>/dev/null | tail -4 || true
say "=== 31B ON ARCHIVE BOX DONE $(date -u) ==="
