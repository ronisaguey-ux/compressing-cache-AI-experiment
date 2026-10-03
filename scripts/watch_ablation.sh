#!/usr/bin/env bash
# Wait for the ablation to write its result file on the old box, then exit.
# Run under `ocbg` so the session is woken the moment the result exists instead of polling.
#
# DETECT ONLY -- it deliberately does NOT pull. The post-ablation coordinator pulls the same file
# to a canonical path, and two concurrent scp writes to one path can corrupt it. This waits for the
# RESULT FILE (the evidence the run finished AND wrote output), not a marker (which only says a
# shell exited).
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
RF=/root/ccai/results/incremental_gemma-4-12b_ablate-recent.json
for i in $(seq 1 240); do
  if timeout 45 $SSH "test -s $RF"; then
    echo "[$(date -u +%H:%M:%S)] ABLATION RESULT READY on the box"
    exit 0
  fi
  sleep 60
done
echo "[$(date -u +%H:%M:%S)] TIMEOUT after 240m -- inspect the box"
exit 1
