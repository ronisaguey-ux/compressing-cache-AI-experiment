#!/usr/bin/env bash
# Wake when RUNTIMEFIX writes its result on the old box. Detect only; the coordinator pulls.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
RF=/root/ccai/results/incremental_gemma-4-12b_runtime_fixed.json
for i in $(seq 1 240); do
  timeout 45 $SSH "test -s $RF" && { echo "[$(date -u +%H:%M:%S)] RUNTIMEFIX RESULT READY"; exit 0; }
  sleep 60
done
echo "[$(date -u +%H:%M:%S)] timeout"; exit 1
