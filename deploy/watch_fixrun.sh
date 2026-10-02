#!/usr/bin/env bash
# Wait for both arms of the fix-mode run to finish, then exit (so ocbg wakes the session).
SSH="ssh -p 29980 root@ssh4.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20"
for i in $(seq 1 600); do
  S=$(timeout 60 $SSH 'tail -c 3000 /root/runfix.log 2>/dev/null' 2>/dev/null)
  case "$S" in *"BOTH ARMS DONE"*) echo "DONE after $((i*60))s"; timeout 60 $SSH 'tail -25 /root/runfix.log; echo ---; grep -aE "PASSED|PER-TURN" /root/arm_linear.log /root/arm_runtime.log' 2>/dev/null; exit 0;; esac
  # also stop early if the box is unreachable for many consecutive tries
  sleep 60
done
echo "TIMEOUT waiting for run"; exit 1
