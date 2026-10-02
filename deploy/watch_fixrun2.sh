#!/usr/bin/env bash
SSH="ssh -p 29980 root@ssh4.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20"
for i in $(seq 1 900); do
  S=$(timeout 60 $SSH 'tail -c 2000 /root/runfix2.log 2>/dev/null' 2>/dev/null)
  case "$S" in *"BOTH ARMS DONE"*)
    echo "DONE after $((i*60))s"
    timeout 90 $SSH 'grep -aE "PASSED|PER-TURN|codes" /root/arm2_linear.log /root/arm2_runtime.log; echo ---; tail -6 /root/runfix2.log' 2>/dev/null
    exit 0;; esac
  sleep 60
done
echo "TIMEOUT"; exit 1
