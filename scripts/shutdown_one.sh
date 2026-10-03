#!/usr/bin/env bash
# Destroy ONE rented box once its result is pulled locally. Cost discipline: an idle GPU is money.
set -u
S=/home/roni-saguey/.local/share/ccai-results/swe
LOG=/home/roni-saguey/.local/share/ccai-results/shutdown_one.log
KEY=$(cat /tmp/opencode/vast_session_key 2>/dev/null || echo "")
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
say "=== single-box shutdown watch $(date -u) ==="
[ -z "$KEY" ] && { say "no key"; exit 1; }
for i in $(seq 1 360); do
  if timeout 45 ssh -p 36882 root@ssh5.vast.ai -i "$HOME/.ssh/vast_ed25519" -o StrictHostKeyChecking=no \
     -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
     'ls /root/ccai/swe_results/swe_gemma-4-12b_runtime_*.json >/dev/null 2>&1'; then
    # pull the newest
    RF=$(timeout 45 ssh -p 36882 root@ssh5.vast.ai -i "$HOME/.ssh/vast_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes 'ls -t /root/ccai/swe_results/swe_gemma-4-12b_runtime_*.json | head -1')
    scp -P 36882 -i "$HOME/.ssh/vast_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@ssh5.vast.ai:$RF" "$S/n20_runtime.json" >/dev/null 2>&1 && say "pulled n20 runtime"
    break
  fi
  sleep 60
done
/tmp/opencode/vastvenv/bin/python - <<'PY' 2>&1 | tail -1
from vastai import VastAI
v=VastAI(api_key=open('/tmp/opencode/vast_session_key').read().strip())
try: print("destroy 53978881 ->", v.destroy_instance(id=53978881))
except Exception as e: print("destroy failed:", str(e)[:100])
PY
say "=== done $(date -u) ==="
