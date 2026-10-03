#!/usr/bin/env bash
# ★★★ FINALIZE THE SWE-BENCH n=20 RUNS (detached).
#
# Waits for both arms' result files on the two boxes, pulls them, computes the pairs, writes the
# analysis, and messages the owner. Run detached so the result lands whether or not a session is up.
set -u
OUT=/home/roni-saguey/.local/share/ccai-results/swe
LOG=/home/roni-saguey/.local/share/ccai-results/finalize_swe.log
REPO=/home/roni-saguey/.local/share/ccai-repo
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
SSH2="ssh -p 20554 root@ssh2.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
SSH5="ssh -p 36882 root@ssh5.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"

say "=== SWE FINALIZER START $(date -u) ==="
# wait for both arms to print their result
for i in $(seq 1 480); do
  a=$(timeout 45 $SSH2 'ls /root/ccai/swe_results/swe_gemma-4-12b_linear_*.json 2>/dev/null | wc -l')
  b=$(timeout 45 $SSH5 'ls /root/ccai/swe_results/swe_gemma-4-12b_runtime_*.json 2>/dev/null | wc -l')
  if [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ]; then say "both result files present"; break; fi
  [ $((i % 30)) -eq 0 ] && say "waiting linear=$a runtime=$b"
  sleep 60
done

# pull the newest n=20 file from each box (by exact newest name on its own box)
rf2=$(timeout 45 $SSH2 "ls -t /root/ccai/swe_results/swe_gemma-4-12b_linear_*.json 2>/dev/null | head -1")
[ -n "$rf2" ] && scp -P 20554 -i "$HOME/.ssh/vast_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@ssh2.vast.ai:$rf2" "$OUT/n20_linear.json" >/dev/null 2>&1 && say "pulled linear n20"
rf5=$(timeout 45 $SSH5 "ls -t /root/ccai/swe_results/swe_gemma-4-12b_runtime_*.json 2>/dev/null | head -1")
[ -n "$rf5" ] && scp -P 36882 -i "$HOME/.ssh/vast_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@ssh5.vast.ai:$rf5" "$OUT/n20_runtime.json" >/dev/null 2>&1 && say "pulled runtime n20"


