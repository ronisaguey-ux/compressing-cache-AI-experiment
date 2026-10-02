#!/usr/bin/env bash
# Pull both arms off the GPU box and run the FAIRNESS GATE, then archive durably.
# /tmp is wiped at boot, so the results are copied into ~/.local/share/ccai-results/.
set -u
OUT="$HOME/.local/share/ccai-results/fixmode-v2-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
SSH="ssh -p 29980 root@ssh4.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -P 29980 -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "=== pulling results to $OUT ==="
timeout 300 $SCP root@ssh4.vast.ai:/root/ccai/results/*.json "$OUT/" 2>&1 | tail -2
timeout 200 $SCP root@ssh4.vast.ai:/root/arm2_linear.log root@ssh4.vast.ai:/root/arm2_runtime.log "$OUT/" 2>&1 | tail -2

echo
echo "=== FAIRNESS GATE + HEAD-TO-HEAD ==="
# ★ PIPESTATUS, NOT $?. `$?` after a pipeline is the LAST command's status -- tee's -- so the
# gate could refuse an unfair comparison and this script would still report exit 0. The gate
# exists precisely to stop bad numbers being quoted; reading its exit code through a pipe
# defeats it.
set -o pipefail
timeout 200 python3 /home/roni-saguey/.local/share/ccai-repo/tools/run_compare.py "$OUT/*.json" | tee "$OUT/COMPARE.txt"
GATE=${PIPESTATUS[0]}
set +o pipefail
echo
echo "gate exit=$GATE  (0 = fair comparison; nonzero = REFUSED, do not quote the numbers)"

echo
echo "=== per-turn success, both arms ==="
for a in linear runtime; do
  f="$OUT/arm2_$a.log"
  [ -f "$f" ] || continue
  ok=$(grep -ac "  turn [0-9]* OK"  "$f")
  fl=$(grep -ac "  turn [0-9]* FAIL" "$f")
  echo "  $a: OK=$ok FAIL=$fl"
done
echo
echo "saved: $OUT"
ls -la "$OUT"
