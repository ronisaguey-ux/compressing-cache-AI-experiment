#!/usr/bin/env bash
# Wake when the CLEAN ablation result has been pulled locally (by the box-2 chain).
set -u
OUT=/home/roni-saguey/.local/share/ccai-results/fixmode-v3
RF=$OUT/incremental_gemma-4-12b_ablate-recent.json
ARCH=$OUT/CONTAMINATED_incremental_gemma-4-12b_ablate-recent.resume-artifact.json
for i in $(seq 1 240); do
  # the clean result is "new" if it exists and the archive was written (i.e. a re-run happened)
  if [ -s "$RF" ] && [ -s "$ARCH" ]; then
    python3 - "$RF" "$ARCH" <<'PY'
import json,sys,hashlib
a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read()
print("clean==archive(identical):", hashlib.md5(a).hexdigest()==hashlib.md5(b).hexdigest())
PY
    echo "[$(date -u +%H:%M:%S)] clean ablation result present"
    exit 0
  fi
  sleep 60
done
echo "[$(date -u +%H:%M:%S)] timeout"; exit 1
