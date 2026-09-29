#!/bin/bash
cd "$(dirname "$0")"
PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
OUT=data/depth_sweep.jsonl
: > "$OUT"
for d in 0.15 0.35 0.55 0.75; do
  for m in baseline posnorm line both; do
    avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)
    echo "[$(date +%H:%M:%S)] depth=$d mode=$m avail=${avail}MB" >> data/sweep_progress.log
    if [ "$m" = "baseline" ]; then EXTRA="--check-causal"; else EXTRA=""; fi
    timeout 900 $PY src/depth_sweep.py --depth $d --mode $m $EXTRA 2>/dev/null | tail -1 >> "$OUT"
    echo "    rc=$? avail_after=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)MB" >> data/sweep_progress.log
  done
done
echo "[$(date +%H:%M:%S)] DONE" >> data/sweep_progress.log
