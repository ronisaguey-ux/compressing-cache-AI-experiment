#!/bin/bash
cd "$(dirname "$0")"
PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
OUT=data/keep_sweep.jsonl
: > "$OUT"
for k in 0.75 0.90; do
  for d in 0.15 0.35 0.55 0.75; do
    for m in line baseline; do
      timeout 900 $PY src/depth_sweep.py --depth $d --mode $m --keep-frac $k 2>/dev/null | tail -1 >> "$OUT"
      echo "[$(date +%H:%M:%S)] keep=$k depth=$d mode=$m avail=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo)MB" >> data/keep_progress.log
    done
  done
done
echo "[$(date +%H:%M:%S)] KEEPSWEEP DONE" >> data/keep_progress.log
