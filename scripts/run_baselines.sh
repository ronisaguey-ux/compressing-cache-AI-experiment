#!/bin/bash
cd "$(dirname "$0")"
PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
OUT=data/baseline_causal.jsonl
: > "$OUT"
for d in 0.15 0.35 0.55 0.75; do
  avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)
  echo "[$(date +%H:%M:%S)] baseline depth=$d avail=${avail}MB" >> data/sweep_progress.log
  timeout 900 $PY src/depth_sweep.py --depth $d --mode baseline --check-causal 2>/dev/null | tail -1 >> "$OUT"
  echo "    done avail_after=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)MB" >> data/sweep_progress.log
done
echo "[$(date +%H:%M:%S)] BASELINES DONE" >> data/sweep_progress.log
