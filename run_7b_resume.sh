#!/bin/bash
# Resume the 7B sweep, skipping (depth, mode) pairs already recorded.
# The heavy_lock inside Harness now refuses a second 7B process, so this cannot
# repeat the 2026-09-29 crash by being started twice.
cd "$(dirname "$0")"
PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
MODEL=${CCAI_MODEL_7B:-Qwen/Qwen2.5-7B}
OUT=data/run_7b.jsonl
LOG=data/run_7b.log
echo "[$(date +%H:%M:%S)] RESUME start" >> "$LOG"

for d in 0.15 0.35 0.55 0.75; do
  for m in baseline line posnorm; do
    if grep -q "\"depth\": $d, \"mode\": \"$m\"" "$OUT" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] skip $d $m (already recorded)" >> "$LOG"
      continue
    fi
    avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)
    echo "[$(date +%H:%M:%S)] run depth=$d mode=$m avail=${avail}MB" >> "$LOG"
    CCAI_QUANT=8bit timeout 5400 $PY src/depth_sweep.py \
        --model "$MODEL" --depth $d --mode $m --keep-frac 0.60 \
        > "data/_7b_${d}_${m}.out" 2> "data/_7b_${d}_${m}.err"
    rc=$?
    if grep -q "^{" "data/_7b_${d}_${m}.out" 2>/dev/null; then
      grep "^{" "data/_7b_${d}_${m}.out" | tail -1 >> "$OUT"
      echo "    ok rc=$rc" >> "$LOG"
    else
      echo "[$(date +%H:%M:%S)] FAILED $d $m rc=$rc: $(tail -1 data/_7b_${d}_${m}.err 2>/dev/null | cut -c1-120)" >> "$LOG"
    fi
  done
done
echo "[$(date +%H:%M:%S)] RESUME DONE" >> "$LOG"
