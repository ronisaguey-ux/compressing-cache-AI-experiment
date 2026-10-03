#!/bin/bash
# The 7B validation run. Bob's instruction: freeze the 0.5B harness, run the exact
# pipeline on Qwen2.5-7B at 8-bit, log the depth breakdown, see whether the
# conditioning wall scales away.
#
# 7B at 8-bit needs ~7GB weights + activations on a 15.7GB box shared with
# laya-serve. One process per combination, memory-checked before each.
cd "$(dirname "$0")"
PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
MODEL=${CCAI_MODEL_7B:-Qwen/Qwen2.5-7B}
OUT=data/run_7b.jsonl
LOG=data/run_7b.log
: > "$OUT"
echo "[$(date +%H:%M:%S)] 7B run start, model=$MODEL" >> "$LOG"

for d in 0.15 0.35 0.55 0.75; do
  for m in baseline line posnorm; do
    avail=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)
    echo "[$(date +%H:%M:%S)] depth=$d mode=$m avail=${avail}MB" >> "$LOG"
    CCAI_QUANT=8bit timeout 5400 $PY src/depth_sweep.py \
        --model "$MODEL" --depth $d --mode $m --keep-frac 0.60 \
        > "data/_7b_${d}_${m}.out" 2> "data/_7b_${d}_${m}.err"
    rc=$?
    if grep -q "^{" "data/_7b_${d}_${m}.out"; then
        grep "^{" "data/_7b_${d}_${m}.out" | tail -1 >> "$OUT"
    else
        echo "{\"depth\": $d, \"mode\": \"$m\", \"pass_\": false, \"error\": \"rc=$rc or no json\", \"stderr_tail\": \"$(tail -1 data/_7b_${d}_${m}.err 2>/dev/null | tr -d '\"' | cut -c1-160)\"}" >> "$OUT"
    fi
    echo "    rc=$? avail_after=$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo)MB" >> "$LOG"
  done
done
echo "[$(date +%H:%M:%S)] 7B RUN DONE" >> "$LOG"
