#!/bin/bash
# run_gemma4.sh — the budget-safe Gemma 4 run, staged for the Lightning Studio.
#
# WHY THESE NUMBERS (all measured, not guessed):
#   - Balance is $4.93 (queried from /v1/users/billing/balance), A100-40GB is $2.19/hr
#     => 2.25 GPU-hours total. An unflagged 200-turn x2-arm run would cost ~$7 and
#        overdraw the account.
#   - The linear arm loses the turn-1 contract once its context fills. It fills at
#       383 tok/turn (measured), so the loss point is a function of the window:
#           12000 -> loses it at ~turn 31
#            6000 -> loses it at ~turn 16
#        A SMALLER WINDOW makes the pressure build sooner, so a SHORTER run still
#        discriminates. That is the legitimate way to cut cost, not cutting corners.
#   - Runtime anchors turn 1, so it never loses the contract at any window size.
#
# Defaults below: 6000 window, 40 turns  => ~16 turns with the contract gone, ~$1.50.
# Usage:  bash run_gemma4.sh [turns] [window]
set -u
TURNS="${1:-40}"
WINDOW="${2:-6000}"
MODEL="${MODEL:-gemma-4-12b}"
QUANT="${QUANT:-4}"
CKPT=/teamspace/studios/this_studio/ccai_ckpt
mkdir -p "$CKPT"

run_arm() {
  local ARM=$1 LOG=/teamspace/studios/this_studio/gemma4_${MODEL}_${ARM}.log
  for attempt in 1 2; do
    {
      echo "### $MODEL/$ARM turns=$TURNS window=$WINDOW attempt=$attempt $(date)"
      CCAI_DEVICE=cuda \
      CCAI_QUANT="$QUANT" \
      CCAI_MAX_PROMPT_TOKENS="$WINDOW" \
      CCAI_RUNTIME_TOKENS=2048 \
      CCAI_CKPT_DIR="$CKPT" \
      CCAI_RESUME=1 \
      HF_HOME=/teamspace/studios/this_studio/hf \
      python /teamspace/studios/this_studio/ccai/lightning/run_incremental.py \
        --model "$MODEL" --arm "$ARM" --features "$TURNS" --max-new 3072
      echo "### EXIT=$? $(date)"
    } >> "$LOG" 2>&1
    grep -q "RUN-COMPLETE" "$LOG" && return 0
    echo "### retrying $ARM (no RUN-COMPLETE marker)" >> "$LOG"
    sleep 30
  done
}

# Sequential, not parallel: one GPU, and two concurrent 12B models would OOM.
run_arm linear
run_arm runtime
echo "GEMMA4 DONE $(date)" > /teamspace/studios/this_studio/gemma4_done.log
