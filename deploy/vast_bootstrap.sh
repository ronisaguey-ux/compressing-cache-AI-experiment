#!/usr/bin/env bash
# One-shot setup for a rented GPU: deps, model, then run ONE arm.
# Written to be idempotent -- a resume re-runs it and skips what is done.
#
#   ARM=linear  CCAI_MODEL=gemma-4-26b-a4b ./vast_bootstrap.sh
#   ARM=runtime CCAI_MODEL=gemma-4-26b-a4b ./vast_bootstrap.sh
set -u

ARM="${ARM:-linear}"
MODEL_KEY="${CCAI_MODEL:-gemma-4-26b-a4b}"
TURNS="${CCAI_TURNS:-200}"
FEATURES="${CCAI_FEATURES:-200}"
WORK="${CCAI_WORK:-/root/ccai}"
CKPT="${CCAI_CKPT_DIR:-/root/ccai/ckpt}"
LOG="/root/ccai_${ARM}.log"

mkdir -p "$WORK" "$CKPT"
cd "$WORK" || exit 1

echo "=== [$(date -u +%H:%M:%S)] bootstrap: arm=$ARM model=$MODEL_KEY turns=$TURNS ==="

# ---------------------------------------------------------------- deps
if [ ! -f .deps_ok ]; then
  echo "--- installing deps (this is the slow part) ---"
  python3 -m pip install -q --upgrade pip 2>&1 | tail -1
  # transformers pinned: Gemma4ForCausalLM / Gemma4UnifiedForCausalLM need 5.17.0
  python3 -m pip install -q "transformers==5.17.0" accelerate bitsandbytes \
      safetensors sentencepiece 2>&1 | tail -3
  python3 -c "import torch, transformers, bitsandbytes as bnb; \
print('torch', torch.__version__, '| transformers', transformers.__version__, '| bnb', bnb.__version__); \
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    || { echo "!! dependency import failed"; exit 1; }
  touch .deps_ok
else
  echo "--- deps already installed, skipping ---"
fi

# ---------------------------------------------------------------- fatal if no GPU
python3 - <<'PY' || exit 1
import torch, sys
if not torch.cuda.is_available():
    print("!! NO CUDA DEVICE -- this instance is CPU-only, refusing to run"); sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | {p.total_memory/1024**3:.1f} GB | sm_{p.major}{p.minor}")
PY

# ---------------------------------------------------------------- run
echo "--- starting arm=$ARM (logging to $LOG) ---"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CCAI_MODEL="$MODEL_KEY"
export CCAI_ARM="$ARM"
export CCAI_TURNS="$TURNS"
export CCAI_FEATURES="$FEATURES"
export CCAI_CKPT_DIR="$CKPT"
export CCAI_RESUME=1
export CCAI_QUANT="${CCAI_QUANT:-8}"
export PYTHONUNBUFFERED=1
export HF_HOME="/root/hf"

python3 incremental_coding.py 2>&1 | tee "$LOG"
echo "=== [$(date -u +%H:%M:%S)] arm=$ARM finished, exit=${PIPESTATUS[0]} ==="
