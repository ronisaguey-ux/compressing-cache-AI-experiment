#!/usr/bin/env bash
# ★★ THE COMPETITION CHECKPOINT RUN.
#
# The main Gemma 4 Developer Agent competition (comp 149921) mandates
# `google/gemma-4-31B-it-qat-w4a16-ct`. The paper track does not, but running the SAME two
# policies on the exact checkpoint the host ships forecloses the single most likely reviewer
# objection -- "this is a small-model artifact" -- and makes the result a statement about Gemma 4
# as the competition defines it.
#
# 23.3 GB on disk (pack-quantized w4a16 via compressed-tensors). It must be loaded with
# CCAI_QUANT=pre: the checkpoint carries its own quantization_config, and handing the loader a
# BitsAndBytesConfig fights it and silently loads UNQUANTISED (the measured failure on 26B-A4B,
# where bytes/param came back 1.94 = bf16 and the first forward OOM'd).
#
# Usage on the box:  bash /root/ccai/run31b.sh [features]
set -u
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok 2>/dev/null || echo '')"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N="${1:-${CCAI_FEATURES:-60}}"
MODEL="${CCAI_MODEL:-gemma-4-31b-qat}"
LOG=/root/run31b.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 31B QAT (competition checkpoint)  N=$N  quant=pre  arms=runtime,linear ==="
say "python: $(/venv/main/bin/python -c 'import torch,sys;print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))' 2>&1 | tail -1)"

# compressed-tensors is REQUIRED for the pack-quantized checkpoint. Install it before the first
# load rather than discovering the ImportError mid-run, and verify the import afterwards.
/venv/main/bin/python -m pip install -q compressed-tensors 2>&1 | tail -2
/venv/main/bin/python -c "import compressed_tensors; print('compressed_tensors', compressed_tensors.__version__)" | tee -a "$LOG"

# ★ PREFLIGHT: pre-download so a network failure costs minutes, not a half-finished arm, and so the
# first forward is a load test rather than a download-and-pray.
say "--- pre-downloading $MODEL ---"
/venv/main/bin/python - <<'PY' 2>&1 | tail -3 | tee -a "$LOG"
import os
from huggingface_hub import snapshot_download
p = snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct",
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)
print("downloaded to %s  (%.1f GB)" % (p, tot / 1e9))
PY

for ARM in runtime linear; do
  rm -rf /work/inc
  mkdir -p /root/ccai/ckpt31_$ARM
  say "--- arm=$ARM starting on $MODEL ---"
  # CCAI_ARM=runtime/linear selects the context policy; the task, gate and budgets are identical
  # to the 12B run so the two model sizes are directly comparable.
  CCAI_TASK=fix CCAI_MODEL="$MODEL" CCAI_QUANT=pre CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=$ARM CCAI_FEATURES=$N CCAI_GATE=1 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
  CCAI_TIME_BUDGET_S="${CCAI_TIME_BUDGET_S:-16200}" CCAI_CKPT_DIR=/root/ccai/ckpt31_$ARM \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arm31_$ARM.log 2>&1
  rc=$?
  say "arm=$ARM exit=$rc"
  grep -aE "PASSED|PER-TURN|GRADER FAILED|Error|error" /root/arm31_$ARM.log | tail -3 | tee -a "$LOG"
done
say "=== 31B ARMS DONE ==="
ls -la /root/ccai/results/ | tee -a "$LOG"
