#!/usr/bin/env bash
# Run ONE arm of the incremental-coding benchmark.
#
# A separate process per arm, deliberately: the 26B model is fully released between arms.
# A second load inside one process was measured to OOM during loading.
#
# ★ QUANTISATION IS `pre`, NOT `8`, AND THAT IS NOT A STYLE CHOICE.
#   `CCAI_QUANT=8` -> BitsAndBytesConfig(load_in_8bit=True) on google/gemma-4-26B-A4B-it.
#   MEASURED: that does NOT produce an 8-bit model. bnb quantises only nn.Linear, and in this
#   MoE the expert weights are raw parameters, so they (and the clippable linears) stay bf16.
#   426 Linear8bitLt vs 189 Gemma4ClippableLinear; bytes/param came out 1.94, i.e. bf16;
#   48.8 GB allocated on a 48 GB card -> OOM on the FIRST forward pass.
#   `pre` loads cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit, quantised at conversion time
#   (compressed-tensors w8a8, group_size 32) so the experts are covered: ~29 GB, real int8.
ARM="${1:?usage: run_arm.sh <linear|runtime>}"
cd /root/ccai || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok)"

export CCAI_MODEL="${CCAI_MODEL:-gemma-4-26b-a4b-awq8}"
export CCAI_QUANT=pre
export CCAI_ARM="$ARM"
export CCAI_FEATURES="${CCAI_FEATURES:-200}"
export CCAI_KEEP_TURNS="${CCAI_KEEP_TURNS:-4}"
export CCAI_CKPT_DIR=/root/ccai/ckpt
export CCAI_RESUME=1
export CCAI_RUNTIME_TOKENS=2048
# Emergency break only -- both arms stop on TURN COUNT. A wall-clock stop is arm-dependent
# because the runtime arm is faster per turn. See tools/run_compare.py.
export CCAI_TIME_BUDGET_S="${CCAI_TIME_BUDGET_S:-21600}"

if [ -n "$HF_TOKEN" ]; then AUTH=yes; else AUTH=NO; fi
echo "=== [$(date -u +%H:%M:%S)] arm=$ARM model=$CCAI_MODEL quant=$CCAI_QUANT features=$CCAI_FEATURES hf_auth=$AUTH ==="
/venv/main/bin/python /root/ccai/vast_entry.py
rc=$?
echo "=== [$(date -u +%H:%M:%S)] arm=$ARM EXIT=$rc ==="
exit $rc
