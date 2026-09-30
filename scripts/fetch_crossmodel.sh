#!/usr/bin/env bash
# Fetch a non-Qwen 7B-class model for the cross-architecture test (Bob's Test 2).
#
# Llama-3.1-8B is the named target but it is a GATED repo -- it needs both a token and a
# license acceptance on the HF account. Try it first; fall back to Mistral-7B-v0.3, which
# is Apache-2.0 and ungated, so the test cannot be blocked by an account setting.
#
# Download happens in the background because it is the long pole (14-16 GB); the other
# three validation tests do not need it and run meanwhile.
set -u
cd /tmp/opencode/ccai
exec >> data/crossmodel_fetch.log 2>&1
export HF_HOME=/tmp/opencode/hfhome
export TOKENIZERS_PARALLELISM=false

PY=/home/roni/Roni_workspace/humanizer/.venv/bin/python
TOKEN_FILE=/home/roni/Roni_workspace/text-improver/.secrets/hf_token
if [ -f "$TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '\n' < "$TOKEN_FILE")"
  echo "[fetch] using HF token from $TOKEN_FILE"
fi

echo "=== fetch start $(date +%H:%M:%S) ==="
for MODEL in "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-v0.1"; do
  echo "[fetch] trying $MODEL"
  if $PY - <<PYEOF
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("$MODEL", allow_patterns=["*.safetensors","*.json","*.model","tokenizer*","*.txt"])
    print("OK", p)
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:200])
    sys.exit(1)
PYEOF
  then
    echo "[fetch] SUCCESS $MODEL" | tee data/crossmodel_model.txt
    echo "$MODEL" > data/crossmodel_model.txt
    break
  else
    echo "[fetch] unusable: $MODEL"
  fi
done
echo "=== fetch done $(date +%H:%M:%S) ==="
