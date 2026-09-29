#!/bin/bash
export HF_HOME=/tmp/opencode/hfhome
cd /tmp/opencode/ccai
: > data/dl7b.log
exec /home/roni/Roni_workspace/humanizer/.venv/bin/python - >> data/dl7b.log 2>&1 <<'PY'
import os, time
os.environ.setdefault("HF_HOME","/tmp/opencode/hfhome")
from huggingface_hub import snapshot_download
t0=time.time()
try:
    p = snapshot_download("Qwen/Qwen2.5-7B",
          allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*"],
          max_workers=4)
    print("DOWNLOADED", p, "in %.0fs" % (time.time()-t0))
except Exception as e:
    print("DOWNLOAD FAILED:", type(e).__name__, str(e)[:300])
PY
echo "EXIT=$?" >> /tmp/opencode/ccai/data/dl7b.log
