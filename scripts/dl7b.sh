#!/bin/bash
export HF_HOME=/tmp/opencode/hfhome
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
cd /tmp/opencode/ccai
LOG=data/dl7b.log
echo "[$(date +%H:%M:%S)] start" >> "$LOG"
/home/roni/Roni_workspace/humanizer/.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import os, time
os.environ.setdefault("HF_HOME","/tmp/opencode/hfhome")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
from huggingface_hub import snapshot_download
for attempt in range(1, 9):
    try:
        t0=time.time()
        p = snapshot_download("Qwen/Qwen2.5-7B",
              allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*"],
              max_workers=8)
        print("DOWNLOADED on attempt %d in %.0fs" % (attempt, time.time()-t0))
        break
    except Exception as e:
        print("attempt %d failed: %s %s" % (attempt, type(e).__name__, str(e)[:160]))
        time.sleep(20)
PY
echo "[$(date +%H:%M:%S)] EXIT=$?" >> /tmp/opencode/ccai/data/dl7b.log
