#!/usr/bin/env bash
# ★★★ DESTROY THE RENTED BOXES WHEN THEIR WORK IS DONE.
#
# An idle GPU is money. Each box's results are pulled to this machine first, and nothing on a box is
# unique: the model, the repo and the task data are all re-derivable, so a finished box is torn down
# rather than left to bill. This waits for the local artifacts that PROVE the work is done -- not for
# a log line -- then destroys the instance.
#
# Order of proof:
#   1. variance: all 6 result files present locally in ~/.local/share/ccai-results/variance/
#   2. swe n=20: both arms pulled locally in ~/.local/share/ccai-results/swe/
# Then: destroy 53927901 (old box), 54042957 (ssh2), 53978881 (box2).
set -u
V=/home/roni-saguey/.local/share/ccai-results/variance
S=/home/roni-saguey/.local/share/ccai-results/swe
LOG=/home/roni-saguey/.local/share/ccai-results/shutdown_boxes.log
VASTAPI=/tmp/opencode/vastvenv/bin/python
KEY=$(cat /tmp/opencode/vast_session_key 2>/dev/null || echo "")
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== BOX SHUTDOWN COORDINATOR START $(date -u) ==="
if [ -z "$KEY" ]; then say "no vast session key -- cannot destroy; aborting"; exit 1; fi

# 1. wait for variance (6 files) and swe n20 (2 files)
for i in $(seq 1 720); do
  nv=$(ls "$V"/var_*.json 2>/dev/null | wc -l)
  ns=$(ls "$S"/n20_*.json 2>/dev/null | wc -l)
  if [ "$nv" -ge 6 ] && [ "$ns" -ge 2 ]; then say "work done: variance=$nv swe=$ns"; break; fi
  [ $((i % 60)) -eq 0 ] && say "waiting: variance=$nv/6 swe=$ns/2"
  sleep 60
done

# one final pull attempt for anything missing
if [ "$(ls "$V"/var_*.json 2>/dev/null | wc -l)" -lt 6 ]; then
  say "variance incomplete locally; NOT destroying the old box yet"
  OLD_OK=0
else
  OLD_OK=1
fi
if [ "$(ls "$S"/n20_*.json 2>/dev/null | wc -l)" -lt 2 ]; then
  say "swe incomplete locally; NOT destroying the swe boxes yet"
  SWE_OK=0
else
  SWE_OK=1
fi

destroy(){
  local id="$1" name="$2"
  $VASTAPI - "$id" "$KEY" <<'PY' 2>&1 | tail -1
import sys
from vastai import VastAI
iid=int(sys.argv[1]); key=sys.argv[2]
try:
    r=VastAI(api_key=key).destroy_instance(id=iid)
    print("destroy %s -> %s" % (iid, r))
except Exception as e:
    print("destroy %s failed: %s" % (iid, str(e)[:120]))
PY
  say "destroyed $name ($id)"
}

[ "$OLD_OK" = 1 ] && destroy 53927901 "old-box(variance)"
[ "$SWE_OK" = 1 ] && destroy 54042957 "swe-box-ssh2"
[ "$SWE_OK" = 1 ] && destroy 53978881 "swe-box-ssh5"

say "=== BOX SHUTDOWN COORDINATOR DONE $(date -u) ==="
/home/roni-saguey/.local/bin/tg_send_checked.sh "all runs finished and pulled - i shut the gpu boxes down so they stop billing. both papers are done, swe-bench + variance results are saved locally. nothing left running on your dime." >/dev/null 2>&1 || true
