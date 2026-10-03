#!/usr/bin/env bash
# Collect the third arm when it finishes, then regenerate §5 from the REAL results and
# rebuild the Kaggle notebook. Wakes this session via oc_send.js at the end.
#
# Usage: watch_prune_and_finalize.sh <host> <port> <arm>
# Durable on purpose — /tmp is wiped at every boot, so this lives in the repo.
set -uo pipefail

HOST="${1:?host}"; PORT="${2:?port}"; ARM="${3:?arm}"
REPO="$HOME/.local/share/ccai-repo"
RES="$HOME/.local/share/ccai-results/fixmode-v3"
LOG="/tmp/opencode/watch_${ARM}.log"
SSH="ssh -p $PORT root@$HOST -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
SCP="scp -P $PORT -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes"

mkdir -p "$RES"
exec >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] watching $ARM on $HOST:$PORT"

for i in $(seq 1 300); do
  S=$(timeout 60 $SSH "tail -c 3000 /root/runfix3.log 2>/dev/null" 2>/dev/null)
  case "$S" in
    *"ALL ARMS DONE"*|*"${ARM} DONE"*|*"DONE"*) break ;;
  esac
  # a per-arm marker is the reliable one: the arm's own log printed its final line
  LAST=$(timeout 60 $SSH "grep -aoE 'turn [0-9]+/60' /root/arm3_${ARM}.log 2>/dev/null | tail -1" 2>/dev/null)
  if [ "$LAST" = "turn 60/60" ]; then break; fi
  sleep 60
done
echo "[$(date -u +%H:%M:%S)] $ARM appears finished (loop=$i)"

# ---- collect ---------------------------------------------------------------
timeout 180 $SCP "root@$HOST:/root/ccai/results/incremental_*_${ARM}.json" "$RES/" && echo "scp ok" || echo "scp FAILED"
timeout 120 $SCP "root@$HOST:/root/arm3_${ARM}.log" "$RES/" >/dev/null 2>&1 || true

# ---- regenerate §5 from whatever arms we actually have ----------------------
cd "$REPO" || exit 1
# ★ THESE TOOLS TAKE ENV VARS AND POSITIONALS, NOT FLAGS. Passing --results-dir/--paper/--out was
# silently ignored: the generator fell back to a FIXTURE directory and the figure tool ran with an
# empty path (FileNotFoundError: ''). The docstrings said so; the watcher contradicted them.
cp paper/PAPER.md /tmp/opencode/paper_live.md
RESULTS_DIR="$RES" PAPER=/tmp/opencode/paper_live.md python3 tools/make_results_section.py
mkdir -p paper/figs
RESULTS_DIR="$RES" python3 tools/make_figures.py "$RES" "$REPO/paper/figs/trajectory.svg" || true
PAPER=/tmp/opencode/paper_live.md NB_OUT="$REPO/paper/gemma4-paper-track.ipynb" \
  python3 tools/make_notebook.py || true
# §5 must not be a placeholder in the artifact that gets submitted
if grep -q "to be completed from the run in flight" paper/gemma4-paper-track.ipynb 2>/dev/null; then
  echo "WARNING: notebook still carries a placeholder §5"
fi
python3 tools/check_citations.py paper/PAPER.md 2>/dev/null || true

# ---- report ----------------------------------------------------------------
SUMMARY=$(python3 - <<'PY'
import json, glob, os
res = os.environ["RES"]
rows = {}
for p in glob.glob(os.path.join(res, "incremental_*.json")):
    d = json.load(open(p))
    if d.get("arm"): rows[d["arm"]] = d
def m(d, k):
    v = d.get(k)
    return v if v is not None else (d.get("metrics") or {}).get(k)
out = []
for a in ("runtime", "linear", "prune"):
    if a not in rows: out.append("%s: missing" % a); continue
    d = rows[a]
    out.append("%s: passed %s/%s codes %s gate %s cost %s hit %.3f" % (
        a, m(d,"passed"), m(d,"features_completed"), m(d,"codes_n"),
        m(d,"gate_ok"), m(d,"cost_units"),
        (m(d,"cache_hit_rate") or 0)))
r, l = rows.get("runtime"), rows.get("linear")
if r and l and m(r,"cost_units") and m(l,"cost_units"):
    out.append("cost ratio linear/runtime: %.2fx" % (m(l,"cost_units")/m(r,"cost_units")))
print(" | ".join(out))
PY
)
echo "$SUMMARY"
HOME=/home/roni-saguey /home/roni-saguey/.local/bin/tg_send_checked.sh "ccai $ARM finished. $SUMMARY" >/dev/null 2>&1 || true
node "$HOME/.local/lib/ocbridge/oc_send.js" "ccai run: $ARM finished. $SUMMARY" --async >/dev/null 2>&1 || true
echo "[$(date -u +%H:%M:%S)] done"
