#!/usr/bin/env bash
# Wait for the 3-arm run to finish, then pull the results and report.
#
# ★ WHY A WATCHER AND NOT A POLL LOOP IN THE SESSION: the run takes hours. Polling it from a turn
# burns context on sleep, and if the session moves on the result is never collected. This runs as a
# background job (`ocbg`) so it survives the turn, and it wakes the session when there is something
# real to say -- the same rule as `ocbg` itself: never block on a long job, be told when it's done.
set -u
BOX="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20"
OUT="$HOME/.local/share/ccai-results/fixmode-v3"
mkdir -p "$OUT"
LOG="$OUT/watcher.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "watching for 3-arm completion"
DEADLINE=$(( $(date +%s) + 15*3600 ))     # 15h ceiling: the run cannot outlive its own budget

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # the outer chain log prints this only after all three arms have exited
    if timeout 60 $BOX 'grep -q "ALL ARMS DONE" /root/runfix3_outer.log 2>/dev/null' 2>/dev/null; then
        say "ALL ARMS DONE -- collecting"
        break
    fi
    # heartbeat: record where each arm is, so a stall is visible as a frozen number
    ST=$(timeout 60 $BOX 'for a in runtime linear prune; do printf "%s:%s " "$a" "$(grep -acE "turn [0-9]+/60" /root/arm3_$a.log 2>/dev/null || echo 0)"; done' 2>/dev/null | tr -d '\r')
    say "progress $ST"
    sleep 300
done

say "pulling results"
for f in runtime linear prune; do
    timeout 120 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" \
      -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "root@ssh6.vast.ai:/root/ccai/results/incremental_*_$f.json" "$OUT/" 2>/dev/null
    timeout 120 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" \
      -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "root@ssh6.vast.ai:/root/arm3_$f.log" "$OUT/" 2>/dev/null
done
timeout 60 $BOX 'cat /root/runfix3_outer.log' > "$OUT/chain.log" 2>/dev/null
ls -la "$OUT" | tee -a "$LOG"

# The fairness gate. It does NOT simply report the arms side by side: it reconciles unequal arms to
# the COMMON PREFIX of turns both completed, and refuses only when fewer than two turns are
# comparable. A comparison script that cannot refuse is just a formatter.
if command -v python3 >/dev/null; then
    say "--- gate ---"
    python3 "$HOME/.local/share/ccai-repo/tools/run_compare.py" "$OUT"/incremental_*.json 2>&1 | tee -a "$LOG" | tail -80
    say "--- results section (generated, never hand-typed) ---"
    python3 "$HOME/.local/share/ccai-repo/tools/make_results_section.py" 2>&1 | tee -a "$LOG" | tail -30
    say "--- figure (SVG, from the same cache_rows as the tables) ---"
    python3 "$HOME/.local/share/ccai-repo/tools/make_figures.py" "$OUT" \
        "$HOME/.local/share/ccai-repo/paper/figs/trajectory.svg" 2>&1 | tee -a "$LOG" | tail -5
    # ★ THE RESULTS SECTION MUST REACH THE SUBMISSION ARTIFACT, not just the markdown. The Kaggle
    # notebook is generated from PAPER.md, so leaving it out of this pipeline means the submitted
    # notebook keeps the placeholder §5 while the repo carries the real one -- the two drift exactly
    # where it matters most. Rebuilding here costs a second and closes that.
    say "--- submission notebook (rebuilt from the paper, so §5 is the measured one) ---"
    python3 "$HOME/.local/share/ccai-repo/tools/make_notebook.py" 2>&1 | tee -a "$LOG" | tail -5
fi
say "DONE"
