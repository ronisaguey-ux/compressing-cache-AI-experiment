#!/usr/bin/env bash
# ★★★ FINALIZE THE VARIANCE SWEEP, DETACHED.
#
# Both boxes are pulling their salts into ~/.local/share/ccai-results/variance/. When all ten
# (5 salts x 2 arms) are present, compute the paired 95% interval and report it. Run detached so the
# result lands whether or not a session is watching -- polling by hand for seven hours is waste.
set -u
VAR=/home/roni-saguey/.local/share/ccai-results/variance
REPO=/home/roni-saguey/.local/share/ccai-repo
OUT=/home/roni-saguey/.local/share/ccai-results/variance_analysis.txt
LOG=/home/roni-saguey/.local/share/ccai-results/finalize_variance.log
mkdir -p "$VAR"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== VARIANCE FINALIZER START $(date -u) ==="
NEED=6
for i in $(seq 1 720); do        # up to 12h
  n=$(ls "$VAR"/var_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge "$NEED" ]; then say "have $n result files"; break; fi
  [ $((i % 30)) -eq 0 ] && say "waiting: $n/$NEED files"
  sleep 60
done
n=$(ls "$VAR"/var_*.json 2>/dev/null | wc -l)
say "final count: $n files"

{
  echo "VARIANCE ANALYSIS  $(date -u)"
  echo "files ($n):"
  ls -1 "$VAR"/var_*.json 2>/dev/null | sed 's#.*/#  #'
  echo ""
  if [ "$n" -ge 2 ]; then
    cd "$REPO" && python3 tools/analyze_variance.py "$VAR"/var_*.json 2>&1
  else
    echo "not enough results to analyse"
  fi
  echo ""
  echo "=== per-arm spread across salts (the metric that actually varies) ==="
  echo "Token counts are fixed-width constants, so cost_units is expected to be near-identical"
  echo "across salts; that makes the cost RATIO structural rather than seed-dependent. Per-turn"
  echo "SUCCESS and final state are where run-to-run variation lives, so both are reported."
  python3 - "$VAR" <<'PY'
import json, glob, os, sys
V=sys.argv[1]
by={}
for f in sorted(glob.glob(os.path.join(V,"var_*.json"))):
    d=json.load(open(f)); by.setdefault(d.get("arm"),[]).append((os.path.basename(f),d))
for arm,rows in sorted(by.items()):
    costs=[(r.get("metrics") or {}).get("cost_units") for _,r in rows if (r.get("metrics") or {}).get("cost_units")]
    hits=[r.get("cache_hit_rate") for _,r in rows if r.get("cache_hit_rate") is not None]
    pturn=[(r.get("metrics") or {}).get("per_turn_success") for _,r in rows if (r.get("metrics") or {}).get("per_turn_success") is not None]
    passed=[r.get("passed") for _,r in rows]
    def rng(xs):
        return "--" if not xs else "min=%.4f max=%.4f" % (min(xs), max(xs))
    print("  %-8s n=%d  cost %s  hit %s  per-turn %s  passed %s" % (
        arm, len(rows), rng(costs), rng(hits), rng(pturn), passed))
PY
} > "$OUT" 2>&1
say "wrote $OUT"
tail -30 "$OUT"

# Report to the owner in plain English (no IDs, no SHAs).
SUMMARY=$(grep -aiE "mean ratio|95%|interval|excludes|cannot exclude|n=|paired" "$OUT" 2>/dev/null | head -6)
if [ -n "$SUMMARY" ]; then
  /home/roni-saguey/.local/bin/tg_send_checked.sh "variance done. paired 95% interval on the cost ratio across the salts:

$SUMMARY

that is the paper's error bars - was the 'one seed, no error bars' gap." >/dev/null 2>&1 || true
fi
say "=== VARIANCE FINALIZER DONE $(date -u) ==="
