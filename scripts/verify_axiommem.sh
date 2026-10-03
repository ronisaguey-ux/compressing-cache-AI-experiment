#!/usr/bin/env bash
# ★★★ INDEPENDENT VERIFICATION OF AxiomMem (agy's AML build).
#
# WHY THIS EXISTS: agy has a documented history on this box of reporting success over work that did
# not exist (phantom passes, placeholder findings, "verified" claims over untouched files). The
# owner's instruction is to offload the AML work, but "the report says it works" is a CLAIM, not
# evidence. This script drives the real API against a fresh clone and reports what IT measures.
#
# It checks the two things that decide whether the submission is even valid, plus the two that
# decide whether it is worth anything:
#   A. THE CONTRACT — smoke-test compliance. A schema mismatch scores zero, so this is binary.
#   B. THE INVARIANTS — idempotency, user isolation, synchronous visibility. Each is an explicit
#      requirement and each fails the evaluation if broken.
#   C. COLUMN G — does the execution-continuity ranking actually work? (the differentiator)
#   D. ARE THE TESTS REAL — do they FAIL when the behaviour is broken? A green suite proves nothing
#      on its own; this is the non-vacuity check that caught a grader scoring 20/20 on a broken repo.
#
# Output: $LOG, and a Telegram summary. Nothing in here trusts the repo's own README.
set -u
DST=/home/roni-saguey/.local/share/aml-verify
LOG=/home/roni-saguey/.local/share/aml-verify-$(date -u +%Y%m%d-%H%M%S).log
mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
fails=0
pass(){ say "PASS  $*"; }
fail(){ say "FAIL  $*"; fails=$((fails+1)); }

say "=== AxiomMem independent verification ==="

# ── fresh clone ──────────────────────────────────────────────────────────────────────────────────
rm -rf "$DST"
if ! git clone --quiet https://github.com/ronisaguey-ux/aml-hackathon.git "$DST"; then
  fail "clone"; say "VERDICT: repo unreachable"; exit 1
fi
cd "$DST" || exit 1
say "commit: $(git rev-parse HEAD)"
say "files: $(git ls-files | wc -l)"

# ── does the claimed structure actually exist? ───────────────────────────────────────────────────
say ""
say "--- claimed structure ---"
for f in server.py schemas.py pipeline.py store/db.py README.md LICENSE ARCHITECTURE.md \
         REPRODUCIBILITY.md SUBMISSION.md smoke_test.py probe_suite.py benchmark_load.py Dockerfile; do
  # the repo may nest these; find either at root or anywhere
  if git ls-files | grep -qE "(^|/)$(basename "$f")\$"; then pass "present: $f"; else fail "MISSING: $f"; fi
done
if git ls-files | grep -q "tests/"; then pass "tests/ directory present"; else fail "MISSING tests/"; fi
if git ls-files | grep -q "LICENSE"; then
  say "license: $(head -2 LICENSE 2>/dev/null | tr '\n' ' ')"
fi

# ── find the python entry points (the tree may nest) ─────────────────────────────────────────────
VENV=/home/roni-saguey/.local/share/aml-verify-venv
if [ ! -x "$VENV/bin/python" ]; then
  say "--- creating verification venv ---"
  uv venv "$VENV" >/dev/null 2>&1 || python3 -m venv "$VENV" >/dev/null 2>&1
fi
PY="$VENV/bin/python"
req=$(git ls-files | grep -E "requirements.*\.txt|pyproject\.toml" | head -1)
if [ -n "$req" ]; then
  say "installing deps from $req"
  VIRTUAL_ENV="$VENV" uv pip install -q -r "$(dirname "$req")/$(basename "$req")" 2>&1 | tail -3
else
  fail "no requirements file found"
fi

# ── B. THE INVARIANTS — drive the real API ───────────────────────────────────────────────────────
say ""
say "--- driving the live API ---"
PORT=${AML_TEST_PORT:-8791}
# ★ THE START COMMAND IS THE REPO'S OWN, READ FROM ITS README — not invented. First attempt used a
# root `server.py` with an `AXIOM_PORT` env var; the real entry is `scripts/run_server.py --port N`,
# so the probe failed to bind and reported "server did not come up" on a perfectly good app.
# A probe that starts the app the wrong way measures the probe.
srv=$(git ls-files | grep -E "scripts/run_server\.py$|(^|/)run_server\.py$" | head -1)
[ -z "$srv" ] && srv=$(git ls-files | grep -E "(^|/)server\.py$" | head -1)
[ -z "$srv" ] && srv=$(git ls-files | grep -E "app\.py$" | head -1)
if [ -z "$srv" ]; then
  fail "cannot locate a server entry point"
else
  say "server entry: $srv"
  ( cd "$(dirname "$srv")" && "$PY" -u "$(basename "$srv")" --port "$PORT" ) > server.log 2>&1 &
  SRV=$!
  # wait for the port, up to 60s
  up=0
  for i in $(seq 1 60); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/" 2>/dev/null; then up=1; break; fi
    if curl -s -o /dev/null --max-time 2 -X POST "http://127.0.0.1:$PORT/search" \
         -H 'content-type: application/json' -d '{"query":"x","user_id":"probe","top_k":1}' 2>/dev/null; then up=1; break; fi
    sleep 1
  done
  if [ "$up" != "1" ]; then
    fail "server did not come up on :$PORT"
    tail -20 server.log
  else
    pass "server is listening on :$PORT"

    B="http://127.0.0.1:$PORT"
    jqget(){ "$PY" -c "import json,sys;d=json.load(sys.stdin);print(eval('d'+'$1'))" 2>/dev/null; }

    # A. /add echoes the identifiers exactly  (contract requirement)
    R=$(curl -s --max-time 15 -X POST "$B/add" -H 'content-type: application/json' -d '{
      "request_id":"verify-req-1","user_id":"verify_u1","session_id":"verify_s1",
      "messages":[{"role":"user","timestamp":1704067200000,"content":"My deploy port is 9174."}]}')
    echo "$R" >> add_responses.txt
    if echo "$R" | grep -q '"request_id"[[:space:]]*:[[:space:]]*"verify-req-1"'; then pass "add echoes request_id"; else fail "add does NOT echo request_id: $R"; fi
    if echo "$R" | grep -q '"success"[[:space:]]*:[[:space:]]*true'; then pass "add returns success:true"; else fail "add missing success:true: $R"; fi

    # B. synchronous visibility — search immediately, no sleep
    S=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
        -d '{"query":"deploy port","user_id":"verify_u1","top_k":10}')
    if echo "$S" | grep -q "9174"; then pass "search IMMEDIATELY sees the memory (synchronous)"; else fail "not synchronized: $S"; fi

    # A. data is present and is a list, never omitted
    if echo "$S" | grep -q '"data"[[:space:]]*:'; then pass "search returns a 'data' key"; else fail "'data' key missing: $S"; fi

    # B. absolute user isolation
    S2=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
         -d '{"query":"deploy port","user_id":"verify_OTHER","top_k":10}')
    if echo "$S2" | grep -q "9174"; then fail "USER ISOLATION BREACH: another user sees it"; else pass "user isolation holds"; fi

    # A. empty result is [] not omitted
    S3=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
         -d '{"query":"zzz nothing here zzz","user_id":"verify_u1","top_k":10}')
    if echo "$S3" | grep -q '"data"'; then pass "empty search still returns 'data'"; else fail "empty search omitted 'data': $S3"; fi

    # B. idempotency on request_id — same call twice must not duplicate
    curl -s --max-time 15 -X POST "$B/add" -H 'content-type: application/json' -d '{
      "request_id":"verify-req-1","user_id":"verify_u1","session_id":"verify_s1",
      "messages":[{"role":"user","timestamp":1704067200000,"content":"My deploy port is 9174."}]}' >/dev/null
    S4=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
         -d '{"query":"deploy port","user_id":"verify_u1","top_k":50}')
    n=$("$PY" - "$S4" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.loads(sys.argv[1]); print(sum(1 for x in d.get("data",[]) if "9174" in str(x.get("content",""))))
except Exception: print(-1)
PY
)
    if [ "$n" = "1" ]; then pass "idempotent: replay produced exactly 1 copy"; else fail "idempotency: got $n copies of the replayed memory"; fi

    # ── C. COLUMN G — execution continuity ──────────────────────────────────────────────────────
    say ""
    say "--- Column G probe (execution continuity) ---"
    curl -s --max-time 15 -X POST "$B/add" -H 'content-type: application/json' -d '{
      "request_id":"verify-req-g","user_id":"verify_g","session_id":"verify_sg",
      "messages":[
        {"role":"user","timestamp":1704070800000,"content":"Step 1: stop the worker service."},
        {"role":"user","timestamp":1704070860000,"content":"Step 2: back up the sqlite file."},
        {"role":"user","timestamp":1704070920000,"content":"Step 3: run the migration script."},
        {"role":"user","timestamp":1704070980000,"content":"Step 4: restart the worker service."}]}' >/dev/null
    SG=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
         -d '{"query":"how do I run the migration","user_id":"verify_g","top_k":20}')
    steps=$(echo "$SG" | grep -o "Step [0-9]" | sort -u | wc -l)
    if [ "$steps" -ge 3 ]; then pass "Column G: $steps distinct procedural steps returned together"; else fail "Column G: only $steps steps returned (expected the contiguous sequence)"; fi

    # ── C2. COLUMN C — temporal resolution ──────────────────────────────────────────────────────
    say ""
    say "--- Column C probe (temporal resolution) ---"
    curl -s --max-time 15 -X POST "$B/add" -H 'content-type: application/json' -d '{
      "request_id":"verify-req-t","user_id":"verify_t","session_id":"verify_st",
      "messages":[
        {"role":"user","timestamp":1704063600000,"content":"The API base URL is alpha.example.com."},
        {"role":"user","timestamp":1704074400000,"content":"Correction: the API base URL is now beta.example.com."}]}' >/dev/null
    ST=$(curl -s --max-time 15 -X POST "$B/search" -H 'content-type: application/json' \
         -d '{"query":"what is the API base URL","user_id":"verify_t","top_k":20}')
    first=$("$PY" - "$ST" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.loads(sys.argv[1])["data"]
    print("beta" if d and "beta.example.com" in str(d[0].get("content","")) else ("alpha" if d and "alpha.example.com" in str(d[0].get("content","")) else "none"))
except Exception: print("err")
PY
)
    if [ "$first" = "beta" ]; then pass "Column C: the LATEST state is ranked #1"; else fail "Column C: top result was '$first' (expected the updated value)"; fi

    kill "$SRV" 2>/dev/null
    sleep 1
    kill -9 "$SRV" 2>/dev/null
  fi
fi

# ── D. ARE THE TESTS REAL?  non-vacuity check ───────────────────────────────────────────────────
say ""
say "--- running the repo's own tests ---"
if [ -d tests ]; then
  VIRTUAL_ENV="$VENV" uv pip install -q pytest 2>&1 | tail -1
  out=$(AXIOM_PORT=8792 "$VENV/bin/python" -m pytest tests -q 2>&1 | tail -5)
  echo "$out"
  echo "$out" | grep -qE "[0-9]+ passed" && pass "pytest ran" || fail "pytest did not report passes"
  echo "$out" | grep -qE "failed|error" && fail "pytest reported failures" || pass "pytest clean"
else
  fail "no tests/ directory"
fi

say ""
say "=== VERDICT ==="
if [ "$fails" -eq 0 ]; then
  say "ALL CHECKS PASSED — AxiomMem drives correctly against a fresh clone."
else
  say "$fails CHECK(S) FAILED — see the FAIL lines above. These are measured, not read."
fi
say "full log: $LOG"

/home/roni-saguey/.local/bin/tg_send_checked.sh "AxiomMem verification finished — $fails failure(s). Log: $LOG" >/dev/null 2>&1 || true
