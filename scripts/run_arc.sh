#!/usr/bin/env bash
# ★★★ THE ARC RUN — the same two policies, on a real ARC-AGI-2 task.
#
# WHY: the paper's ARC claim was an argument. "If a long search re-reads its context every attempt
# and each attempt is 4x cheaper under the anchored policy, the search can try 4x more candidates in
# the same budget." That step was labelled a hypothesis because it had never been run on ARC data.
# This runs it. Whatever comes out -- including a low solve rate -- is a measured ARC result and
# replaces an argued one.
#
# WHAT IT IS NOT: a competitive ARC entry. One puzzle, sixty attempts, no large candidate search.
# The per-turn solve rate is reported honestly alongside the cost and cache numbers.
#
# DESIGN CONSTRAINTS
#   * SAME task, same 60 turns, same budgets as the fix run, so the two are comparable. Only
#     CCAI_TASK changes, and the whole policy/cache/grader machinery is shared.
#   * NO correctness feedback between turns. Feedback would itself carry information: a turn that
#     lost the examples could still infer its standing from it, which would defeat the retention
#     measurement.
#   * The gold output is on disk for the grader and is NEVER in a prompt.
#   * CCAI_ARC_INDEX pins the puzzle so both arms attempt the identical task. Index 287 is a
#     27x27 task whose brief renders to ~1009 tokens -- real context pressure, comparable to the
#     fix task's brief, so the two experiments sit in the same regime rather than one being trivial.
#
# Usage on the box:  bash /root/ccai/run_arc.sh [features] [index]
set -u
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok 2>/dev/null || echo '')"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N="${1:-${CCAI_FEATURES:-60}}"
IDX="${2:-287}"
LOG=/root/run_arc.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== ARC-AGI-2  N=$N  index=$IDX  arms=runtime,linear ==="

# PREFLIGHT: prove the task loads and the grader is satisfiable BEFORE spending GPU hours. A missing
# data dir or a task-index typo would otherwise burn a whole arm and read as "the model failed".
say "--- preflight: task loads, grader accepts the reference and rejects identity ---"
# ★ `set -o pipefail` + PIPESTATUS: `$?` after `... | tee` is TEE's status, which is always 0, so the
# previous version reported a vacuous grader and then ran anyway. The assertion below is only worth
# anything if its failure actually stops the run.
set -o pipefail
CCAI_ARC_DIR=/root/arc/data CCAI_ARC_SPLIT=training CCAI_ARC_INDEX="$IDX" \
/venv/main/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import sys, os, json, subprocess, tempfile
sys.path.insert(0, "/root/ccai")
import arc_task as at
work = tempfile.mkdtemp()
at.write_gold(work)
seed, brief, tests, ref = at.build(3)
tid, train, ti, to = at.load_task()
print("task=%s  examples=%d  test=%dx%d  brief~%d tok" % (tid, len(train), len(ti), len(ti[0]), len(brief[0])//4))
prelude = at.arc_prelude(work)
# ★ tests entries are (name, expr) PAIRS -- take the expression, not the tuple. Formatting the tuple
# yields a valid Python expression of two strings, which exits 0 and makes the grader look vacuous.
tcheck = tests[0][1]
def rc(src):
    open(os.path.join(work, "solution.py"), "w").write(src)
    code = "import sys; sys.path.insert(0,%r)\nimport solution as m\n%s\n%s\n" % (work, prelude, tcheck)
    return subprocess.run(["python3","-c",code], capture_output=True, text=True, timeout=60).returncode
wrong = rc(at.SEED_SOURCE)
right = rc("import json as _j,os as _o\n_G=_j.load(open(_o.path.join(%r,%r)))\ndef solve(grid):\n    return _G['test_output']\n" % (work, at.GOLD_NAME))
print("identity rc=%d (want non-zero)   reference rc=%d (want 0)" % (wrong, right))
assert wrong != 0, "GRADER IS VACUOUS: identity passed"
assert right == 0, "GRADER IS BROKEN: the reference failed"
print("preflight OK")
PY
PRE=${PIPESTATUS[0]}
if [ "$PRE" -ne 0 ]; then say "PREFLIGHT FAILED (rc=$PRE) -- refusing to run"; exit 1; fi

# ★ MODEL IS OVERRIDABLE. The owner's direction: run the ARC task on a stronger coding model that can
# actually solve some of it, so the paper shows a real solve difference between the policies rather
# than two zeros. `CCAI_ARC_MODEL` (and its quantisation) select it; the default stays Gemma so an
# unadorned invocation reproduces the recorded run.
ARC_MODEL="${CCAI_ARC_MODEL:-gemma-4-12b}"
ARC_QUANT="${CCAI_ARC_QUANT:-none}"
say "model=$ARC_MODEL quant=$ARC_QUANT"

for ARM in runtime linear; do
  rm -rf /work/inc
  mkdir -p /root/ccai/ckpt_arc_$ARM
  say "--- arm=$ARM starting on ARC task $IDX model=$ARC_MODEL ---"
  CCAI_TASK=arc CCAI_MODEL="$ARC_MODEL" CCAI_QUANT="$ARC_QUANT" CCAI_MAX_GPU_MEMORY=44GiB \
  CCAI_ARM=$ARM CCAI_FEATURES=$N CCAI_GATE=0 CCAI_PRUNE_EVERY=30 CCAI_RUNTIME_TOKENS=10240 \
  CCAI_ARC_DIR=/root/arc/data CCAI_ARC_SPLIT=training CCAI_ARC_INDEX="$IDX" \
  CCAI_TIME_BUDGET_S="${CCAI_TIME_BUDGET_S:-10800}" CCAI_CKPT_DIR=/root/ccai/ckpt_arc_$ARM \
    /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/arc_$ARM.log 2>&1
  rc=$?
  say "arm=$ARM exit=$rc"
  grep -aoE "turn [0-9]+/[0-9]+" /root/arc_$ARM.log | tail -1 | sed 's/^/  last: /' | tee -a "$LOG"
  echo "ARC $ARM EXIT=$rc" >> /root/chain_arc.marker
done
say "=== ARC ARMS DONE ==="
