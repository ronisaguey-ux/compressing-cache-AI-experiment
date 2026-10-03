#!/usr/bin/env bash
# ★★★ VARIANCE RUNS — N=5 task instances per arm, for error bars.
#
# WHY: every number in the paper is n=1 per arm. "No error bars" is a named reject trigger, and it is
# the largest remaining methodological gap. Decoding is greedy, so re-running the SAME task is
# bit-identical and a seed would fabricate variance; the honest axis is a different task instance. The
# harness already supports this: CCAI_TASK_SALT re-derives every constant into a new instance of the
# same broken shape, while the retention probes are deliberately NOT salted, so runs stay comparable
# while the work underneath varies.
#
# DESIGN:
#   * 5 salts x 2 arms (runtime, linear) = 10 runs, 60 features each, same budget as the headline run.
#   * Every run is a COMPLETE 60-turn session, so each is a paired observation, not a fragment.
#   * Paired by salt: for salt i we compute runtime_i and linear_i under the identical task, then the
#     per-salt cost ratio. The report is the MEAN RATIO with a 95% Student-t interval over the 5
#     instances -- the ratio is the quantity the paper claims, so that is what gets the interval.
#   * Results land in /root/ccai/results/ and are scp'd back per run, so a box death mid-sweep keeps
#     every completed instance rather than losing the batch.
#
# Waits for the current chain (runtime_fixed) to finish, because the box is serial: prune/31B/12B each
# want most of the 48GB card and two at once is how this project once killed the machine.
set -u
SSH="ssh -p 15322 root@ssh6.vast.ai -i $HOME/.ssh/vast_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
OUT=/home/roni-saguey/.local/share/ccai-results/variance
LOG=/home/roni-saguey/.local/share/ccai-results/chain_variance.log
mkdir -p "$OUT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

say "waiting for 'RUNTIME_FIXED EXIT' (the current chain's last job)"
for i in $(seq 1 1440); do
  if timeout 60 $SSH 'grep -qa "RUNTIME_FIXED EXIT" /root/chain_runtime_fixed.marker 2>/dev/null'; then
    say "current chain finished"; break
  fi
  sleep 60
done

for SALT in salt1 salt2 salt3 salt4 salt5; do
  for ARM in runtime linear; do
    say "--- salt=$SALT arm=$ARM ---"
    timeout 60 $SSH "cd /root && setsid nohup bash -lc '
      export HF_HOME=/root/hf HF_TOKEN=\$(cat /root/.hftok 2>/dev/null || echo \"\") HF_XET_HIGH_PERFORMANCE=1
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      rm -rf /work/inc; mkdir -p /root/ccai/ckpt_${SALT}_${ARM}
      CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none CCAI_MAX_GPU_MEMORY=44GiB \
      CCAI_ARM=${ARM} CCAI_TASK_SALT=${SALT} CCAI_FEATURES=60 CCAI_GATE=1 CCAI_PRUNE_EVERY=30 \
      CCAI_RUNTIME_TOKENS=10240 CCAI_TIME_BUDGET_S=10800 CCAI_CKPT_DIR=/root/ccai/ckpt_${SALT}_${ARM} \
        /venv/main/bin/python -u /root/ccai/vast_entry.py > /root/var_${SALT}_${ARM}.log 2>&1
      echo \"VAR ${SALT} ${ARM} EXIT=\$?\" >> /root/chain_variance.marker
    ' > /dev/null 2>&1 < /dev/null & echo launched" || say "launch failed"

    # wait for THIS run's marker, then collect the result while the next one is not yet started
    for j in $(seq 1 360); do
      if timeout 60 $SSH "grep -qa 'VAR ${SALT} ${ARM} EXIT' /root/chain_variance.marker 2>/dev/null"; then
        say "  ${SALT}/${ARM} done"
        break
      fi
      sleep 60
    done
    f=$(timeout 60 $SSH "ls /root/ccai/results/*${ARM}*.json 2>/dev/null | tail -1")
    if [ -n "$f" ]; then
      timeout 120 scp -P 15322 -i "$HOME/.ssh/vast_ed25519" -o IdentitiesOnly=yes \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        "root@ssh6.vast.ai:$f" "$OUT/var_${SALT}_${ARM}.json" >/dev/null 2>&1 \
        && say "  collected $(basename "$f")" || say "  scp failed"
    else
      say "  no result file found for ${SALT}/${ARM}"
    fi
  done
done

say "=== VARIANCE SWEEP DONE ==="
ls -la "$OUT"
