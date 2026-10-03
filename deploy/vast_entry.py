"""Run incremental_coding.py on a plain GPU box (Vast / any ssh host).

The benchmark was written for Modal, so `incremental_coding.py` does `import modal`
at module scope and decorates its worker with `@app.function(...)`. Modal is not
installed here and is not wanted -- this stubs the module so the decorators become
no-ops, then calls the worker DIRECTLY instead of `.remote()`.

Nothing about the experiment changes: same function, same env vars, same grader.
Only the transport (Modal's scheduler) is removed.

Usage:
    CCAI_ARM=linear CCAI_MODEL=gemma-4-26b-a4b python vast_entry.py
"""
import json, os, sys, time, types

# ------------------------------------------------------------------ modal stub
class _Chain:
    """Chainable no-op: modal.Image.debian_slim(..).pip_install(..).env(..)."""
    def __getattr__(self, _n):
        return _Chain()
    def __call__(self, *a, **k):
        return _Chain()


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        def deco(fn):
            return fn                      # <- the worker itself, undecorated
        return deco

    def local_entrypoint(self, *a, **k):
        def deco(fn):
            return fn
        return deco


_stub = types.ModuleType("modal")
_stub.App = _App
_stub.Image = _Chain()
_stub.Volume = _Chain()
_stub.Secret = _Chain()
# NO __getattr__ ON THE MODULE. That would make getattr(modal, "__file__") return a
# _Chain, and inspect.getsourcefile() then calls os.path.splitext() on it -- which raised
# "TypeError: expected str, bytes or os.PathLike object, not _Chain" from deep inside TORCH's
# import. A stub that answers every attribute breaks introspection in code that has nothing
# to do with it. Only the four names incremental_coding.py actually uses are defined.
sys.modules["modal"] = _stub

# ------------------------------------------------------------------ run
import incremental_coding as inc

model      = os.environ.get("CCAI_MODEL", "gemma-4-26b-a4b")
arm        = os.environ.get("CCAI_ARM", "linear")
features   = int(os.environ.get("CCAI_FEATURES", "200"))
keep_turns = int(os.environ.get("CCAI_KEEP_TURNS", "4"))

print("[vast] model=%s arm=%s features=%d quant=%s"
      % (model, arm, features, os.environ.get("CCAI_QUANT", "auto")), flush=True)

t0 = time.time()
r = inc.run_inc(model, arm, features, keep_turns)

out = "/root/ccai/results"
os.makedirs(out, exist_ok=True)
# ★ THE SALT MUST BE IN THE FILENAME. Without it every variance run writes the SAME path and each
# instance silently overwrites the last, so a sweep of five salts produces one file repeated five
# times -- which is exactly how this went wrong: six md5-identical files that all turned out to be
# the prune arm from an earlier run. A run's identity belongs in its artifact name.
_salt = os.environ.get("CCAI_TASK_SALT", "")
_tag = ("_%s" % _salt) if _salt else ""
_sfx = os.environ.get("CCAI_RUN_TAG", "")
p = os.path.join(out, "incremental_%s_%s%s%s.json" % (model.replace("/", "-"), arm, _tag, _sfx))
with open(p, "w") as f:
    json.dump(r, f, indent=2)

print("\n%s  arm=%s  features=%d" % (r["model"], r["arm"], r["features"]))
if r.get("stopped_early") is not None:
    print("  PARTIAL: stopped at turn %s after %.0f s -- grading only %d features attempted"
          % (r.get("stopped_early"), r.get("seconds", 0), r.get("features_completed", 0)))
print("  PASSED %d/%d  (%.0f%%)" % (r["passed"], r["total"],
                                    100.0 * r["passed"] / max(1, r["total"])))
# ★ PER-TURN SUCCESS -- the owner's metric for the fix task.
if r.get("turns_ok") is not None:
    _t = r.get("turn_ok") or []
    print("  PER-TURN SUCCESS %d/%d turns applied their own fix correctly"
          % (r["turns_ok"], len(_t)))
print("  failed: %s" % (", ".join(r["failed"]) if r["failed"] else "none"))
print("  TTFT  %6.0f -> %6.0f ms     prompt %5d -> %5d tok"
      % (r["ttft_first"], r["ttft_last"], r["prompt_first"], r["prompt_last"]))
print("  KV    peak %6.1f MB   solution.py %d chars"
      % (r["kv_peak"] / 1e6, r["solution_chars"]))
print("  wall  %.0f s (%.1f min)" % (time.time() - t0, (time.time() - t0) / 60))
print("wrote %s" % p)
