#!/usr/bin/env python3
"""Cross-process tests for the model-resident slot.

The defect these exist for: `heavy_slot` keyed its lock file by model NAME, so
`model-7b` and `model-small` were independent and a 7B and a 0.5B could both load.
That is the 2026-09-29 crash shape with a smaller second half. A test that only
entered the context manager twice IN ONE PROCESS would have passed throughout,
because reentrancy is deliberately allowed within a process -- the danger is a
second PROCESS. So every exclusion test here forks.

Run:  python tests/test_single_model_slot.py
"""
import os, sys, json, time, shutil, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

LOCKDIR = tempfile.mkdtemp(prefix="ccai_locktest_")
os.environ["CCAI_LOCK_DIR"] = LOCKDIR

import heavy_lock
from heavy_lock import heavy_slot, HeavySlotError

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print("%-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))


def refused(fn):
    """Return (refused, message)."""
    try:
        fn()
        return False, ""
    except HeavySlotError as e:
        return True, str(e)


def reset():
    """Clear the lock dir so tests do not inherit each other's state."""
    if os.path.isdir(LOCKDIR):
        for f in os.listdir(LOCKDIR):
            try:
                os.unlink(os.path.join(LOCKDIR, f))
            except FileNotFoundError:
                pass


def try_acquire(name, need_mb=100, wait_s=0):
    """Return (refused, message), ALWAYS releasing on success.

    Calling `heavy_slot(...).__enter__()` directly leaks the lock when it is not
    refused -- the first version of this test did exactly that and made a later
    test fail for the wrong reason.
    """
    try:
        with heavy_slot(name, need_mb=need_mb, wait_s=wait_s):
            return False, ""
    except HeavySlotError as e:
        return True, str(e)


def child_holds(name, need_mb, hold_s=25):
    """Fork a child that takes the slot and holds it. Returns (pid, ready_r, stop_w).

    The child writes a byte when the slot is held, so the parent never races the
    acquisition.
    """
    r, w = os.pipe()      # child -> parent: ready
    r2, w2 = os.pipe()    # parent -> child: release
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            os.close(w2)
            with heavy_slot(name, need_mb=need_mb, wait_s=0):
                os.write(w, b"1")
                # hold until the parent says stop, or time out
                import select
                select.select([r2], [], [], hold_s)
            os._exit(0)
        except BaseException:
            traceback.print_exc()
            os._exit(1)
    os.close(w)
    os.close(r2)
    return pid, r, w2


def wait_ready(r):
    import select
    if not select.select([r], [], [], 30)[0]:
        raise RuntimeError("child never acquired the slot")
    os.read(r, 1)


def reap(pid, w2):
    os.write(w2, b"x")
    os.close(w2)
    os.waitpid(pid, 0)


print("lock dir:", LOCKDIR)
print()

# ---------------------------------------------------------------- 1. the defect
try:
    reset()
    pid, r, w2 = child_holds("model-small", 100)
    wait_ready(r)
    ok, msg = try_acquire("model-7b")
    check("1. a 7B is refused while a small model is resident", ok,
          "(was allowed before the global slot)" if ok else "STILL ALLOWED: " + msg[:60])
    reap(pid, w2)
except Exception:
    traceback.print_exc()
    check("1. a 7B is refused while a small model is resident", False, "exception")

# --------------------------------------------------- 2. release frees the slot
try:
    reset()
    with heavy_slot("model-small", need_mb=100, wait_s=0):
        pass
    ok, msg = try_acquire("model-7b")
    check("2. after the holder exits, the next model starts", not ok, msg[:60])
except Exception:
    traceback.print_exc()
    check("2. after the holder exits, the next model starts", False, "exception")

# ------------------------------------------ 3. the same name is still exclusive
try:
    reset()
    pid, r, w2 = child_holds("model-7b", 100)
    wait_ready(r)
    ok, msg = try_acquire("model-7b")
    check("3. two runs of the SAME model are still mutually exclusive", ok, msg[:50])
    reap(pid, w2)
except Exception:
    traceback.print_exc()
    check("3. two runs of the SAME model are still mutually exclusive", False, "exception")

# --------------------------------------------- 4. reentrancy within a process
try:
    reset()
    with heavy_slot("model-small", need_mb=100, wait_s=0):
        ok = True
        try:
            with heavy_slot("model-small", need_mb=100, wait_s=0):
                pass
        except HeavySlotError:
            ok = False
        check("4. a second Harness in the SAME process is allowed", ok,
              "refusing it would be a self-deadlock" if ok else "self-deadlocked")
except Exception:
    traceback.print_exc()
    check("4. a second Harness in the SAME process is allowed", False, "exception")

# ------------------------- 5. an inner exit must not release the outer's lock
# The old code unlinked the file on every exit, so this released early.
try:
    reset()
    with heavy_slot("model-small", need_mb=100, wait_s=0):
        with heavy_slot("model-small", need_mb=100, wait_s=0):
            pass
        held = sorted(f for f in os.listdir(LOCKDIR) if f.endswith(".lock"))
        want = sorted([heavy_lock.GLOBAL_SLOT + ".lock", "model-small.lock"])
        check("5. inner exit does not release the outer's lock", held == want,
              "held: %s" % held)
except Exception:
    traceback.print_exc()
    check("5. inner exit does not release the outer's lock", False, "exception")

# ------------------------------------------------- 6. a dead holder is reclaimed
try:
    reset()
    dead = os.path.join(LOCKDIR, heavy_lock.GLOBAL_SLOT + ".lock")
    # a pid that cannot be alive: fork and reap, so the pid is definitively gone
    p = os.fork()
    if p == 0:
        os._exit(0)
    os.waitpid(p, 0)
    with open(dead, "w") as f:
        json.dump({"pid": p, "started": time.time(), "name": "ghost"}, f)
    ok, msg = refused(lambda: heavy_slot("model-7b", need_mb=100, wait_s=6).__enter__())
    check("6. a lock whose holder is dead is reclaimed", not ok, msg[:50])
except Exception:
    traceback.print_exc()
    check("6. a lock whose holder is dead is reclaimed", False, "exception")

# ------------------------------------------------ 7. the memory check still runs
try:
    reset()
    ok, msg = refused(lambda: heavy_slot("model-7b", need_mb=10 ** 9, wait_s=0).__enter__())
    check("7. an impossible memory request is refused", ok, msg[:60])
except Exception:
    traceback.print_exc()
    check("7. an impossible memory request is refused", False, "exception")

# ------------------------------------ 8. a refusal leaves no lock behind
try:
    reset()
    try:
        heavy_slot("model-7b", need_mb=10 ** 9, wait_s=0).__enter__()
    except HeavySlotError:
        pass
    left = [f for f in os.listdir(LOCKDIR) if f.endswith(".lock")]
    check("8. a refused start leaves no lock behind", not left, "left: %s" % left)
except Exception:
    traceback.print_exc()
    check("8. a refused start leaves no lock behind", False, "exception")

shutil.rmtree(LOCKDIR, ignore_errors=True)

print()
npass = sum(1 for _, ok, _ in RESULTS if ok)
print("%d/%d passed" % (npass, len(RESULTS)))
sys.exit(0 if npass == len(RESULTS) else 1)
