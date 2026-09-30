#!/usr/bin/env python3
"""Mutual exclusion + memory precheck for any run that loads a large model.

WHY THIS EXISTS (2026-09-29). A 7B sweep was running when a diagnostic that also
loaded 7B was started. Two ~9GB models on a 15.7GB box took MemAvailable to 788MB,
swap to 99%, and crashed the session. The harness already HAD a memory guard
(`require_memory`), but the diagnostic was launched through a path that never
called it -- a guard on one door while another door stood open.

So this is two things in one place:
  1. a LOCK, so a second large-model process cannot start while one is live.
  2. the memory precheck, so a first one cannot start without headroom either.

Both are required. A lock alone lets the first process overcommit; a memory check
alone races -- two processes can both read "9GB free" within the same second and
both proceed. The lock is taken BEFORE the check so only one process is ever
deciding.

TWO LOCKS, AND THE GLOBAL ONE IS THE ONE THAT MATTERS (2026-09-30). Keying only by
model name meant `model-7b` and `model-small` were INDEPENDENT locks, so a 7B and a
small model could both load -- the same shape as the crash above, just with a
smaller second half. Measured on this box: a resident 7B leaves ~6.3GB available,
which clears a small model's 3.5GB request, and the pair lands at ~13GB against a
watchdog that pauses at 85%. A single exclusive slot for "a model is resident"
closes it; the per-name lock is kept because it produces a better error message.

Usage:
    from heavy_lock import heavy_slot
    with heavy_slot("7b-sweep", need_mb=9000):
        ...load model, run...
"""
import os, sys, json, time, errno, atexit, signal

LOCK_DIR = os.environ.get("CCAI_LOCK_DIR", "/tmp/opencode/ccai/locks")
DEFAULT_STALE_S = 4 * 3600  # a run older than this is assumed dead

# One slot for "some model is resident", independent of which one. Without this a
# 7B and a 0.5B take different locks and can coexist.
GLOBAL_SLOT = "_model_resident"

# path -> acquisition count for THIS process. Reentrancy is per-path so an inner
# context manager cannot release the outer one's lock. The previous version set a
# bare `held` flag and unlinked the file on EVERY exit, so a second Harness in the
# same process took the lock without creating anything and then deleted the first
# holder's file on its way out -- a premature release.
_OWNED = {}

# ★ `Harness` calls `heavy_slot(...).__enter__()` directly and NEVER calls `__exit__`, so
# the lock files were only ever cleaned up by the pid-death reclaim on the NEXT run. That
# works -- the reclaim is why nothing broke -- but it leaves stale locks lying around and
# makes an occupied-looking slot for however long the box is idle. Registered once, at
# import, so every acquisition is released at interpreter exit however the process ends.
_ATEXIT_REGISTERED = False


def _release_all():
    for path in list(_OWNED):
        _OWNED[path] = 1
        _release_one(path)


def _register_atexit():
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_release_all)
    # SIGTERM/SIGINT too: a killed run should not leave the slot occupied either.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)

            def _handler(signum, frame, _prev=prev):
                _release_all()
                if callable(_prev):
                    return _prev(signum, frame)
                raise SystemExit(128 + signum)

            signal.signal(sig, _handler)
        except Exception:
            pass
    _ATEXIT_REGISTERED = True


def _mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as e:
        # EPERM means it exists but is not ours -- still alive
        return e.errno == errno.EPERM
    except Exception:
        return False


class HeavySlotError(SystemExit):
    pass


def _take_one(path, name, need_mb):
    """Create one exclusive lock file, or report that it is held elsewhere.

    Reentrant within one process: a second Harness shares both the pid and the
    memory, so refusing it would be a self-deadlock, not protection. The danger
    being guarded is a SECOND PROCESS.
    """
    if path in _OWNED:
        _OWNED[path] += 1
        return True
    _register_atexit()
    os.makedirs(LOCK_DIR, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        info = None
        try:
            with open(path) as f:
                info = json.load(f)
        except Exception:
            pass
        if info and info.get("pid") == os.getpid():
            # Survived from a crashed predecessor in this same pid (pid reuse).
            _OWNED[path] = 1
            return True
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "started": time.time(),
                   "name": name, "need_mb": need_mb}, f)
    _OWNED[path] = 1
    return True


def _release_one(path):
    n = _OWNED.get(path)
    if not n:
        return
    if n > 1:
        _OWNED[path] = n - 1
        return
    del _OWNED[path]
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _describe(path):
    info = None
    try:
        with open(path) as f:
            info = json.load(f)
    except Exception:
        return None
    if not info:
        return None
    return (info.get("pid"), time.time() - info.get("started", time.time()),
            info.get("name"))


class heavy_slot:
    """Context manager holding an exclusive slot for a large-model run."""

    def __init__(self, name, need_mb=9000, stale_after=DEFAULT_STALE_S, wait_s=0):
        self.name = name
        self.need_mb = need_mb
        self.stale_after = stale_after
        self.wait_s = wait_s
        # Global first, then the per-name lock.
        self.paths = [os.path.join(LOCK_DIR, GLOBAL_SLOT + ".lock"),
                      os.path.join(LOCK_DIR, name + ".lock")]
        self.held = []  # paths acquired, in order

    def __enter__(self):
        deadline = time.time() + max(0, self.wait_s)
        for path in self.paths:
            while True:
                if _take_one(path, self.name, self.need_mb):
                    self.held.append(path)
                    break
                d = _describe(path)
                pid, age, holder = d if d else (None, 0.0, None)
                if d and (not _pid_alive(pid) or age > self.stale_after):
                    # Holder is gone (or impossibly old): reclaim rather than wedge.
                    print("[heavy_lock] reclaiming stale lock %s (pid=%s age=%.0fs)"
                          % (path, pid, age), file=sys.stderr)
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                    continue
                if time.time() >= deadline:
                    self.__exit__(None, None, None)
                    if path.endswith(GLOBAL_SLOT + ".lock"):
                        why = ("another model is already resident (held by %s, pid=%s, "
                               "%.0fs ago)" % (holder, pid, age))
                    else:
                        why = ("another run holds slot %r (pid=%s, started %.0fs ago)"
                               % (self.name, pid, age))
                    raise HeavySlotError(
                        "REFUSING TO START %s: %s. Two large models at once is what "
                        "crashed this box on 2026-09-29. Wait for it, or stop pid %s."
                        % (self.name, why, pid))
                time.sleep(2)

        # Memory is checked AFTER the lock, so only one process is ever deciding.
        avail = _mem_available_mb()
        if avail < self.need_mb:
            self.__exit__(None, None, None)
            raise HeavySlotError(
                "REFUSING TO START %s: %.0f MB available, need %.0f MB. "
                "Reclaim memory first; a partial load here is how the session died."
                % (self.name, avail, self.need_mb))
        print("[heavy_lock] %s acquired (%.0f MB available, need %.0f MB)"
              % (self.name, avail, self.need_mb), file=sys.stderr)
        return self

    def __exit__(self, *exc):
        # Reverse order: drop the name lock, then the global slot.
        while self.held:
            _release_one(self.held.pop())
        return False
