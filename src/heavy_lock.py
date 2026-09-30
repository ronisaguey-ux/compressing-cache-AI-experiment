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

Usage:
    from heavy_lock import heavy_slot
    with heavy_slot("7b-sweep", need_mb=9000):
        ...load model, run...
"""
import os, sys, json, time, errno, atexit, signal

LOCK_DIR = os.environ.get("CCAI_LOCK_DIR", "/tmp/opencode/ccai/locks")
DEFAULT_STALE_S = 4 * 3600  # a run older than this is assumed dead


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


class heavy_slot:
    """Context manager holding an exclusive slot for a large-model run."""

    def __init__(self, name, need_mb=9000, stale_after=DEFAULT_STALE_S, wait_s=0):
        self.name = name
        self.need_mb = need_mb
        self.stale_after = stale_after
        self.wait_s = wait_s
        self.path = os.path.join(LOCK_DIR, name + ".lock")
        self.held = False

    def _read(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return None

    def _take(self):
        os.makedirs(LOCK_DIR, exist_ok=True)
        # Reentrant within one process: a second Harness in the same process shares
        # both the pid and the memory, so refusing it would be a self-deadlock, not
        # protection. The danger this guards is a SECOND PROCESS.
        info = self._read()
        if info and info.get("pid") == os.getpid():
            self.held = True
            return True
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "started": time.time(),
                       "name": self.name, "need_mb": self.need_mb}, f)
        self.held = True
        return True

    def __enter__(self):
        deadline = time.time() + max(0, self.wait_s)
        while True:
            if self._take():
                break
            info = self._read()
            pid = (info or {}).get("pid")
            age = time.time() - (info or {}).get("started", time.time())
            if info and (not _pid_alive(pid) or age > self.stale_after):
                # Holder is gone (or impossibly old): reclaim rather than wedge.
                print("[heavy_lock] reclaiming stale lock %s (pid=%s age=%.0fs)"
                      % (self.name, pid, age), file=sys.stderr)
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
                continue
            if time.time() >= deadline:
                raise HeavySlotError(
                    "REFUSING TO START %s: another heavy run holds the slot "
                    "(pid=%s, started %.0fs ago). Two large models at once is what "
                    "crashed this box on 2026-09-29. Wait for it, or stop pid %s."
                    % (self.name, pid, age, pid))
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
        if self.held:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.held = False
        return False
