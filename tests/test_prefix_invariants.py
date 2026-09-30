#!/usr/bin/env python3
"""Model-free tests for the prefix invariants (Findings 21-24).

Everything here is pure logic -- no torch, no model, no memory. That is deliberate:
the invariant at the centre of the prefix work ("the reusable prefix is set by the
EARLIEST eviction index") is a property of the index set, not of a forward pass, so
it can be pinned exhaustively without loading anything.

What these test, and the claim each one protects:

  1. prefix_len_of is exactly "first k where kept[k] != k" -- Finding 21's boundary.
  2. evicting the TAIL keeps every earlier token -> the longest possible prefix.
  3. evicting the FRONT keeps none of the leading run -> prefix 0.
  4. a SCATTERED cut is bounded by the earliest dropped index, which is why forcing
     the first f of the region raises the prefix (Finding 22).
  5. the reuse ranking is a function of the earliest drop POSITION only, and is
     independent of the number of tokens kept -- i.e. a policy cannot buy prefix
     reuse by keeping more tokens if it cuts early.

Claim 5 is the one that makes Finding 23 a conflict rather than a tuning problem:
efficiency and the early-drop position are not the same knob.

Run:  python tests/test_prefix_invariants.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from prefix_cache import prefix_len_of  # noqa: E402

RESULTS = []


def check(name, got, want):
    ok = got == want
    RESULTS.append((name, ok, got, want))
    print("%-66s %s  got=%s want=%s" % (name, "PASS" if ok else "FAIL", got, want))


def check_true(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail, ""))
    print("%-66s %s %s" % (name, "PASS" if cond else "FAIL", detail))


print("=" * 88)
print("1. prefix_len_of is the first index where kept[k] != k")
print("=" * 88)
check("nothing dropped -> full hit", prefix_len_of(list(range(10)), 10), 10)
check("first token dropped -> 0", prefix_len_of(list(range(1, 10)), 10), 0)
# NOTE: dropping the first four gives prefix 0, NOT 4. prefix_len counts the leading
# CONTIGUOUS run starting at 0, so a cut at index 0 leaves a prefix of length 0 however
# many tokens are kept afterwards. My first version of this test asserted 4 and was wrong
# -- the function was right and the expectation was the defect.
check("first four dropped -> 0 (not 4)", prefix_len_of(list(range(4, 10)), 10), 0)
check("hole at index 3 -> 3", prefix_len_of([0, 1, 2, 4, 5, 6, 7, 8, 9], 10), 3)
check("hole at the last index only -> 9", prefix_len_of(list(range(9)), 10), 9)
check("empty selection -> 0", prefix_len_of([], 10), 0)
check("single early token -> 1", prefix_len_of([0], 10), 1)
check("n=0 -> 0", prefix_len_of([], 0), 0)

print()
print("=" * 88)
print("2/3. tail vs front eviction -- the Finding 23 conflict")
print("=" * 88)
N = 1000
KEEP = 600
tail = list(range(KEEP))                 # drop the last 400
front = list(range(N - KEEP, N))         # drop the first 400
check("evict the TAIL -> prefix == keep (100% reuse)", prefix_len_of(tail, N), KEEP)
check("evict the FRONT -> prefix == 0 (0% reuse)", prefix_len_of(front, N), 0)
check_true("tail reuse is 100%% by construction", prefix_len_of(tail, N) == len(tail),
           "prefix %d of %d kept" % (prefix_len_of(tail, N), len(tail)))
check_true("front reuse is 0%% by construction", prefix_len_of(front, N) == 0,
           "prefix %d of %d kept" % (prefix_len_of(front, N), len(front)))

print()
print("=" * 88)
print("4. a scattered cut is bounded by the EARLIEST dropped index")
print("=" * 88)
# 600 tokens kept out of 1000, but the first drop is at index 10
scatter_late = list(range(10)) + list(range(11, 10 + 1 + 590))
check("scattered, first drop at 10 -> prefix 10", prefix_len_of(scatter_late, N), 10)
# forcing the first 150 contiguous costs 150 slots but lifts the prefix
forced = list(range(150)) + list(range(160, 160 + 450))
check("forced prefix of 150, first drop at 150 -> 150", prefix_len_of(forced, N), 150)
check_true("forcing the front raises the prefix (Finding 22)",
           prefix_len_of(forced, N) > prefix_len_of(scatter_late, N),
           "%d -> %d" % (prefix_len_of(scatter_late, N), prefix_len_of(forced, N)))

print()
print("=" * 88)
print("5. the prefix depends on the earliest drop POSITION, not on how much is kept")
print("=" * 88)
# Same early cut at index 20; very different amounts kept.
small_keep = list(range(20)) + [100, 101]
big_keep = list(range(20)) + list(range(30, 900))
check("cut at 20, 22 kept -> prefix 20", prefix_len_of(small_keep, N), 20)
check("cut at 20, 890 kept -> prefix 20", prefix_len_of(big_keep, N), 20)
check_true("keeping 40x more tokens does NOT raise the prefix",
           prefix_len_of(small_keep, N) == prefix_len_of(big_keep, N),
           "both %d" % prefix_len_of(big_keep, N))
check_true("=> efficiency and early-drop position are DIFFERENT knobs",
           prefix_len_of(big_keep, N) == 20 and len(big_keep) > 40 * len(small_keep),
           "%d kept vs %d kept, same prefix" % (len(big_keep), len(small_keep)))

print()
print("=" * 88)
print("6. the reuse ranking is monotone in where the first drop lands")
print("=" * 88)
for cut in [0, 5, 50, 200, 500]:
    kept = list(range(cut)) + list(range(cut + 10, 1000))
    p = prefix_len_of(kept, 1000)
    check_true("first drop at %-4d -> prefix %-4d" % (cut, p), p == cut, "")

print()
npass = sum(1 for r in RESULTS if r[1])
print("%d/%d passed" % (npass, len(RESULTS)))
sys.exit(0 if npass == len(RESULTS) else 1)
