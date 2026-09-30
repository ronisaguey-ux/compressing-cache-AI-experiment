#!/usr/bin/env python3
"""Verify the block-causal mask construction. Model-free.

A WRONG MASK INVALIDATES EVERYTHING. If block_causal_mask accidentally leaves
cross-block attention enabled, the survival test passes for the wrong reason (nothing
was isolated) and the whole tiered architecture looks like a free win when it is
actually full causal attention with extra steps. So the masks are checked as boolean
patterns against an independently-written reference, not eyeballed.

The property being asserted:

    token i may attend to token j  <=>  j <= i  AND  block(i) == block(j)

for the masked case, and j <= i for the causal case.

Run:  python tests/test_block_masks.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import torch  # noqa: E402
from block_causal import (split_blocks, block_causal_mask, block_diag_mask_for,  # noqa: E402
                          causal_mask, NEG)

RESULTS = []


def check(name, got, want):
    ok = got == want
    RESULTS.append(ok)
    print("%-70s %s" % (name, "PASS" if ok else "FAIL"))


def allowed(mask):
    """mask -> boolean allow-matrix. 0.0 allows, NEG blocks."""
    return (mask[0, 0] == 0.0)


def reference_block_diag(blocks):
    """Independently written: allow iff same block and j <= i."""
    n = blocks[-1][1]
    ref = torch.zeros((n, n), dtype=torch.bool)
    for (lo, hi) in blocks:
        for i in range(lo, hi):
            for j in range(lo, hi):
                if j <= i:
                    ref[i, j] = True
    return ref


def reference_causal(n):
    ref = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1):
            ref[i, j] = True
    return ref


print("=" * 78)
print("MASK CONSTRUCTION")
print("=" * 78)

n = 12
blocks = split_blocks(list(range(n)), 3)
print("blocks: %s" % blocks)

got = allowed(block_causal_mask(blocks))
want = reference_block_diag(blocks)
check("block_causal_mask matches the reference pattern", bool(torch.equal(got, want)), True)
check("  total allowed entries", int(got.sum()), int(want.sum()))

got_c = allowed(causal_mask(n))
want_c = reference_causal(n)
check("causal_mask matches the reference pattern", bool(torch.equal(got_c, want_c)), True)

print()
print("=" * 78)
print("THE PROPERTIES THAT DECIDE THE EXPERIMENT")
print("=" * 78)

bc = allowed(block_causal_mask(blocks))
# cross-block attention must be impossible
cross = 0
for (lo1, hi1) in blocks:
    for (lo2, hi2) in blocks:
        if (lo1, hi1) == (lo2, hi2):
            continue
        cross += int(bc[lo1:hi1, lo2:hi2].sum())
check("NO cross-block attention is allowed", cross, 0)

# within a block, causality still holds
within_viol = 0
for (lo, hi) in blocks:
    sub = bc[lo:hi, lo:hi]
    within_viol += int(torch.triu(sub, diagonal=1).sum())
check("causality holds WITHIN each block", within_viol, 0)

# full causal, by contrast, DOES allow cross-block
fc = allowed(causal_mask(n))
cross_fc = int(fc[6:9, 0:3].sum())   # block 2 looking at block 0
check("full causal allows block 2 to see block 0 (contrast)", cross_fc > 0, True)

print()
print("=" * 78)
print("THE EVICTED-SEQUENCE MASK (kept blocks laid end to end)")
print("=" * 78)

evicted = [b for b in blocks if b != blocks[1]]      # drop the middle block
print("kept blocks: %s" % evicted)
bd = allowed(block_diag_mask_for(evicted, n))
n_new = bd.shape[0]
check("new sequence length == tokens of kept blocks",
      n_new, sum(hi - lo for lo, hi in evicted))

# each kept block's slice must be block-diagonal IN ITS NEW POSITION
cur = 0
ok_all = True
for (lo, hi) in evicted:
    L = hi - lo
    sub = bd[cur:cur + L, cur:cur + L]
    ref = reference_causal(L)
    if not torch.equal(sub, ref):
        ok_all = False
    # and nothing outside its own new slice
    outside = int(bd[cur:cur + L, :cur].sum()) + int(bd[cur:cur + L, cur + L:].sum())
    if outside != 0:
        ok_all = False
    cur += L
check("each kept block is causal within its NEW slice and blind outside it", ok_all, True)

print()
print("=" * 78)
print("SHAPE / DTYPE SANITY")
print("=" * 78)
m = block_causal_mask(blocks)
check("mask is 4D broadcastable (1,1,N,N)", tuple(m.shape), (1, 1, n, n))
check("mask dtype is float", m.dtype, torch.float32)
b2 = split_blocks(list(range(10)), 3)
check("split_blocks covers every index exactly once",
      sorted(i for lo, hi in b2 for i in range(lo, hi)), list(range(10)))
check("split_blocks blocks are contiguous and ordered",
      all(b2[k][1] == b2[k + 1][0] for k in range(len(b2) - 1)), True)
check("split_blocks on an empty sequence", split_blocks([], 3), [(0, 0), (0, 0), (0, 0)])

print()
npass = sum(1 for r in RESULTS if r)
print("%d/%d passed" % (npass, len(RESULTS)))
sys.exit(0 if npass == len(RESULTS) else 1)
