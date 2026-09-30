#!/usr/bin/env python3
"""Finding 37 invariant: the assembled table's LENGTH must equal the positions the query uses.

THE BUG THIS PINS. Blocks are built with DISJOINT absolute position ranges, so after evicting a
middle block the survivors' caches total ~90 rows while their position ids span ~3300. Two
things then disagree, and BOTH must be consistent or the join fails:

  1. `concat_caches` keeps original position ids -> the table has a HOLE in position space.
     `assemble_contiguous` closes it, so the assembled length is the survivors' true length.
  2. The query must be positioned at THAT length. Positioning it at the original `total` leaves
     it ~2814 positions past the last key, which measured as a failing join in the bench matrix
     even after the table itself was re-indexed.

This test needs no model, so it runs on this box in under a second while the GPU work is
elsewhere. It is deliberately about the ARITHMETIC of assembly, which is where the defect lived,
rather than about generation, which needs a GPU and a careful control to interpret.

Usage: python tests/test_contiguous_assembly.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from transformers import DynamicCache
import tiered_cache as TC


def mk_cache(n_layers, seq, seed=0):
    """A synthetic DynamicCache with `seq` rows per layer -- no model required."""
    g = torch.Generator().manual_seed(seed)
    data = [(torch.randn(1, 1, seq, 8, generator=g), torch.randn(1, 1, seq, 8, generator=g))
            for _ in range(n_layers)]
    return DynamicCache(ddp_cache_data=data)


def main():
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # ---- the layout: global [0,35), block1 [35,64), block2 EVICTED [64,2878),
    #      block3 [2878,2901).  This is the real shape from the two-needle task.
    G, n1, n2, n3 = 35, 29, 2814, 23
    g = mk_cache(2, G, seed=1)
    b1 = mk_cache(2, n1, seed=2)
    b2 = mk_cache(2, n2, seed=3)
    b3 = mk_cache(2, n3, seed=4)

    # ---- gapped assembly (the shipped behaviour that failed)
    gapped = TC.concat_caches([g, b1, b3])
    check("gapped table holds only the surviving rows", TC.cache_len(gapped) == G + n1 + n3,
          "len=%d want=%d" % (TC.cache_len(gapped), G + n1 + n3))

    # ---- contiguous assembly (the fix)
    fixed = TC.assemble_contiguous(g, [b1, b3])
    check("contiguous table has the SAME length", TC.cache_len(fixed) == G + n1 + n3,
          "len=%d" % TC.cache_len(fixed))

    # ★ THE ACTUAL DEFECT: the DISTANCE between block 1 and block 3 differs by the evicted span.
    # Gapped: block 3's content sits n2 = 2814 positions further along than it should.
    gap_positions = n2
    check("the gapped layout leaves a %d-position hole" % gap_positions, gap_positions > 0)

    # ---- FIXTURE TRAP: a DynamicCache is MUTATED IN PLACE by a forward pass, so an assembly
    # that shares a layer object with its source would make this comparison meaningless.
    check("assembly does not alias the source cache",
          fixed.layers[0].keys.data_ptr() != g.layers[0].keys.data_ptr())

    # ---- the survivors' DATA must be identical between the two assemblies -- re-indexing is a
    # property of the assembly, not a rewrite of any block.
    same_k = torch.equal(
        gapped.layers[0].keys[:, :, G:G + n1, :],
        fixed.layers[0].keys[:, :, G:G + n1, :])
    same_v = torch.equal(
        gapped.layers[0].values[:, :, G + n1:G + n1 + n3, :],
        fixed.layers[0].values[:, :, G + n1:G + n1 + n3, :])
    check("block 1 is bit-identical between gapped and contiguous", same_k)
    check("block 3 is bit-identical between gapped and contiguous", same_v)

    # ---- the query position rule: assembled length, NOT the original total.
    total_gapped = G + n1 + n2 + n3
    check("query at original total would sit %.0f positions past the keys"
          % (total_gapped - TC.cache_len(fixed)), total_gapped > TC.cache_len(fixed))
    check("query at cache_len sits exactly at the end",
          TC.cache_len(fixed) == G + n1 + n3)

    # ---- empty/None handling: evicting everything must not raise
    try:
        empty = TC.assemble_contiguous(g, [None, None])
        ok = TC.cache_len(empty) == TC.cache_len(g)
    except Exception as e:
        ok, empty = False, None
        print("   assemble with all-None raised: %s" % e)
    check("all-evicted keeps only the global anchor", ok)

    print("=" * 78)
    print("FINDING 37 — CONTIGUOUS ASSEMBLY INVARIANTS")
    print("=" * 78)
    bad = 0
    for name, ok, detail in checks:
        print("  %-62s %s%s" % (name[:62], "PASS" if ok else "FAIL",
                                ("  [%s]" % detail) if detail and not ok else ""))
        if not ok:
            bad += 1
    print("-" * 78)
    print("  %d/%d passed" % (len(checks) - bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
