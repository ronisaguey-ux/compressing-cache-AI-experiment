#!/usr/bin/env python3
"""Verify that every in-text citation resolves to a reference entry, and vice versa.

Usage: python3 tools/check_citations.py [paper.md]

Exits non-zero on any mismatch so a run watcher cannot treat a broken reference list as fine.
This exists because the watcher referenced a checker that did not exist, and `|| true` made that
invisible -- a check that never runs is worse than no check, because it is reported as one.
"""
import re
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(REPO, "paper", "PAPER.md")


def check(path):
    text = open(path, encoding="utf-8").read()
    if "## References" not in text:
        return ["no '## References' section"]
    body, refs = text.split("## References", 1)
    # ★ A citation can be a LIST: "[21, 22]" or a range "[21-24]". The original regex only matched a
    # single number per bracket, so a grouped citation was invisible and the checker reported the
    # reference as "listed but not cited" -- a false failure on a correct paper. Parse every number
    # inside each bracket group.
    cited = set()
    for group in re.findall(r"\[([\d,\s\-]+)\]", body):
        for part in group.split(","):
            part = part.strip()
            if "-" in part:
                lo, _, hi = part.partition("-")
                if lo.strip().isdigit() and hi.strip().isdigit():
                    cited.update(range(int(lo), int(hi) + 1))
            elif part.isdigit():
                cited.add(int(part))
    cited = sorted(cited)
    listed = sorted({int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", refs, re.M)})
    problems = []
    for c in cited:
        if c not in listed:
            problems.append("cited but not listed: [%d]" % c)
    for l in listed:
        if l not in cited:
            problems.append("listed but not cited: [%d]" % l)
    # a reference entry with no author, or no URL/identifier, is likely a fabricated citation
    for m in re.finditer(r"^\[(\d+)\]\s*(.+?)\s*$", refs, re.M):
        n, entry = int(m.group(1)), m.group(2)
        if not re.search(r"arxiv|doi|https?://|ISBN|pp\.|In ", entry, re.I):
            problems.append("[%d] has no identifier: %s" % (n, entry[:60]))
    return problems


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print("no paper at %s" % path)
        return 2
    problems = check(path)
    if problems:
        print("CITATION CHECK FAILED (%d):" % len(problems))
        for p in problems:
            print("   ", p)
        return 1
    text = open(path, encoding="utf-8").read()
    listed = sorted({int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", text.split("## References", 1)[1], re.M)})
    print("citation check OK: %d references, all cited, all listed" % len(listed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
