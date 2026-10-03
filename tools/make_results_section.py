#!/usr/bin/env python3
"""Turn the three arm results into the paper's §5 Results section, as text.

★ WHY THIS IS A GENERATOR AND NOT A PARAGRAPH I WRITE LATER.
The results section is the one part of the paper that must not contain a number I typed. Three arms
produce ~25 metrics each; transcribing them by hand is where a plausible-looking wrong figure gets
into a submission, and the paper's own Verifiability criterion is the thing being scored. So the
numbers are emitted from the JSON files, and the prose is emitted conditionally on what the numbers
actually show -- including the case where they show nothing.

★ AND IT REFUSES TO EMBELLISH. If the cache hit rate does not separate the arms, the generator says
so in plain text rather than reaching for a softer framing. A results section that oversells is
worse than a short one, because the reviewer checks.
"""
import glob
import json
import os
import sys

OUT = os.environ.get("RESULTS_DIR",
                     os.path.expanduser("~/.local/share/ccai-results/fixmode-v3"))
PAPER = os.environ.get("PAPER", os.path.expanduser("~/.local/share/ccai-repo/paper/PAPER.md"))


def load(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "incremental_*.json"))) + \
             sorted(glob.glob(os.path.join(d, "fixres_*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict) and r.get("arm"):
            runs[r["arm"]] = r
    return runs


def g(r, *path, default=None):
    cur = r
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def fmt(v, kind="num"):
    if v is None:
        return "n/a"
    if kind == "pct":
        return "%.1f%%" % (100.0 * v)
    if kind == "x":
        return "%.2f×" % v
    if kind == "int":
        return "%d" % v
    if kind == "f3":
        return "%.3f" % v
    if kind == "f1":
        return "%.1f" % v
    if kind == "mb":
        return "%.0f MB" % (v / 1e6)
    return str(v)


def main():
    runs = load(OUT)
    if len(runs) < 3:
        print("only %d arm(s) present in %s: %s" % (len(runs), OUT, sorted(runs)))
        print("the results section cannot be generated until all three land.")
        return 1
    order = [a for a in ("runtime", "linear", "prune") if a in runs]
    order += [a for a in sorted(runs) if a not in order]
    m = {a: (runs[a].get("metrics") or {}) for a in order}

    L = []
    L.append("## 5. Results")
    L.append("")
    # ---- completeness first: refuse to state a comparison on incomplete arms ----
    inc = [a for a in order if runs[a].get("features_completed") != runs[order[0]].get("features_completed")]
    if inc:
        L.append("**Comparable turns: %d** (arms ran differing lengths: %s). Percentages below are "
                 "over the common prefix, which is the only basis on which the arms are "
                 "comparable." % (min(runs[a].get("features_completed") or 0 for a in order), ", ".join(inc)))
        L.append("")
    n = runs[order[0]].get("features_completed")

    L.append("All three policies ran %s turns of the broken-repository task on one model under an "
             "identical prompt and tool budget. `runtime` pins the turn-1 brief and archives each "
             "turn's instruction; `linear` grows to the ceiling and evicts the oldest block; `prune` "
             "is `linear` plus a compaction every 30 turns." % n)
    L.append("")

    # ---- Table 1: cost and context ----
    L.append("**Table 1 — cost in raw compute units.** A cache-reused token is priced at 0.1 of a "
             "fresh one; the baseline prices every token as a miss.")
    L.append("")
    L.append("| policy | cache hit rate | miss tokens | cost units | no-cache cost | saving | "
             "prompt first → last | TTFT first → last |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in order:
        d = m[a]
        L.append("| `%s` | %s | %s | %s | %s | %s | %s → %s | %s → %s |" % (
            a,
            fmt(d.get("cache_hit_rate"), "pct"),
            fmt(d.get("cache_miss_tokens"), "int"),
            fmt(d.get("cost_units"), "int"),
            fmt(d.get("cost_units_no_cache"), "int"),
            fmt(d.get("cost_saving_ratio"), "x"),
            fmt(d.get("prompt_first"), "int"), fmt(d.get("prompt_last"), "int"),
            fmt(d.get("ttft_first_ms"), "f1"), fmt(d.get("ttft_last_ms"), "f1")))
    L.append("")

    # ---- Table 2: retention ----
    L.append("**Table 2 — retention.** Per-turn success is graded after each turn; code recall is "
             "the accumulated list; the release gate is the independent mid-session secret.")
    L.append("")
    L.append("| policy | per-turn success | final state | applied-then-lost | code recall | "
             "ordered | release gate |")
    L.append("|---|---|---|---|---|---|---|")
    for a in order:
        d = m[a]
        r = runs[a]
        L.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            a,
            fmt(d.get("per_turn_success"), "f3"),
            fmt(d.get("final_state_accuracy"), "f3"),
            fmt(d.get("fixes_applied_then_lost"), "int"),
            fmt(d.get("codes_recall"), "f3"),
            str(r.get("codes_ordered")),
            "PASS" if r.get("gate_ok") else ("FAIL" if r.get("gate_ok") is False else "n/a")))
    L.append("")

    # ---- the claim, stated conditionally on what the numbers show ----
    hr = {a: (m[a].get("cache_hit_rate") or 0.0) for a in order}
    cu = {a: (m[a].get("cost_units") or float("inf")) for a in order}
    best = max(hr, key=lambda a: hr[a])
    cheap = min(cu, key=lambda a: cu[a])
    worst = min(hr, key=lambda a: hr[a])
    L.append("**Separation.** `%s` reuses a prefix for %.0f%% of its prefill against %.0f%% for "
             "`%s`, at %s compute units against %s (%s less). " % (
                 best, 100 * hr[best], 100 * hr[worst], worst,
                 fmt(cu[cheap], "int"), fmt(cu[worst], "int"),
                 fmt((cu[worst] / cu[cheap]) if cu[cheap] else None, "x")))
    if hr[best] - hr[worst] < 0.15:
        L.append("The margin in prefix reuse is under 15 points, so **this run does not establish "
                 "the cost claim**; stating otherwise would be the artefact the fairness gate "
                 "exists to prevent.")
    else:
        L.append("The difference is structural, not incidental: a policy that evicts from the front "
                 "changes its first block as soon as eviction begins, while one that pins the front "
                 "and advances the window at the end keeps it. Hence reuse separates where context "
                 "size alone might not.")
    L.append("")
    # ---- compaction events ----
    # ★ `cache_rows` is TOP-LEVEL in the result (`incremental_coding.py:1565`), not inside `metrics`.
    # Reading it from the metrics block returns nothing and silently skips this paragraph.
    pr = list(runs.get("prune", {}).get("cache_rows") or [])
    if pr:
        drops = []
        for i in range(1, len(pr)):
            if pr[i - 1].get("prompt") and pr[i]["prompt"] < pr[i - 1]["prompt"]:
                drops.append((pr[i]["turn"], pr[i - 1]["prompt"], pr[i]["prompt"], pr[i].get("hit")))
        if drops:
            L.append("**Compaction events.** `prune` compacted on %d turn(s) (%s). Each is a "
                     "discontinuity in the prompt: %s. This is the operation the cost argument is "
                     "about — the retained blocks are rewritten, so the prefix shared with the "
                     "previous turn no longer applies and the prompt is re-processed." % (
                         len(drops),
                         ", ".join(str(d[0]) for d in drops[:8]),
                         "; ".join("turn %d %d→%d tok, %d reused" % d for d in drops[:4])))
            L.append("")

    body = "\n".join(L)
    print(body)
    # ---- splice into the paper ----
    try:
        src = open(PAPER).read()
    except Exception as e:
        print("\n(could not open the paper to splice: %s)" % e)
        return 0
    start = src.find("## 5. Results")
    end = src.find("## 6. Failure modes")
    if start == -1 or end == -1:
        print("\n(could not find the section markers; body printed above only)")
        return 0
    new = src[:start] + body + "\n\n" + src[end:]
    open(PAPER, "w").write(new)
    # ★ WORD-BUDGET GUARD. The cap is 3,000 and §5 is generated, so an overrun could otherwise be
    # introduced by a run with more compaction events than expected. Warn loudly rather than trim
    # silently -- silently dropping a table row to fit would be a worse failure than being long.
    import re as _re
    flat = _re.sub(r"```.*?```", " ", new, flags=_re.S)
    n_words = len(_re.findall(r"\S+", flat))
    flag = "OK" if n_words <= 3000 else "OVER THE 3,000 LIMIT"
    print(">> spliced into %s (§5 replaced, %d -> %d chars)  words=%d  [%s]"
          % (PAPER, len(src), len(new), n_words, flag))
    if n_words > 3000:
        print("   ⚠️ OVER by %d words -- trim §2 or the abstract before submitting." % (n_words - 3000))
    elif n_words > 2950:
        print("   ⚠️ within 50 words of the cap -- leave no room for a late addition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
