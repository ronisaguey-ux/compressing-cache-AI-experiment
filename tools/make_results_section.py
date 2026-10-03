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
    if not runs:
        print("no arms with cache_rows in %s" % OUT)
        return 1
    if len(runs) < 3:
        # ★ DEGRADE, DO NOT REFUSE. A box that dies mid-run leaves the later arms short or absent,
        # and refusing here would leave the paper with no results section at all on exactly the run
        # that cost the most. The section states which arms are present and how many comparable
        # turns they share, so a partial result is legible as partial instead of being withheld.
        print("!! only %d arm(s) present in %s: %s -- generating a PARTIAL section"
              % (len(runs), OUT, sorted(runs)))
    order = [a for a in ("runtime", "linear", "prune") if a in runs]
    order += [a for a in sorted(runs) if a not in order]
    m = {a: (runs[a].get("metrics") or {}) for a in order}

    L = []
    L.append("## 5. Results")
    L.append("")
    n = runs[order[0]].get("features_completed")
    inc = [a for a in order if runs[a].get("features_completed") != n]
    if inc:
        L.append("Common prefix: **%d** turns (arms ran unequal lengths: %s)." % (
            min(runs[a].get("features_completed") or 0 for a in order), ", ".join(inc)))
        L.append("")
    # ★ THE COUNT MUST BE THE COMPARABLE PREFIX, NOT THE FIRST ARM'S LENGTH. On a partial run the
    # first arm is the long one, so "all three ran 60 turns" contradicted the common-prefix line
    # printed directly above it and overstated what was compared. "three" is also wrong when an arm
    # is absent -- the section says what it has.
    n_cmp = min((runs[a].get("features_completed") or 0) for a in order)
    _lead = {1: "One arm ran", 2: "Both arms ran", 3: "All three arms ran"}.get(
        len(order), "All %d arms ran" % len(order))
    L.append("%s %s comparable turns under an identical prompt and tool budget "
             "(\u00a73.1)." % (_lead, n_cmp))
    L.append("")
    L.append("**Figure 1** (`figs/trajectory.svg`) — growth (a) and reuse (b), per turn.")

    # ---- Table 1: cost and context ----
    L.append("**Table 1 — cost in raw compute units.** A cache-reused token costs 0.1 of a fresh "
             "one; the baseline prices every token as a miss.")
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
    def _resume_artifact(r):
        """Detect a resumed run's phantom per-turn zeros.

        MEASURED, ablate-recent: the run crashed at turn 27, resumed, and -- because `turn_ok` was
        not checkpointed at the time -- reported turns 1..27 as ZEROS. The tell is a long LEADING run
        of zeros followed by a perfect tail, together with a `fixes_applied_then_lost` that is
        negative. A real policy failure scatters its failures; it does not fail a clean prefix and
        then succeed on every remaining turn. Reporting such a number as a result would put a harness
        artifact into the paper, so it is suppressed and named instead.
        """
        tok = r.get("turn_ok") or []
        if len(tok) < 10:
            return False
        if tok[0] != 0 or sum(tok) == len(tok):
            return False
        lead = 0
        for x in tok:
            if x:
                break
            lead += 1
        # a long zero prefix that then never fails again, plus a negative applied-then-lost
        tail_clean = all(tok[i] for i in range(lead, len(tok)))
        lost = r.get("metrics", {}).get("fixes_applied_then_lost")
        return lead >= 5 and tail_clean and (lost is not None and lost < 0)

    L.append("**Table 2 — retention.** Per-turn success is graded after each turn; code recall is "
             "the accumulated list; the release gate is the independent mid-session secret.")
    L.append("")
    L.append("| policy | per-turn success | final state | applied-then-lost | code recall | "
             "ordered | release gate |")
    L.append("|---|---|---|---|---|---|---|")
    _artifact_arms = []
    for a in order:
        d = m[a]
        r = runs[a]
        if _resume_artifact(r):
            _artifact_arms.append(a)
            pts, lost = "INVALID", "INVALID"
        else:
            pts = fmt(d.get("per_turn_success"), "f3")
            lost = fmt(d.get("fixes_applied_then_lost"), "int")
        L.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            a,
            pts,
            fmt(d.get("final_state_accuracy"), "f3"),
            lost,
            fmt(d.get("codes_recall"), "f3"),
            str(r.get("codes_ordered")),
            "PASS" if r.get("gate_ok") else ("FAIL" if r.get("gate_ok") is False else "n/a")))
    L.append("")
    if _artifact_arms:
        L.append("**Suppressed as a harness artifact:** %s. The run resumed after a crash and its "
                 "per-turn grades for the pre-crash turns were not checkpointed, so they are "
                 "recorded as failures the model did not commit. `code recall` and `final state` "
                 "are computed from the final artifact and are unaffected; only the per-turn "
                 "column is suppressed." % ", ".join("`%s`" % a for a in _artifact_arms))
        L.append("")

    # ---- the claim, stated conditionally on what the numbers show ----
    hr = {a: (m[a].get("cache_hit_rate") or 0.0) for a in order}
    cu = {a: (m[a].get("cost_units") or float("inf")) for a in order}
    best = max(hr, key=lambda a: hr[a])
    cheap = min(cu, key=lambda a: cu[a])
    worst = min(hr, key=lambda a: hr[a])
    # ★ THE COMPARISON ARM MUST NOT BE THE BEST ARM ITSELF. Picking `worst` as the second value is
    # degenerate whenever only one arm carries cache data: it resolves to the same arm as `best`
    # and the sentence reads "`runtime` ... against 81% for `runtime`", which is a comparison of an
    # arm with itself dressed up as a separation. The contrast arm is the strongest OTHER arm.
    others = [a for a in order if a != best]
    other = max(others, key=lambda a: hr[a]) if others else None
    if other is None:
        L.append("**Separation.** Only one arm reported prefix-cache data, so there is no contrast "
                 "to state. The retention table above stands on its own.")
        other = best
        gap = 0.0
    else:
        gap = hr[best] - hr[other]
        L.append("**Separation.** `%s` reuses a prefix for %.0f%% of its prefill against %.0f%% for "
                 "`%s`, at %s compute units against %s (%s)." % (
                     best, 100 * hr[best], 100 * hr[other], other,
                     fmt(cu[cheap], "int"), fmt(cu[other], "int"),
                     fmt((cu[other] / cu[cheap]) if cu[cheap] else None, "x")))
    if other is not None and gap < 0.15:
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
            L.append("**Compaction events.** `prune` compacted on %d turn(s) (%s), each a "
                     "discontinuity: %s." % (
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
    # ★ COUNT EVERYTHING, NOT JUST THE PROSE. Stripping fenced code blocks before counting makes
    # the guard read ~18 words lower than a counter that includes them, and the two readings
    # disagreed on which side of the cap the paper was on. Which one Kaggle applies is unknown, so
    # the guard takes the STRICTER reading: a cap you might be over is not a cap you are under.
    import re as _re
    n_words = len(_re.findall(r"\S+", new))
    n_lenient = len(_re.findall(r"\S+", _re.sub(r"```.*?```", " ", new, flags=_re.S)))
    flag = "OK" if n_words <= 3000 else "OVER THE 3,000 LIMIT"
    print(">> spliced into %s (§5 replaced, %d -> %d chars)  words=%d (all text) / %d (code blocks "
          "excluded)  [%s]" % (PAPER, len(src), len(new), n_words, n_lenient, flag))
    if n_words > 3000:
        print("   ⚠️ OVER by %d words under the STRICTER count (code blocks included); %d under the "
              "lenient one. Trim §2 or the abstract." % (n_words - 3000, n_lenient - 3000))
    elif n_words > 2950:
        print("   ⚠️ within 50 words of the cap under the stricter count -- leave no room for a "
              "late addition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
