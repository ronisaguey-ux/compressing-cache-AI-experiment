# Submitting

Two paper tracks plus the ARC participation entry. This file records what is ready, what is blocked,
and the exact evidence for each claim.

State as of 2026-10-03.

## Resolved gates

Identity verification is done, and the first ARC submission landed:

| what | result | evidence |
|---|---|---|
| Identity verification | resolved | `CreateCodeSubmission` no longer returns `IdentityVerificationRequired` |
| ARC-AGI-2 submission | **submitted** | id `56788985`, 2026-10-03 04:02:43, status PENDING, daily limit spent (1/day) |
| Repository visibility | **public** | anonymous `git ls-remote` / HTTP fetch returns 200 |

The earlier `403` was identity, not rules: rules acceptance showed `userHasEntered = True` on
arc-agi-2, arc-agi-3, paper-track and gemma-paper throughout. Read that error from `api.kaggle.com`;
`www.kaggle.com` serves a 404 HTML page for the endpoint, which reads as a routing bug.

## Gemma 4 Developer Agent — Paper Track (comp 163111, $35k, closes 2026-11-12)

Prerequisites are met: entered, rules accepted, identity verified, repository public.

Artifact: `paper/gemma4-paper-track.ipynb`, generated from `paper/PAPER.md` by
`tools/make_notebook.py`. The paper is written in continuous academic prose and is **2,986 words of
body** against the track's 3,000-word maximum (3,352 including references). Section 5 carries the
measured results, including the ablation.

Before submitting, re-run the comparison so the notebook matches the committed code:

```
python3 tools/run_compare.py benchmarks/results/incremental_gemma-4-12b_{linear,runtime}.json
```

## ARC Prize 2026 — Paper Track (comp 133724, $450k, closes 2026-11-09)

**This track has a 1,500-word cap**, verified on the competition page: the Writeup "should not exceed
1,500 words". It is a separate document from the Gemma paper.

Artifact: `paper/ARC_PAPER.md`, **1,485 words**, built to the official six-part structure. It reports
our own ARC-AGI-2 score as 0.0%, and now reports the retention measurement **run inside an ARC solve
loop**: on an 8x8 ARC-AGI-2 puzzle neither policy solved any of sixty attempts, and the anchored
policy cost 53,849 compute units against 392,390, a factor of 7.29 at identical accuracy. The
remaining step, from cheaper turns to a larger search budget, is labelled a hypothesis with its
falsification test named.

Required assets: a Writeup, a cover image in the Media Gallery, and a public notebook attached in
Project Links.

## What still needs a human

1. **Submit the ARC writeup** — paste `paper/ARC_PAPER.md` into a new Writeup, select the ARC-AGI-2
   track, and attach the public notebook and a cover image.
2. **Submit the Gemma paper notebook.**
3. **Eligibility.** Both tracks' rules require the entrant to be the older of 18 or the age of
   majority, *"unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian
   consents have been obtained by Competition Sponsor"*. This is two conditions and the sponsor must
   agree. Kaggle's private route is `kaggle.com/contact#/privacy/minor`: a parent creates a Kaggle
   account and signs a consent form per competition. The request to `team@arcprize.org` has been sent;
   the Kaggle form still needs the parent. Do this before any result exists — the rules state that
   failure can bar a prize.

## Known defects that affect a reviewer's reproduction

1. **The runtime numbers are the corrected re-run, not the historical ones.** The anchored policy
   carried the turn-1 brief twice on every turn (the archive's first entry is the same object as the
   transcript's first entry); the fix is committed and the reported figures are from the re-run with
   that fix: 76.0% reuse, 106,678 compute units against 484,957 for the linear policy, a factor of
   4.55 (4.49 against prune).
2. **The ablation is reported from a clean re-run.** `ablate-recent` (anchored minus the instruction
   archive) isolates the archive as the mechanism: code recall collapses from 60/60 to 8/60, per-turn
   success 0.917, final artifact 55/60. A prior run reached 60/60 at the end but its per-turn grades
   were corrupted by a resume (turn_ok was not checkpointed at the time), so it is archived as
   `CONTAMINATED_...resume-artifact.json` and only the clean run is reported. Final state varies
   across the two runs; the recall collapse does not.
3. **All 17 arXiv identifiers in the paper were verified against the arXiv API** and resolve to the
   cited titles, including `[20]` SinkTrack (arXiv:2604.10027, Liu, Chen and Wang).
4. **One model, one seed per arm.** Results carry no error bars; the sensitivity of the cost result to
   the cache-hit multiplier is swept from 0.05 to 1.00 and reported in Section 7.

## Historical defects, kept because they generalise

Getting the ARC kernel to produce a grader-valid file took three corrections, each of which produced a
valid-looking wrong file:

1. The mount is nested at `/kaggle/input/competitions/<slug>`. A flat glob finds nothing, and the
   fail-loud guard then refused to write an empty submission, which read as "the data is not
   mounting".
2. Five JSONs are mounted and three carry a `test` key. "First file with a test key" silently selected
   the evaluation set (120 tasks) where the grader expects the test set (240). Disambiguate by
   filename and validate the key set against `sample_submission.json`.
3. The library emitted the grader's dict shape while the notebook driver still wrote raw
   `[[a1, a2]]`. A fix in one of two producers is not a fix; grep every writer of the format.

Verified end to end: kernel COMPLETE, `submission.json` 240 keys, 259 entries, 0 malformed, key set
identical to the official sample. The kernel is public because code competitions reject a private one.
