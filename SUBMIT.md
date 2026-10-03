# Submitting (owner actions)

Two competitions are entered. This file is the checklist for what still needs a human click.

**State as of 2026-10-02 23:59 UTC — everything on our side is ready; ONE owner click remains.**

## ★ THE ONLY BLOCKER: Kaggle identity verification

`CreateCodeSubmission` returns, on both competitions:

```
403 PERMISSION_DENIED
"You must Identity Verify your account to perform this action."
reason: IdentityVerificationRequired   metadata.url: /settings
```

**This is the whole blocker.** Rules ARE accepted (`userHasEntered = True` on arc-agi-2,
arc-agi-3, paper-track and gemma-paper). The account is entered, the kernels run, the submission
files validate against the official sample. The submit call is refused on identity alone.

- **Owner action:** kaggle.com → Settings → Identity verification. Requires documents; cannot be
  scripted, and could not be completed from this box.
- **Read the error from `api.kaggle.com`, not `www.kaggle.com`** — the latter serves a 404 HTML page
  for this endpoint, which reads as a routing bug. The real body is only on the `api.` host.
- Requests were checked at `/api/i/competitions.CompetitionService/GetCompetition` and at
  `api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission`; both agree.

**One click unlocks both tracks** — ARC-AGI-2 needs it for its own submission, and the ARC Paper
Track (§2.1.b) needs an ARC-AGI-2/3 submission to exist.

## Historical gate: rules acceptance (RESOLVED 2026-10-02)

Rules acceptance is a web-session action and is **not API-reachable** — seven
`competitions.CompetitionService/*` endpoint name variants probed, all 404. It was completed by the
owner, and the signature that it had worked was **`competitions download` going 403 → 302** on the
same token (`competitions files` worked throughout, so a 403 on download alone was never a key
problem).

A ruled-out theory worth keeping: with rules unaccepted the competition data does **not mount** into
a kernel at all, so an early ARC run failed with `input dir: None` and the guard refused to write an
empty submission. That was the rules gate, not a path bug. **Once rules were accepted the data
mounted** — and then the nest below turned out to be the real discovery problem.

## Gemma 4 Developer Agent — Paper Track (comp 163111, $35k, closes 2026-11-12)

**Prerequisites**
- [x] Entered (`userHasEntered = True`)
- [x] Rules accepted
- [ ] **Identity verified** ← the only remaining gate
- [ ] Repository public (Verifiability criterion; currently private — owner: *"we will make the repo
      public later"*). Scanned safe: 248 files, 2.6 MB, zero secret-shaped strings in the history.

**Artifact:** `paper/gemma4-paper-track.ipynb` — 80 cells, generated from `paper/PAPER.md` by
`tools/make_notebook.py` so the two cannot drift; figure inlined as SVG (a relative path renders as
a broken image on Kaggle). **2,946 words strict / 2,930 excluding code blocks**, against a 3,000 cap.
§5 carries the measured two-arm results, not a placeholder.

**Submit:** `kaggle competitions submit gemma-4-developer-agent-paper -k <kernel> -v <ver> -f file`,
or "Submit to competition" from the notebook page. `maxDailySubmissions` is 5.

## ARC Paper Track (comp 133724, $450k, closes 2026-11-09)

- [x] Entered
- [x] Rules accepted
- [ ] **Identity verified** ← same gate
- [ ] ARC-AGI-2 submission made (§2.1.b — *participation*, no minimum score)
- [ ] Paper submitted to the track

**The baseline is built, validated, and has RUN ON KAGGLE.** `tools/arc_baseline.py` +
`tools/make_arc_notebook.py`; measured 0.0% on ARC-AGI-2 with a non-zero control on ARC-AGI-1
training (9.8%) proving the solver and scorer work. It is a formality to satisfy the participation
rule, not a competitive entry — and the notebook says so about itself.

## ★ THREE DEFECTS FOUND GETTING THE ARC KERNEL TO RUN (each produced a valid-looking wrong file)

1. **The 2026 mount is nested: `/kaggle/input/competitions/<slug>`.** A flat
   `glob("/kaggle/input/*arc*")` finds nothing there, and the fail-loud guard then refused to write
   an empty submission — which read for two runs as *"the data is not mounting"* when it was one
   directory deeper. Discovery is now recursive.
2. **Five JSONs are mounted and THREE look eligible** — training (1000), evaluation (120) and test
   (240) all carry `"test"` entries, so *"the first file with a test key"* silently picked
   **evaluation**, producing 120 tasks where the grader expects 240. Now disambiguated by filename
   (`test_challenges`) and validated against **`sample_submission.json`** (19,936 B, 240 keys), which
   is the oracle.
3. **The library emitted the grader's dict shape but the notebook driver still wrote
   `submission[tid] = attempts`** — raw `[[a1,a2],…]` → `[[[grid]]]`. A fix in one of two producers
   is not a fix; grep every writer of a format, not just the obvious one.

**Verified end to end:** kernel COMPLETE, output `submission.json` **240 keys / 259 entries / 0
malformed**, key set identical to the official sample. Kernel is **public**
(`is_private: false`) because code competitions reject a private one.

## Verified kernel state (2026-10-02)

| what | result |
|---|---|
| Gemma paper notebook pushed | `roni9999/bounded-context-without-context-rewrites` |
| its Kaggle run | **COMPLETE** — 48 KB notebook / 315 KB rendered HTML, no errors |
| its submission attempt | **403 `IdentityVerificationRequired`** |
| ARC notebook pushed | `roni9999/arc-agi-2-baseline-submission` (public, v6) |
| its last Kaggle run | **COMPLETE** — wrote `submission.json`, 240 keys / 259 entries / 0 malformed |
| its submission attempt | **403 `IdentityVerificationRequired`** |
| submissions so far | `No submissions found` (confirmed via `competitions submissions`) |
| Gemma notebooks in the field | 11 public for 101 teams |
| repo flag name | `is_private` in kernel metadata |

Both notebooks run clean on Kaggle. **Nothing further can be done from this box until the account
is identity-verified.**

## Order

1. **Verify identity at kaggle.com/settings** → unblocks both submissions
2. Submit the ARC-AGI-2 kernel (version 6, `submission.json`) — 1/day limit, so one shot
3. Submit the Gemma paper notebook
4. Decide repo visibility → unblocks the Verifiability criterion on both
