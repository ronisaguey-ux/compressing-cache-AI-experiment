# Submitting (owner actions)

Two competitions are entered. This file is the checklist for what still needs a human click.

## Gate: competition rules (NOT API-reachable)

Rules acceptance is a web-session action. Probed seven `competitions.CompetitionService/*` endpoint
name variants — all 404. Identity verification is already satisfied, so this is only a checkbox:

- `kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules` → "I Understand and Accept"
- `kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules` → same

Until then the token returns **403 on `competitions download`** while `competitions files` works on
the same token — that asymmetry is the signature of rules-not-accepted, not a broken key.

### ★ VERIFIED END TO END: without rules acceptance the data does not even MOUNT

The submission notebook was pushed and run on Kaggle (kernel
`roni9999/arc-agi-2-baseline-submission`, `competition_sources: ["arc-prize-2026-arc-agi-2"]`). It
ran, and the log shows:

```
input dir: None
challenges: None
SystemExit: no challenge file found under None -- refusing to write an empty submission
```

So the failure is upstream of the solver: **an unaccepted-rules competition does not mount its data
into the kernel at all**, and `enable_internet` is false so it cannot be fetched at runtime. This is
not a path bug -- the notebook discovers the directory rather than hardcoding it, and there is
nothing to discover.

★ The refusal is the correct outcome. Without the guard the notebook would have written an empty
`submission.json`, exited 0, and scored 0.0 — indistinguishable from a solver that ran and failed.
That distinction is the subject of the Gemma paper, so it would have been a poor way to lose the
$450k track.

Why ARC-AGI-2 matters: the ARC Prize 2026 **Paper Track** ($450k) requires the team to have
submitted to ARC-AGI-2 or ARC-AGI-3. Participation, not score.

## Gemma 4 Developer Agent — Paper Track (comp 163111, $35k, closes 2026-11-12)

**Prerequisites**
- [x] Entered (verified: `userHasEntered = True`)
- [ ] Rules accepted — `kaggle.com/competitions/gemma-4-developer-agent-paper/rules`
- [ ] Repository public (verifiability criterion; currently private)

**Artifact:** `paper/gemma4-paper-track.ipynb` — generated from `paper/PAPER.md` by
`tools/make_notebook.py`. 65 cells, figure inlined as SVG, 2,991 words against a 3,000 cap.

**Submit:** push the notebook to Kaggle (or upload the .ipynb), then "Submit to competition" from
the notebook page. `maxDailySubmissions` is 5.

## ARC Paper Track (comp 133724, $450k, closes 2026-11-09)

- [x] Entered
- [ ] ARC-AGI-2 or ARC-AGI-3 submission made (the §2.1.b requirement)
- [ ] Paper submitted to the track

**Baseline is built and measured** (`tools/arc_baseline.py`, `tools/ARC-VALIDATION.md`): 0.0% on
ARC-AGI-2, with a non-zero control on ARC-AGI-1 training (9.8%) proving the tool works. It is a
formality to satisfy the participation rule, not a competitive entry — and it says so about itself.

## Verified kernel state (2026-10-02)

| what | result |
|---|---|
| Gemma paper notebook pushed | `roni9999/bounded-context-without-context-rewrites` |
| its Kaggle run | **COMPLETE** — rendered 48 KB notebook / 315 KB HTML, no errors |
| submission attempt | **403** on `CreateCodeSubmission` — rules gate, as expected |
| ARC notebook pushed | `roni9999/arc-agi-2-baseline-submission` |
| its Kaggle run | **ERROR**, and correctly so: `input dir: None` → guard refused an empty submission |
| Gemma notebooks in the field | 11 public for 101 teams |
| repo flag name | `is_private: true` in kernel metadata — set false at submission time |

The Gemma notebook is **ready to submit the moment rules are accepted**; everything on our side
already runs clean on Kaggle. The ARC notebook cannot run until then, because an unaccepted-rules
competition does not mount its data.

## Order

1. Accept ARC-AGI-2 rules → unblocks the ARC Paper Track
2. Accept the Gemma paper-track rules → unblocks the submission above
3. Decide repo visibility → unblocks the Verifiability criterion on both
4. Flip `is_private` to false in the kernel metadata, re-push, submit
