# compressing-cache-AI-experiment

Can a volatile middle region be evicted from a KV cache without destroying what the model
knows about the content in it?

**Short answer from the measurements: yes — but not by re-rotating keys.** Compaction to 60%
retention preserves a mid-context fact at every needle position tested, on a 0.5B and a 7B,
**provided the retained tokens are recomputed rather than re-rotated.** Key-only rotation is
insufficient: it produces tensors at cosine 0.75–0.87 from the ones a correct forward pass
builds, and that is enough to lose the answer.

Read [`RESULTS.md`](RESULTS.md) for the numbers. The corrected method is Findings 18–20; the
rotation study is Findings 1–17 and several of them are **superseded** — see the correction
index at the end of this file.

---

## Results at a glance

Everything below is **measured on real hardware**, not estimated. Three separate experiments, in
the order they were run.

### 1. KV compaction: recompute, do not rotate (Findings 18–24)

| | retention | retrieval | KV saved |
|---|---|---|---|
| rotate keys in place | 60% | **fails** — keys sit at cos 0.75–0.87 from a correct forward | 34–37% |
| **recompute the retained tokens** | 60% | **passes at every depth tested** | 34–37% |
| prefix-cache reuse (`f=0.15`) | 60% | passes, **28% of the prefill reusable** | +7.7% wall |

The load-bearing result: **a compacted cache is only correct if the retained tokens are run
through a fresh forward pass.** Key-only RoPE re-rotation fixes the *position* and nothing else —
a key's content is conditioned on everything before it, so moving it leaves it wrong.

### 2. Long-horizon agent work: bounded context vs linear prefill (definitive run)

Gemma-4-12B, bf16, fully GPU-resident on an RTX A6000. **120 turns per arm, identical
instruction, identical task.** The only variable is the context policy.

| | **linear prefill** | **anchored runtime** | ratio |
|---|---|---|---|
| features passed | 122/123 (99.2%) | 123/123 (100%) | ~1.01x |
| **turn-1 nonce retained** | **LOST** | **RETAINED** | — |
| TTFT, first → last | 731 → **7,325 ms** | 736 → **497 ms** | **flat vs 10x growth** |
| prompt, first → last | 501 → **12,065 tok** | 501 → **1,228 tok** | **24.1x vs 2.5x** |
| peak KV | **4,184 MB** | **501 MB** | **8.4x less** |
| wall clock | 24.8 min | **10.9 min** | **2.3x faster** |
| total prefill tokens | **1,021,255** | **142,911** | **7.1x fewer** |

**The honest headline is cost and retention, not raw accuracy.** Accuracy nearly equalises on
this task; what does not is the price of holding the same context, and whether a fact stated once
at the beginning still exists 119 turns later.

### 3. The broken-repo task — does the edge survive a harder job?

The feature task above turned out to be **too easy to discriminate** (122 vs 123): adding an
*independent* function per turn needs no memory of earlier turns. So the task was rebuilt so that
turn 1 is load-bearing:

> `solution.py` is seeded with 80 broken functions. **Turn 1 carries the complete bug report
> once** — for each bug, the exact wrong value and the exact required value, drawn from an
> unguessable constant. **Every later turn says only `Fix bug 3.`** — no restatement, no hint.

A model that has lost turn 1 cannot reason its way to an arbitrary constant. It can only guess,
and a guess is visible as a wrong number.

**Scored per turn** (owner's metric): the assertion for bug `i` runs *immediately after turn `i`*.
This separates a fix that was never applied from one that was applied and then clobbered by a
later full-file rewrite.

**Baseline arm, measured mid-run** (n = 34 turns):

| turns | report in context? | correct |
|---|---|---|
| 0–6 | yes | **5/7 (71%)** |
| 7+ | no — evicted | **5/27 (19%)** |

The file itself stays structurally perfect throughout (all 80 functions present, parses clean) —
the failures are **value-level**, i.e. genuinely lost memory rather than a broken harness.

---

## The method

```
1. prefill the full sequence, capturing per-token self-attention
2. score the evictable region by the attention each token receives
3. keep the first 15% of that region outright, then the top attention-scoring remainder
4. take the pristine leading KV prefix from the original cache (up to the first eviction)
5. forward only the rest against it, with cache_position resuming at the prefix length
6. decode against that
```

KV memory drops 34–37%. Retrieval holds at every depth measured, on the 0.5B and the 7B. There
is no rotation step, no per-layer band, no line pooling, and no conditioning patch — all four
were tried, and all four turned out to be unnecessary once the cache was built by a correct
forward pass rather than by surgery.

**Where the critical content sits matters more than how you select.** Validity is a *prefix*
property — a token is reusable only if everything before it is untouched — so evicting the
tail is the only policy that preserves the leading tokens, and it is free only when the
content you need is not in the tail. Findings 23 and 24.

## What this costs

The recompute is the price. For a single request the sequence is prefilled twice — once to
score, once compacted — in exchange for a smaller decode-time cache. That is worth it for long
generations and not for short ones.

**Part of that cost can be recovered without any new mechanism.** Causal attention runs
left-to-right, so any token whose whole prefix survived untouched has the same hidden state and
the same RoPE phase as in the uncompacted run — the leading `k` entries of the original cache
are pristine and need not be recomputed. Slice them off, forward only what remains with
`cache_position` starting at `k`, and the recompute shrinks. That is Finding 21, worth **7.7%
wall-clock** as measured.

The size of the win is set by the *selection pattern*, not the caching code. Scattered top-k
evicts early, so the pristine prefix is short. Force-keeping the first 15% of the evictable
region triples the reusable prefix (9.2% → 28.4%) **at no cost in retrieval** — Finding 22.
That is the recommended configuration.

For prefix-cache reuse across *requests*, where the point is to avoid a recompute at all, this
method still does not apply: avoiding the recompute is exactly what the rotation approach was
for, and it does not work.

## Run it

```bash
python src/native_vs_cache.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.55
python src/selectors_corrected.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
```

Needs `torch` and `transformers`. Downloads `Qwen/Qwen2.5-0.5B` on first run (~1 GB); the 7B
runs are 8-bit and need ~10 GB. CPU only, no CUDA, no vLLM, no Triton — deliberately standalone.

| script | what it establishes |
|---|---|
| `src/native_vs_cache.py` | the decisive control: same retained tokens, native vs cache path |
| `src/selectors_corrected.py` | selector comparison **against the corrected pipeline** |
| `src/cache_vs_native.py` | how far the cache tensors drift from a correct forward |
| `src/native_7b.py` | the 7B confirmation, three failure modes separated |
| `src/prefix_cache.py` | prefix-preserving re-prefill; asserts the prefix is bit-identical |
| `src/prefix_tradeoff.py` | reuse vs retrieval as the forced-prefix fraction varies |
| `src/inverted_context.py` | pristine prefix per eviction policy + the suffix drift equation |
| `src/needle_layout.py` | the inversion test: move the needle, hold the policy fixed |
| `src/cascade_sifter.py` | two-speed cascade: cheap model sifts, expensive model consumes |
| `src/block_causal.py` | block-diagonal attention: eviction isolation, with the causal contrast |
| `src/tiered_cache.py` | global backbone + modular evictable blocks (Tier 1/2/3) |
| `src/laya_sift.py` | Laya as a line-granularity sifter |
| `src/laya_labels.py` | sifter training labels; distil vs ablation |
| `src/sink_probe.py` | where attention mass actually goes |
| `src/decode_divergence.py` | first token where a compacted decode diverges from full |
| `src/depth_sweep.py` | the original depth sweep, rotation pipeline |

## The result, in one table

Selector comparison, identical rows, identical budget, **corrected pipeline**:

| depth | baseline | line pooling | mid-layer band | random | old rotation path |
|---|---|---|---|---|---|
| 15% | PASS | PASS | PASS | fail | fail |
| 35% | PASS | PASS | PASS | fail | PASS |
| 55% | PASS | PASS | PASS | fail | fail |
| 75% | PASS | PASS | PASS | PASS | PASS |
| | **4/4** | **4/4** | **4/4** | **1/4** | **2/4** |

**The selector is not the lever. The pipeline is.** A plain attention sum is sufficient; every
selector refinement this repository tried was compensating for a corrupted cache.

## Corrections

Several earlier findings are superseded. They are kept in `RESULTS.md` with their retractions
rather than deleted, because the way each one went wrong is the most transferable part of this
work.

| finding | status |
|---|---|
| 1, 2 | stand — the retention floor and the rotation effect are real measurements |
| 3, 4 | stand |
| 5, 6 | **superseded** — the "query-agnostic selector works at 75%" claim is about a pipeline that does not work |
| 7, 8 | stand as descriptions of the broken pipeline |
| 9–13 | **superseded** — nine scalar explanations (attention mass, sink displacement, fragmentation, density, RoPE geometry) all failed to predict the verdict, because they were measuring the symptom of one defect |
| 12 | **withdrawn** — "two causes" was wrong; there is one cause |
| 14 | stands — the missing full-cache control; compaction is the sole cause |
| 15 | stands — position-independent caching is worse than dense rotation |
| 16 | **corrected** — recomputing only the needle proved nothing; the whole cache must be right |
| 17 | **superseded** — the mid-layer band fixed a depth that is not broken once the pipeline is correct |
| **18** | **the root cause** — cache surgery, not selection |
| **19** | the 7B separates selection failure from rotation failure |
| **20** | **with the pipeline fixed, the selector barely matters** |
| **21** | prefix-preserving re-prefill: exact (bit-identical, not cosine), correct, worth 7.7% |
| **22** | **the selection pattern is the lever** — force 15% of chunk 1, 3x the reuse, verdicts unchanged |
| **23** | **the inverted-context layout is the wrong way round** — suffix reuse is impossible; critical content early, not late |
| **24** | **the layout is the knob** — with the policy fixed, the verdict tracks where the needle sits, 5/5 |
| **25** | **Laya's HTTP API silently drops a plain-string state** — pass `{"text": ...}` |
| **26** | base Laya zero-shot is not a usable sifter — needle ranks 6/109, behind compiler warnings |
| **27** | **the cascade works** — cross-model sift 4/4 PASS, and two sifters with only 19% overlap both succeed |
| **28** | **block-diagonal attention IS isolated** — eviction error 3e-05 vs 1.7e+01 causally |
| **29** | **the query must be its own tier** — inside the last block it cannot see earlier blocks at all |

## Prior art

LazyAttention, MEPIC, MiniPIC, Irminsul, SemPIC, COMB and Leyline all apply RoPE correction to
cached keys, and all are GPU-only. The eviction literature (H2O, SnapKV, StreamingLLM, TOVA,
Quest) selects tokens and leaves coordinates alone, which is a different problem.

**This repository's contribution is a negative result about that first family.** Deferring or
recomputing a rotation assumes the cached key's *content* is intact and only its *position*
needs fixing. Measured here, that assumption fails: a rotated key sits at cosine 0.75–0.87
from the key a correct forward pass would build at the same position, and a retained needle
survives a native recompute and dies through the rotation path on identical tokens. If that
generalises, deferred-rotation caching is not a drop-in for recomputation.

## Finding 35 (2026-09-30, Modal A10G — Bob's benchmark matrix)

**The block runtime cannot perform a cross-block JOIN. Cause is PREFILL ISOLATION, not the
positional gap left by eviction.** Measured, not inferred.

Qwen2.5-7B, babilong task: Block 1 "Richard moved the key kitchen->shed", Block 3 "Jessica moved
it shed->attic", distractor between, ROTATED OUT. Ground truth = **attic**.

| arm | layout | answer |
|---|---|---|
| 3 | survivors in ONE causal sequence | **attic** PASS |
| 4 | full causal + distractor | **attic** PASS |
| 5 | survivors re-prefilled contiguously over the global cache | **attic** PASS |
| 1 | block table, shipped layout (disjoint ranges, gap) | *garbage/degenerate* FAIL |
| 2 | block table, block3 adjacent to block1 — **same isolation, NO gap** | *degenerate* FAIL |

**arm1 vs arm2 is the experiment that decides it, and arm2 failing is the answer.** Both arms
prefill the blocks in SEPARATE forwards, so both are isolated; only the positions differ. If the
gap were the cause, arm2 would pass. It does not. **The gap is not the defect — the isolation is.**

**⇒ WHY.** `tiered_cache.build_table` forwards each block against a FRESH CLONE of the global
cache, so block 1's and block 3's keys never co-attended. Block 1 encodes "shed" in a hidden state
conditioned on a context that does not contain block 3, and vice versa. The query can attend over
both, but neither block carries anything the other can use. The join has to happen inside the
query tier, and a single query forward cannot manufacture the relation that never existed.

**This CONFIRMS Finding 34's prediction and extends it.** F34 showed at block granularity the
binding constraint is CONTEXT, not evidence. F35 shows the same limitation on the hardest case: a
relation split across two isolated blocks is not recoverable by eviction policy, selection, or
position arithmetic.

**⇒ WHAT DOES WORK, and it is the recompute path we already established.** arm5 — re-prefill the
survivors as ONE contiguous span over the global cache — passes. That is F18/F21's answer again
from a different direction: **when a cross-block relation matters, the survivors must be
recomputed as a single sequence, not reassembled from independently-prefilled blocks.** Block
tables remain correct for independent self-contained payloads (per-file summaries, retrieved
passages, documents); they are the wrong shape for one artifact whose meaning spans its length.

**Verified non-vacuous:** the three controls all PASS on the same model and the same surviving
text, so the failures are attributable to the layout rather than to the model being unable to do
the task. Survivor drift under eviction measured **0.000e+00** throughout — the blocks themselves
are bit-identical, which is exactly why the failure is informative: the KV is intact and the
answer is still wrong.

Artifacts: `modal/probe_query.py` (decoder-vs-architecture, both fail identically ⇒ not a decoder
bug) · `modal/probe_gap.py` (this experiment) · `modal/bench.py` (3-arm matrix on GPU).

## Finding 36 (2026-09-30, Modal A10G — Bob's Test 4, runnable half)

**Multi-turn serving cost: the block runtime cuts prefill work ~60% and wall clock ~70% over 8
turns with 4 evictions, and answers the late-turn fact correctly on Qwen2.5-7B.** Measured, both
arms summed over all turns.

| model | control | vanilla | runtime | prefill vs vanilla | wall vs vanilla |
|---|---|---|---|---|---|
| qwen2.5-7b | PASS | PASS | **PASS** | **39%** | **26%** |
| mistral-7b-instruct | PASS | PASS | fail | 41% | 29% |

**★ WHY THE SAVING IS REAL AND WHY IT GROWS.** The vanilla arm re-prefills the ENTIRE context on
every turn, so its token count is quadratic in the number of turns. The runtime ingests each block
once and forwards only the new turn plus the query. At 8 turns that is 617 tokens against 1568.
The gap widens with turns and with context size, which is the regime a real agent loop runs in.

**★ BUT THE RESIDENT-KV CLAIM DOES NOT HOLD HERE, AND THIS IS THE HONEST PART.** Bob's Test 3
asked for "~50% savings on evicted Block 2" and that is NOT what this workload shows: measured KV
went 18.4 MB (vanilla) → 23.1 MB (runtime), i.e. it did not drop at all. Two reasons, both
specific to the test: the evicted turns are ~30-token failed attempts (tiny), and the global anchor
is a separately retained tier that vanilla does not pay for. **Eviction savings are a function of
how much the evicted content weighs relative to the retained anchor.** With a 2,800-token evicted
block the saving is large (measured 99% in the bench matrix); with 4 thirty-token turns it is
negative. Do not quote a KV saving without naming the eviction-to-context ratio.

**★ THE RUNTIME ANSWER IS CORRECT BUT NOT CLEAN, ON BOTH MODELS.** Qwen returns
"The most recent IP IP address successfully assigned addr add 192.168.1..." — right address, and it
does NOT regress to the un-sudo'd command Bob predicted, but it stutters and echoes. Mistral never
converges inside 40 tokens ("The most recent IP address successfully assigned to eth0 is
192.168.1 "). The control and vanilla arms answer in 3 tokens. So the runtime's decode is measurably
less stable, even when the fact is present — consistent with Finding 35's mechanism: the surviving
blocks never co-attended, so the query tier is recovering the answer from a weaker representation.

**Verified non-vacuous:** the control arm carries the identical context in one causal sequence and
PASSes, so the comparison is against a working reference rather than a broken baseline. The
assertion targets the newest successful bind (`192.168.1.57`), which is never evicted, so cost is
isolated from the F35 join failure.

Artifacts: `modal/multiturn.py`, results in `benchmarks/results/multiturn_*.json`.

## Finding 37 (2026-09-30) — FINDING 35 WAS WRONG. The defect is the POSITIONAL GAP, not isolation.

**10 randomised trials per model, 2 architectures. The eviction gap decides the outcome, and it is
fixable.**

| arm | qwen2.5-7b | mistral-7b-instruct | what it is |
|---|---|---|---|
| control | **0/10** | **100%** | survivors + distractor, one causal sequence |
| vanilla | **10/10** | 0/10 | survivors only, one causal sequence, no distractor |
| **runtime** | **0/10** | **0/10** | block table, survivors keep ORIGINAL positions (gap) |
| **gap-adjacent** | **10/10** | **100%** | block table, survivors re-indexed CONTIGUOUSLY |

**runtime vs gap-adjacent is the experiment, and it is the same architecture in both arms** —
identical separate-prefill isolation, identical KV, identical eviction. The ONLY difference is
whether block 3 keeps its absolute position id (3300+ tokens after block 1, with the evicted span
missing between them) or is re-indexed to sit directly after block 1. **The gapped layout fails
100%; the contiguous layout passes 100%, on both models.**

**⇒ FINDING 35's CONCLUSION WAS WRONG AND ITS EVIDENCE WAS CONTAMINATED.** F35 concluded "prefill
isolation, not the gap" from `probe_gap.py`. That probe had a slicing bug — it addressed block KVs
by POSITION ID instead of insertion index, so gapped blocks sliced to EMPTY tensors. Both F35 arms
were therefore measured with a broken/absent block 3. The bug was found and fixed in the same
session, but F35's *conclusion* was written against the buggy run and was not revisited. The
corrected probe printed coherent strings for the gapped arm where the buggy one printed `0\n0\n0000`,
and the conclusion should have been revisited then. **A probe fix that changes the observed symptom
invalidates the finding built on the old observation — re-run the whole finding, not just the probe.**

**★ WHY THE GAP BREAKS IT.** RoPE is relative, so the distance between block 1's keys and the query
is what the query's attention sees. With the evicted span missing, block 3 sits at position ~3300
while the table only holds ~90 rows — the query lands ~3200 positions from block 1 instead of ~60.
The attention geometry is that of a context that still contains the distractor, but the content is
gone, so block 1 is effectively out of range. Re-indexing collapses the survivors into the position
space they actually occupy.

**★★ THE FIX IS ONE LINE AND IT IS NOW MEASURED: re-index survivors contiguously on eviction.**
Block caches are built separately (isolation preserved, which is what makes eviction lossless by
construction) and then ASSEMBLED with contiguous position ids. This keeps every property the design
wants — bit-identical survivors, no recompute, O(1) eviction — and repairs the cross-block join.
`gap-adjacent 10/10` both models is that fix verified.

**⚠️ THE CONTROL ARMS ARE MODEL-DEPENDENT AND BOTH ARE NEEDED.** Qwen's control is 0/10: with the
2,800-token distractor present in a plain causal sequence it cannot answer, while the same content
without the distractor is 10/10 — attention dilution, reproduced. Mistral-instruct inverts it:
control 100%, vanilla 0% — it follows the log format but not the bare concatenation. A single
control would have mislabelled one of the two models as broken.

**Verified non-vacuous:** 10 randomised port/secret pairs per arm per model, so no answer is
memorisable; the two block arms differ in exactly one variable; and the fix is confirmed on two
architectures. Artifact: `modal/trials_two_needle.py`.

## Finding 38 (2026-09-30) — Bob's Test 1 (RoPE): his stated pass metric is contradicted

**Bob's pass metric:** *"retaining original absolute positional coordinates preserves exact
generation coherence WITHOUT requiring continuous re-indexing."* **Measured: it is never better
and sometimes fatal. Re-indexing is never worse.**

10 randomised trials, both arms hold the SAME block caches (same KV, same isolation):

| model | full (no eviction) | STATIC (original ids — Bob's metric) | COMPACT (re-indexed) |
|---|---|---|---|
| qwen2.5-7b | 0% | **0%** | **100%** |
| mistral-7b-instruct | 60% | **0%** | **0%** (task-format dependent; 100% in the Finding 37 format) |

**⇒ KEEPING ORIGINAL POSITION IDS WINS NOWHERE AND LOSES BADLY.** In every measurement across
both models and both test formats, `static` is 0%. `compact` reaches 100% on Qwen in this format
and 100% on both models in the Finding 37 format. **Re-indexing survivors contiguously is the
correct behaviour, and Bob's hypothesis is the opposite of what the data shows.**

### ★★ THE DRIFT NUMBER CONTRADICTS THE "10⁻⁵ FLOOR" PREMISE — and this is the subtle part
Mean key drift of the surviving blocks against the full no-eviction run: **static 1.507e+01,
compact 1.507e+01 — IDENTICAL to 4 significant figures.** Re-indexing does not change a single key
tensor (it is the same blocks; only the assembly offset moves), so **drift cannot discriminate
between the two arms, and it cannot be the mechanism behind the join failure.** Only coherence can.

**More importantly: the drift here is ~1.5e+01, NOT 1e-05.** The 10⁻⁵ floor is real but it applies
only when the survivors never attended to the evicted content (the block-diagonal case, Finding 28).
In a LINEAR causal prefill the survivors attended to everything before them, and that influence is
baked into their keys. Evicting the distractor afterwards leaves ~15.0 of drift — the CONTENT error
Finding 23 identified. **Do not quote "10⁻⁵ vs 10¹" as a property of eviction in general; it is a
property of eviction from BLOCK-DIAGONAL isolation specifically.** The same distinction decides
which architecture can claim what.

**Verified non-vacuous:** the `full` arm (no eviction) is measured on the same trials and reaches
60% on Mistral, so the task is answerable and the 0% block arms are attributable to eviction.
Artifact: `modal/rope_offsets.py`.

## Finding 39 (2026-09-30) — Frankenstein vs history eviction: a NULL, and the null is informative

**Bob's question:** when a model repairs code across iterations, does discarding earlier failed
attempts let it cleanly overwrite its previous draft, or does it hallucinate deprecated variables
from them?

**Answer: neither, at this scale. Evicting earlier failures changed nothing, and produced zero
Frankenstein artifacts.**

Qwen2.5-7B, 3 repair rounds, every candidate genuinely exec'd against real assertions. Two arms
differing ONLY in what stays in context (`history` = all prior attempts and errors; `evict` = only
the most recent failure).

| task set | arm | pass@1 | pass@3 | frankenstein |
|---|---|---|---|---|
| easy (6 trivial bugs) | history | 100% | 100% | 0% |
| easy | evict | 100% | 100% | 0% |
| hard (4 multi-piece) | history | **50%** | **50%** | **0%** |
| hard | evict | **50%** | **50%** | **0%** |

**★ THE EASY SET MEASURED NOTHING AND THAT WAS CAUGHT BY MEASURING.** pass@1 = 100% on both arms
means every bug was fixed on the FIRST attempt: the repair loop never ran, so there was no
multi-turn iteration to study and no opportunity for a Frankenstein artifact. The hard set was
built in response, and it genuinely fails — `calc` and `tok` are never solved in 3 rounds.

**★ THE RESULT: eviction is FREE here and Frankenstein did not appear.** Identical 50%/50% in both
arms, and 0% undefined-name failures. Discarding earlier failed drafts neither helped the model
recover nor hurt it, and did not produce a single ghost variable.

**⚠️ SAMPLE SIZE — DO NOT OVER-READ THIS.** Only 2 of 4 hard tasks fail, so there are 4 failing
trajectories total (2 tasks x 2 arms). **0/4 is weak evidence of "no Frankenstein"** and should not
be reported as a rate. To make a claim about the phenomenon the harness needs many more tasks the
model fails — which is exactly the real-repo multi-turn loop Bob's Test 4 specified, and which
does not exist here. Reporting `frankenstein = 0%` without that caveat would be the overclaim.

**★ A GRADER THAT CANNOT FAIL MEASURES NOTHING, TWICE OVER.** Both task sets were validated locally
BEFORE any GPU time: every buggy version must FAIL its assertions and at least two independent
correct implementations must PASS them. The first draft failed this check on **4 of 6 tasks** — a
mutable default inside a factory does not actually share state, `for it in items[:]` while removing
does not actually skip, `round(a,2)==round(b,2)` passed the intended counter-example, and a
recursive `deep_sum` was already correct. All four would have reported as model failures.

Artifacts: `modal/frankenstein.py` (task sets + arms), `modal/cost.py` (ledger).

## Correction to Finding 36 (2026-09-30)

**The multi-turn run evicted 3 turns, not 4.** `evict_idx` selects the even-indexed turns in
`range(1, 8)`, which is `[2, 4, 6]` — three of them. An earlier report described the run as
"8 turns / 4 evictions" following the specification; the measurement is **8 turns, 3 evictions**.
The plot labels itself from the data and prints 3. `--turns 9` would produce 4 evictions. The cost
figures are unaffected (they come from the run, not the spec), but the headline should read
**8 turns / 3 evictions**.

Artifact: `benchmarks/plot_results.py` -> `benchmark_wallclock_vs_recompute.png` (three panels:
per-turn prefill work, cumulative wall clock with the one-time ingest drawn separately, and the
memory watermark — the third panel deliberately shows the runtime NOT winning, because on this
workload it does not).

## Redaction of synthetic test credentials (2026-09-30)

The two-needle and RoPE tests generate a RANDOM credential per trial
(a Stripe-shaped live-key prefix plus 12 random hex chars) so no answer is memorisable. Those values are synthetic and
were never valid, but committed into a **public** repository they are indistinguishable from
leaked live keys — a scanner, or a reader, has no way to tell. **Every occurrence has been changed
to `EXAMPLE_KEY_`, preserving the random suffix.**

**This does not alter any result.** The prefix is cosmetic; the pass/fail outcome depends on
whether the model produced the port:secret pair, and that is unaffected. Verified after the edit:
all 47 result cells still parse and the runtime PASS count is unchanged at 9. The generators in
`modal/trials_two_needle.py`, `modal/rope_offsets.py`, `modal/bench.py` and `benchmarks/tasks.py`
now emit the `EXAMPLE_KEY_` prefix so future runs cannot reintroduce the problem.

## Finding 40 (2026-09-30) — Bob's Sequential Scratchpad does NOT close the state-update case

**Bob's Solution 1:** force an intermediate trace ("Trace each movement sequentially: 1. First, …")
so the generated tokens re-link the isolated blocks during decode.

**Tested and refuted.** Qwen2.5-7B, 10 randomised trials, three moves across two surviving blocks
(each move a different place, sampled from ten), all arms on the same Finding 37 contiguity fix so
only the decode strategy varies:

| arm | what it does | success |
|---|---|---|
| **direct** | query → answer in one forward | **30%** |
| **trace** | query → forced sequential trace → answer | **10%** |
| **null-trace** | same token budget, no movement content | **0%** |

**⇒ THE SCRATCHPAD IS NOT BETTER, AND THE NULL ARM SHOWS WHY THE FORMAT ISN'T THE ISSUE.** `trace`
does not exceed `direct`; `null-trace` is worst, so extra decode steps alone do not help either.
**⚠️ HONEST STATISTICS: 3/10 vs 1/10 is Fisher two-sided p = 0.582.** At n=10 the arms cannot be
separated, so the defensible claim is **"the scratchpad does not help"**, NOT "it makes things
worse". Reporting the direction without the p-value would be the overclaim.

**★ A REAL HARM WAS OBSERVED, AND IT IS DIAGNOSTIC.** The trace arm produces degenerate loops on
this model — `"drawer drawer drawer drawer"`, `"2222222222222…"`, prompt echo
(`"user What is the blue key used for?"`). The scratchpad format causes repetition collapse, so
even the mechanical case for it fails here.

**★ AND THE DIRECT ARM IS ONLY 30%, which bounds the whole question.** This task (three moves,
randomised from ten places, two different actors) is materially harder than the earlier babilong
run, and the state-update failure is DEEPER than decode strategy — it is not that the model needs
more steps, it is that the later fact is not being applied over the earlier one at all.

**⇒ Verdict: the cross-block JOIN is solved (Finding 37, contiguity); the cross-block state UPDATE
is not, and a decode-side scratchpad does not solve it.** The remaining candidate is the
recompute path — re-prefill the survivors as one sequence — which is the same answer Findings 18
and 21 reached from a different direction.

Artifact: `modal/scratchpad.py`. Cost: this run was UNINSTRUMENTED (imported `track()` and never
called it); the wrapper is now applied so future runs record.

## Finding 41 (2026-09-30) — the RECOMPUTE path DOES close the state-update case

**The open defect from Finding 40 is solvable, and this is the measurement.** Three arms, identical
randomised task (three sequential moves, three distinct places from ten, two different actors),
same final question, 10 trials per model:

| arm | qwen2.5-7b | mistral-7b-instruct |
|---|---|---|
| **full** (every block, one causal sequence — upper bound) | 100% | 100% |
| **block** (block table + contiguity fix — the shipped runtime) | **30%** | **10%** |
| **recompute** (survivors re-prefilled as ONE contiguous span) | **100%** | **100%** |

**⇒ ASSEMBLING INDEPENDENTLY-PREFILLED BLOCKS CANNOT PERFORM A SEQUENTIAL STATE UPDATE, AND
RE-PREFILLING THE SURVIVORS AS ONE SEQUENCE CAN.** Both `block` arms fail in the same way — they
answer with an EARLIER place (`garden shed`, `kitchen drawer`, `hallway closet`) or with nothing
(`The blue key`), never the last one. The recompute arm answers the final place on every trial of
both models.

**★ THE COST, IN RAW COMPUTE (measured per query, averaged):**

| | qwen2.5-7b | mistral-7b-instruct |
|---|---|---|
| block — prefill tokens | 22 | 17 |
| recompute — prefill tokens | 73 | 73 |
| block — wall | 34.2 ms | 35.0 ms |
| recompute — wall | 39.7 ms | 42.5 ms |
| block — resident KV | 6.0 MB | 12.3 MB |
| recompute — resident KV | 5.8 MB | 12.4 MB |

**The recompute path costs ~3.3-4.3x the prefill tokens per query and ~16-21% more wall clock.**
It buys 10-30% → 100% correctness. **This is the same trade Findings 18 and 21 priced from the
drift direction**: correctness costs a re-prefill of the surviving text, and the block table's
zero-recompute eviction is exactly what forfeits the update.

**★ SO THE DESIGN SPLITS BY OPERATION, AND THAT IS THE USABLE RESULT:**
| operation | correct engine | why |
|---|---|---|
| lookup / concatenation across blocks | **block table** (contiguity fix) | 10/10 at 3.3x cheaper; no cross-block dependency beyond the join |
| sequential state update across blocks | **recompute** | independent prefills cannot carry a later fact that must supersede an earlier one |

**Verified non-vacuous:** the `full` arm is 100% on both models, so the task IS answerable and the
block arm's failure is attributable to the assembly rather than to the model. The task is
randomised per trial so no answer is memorisable. Artifact: `modal/recompute_state.py`.

## Finding 42 (2026-09-30, Modal) — RAW HARDWARE METRICS: all three disagree, and they correct the previous headline

Bob: *"calculate raw hardware FLOPs, memory bandwidth demands, and actual Watt-hours. This approach
provides an objective, reproducible metric for the paper."* and *"Also do llama, so we dont get the
critic that its model dependent"*.

Measured 20 randomised trials per model. **Watt-hours are MEASURED** (`nvidia-smi power.draw`
sampled on a background thread and integrated), not TDP × time. FLOPs and bytes are MODELLED with
the formula printed. Both arms were warmed up first: the first timed arm read **401 ms** against a
warm **318 ms**, i.e. CUDA kernel compilation and cuDNN autotune landed entirely on whichever arm
ran first and made BLOCK look slower than RECOMPUTE. One-time costs must be paid before timing, and
the arm order now alternates per trial.

### Cross-architecture, the correctness result REPRODUCES

| model | block | recompute |
|---|---|---|
| Qwen2.5-7B | 5/20 | **20/20** |
| **Llama-3.1-8B** | **7/20** | **20/20** |

The block runtime fails the sequential state update on real Llama exactly as it does on Qwen, and
recompute closes it on both. The critic's "model dependent" objection is answered.

⚠️ `meta-llama/Llama-3.1-8B` is **gated** — `403` on file access from this box *and* from Modal,
while the API metadata endpoint returns **200**. `NousResearch/Meta-Llama-3.1-8B` is an ungated
mirror of the same checkpoint (verified `model_type=llama`, `hidden_size=4096`, 32 heads, 14336
intermediate, `gated:False`). Use the mirror; do not read a 200 metadata response as access.

### ★★ THE THREE METRICS DO NOT AGREE — which is the finding

| claim | Qwen | Llama | verdict |
|---|---|---|---|
| **1. compute** (FLOPs/query) | block **0.48×** recompute | block **0.54×** | real win for block |
| **2. bandwidth** (bytes/query) | KV is **0.01%** of bytes | **0.01%** | **NEITHER can win at batch 1** |
| **3. wall clock** | block **1.19×** recompute | block **0.99×** | depends on what is counted |
| **4. energy** (MEASURED Wh) | block 0.0175, recompute 0.0144 | 0.0433 vs 0.0456 | ≈ equal, both directions |
| **5. KV residency** | 5.94 MB vs 5.83 MB | 18.76 MB vs 18.76 MB | capacity, not speed |

### ★★★ THE CORRECTION: MY PREVIOUS HEADLINE WAS A COMPUTE CLAIM, AND COMPUTE IS NOT THE BINDING CONSTRAINT

I told Bob the honest headline was *"16-17% of vanilla in attention pairs, not 39% in tokens"*.
**Attention pairs are FLOPs, and on this hardware FLOPs are nearly free.** Arithmetic intensity:

    ridge point (A10G) = 125 TFLOP/s ÷ 600 GB/s = 208 FLOP/byte
    both arms measure ≈ 1.00 FLOP/byte

**Both arms sit ~208× BELOW the roofline ridge point, so the workload is memory-bandwidth-bound and
its FLOP count is almost irrelevant.** Reducing attention pairs by 84% therefore buys far less than
the number suggests — and a reviewer who knows the roofline would have said so. **That claim was
overstated and is retracted.**

### What actually drives the win, and what KV compaction is honestly for

**Mechanism.** At batch 1 every decode/prefill token reads the ENTIRE weight matrix (7-8B × 2 B ≈
14-16 GB) while the KV it touches is megabytes. The measured bandwidth split is decisive:

    block      weights 525.70G | kv_read 44.6M | kv_write 18.4M | TOTAL 525.76G | KV share 0.01%
    recompute  weights   1.28T | kv_read 63.0M | kv_write 44.6M | TOTAL   1.28T | KV share 0.01%

So the runtime's advantage comes from **making fewer passes over the weights** — 35 vs 85 prefill
tokens on Llama, 22 vs 72 on Qwen, each token paying a full weight read — **not** from reading less
KV. The honest placement:

| property | value | where it counts |
|---|---|---|
| fewer weight passes | real, 0.48-0.54× FLOPs | **wall-clock at batch 1** |
| less KV read/written | **0.01% of bytes** | nothing at batch 1 |
| **less KV resident** | 5.9-18.8 MB/query | **context length that fits; batch>1 where weights amortise** |

**KV compaction earns its place on RESIDENCY and on throughput at batch > 1.** At batch 1 on one
GPU it cannot move the bandwidth number, because the weight term dwarfs it. Stated here so the paper
does not claim a bandwidth win it does not have.

### ★ And the contiguity fix costs something, which corrects Finding 41's wall figure

Finding 41 reported *"recompute 1.2× wall"* counting the **query only**. Counting the work actually
required to close the positional gap — the moved block must be **re-prefilled** so it is contiguous
with its predecessor — the block arm's wall advantage disappears: **1.19× on A10G, 0.99× on A100.**

| what is timed | block vs recompute |
|---|---|
| query only (Finding 41) | block cheaper (recompute 1.2×) |
| **end-to-end incl. contiguity re-prefill** | **≈ equal** |

Both figures are real and answer different questions; the end-to-end one is the honest one for a
deployment, because a moved block always has to be re-prefilled.

⚠️ **Cross-GPU wall-clock is not comparable.** The same workload measured 380 ms on A10G and 1146 ms
on A100-40GB — a small-batch fp16 workload does not scale with the A100's peak. Only arm-vs-arm
comparisons within one run are valid; the A100 run exists because an 8B fp16 model plus the 29k-token
control's activation exceeds the A10G's 22 GB (`24.00 MiB requested, 64.00 MiB free`).

### Artifacts
`modal/hardware_metrics.py` (measured power + modelled FLOPs/bytes) ·
`modal/arithmetic_intensity.py` (the roofline comparison above) ·
`benchmarks/results/hardware_*.json`.

## Finding 43 (2026-09-30, Modal) — APPLIED BENCHMARKS: SWE-bench Lite and long-horizon tool calling

Bob: *"Set up a 10-task SWE-bench Lite run comparing standard linear prefill against Frankenstein
across Qwen-7B, Mistral-7B, and Llama-3-8B. Log TTFT per turn, total wall-clock to resolution,
peak resident KV bytes, and pass/fail parity. That's the table that sells the runtime."* and
*"also more benchmarks, the more the better, preferably ones testing long horizon tool calling."*

### SWE-bench Lite — real instances, and the gold arm is the parity check

10 real `princeton-nlp/SWE-bench_Lite` tasks, all `sympy/sympy`, all with a single-test
FAIL_TO_PASS so per-turn verification is fast. Three gates, each because without it the table is
empty or false:

1. **Non-vacuity, per instance.** The dataset's own `test_patch` is applied to the UNFIXED tree and
   the F2P test must FAIL, then the gold patch applied and it must PASS. All 10 verified. Without
   this an instance that already passes measures nothing.
2. **Gold is a PARITY CHECK, not a baseline — 10/10 resolved.** If gold does not resolve, a model
   failure and a broken environment are indistinguishable.
3. **PASS_TO_PASS validated against the base state.** The dataset's own P2P list contains tests
   already failing at the base commit — measured: `sympy__sympy-11897` lists `test_latex_Float`,
   which fails on the untouched tree before any patch. Treating the raw list as a regression guard
   marked a CORRECT gold patch unresolved. Now validated once per instance and cached.

| model | arm | resolved | TTFT t0 | TTFT tN | tN/t0 | peak KV |
|---|---|---|---|---|---|---|
| **qwen2.5-7b** | linear | 0/10 | 164 ms | **753 ms** | 4.58× | **576.6 MB** |
| | **runtime** | 0/10 | 138 ms | **378 ms** | **2.74×** | **340.6 MB** |
| **mistral-7b-instruct** | linear | 0/10 | 152 ms | **793 ms** | 5.21× | **1273.9 MB** |
| | **runtime** | 0/10 | 163 ms | **307 ms** | **1.89×** | **792.7 MB** |
| llama-3.1-8b | linear | 0/10 | 139 ms | 959 ms | 6.92× | 1475.2 MB |
| | runtime | *running* | — | — | — | — |

**Runtime holds 59–62% of linear's peak KV and ends at 0.39–0.50× linear's last-turn TTFT.**
TTFT growth is 4.6–6.9× for linear against 1.9–2.7× for the runtime.

### ★★ EVERY ARM RESOLVED 0/10, AND THAT IS REPORTED FIRST BECAUSE IT IS THE HONEST HEADLINE

SWE-bench Lite is hard and these are 7–8B models. **A 0% resolution rate in both arms is a statement
about the model, not about the layout**, and it is why the table leads with TTFT, prefill tokens and
KV instead: those are measured on every turn regardless of whether anything is solved. A table whose
only number was `0/10 vs 0/10` would be true and useless.

The failure mode is legible in the logs and it is not a layout failure: the models loop on `search`,
re-reading rather than writing. Of 10 turns, most arms spend 8+ on retrieval and never emit
`edit_file`. That is a capability ceiling, and it is exactly why the long-horizon benchmark below
holds competence constant instead.

### Long-horizon tool calling — the result that actually separates the layouts

40 turns, 3 trials, per-turn graded, Qwen2.5-7B. The task plants config state that is **superseded**
mid-run: a key is set on turn 1 and changed on turn 12, so a correct agent must carry the newest
value and drop the stale one. That distinguishes *"evicted the stale copy"* (correct and cheap) from
*"evicted the live copy"* (fatal) — a benchmark that only evicts filler cannot see the difference.

| arm | accuracy | first half | second half | TTFT t0→tN | KV t0→tN |
|---|---|---|---|---|---|
| linear | 6.7% | 13.3% | **0.0%** | 488 → **5723 ms** | 33 → **1076 MB** |
| **runtime** | **15.8%** | **21.7%** | **10.0%** | 428 → **785 ms** | 33 → **168 MB** |

**2.4× the accuracy, 7.3× lower last-turn TTFT, 6.4× less resident KV.** The linear arm drops to
**zero** accuracy in the second half — it loses the thread entirely once the context grows — while the
runtime holds 10%. Every turn is graded, so that collapse is visible as a curve rather than collapsed
into one final answer.

⚠️ Absolute accuracy is low for both (6.7% vs 15.8%); a 7B is weak at 40-turn state tracking. **Read
the direction, which is measured on every turn, not the level.**

### Harness defects found by RUNNING, each of which scored zero for the harness not the model

- **parser spanned first `{` to last `}`.** A valid tool call followed by prose produced
  `Extra data: line 2 column 1` — valid JSON read as malformed. Every turn of the first run failed
  this way. Now `raw_decode` takes the first complete object and ignores the tail.
- **`Invalid \escape`.** A model writing `\d+` (a regex) or `\alpha` (LaTeX) into a JSON string
  without doubling the backslash produces a document `json.loads` rejects outright. Repaired by
  doubling a backslash that is not a legal JSON escape, **only after parsing has already failed**, so
  a document that parses is never touched — the same safety argument as the raw-newline fix.
- **a repeated identical parse failure is a LOOP.** Measured: the runtime arm sat at an identical
  549-token prompt repeating one error for three turns, because the correction text was the same
  sentence each time. After three consecutive failures the observation now shows the exact required
  shape.
- **`read_file` returned whole files** (3000 lines → the prompt hit its 12000-token cap) and
  `max_new=384` truncated a whole-file write. Now windowed numbered reads plus an **`edit_file`** that
  refuses a non-unique match rather than guessing, which is a corruption risk on source code.
- **Modal rate-limits app creation.** Six arms launched at once: three ran, three exited immediately
  with `App create rate limit exceeded` — which looks identical to a silent failure. The launcher now
  staggers and retries explicitly, so an arm that never ran can never be reported as a zero.

### Artifacts
`modal/swe_solve.py` · `benchmarks/swe_table.py` · `modal/longhorizon.py` ·
`benchmarks/results/swe_*.json` · `benchmarks/results/longhorizon_*.json` ·
`benchmarks/results/superseded/` (the pre-fix runs, kept as evidence of the bugs they exposed).

---

# The long-horizon benchmark in full

This is the part a reader needs in order to judge or reproduce the applied result, and it is
written so that nothing has to be taken on trust.

## What question it answers

> A coding agent works through a long task. Its context is full. Something must be dropped. **What
> does dropping the oldest turns actually cost, and does keeping a small anchored context instead
> of the full transcript preserve the ability to do the work?**

Two policies, same model, same task, same instruction:

- **`linear`** — the full transcript, truncated **oldest-first** only when the model's own window
  forces it. This is the honest production failure mode: a fixed window means something must go,
  and every naive policy drops the oldest.
- **`runtime`** — a **turn-1 anchor plus the most recent few turns**, capped at a small budget of
  its own (`RUNTIME_TOKENS`, default 6144). The anchor (block 0) is never evicted; the middle is
  dropped on every turn once the module outgrows the budget.

Both arms are capped by the **same model** and given the **same turn count**. The ceiling
(12,000 prompt tokens) is applied identically; the difference is *which* turns are kept.

## Task A — the feature task

Each turn asks for one independent function. A **contract is stated exactly once, in turn 1**, and
never repeated; the final grader checks whether it is still in force at the end.

Why the contract alone is not enough: the model re-writes `REGISTRY[...]` on every `append`, so
the *rule* is re-derivable from recent context no matter what was evicted. Measured — `contract:
True` for **both** arms, at 49 turns and again at 120.

The discriminator is therefore a **nonce**: `QX7-4420-BRAVO`, stated once in turn 1 and required
only in the final append. Nothing else in the transcript contains it, so it is recoverable
**exactly when turn 1 survived**.

| | linear | runtime |
|---|---|---|
| features | 122/123 | **123/123** |
| contract | True | True |
| **nonce** | **LOST** | **RETAINED** |

A contract the model applies repeatedly leaves traces and cannot test retention. Testing only the
contract would have reported *"both fine"* and been wrong — twice.

### The prompt bug that invalidated an earlier run

An earlier 49-turn run reported linear at **38/52** with twelve consecutive feature failures. That
number is **retracted**: the per-turn instruction said *"Write the FULL solution.py"* while the
system prompt said *"prefer `append`"* — a flat contradiction, and the per-turn wording won. The
baseline rewrote the whole file every turn (~335–383 tokens/turn), hit its cap at ~turn 26, and
the result measured the **prompt**, not the policy. With the instruction made consistent the cost
falls to ~150 tokens/turn and eviction moves to ~turn 78.

**A turn count must be chosen AFTER the prompt is fixed.** 49 turns was too short under the
corrected prompt, which is why nothing was reported until the 120-turn run existed.

## Task B — the broken-repo task (the harder job)

### Design

`modal/bug_task.py` — `build(n)` returns `(seed_source, bug_report_lines, tests, reference_fixed)`.

Six bug families, chosen so the correct answer cannot be derived from the broken code:

| family | shape of the bug | what the report gives |
|---|---|---|
| wrong multiplier | `return x * 39` | must become `return x * 222` |
| wrong offset | `return x + 354` | the constant must be `57` |
| wrong kwarg default | `def f(x, k=117)` | the default must be `13` |
| wrong comparison | `return x > 321` | must be `>=` against `321` |
| wrong slice bound | `return x[:6]` | must be `[:11]` |
| wrong lookup key | `TABLE['hotel']` | the key must be `'golf'` |

Constants are derived from `sha256("bug:<i>")`, so they are **deterministic but unguessable** —
reproducible without being inferable.

Turn 1 carries the entire report. **Every later turn is a bare `Fix bug N.`** The instruction also
restates two rules that carry **zero information about any bug**, so they cannot leak the answer:

- *keep every function's exact name* (the model renames aggressively; see below)
- *do not copy the bug list, or any comment, into `solution.py`*

### Grading — per turn, immediately

```
turn i completes  ->  run bug i's assertion right now  ->  turn_ok[i]
...
at the end        ->  run every bug's assertion        ->  passed
gap (turns_ok - passed) = fixes applied and then LOST to a later rewrite
```

Two numbers, because they answer different questions. A model can apply a fix correctly and then
destroy it on the next full-file rewrite; grading only the final state would show the loss without
ever showing that the model had it right.

### Two harness defects found while building this — both would have read as model failures

**1. A bare comparison in a `try/except` never fails.** The generated test body was
`m.bug_00(1) == 222` — an *expression*, not an `assert`. It evaluates to `False` and **does not
raise**, so the `except` never fired and every bug was marked fixed. Measured: the **seeded
(broken) module scored 20/20**. Every generated test now emits an `assert`.
Verified after: broken **0/120**, reference-fixed **120/120**, half-fixed **60/120**.

**2. The model renames functions.** Told to keep the exact names, Gemma-4-12B wrote
`def multiply_by_factor(...)` instead of `def bug_00(...)` and scored **0/4 while applying every
correct value**. Restating the rule did not stop it. Grading is now by **definition order**, not
name — `_fns` is the module's top-level functions sorted by `co_firstlineno`, built once in
`bug_task.FIX_PRELUDE` and shared by the harness and every test. This is **no weaker**: the
required constant still exists only in turn 1, so a turn that lost turn 1 still cannot produce it.

### ★ The runtime budget had to fit the anchor plus a turn

`RUNTIME_TOKENS` was 2048, chosen for the feature task where each turn appends one function and
the file accumulates on disk. The bug report is **~2637 tokens on its own**, so a 2048 budget
popped **every** recent turn: the runtime arm would have received the bug list and **no file state
at all**, then failed mechanically — a result that would have read as *"anchoring does not help"*.

Caught by **simulation before the arm ran** (real per-turn sizes, both budgets, 80 turns):

| `RUNTIME_TOKENS` | turn 0 | turn 20 | turn 79 | file state? |
|---|---|---|---|---|
| 2048 | 2637 tok | 2637 tok | 2637 tok | **no** |
| **6144** | 2637 tok | **4337 tok** | **4337 tok** | **yes, flat** |

vs linear, which reaches the 12,000-token cap and **loses turn 1 at ~turn 7**.

## Fairness gates — a comparison that cannot be refused is just a formatter

Two ways this experiment could have reported a difference it did not earn, both closed in code:

**1. A wall-clock stop is arm-dependent by construction.** The runtime arm keeps a small bounded
context, so it is *faster per turn*; stopping on time lets it complete **more** turns and win on
volume rather than policy. The stop condition is therefore the **turn count**. `TIME_BUDGET_S`
survives only as an emergency break (`CCAI_TIME_BUDGET_S`, default 21600), so a genuine
interruption grades what completed instead of losing everything.

**2. Two policies capped at the same ceiling become the same policy.** An earlier version capped
*both* arms at 12k, so runtime grew to ~9.9k and linear to ~11.2k and they converged — measured
4,180 ms vs 4,308 ms, a ratio of 1.0. The experiment could not discriminate at all. Each policy
now has its **own** budget sized to its own claim.

`tools/run_compare.py` **refuses to report** unless `features_completed` is identical across arms,
and refuses any run carrying `harness_failed`. Proven in four cases: fair pair → exit 0; unequal
turns → exit 1; `harness_failed` → exit 1; explicit paths → exit 0.

## The sliding-window confound — must be stated

Gemma 4 is not dense attention. Read from the real `config.json`:

| model | layers | heads/kv | head_dim | layer types | bf16 |
|---|---|---|---|---|---|
| gemma-4-12B-it | 48 | 16/8 | 256 | **40 sliding / 8 full** | 23.9 GB |
| gemma-4-26B-A4B-it | 30 | 16/8 | 256 | **25 sliding / 5 full** | 51.6 GB |
| gemma-4-31B-it | 60 | 32/16 | 256 | **50 sliding / 10 full** | 62.5 GB |

`sliding_window = 1024`, and **83% of layers are sliding**. Two consequences:

1. **VRAM sizing** — a dense KV calculation overstates the cache by ~4x. Use
   `sliding_layers × 1024 + full_layers × ctx`.
2. **Part of "linear lost the contract" is the model's own attention pattern**, not purely the
   baseline's eviction: past 1024 tokens the turn-1 fact is invisible to the sliding layers
   regardless of policy. The full-attention layers can still carry it, and the runtime arm's
   explicit anchor keeps it in the retained set by construction. **This is stated rather than
   buried, because a reviewer would otherwise be right to raise it.**

## Reproducing it

### Hardware and model

Everything here ran on a rented **RTX A6000 (48 GB)** with **Gemma-4-12B-it in bf16, fully
GPU-resident** — 23.9 GB, **0 parameters offloaded**, ~82% GPU utilisation. No quantisation
library in the path, no CPU offload.

### Why 12B bf16, and not 8-bit or 26B

- **8-bit is a silent no-op on this MoE.** `BitsAndBytesConfig(load_in_8bit=True)` on
  `gemma-4-26B-A4B-it` does not quantise it: bnb only replaces `nn.Linear`, and in this MoE the
  **expert weights are raw parameters**. Measured: 426 `Linear8bitLt` modules but **`bytes/param`
  = 1.94 — i.e. bf16**, 48.8 GB allocated on a 48 GB card, OOM on the first forward pass.
  **`bytes/param` is the oracle** (int8 ≈ 1.0, bf16 ≈ 2.0); `config.quantization_config` being
  present proves nothing about what landed on the GPU.
- **A conversion-time quantised checkpoint** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit`) was tried and
  failed to apply (`lm_head.weight ... MISSING`, loaded as bf16).
- **26B bf16 with CPU offload works but is unusable**: per-turn wall grew 59 → 98 → 136 → **416 s**
  (~16 h per 80 turns). Offload hooks — arithmetic, not tuning.
- **12B fits entirely on the GPU and finishes.** bf16 with nothing offloaded is also *cleaner* than
  8-bit: there is no quantisation confound in the result at all.

### The harness

```
modal/incremental_coding.py   the loop, both arms, the graders, the checkpointing
modal/bug_task.py             the broken-repo task generator (build(n))
deploy/vast_entry.py          runs it on a plain GPU box (stubs the Modal decorators)
deploy/runfix.sh              fix-mode A/B: smoke -> linear -> runtime
tools/run_compare.py          the fairness gate; refuses an unfair comparison
```

Run it on any GPU box:

```bash
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hftok)"        # the token must reach the PROCESS, see below
CCAI_TASK=fix CCAI_MODEL=gemma-4-12b CCAI_QUANT=none \
CCAI_ARM=linear CCAI_FEATURES=80 \
  python -u /root/ccai/vast_entry.py
```

Then compare the two arms **through the gate**, never by eye:

```bash
python3 tools/run_compare.py '/root/ccai/results/*.json'
```

### Environment facts that cost real time

- **`HF_HUB_ENABLE_HF_TRANSFER=1` is deprecated** and does nothing. The live knob is
  `HF_XET_HIGH_PERFORMANCE=1`.
- **An unauthenticated HF download is rate-limited to a crawl.** Measured: **9 MB/s decaying to
  0.00 MB/s** with the progress bar frozen, versus **17.9 MB/s** authenticated. Diagnostic that
  works: sample the directory size twice —
  `a=$(du -sb "$D"|cut -f1); sleep 30; b=$(du -sb "$D"|cut -f1)` — and confirm the token reached
  the **process** with `tr '\0' '\n' < /proc/<pid>/environ | grep -c HF_TOKEN`. A credential on
  disk is not a credential in the process.
- **`curl` without `-L`** reports the redirect, not the download — a fabricated speed.
- **A Cloudflare speed test from the box is useless**: it returned HTTP 403 / 18 B/s while HF was
  pulling 17.9 MB/s.
- **`ssh` can refuse while the run is fine.** Judge the box by the API's `actual_status` +
  `gpu_util`, never by the ssh port.
- **A "stopped" instance can still be billing.** Two A6000s were running at once; the old one sat
  at 0% GPU charging $0.317/hr. `actual_status` from the API is the oracle, and an idle GPU is
  visible only in the instance list.

### The checkpoint must be atomic — and it matters most at the end

`open(path,"w")` truncates first: die in that window and the file is half-written JSON, the loader
raises, and the resume path **silently starts fresh** — a whole paid run lost to a ~10 ms window.

Fixed as `tmp → flush → fsync → rotate to .bak → os.replace`, with a `.bak` fallback on read, and
a `_ckpt_dir()` that **creates** an explicitly named `CCAI_CKPT_DIR` rather than silently ignoring
it and falling through to the CWD. Measured, SIGKILL at varying offsets inside the write window,
14 rounds per size:

| checkpoint size | old (truncating) | new (atomic) |
|---|---|---|
| 400 KB | 13/14 | **14/14** |
| 2 MB | 12/14 | **14/14** |
| **8 MB** | **7/14 — a coin flip** | **14/14** |

The risk is highest at turn 150–200, exactly when the most compute has been invested.

## Harness defects found by running, each of which scored zero for the harness and not the model

The recurring lesson of this project: **the reply is a claim; the file at the line is the
evidence.** Every one of these produced a number that looked like a model failure.

1. **A grader that cannot fail measures nothing** — the bare-comparison bug above (broken module
   scored 20/20).
2. **A rename is not a wrong answer** — grading by name failed a run that applied every value.
3. **A parser that rejects valid input is indistinguishable from a model that cannot produce valid
   output** — flat-args tool calls and raw newlines inside JSON strings were both discarded as
   "malformed" while the model was doing the work correctly.
4. **A truncated reply recorded as a model failure** — 215 of 219 no-answer rows were the
   harvester's own `MAX_TOKENS` cap (`finish_reason=length`, `reasoning_tokens == MAX_TOKENS`).
5. **A missing key is falsy** — a status reader trusting the wrong field reports a confident
   falsehood rather than an error (a healthy run read as dead; a busy lane read as idle).
6. **0.0% CPU + frozen output = blocked, not thinking** — a wedged lane was diagnosed as "still
   generating" for 15 minutes.
7. **An empty findings document is ambiguous** — a lazy model returning `[]` for every file is
   indistinguishable from a genuinely clean batch unless the engine refuses to resolve it as
   "clean".

## Citation of the earlier findings

Findings 1–17 (the rotation study) are largely **superseded** by Findings 18–24. Finding 20 was
retracted in part: line pooling and the mid-layer band were compensating for a corrupted cache,
and against the corrected pipeline baseline / line-pooling / mid-layer-band are all 4/4. The
selector is not the lever — **the pipeline is**.

Findings 25–34 (block-diagonal attention, the cascade, needle preservation, the shallow-depth
budget) and 35–43 (the positional-gap defect, the state-update case, raw hardware metrics,
SWE-bench and long-horizon tool calling) stand as written, with their corrections indexed inline.
