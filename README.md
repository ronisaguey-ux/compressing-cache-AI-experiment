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
