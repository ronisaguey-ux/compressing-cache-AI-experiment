# Saliency-Preserving KV Compaction with Dense Delta Re-rotation

Measured results, reproducible with `python src/test_2d_kv_cache.py`.

**Model** Qwen/Qwen2.5-0.5B, CPU, fp32. 24 layers, 14 query heads, 2 KV heads, head_dim 64.
**Context** chunk 0 = 32 tok system prompt holding `SYSTEM_FLAG_A = ALPHA_VERIFIED`;
chunk 1 = 329 tok compiler log holding `0x9AF4_STACK_FAIL` at offset 220 (67% through the region).
**Queries** Q1 asks for the flag (answer in chunk 0, never evicted — the control);
Q2 asks for the failure code (answer in chunk 1, the region being evicted).
**Format** the model's own chat template. An earlier revision drove it as a raw continuer and
several conclusions did not survive the change — see [Corrections](#corrections).

## The control that has to pass first

| arm | cache | nll Q1 | nll Q2 |
|---|---|---|---|
| Arm 0 baseline, untouched | 361 | 2.769 | 2.268 |
| Identity: keep **all** + rotate | 361 | 2.769 | 2.268 |

Re-rotating a complete cache applies `R(0)` to every key, so it must reproduce the baseline
exactly. It does, to 0.000 nats, in every run. Any drift below is caused by eviction, not by the
rotation machinery.

## Finding 1 — a 10% keep budget is below the retention floor for every selector

Pure index arithmetic, no model needed. "Does this selector retain the needle?"

| selector | 5% | 10% | 15% | 25% | 50% | 75% |
|---|---|---|---|---|---|---|
| first-k/2 + last-k/2 | – | – | – | – | – | yes |
| random-k | – | – | – | – | – | yes |
| top-k key-norm | – | – | – | – | – | yes |
| top-k query-attn (all 24 layers) | – | – | – | – | – | yes |
| **top-k query-attn (mid layers 10-19)** | – | – | – | – | **yes** | yes |

**At 10% — the budget the original design assumed — no selector keeps the needle.** A study run
only at 10% would report "every eviction strategy fails" and would have measured the budget, not
the strategy.

Per-layer needle rank:

```
layer  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
rank 303 199 323 314 207 249 221 219 301 134 162  60  86  92  88  87  46 169  88  96 124 140  68 194
```

Layer 0 ranks the error code **303rd of 333**. Layers 11-19 carry it (best: layer 16, rank 46).
Averaging all 24 dilutes that signal below the threshold, which is why the all-layer aggregate
misses the needle while the mid-band finds it. **Any aggregation scheme should check per-layer
ranks before summing.**

## Finding 2 — dense delta re-rotation is what lets a retained needle be used

Keep 50%, same kept set, only the rotation differs:

| kept set | needle | rot on | rot off | gain |
|---|---|---|---|---|
| top-k query-attn (mid layers) | yes | `0x9AF` · nll 2.823 | *"You are a build assistant…"* · nll 4.599 | **+1.776 nats** |
| top-k key-norm | no | `1` · nll 4.554 | *"You are a build assistant…"* · nll 4.503 | −0.051 |

Where the needle survives, switching rotation off does not merely degrade the answer — the model
stops answering altogether and recites the system prompt. **+1.776 nats and a categorical change
in behaviour.** Where the needle is already gone, rotation changes nothing, as expected: correct
positions cannot save content that was evicted.

Rotation improved the rotated-vs-unrotated comparison in the earlier 4-point sweep at
**+0.091 / +0.923 / +2.019 / −0.187 nats**. It is the single highest-leverage component tested.

## Finding 3 — only query-attention selectors produce a needle-bearing answer

At 50% retention (45.7% of KV memory saved). No arm retrieved the **full** literal, so `Q2
lit/norm` is 0/0 everywhere; the informative signal is what the model actually emitted.

| arm | needle kept | Q2 emitted | nll Q2 | Δ |
|---|---|---|---|---|
| Arm 0 baseline | yes | `0x9AF4_STACK_FAIL` ✅ | 2.268 | — |
| **2c query-attn (mid layers)** | **yes** | **`0x9AF`** ◐ | 2.823 | +0.555 |
| **2b query-attn (all layers)** | no | **`9AF`** ◐ | 2.781 | +0.513 |
| 1k first5+last5 | no | `1` ❌ | 4.520 | +2.252 |
| 2a key-norm | no | `1` ❌ | 4.554 | +2.286 |
| R random-k | no | `9` ❌ | 4.595 | +2.327 |
| 3 arm2a, no rotation | no | *system prompt* ❌ | 4.503 | +2.235 |
| 3b arm2c, no rotation | yes | *system prompt* ❌ | 4.599 | +2.331 |

The two query-attention arms land on the real code (truncated by one hex digit); every other arm
produces a wrong number or degenerates. **The query-agnostic selectors available here — key L2
norm and first-k/last-k — do not work at any budget tested.** That is the production blocker: the
selectors that work need the question before they can decide what to keep.

## Finding 4 — text squashing answers the easy question and loses the hard one

| | Arm 1 |
|---|---|
| cache | 50 tok, 86.2% saved |
| Q1 | ✅ `SYSTEM_FLAG_A is ALPHA_VERIFIED.` |
| Q2 | ❌ `The failure code in the build log is "0".` |
| extra prefill | **48.4 GFLOP** — a full re-encode |

Every eviction arm pays **zero** extra prefill; they reuse cached keys. Arm 1 pays for a second
model pass and still loses the answer.

## Corrections

An earlier revision of this harness drove the model as a raw continuer instead of through its
chat template. Two claims from that revision **do not survive** and are retracted here:

1. **"Eviction destroys a question whose answer was never touched."** In the raw-continuer runs,
   Q1 failed in 22 of 24 eviction arms with a 3.089-nat spread, which looked like severe
   conditioning loss propagating into the untouched prefix. With the chat template applied, Q1
   passes in most arms and the spread collapses. The effect was **largely a formatting artefact**,
   not conditioning loss. It should not be cited.
2. **Absolute accuracy figures** from those runs are not meaningful. The qualitative ordering
   between arms was stable, which is why the comparative findings above are kept, but any number
   from the raw-continuer runs should be discarded.

Finding 1 (the retention curve) and Finding 2 (rotation) were unaffected — both are either pure
index arithmetic or a same-cache comparison, so neither depends on the output format.

## Finding 5 — a query-AGNOSTIC selector does work, at 75% retention

The blocker stated in Finding 3 was that the working selectors were oracles. This arm removes
that: it scores chunk-1 positions by **chunk 1's own self-attention**, which is available the
moment the log arrives, before any question exists.

Keep 75% (247 of 329 positions, **22.7% of KV memory saved**):

| arm | needle | Q2 emitted | nll Q2 | Δ |
|---|---|---|---|---|
| Arm 0 baseline | yes | `0x9AF4_STACK_FAIL` ✅ | 2.268 | — |
| **Arm 4a self-attn chunk1 (agnostic)** | yes | **`0x9AF4_STACK_FAIL` ✅** | **2.556** | **+0.288** |
| Arm 2c query-attn (mid, oracle) | yes | `0x9af4` ◐ | 2.632 | +0.364 |
| Arm 4c arm4b, **no rotation** | yes | *"CI failed. CI_FLAG_A = 0…"* ❌ | — | +1.000 |
| Arm 2c arm2c, no rotation | yes | *"CI failed" ×9* ❌ | — | +1.146 |

**Arm 4a produces a complete literal answer at this budget** — see Finding 6, where the
needed keep fraction is measured properly and the query-aware selector turns out to be stronger. The rotation-off row is the control that matters: the same kept set without
phase correction loops on `CI failed`, so the recovery is the rotation, not the selector alone.

### The sink guard is a regression and should not be used

Scoring chunk 1 against the whole prefix made position 4 win — an attention sink, high by
position rather than content. Masking the first 4 positions was the obvious fix. Measured:

| keep | Arm 4a (raw) | Arm 4b (+sink guard) |
|---|---|---|
| 50% | nll 3.970 | nll 3.579 (better) |
| 75% | **nll 2.556, answers correctly** | nll 3.123 (worse) |

The guard helps where nothing worked anyway and **costs the correct answer where it did work**,
because it discards positions 0-4 of the log — and the first lines of a build log carry the
command and target. **The sink is real; masking it by position is the wrong instrument.**
Recorded as a rejected approach rather than deleted, so it is not retried.

## Where this leaves the design

Eviction is viable, and the recipe that works is narrower than the original design assumed:

1. **Budget at 75%, not 10%.** The retention floor for every selector tested is above 50%; 10% is
   below it for all of them.
2. **Score with chunk-1 self-attention**, which needs no query — the only shippable selector found.
3. **Always apply the delta re-rotation.** Without it the same kept set degenerates.
   *(Finding 8 bounds this recipe: it is validated at one needle depth.)*
4. **Do not mask attention sinks by position.**
5. **Dense positions only.** Sparse strides were rejected before implementation (see RESEARCH.md).

## Finding 6 — the budget floor is 59.6%, not 75% (corrects Finding 5)

Finding 5 reported 75% as the working budget. That was an artefact of the retention curve
sampling only 5/10/15/25/50/75% — it showed "no" at 50 and "yes" at 75 and I read the requirement
as 75. The needle's actual **rank** under each selector gives the floor exactly:

| selector | needle rank | keep needed | KV saved at that budget |
|---|---|---|---|
| self-attn (agnostic) | 195 / 329 | 196 (59.6%) | 40.4% |
| key-norm | 235 / 329 | 236 (71.7%) | 28.3% |
| self-attn + sink guard | 191 / 329 | 192 (58.4%) | **41.6%** |

Confirmed by running at 60% and 65%:

| keep | KV saved | arm | Q2 emitted | Δ nll |
|---|---|---|---|---|
| 60% | 36.6% | **2c query-attn (mid layers)** | **`0x9AF4_STACK_FAIL`** ✅ | **+0.051** |
| 60% | 36.6% | 4a self-attn (agnostic) | `0x9AF.` ◐ | +0.558 |
| 65% | 31.9% | **2c query-attn (mid layers)** | **`0x9AF4_STACK_FAIL`** ✅ | **+0.099** |
| 65% | 31.9% | 4b self-attn + sink guard | `0x9AF.` ◐ | +0.387 |
| 60% | 36.6% | 4c (4b, no rotation) | *"You are not allowed to continue."* ❌ | +2.010 |

**The saving figure improves from 22.7% to 36.6% of KV memory**, with the query-aware selector
producing the complete literal at +0.051 nats — essentially baseline.

### A rank is a better instrument than a pass/fail curve

A curve answers "does it work at this budget"; a rank answers "how much budget does it need",
which is the question a tuning decision turns on. The curve said 75%; the rank said 59.6% and was
right. **Report the rank, not just the curve.** The two agree on the binary and disagree by 15
points on the number.

Rotation remains mandatory at every budget measured: at 60% with the same kept set, switching it
off changes the answer from the exact code to a refusal loop, worth +2.010 nats.

## Finding 7 — combining signals makes selection WORSE, and the best-looking rank is a trap

Rank is the floor (Finding 6), so a higher-ranking selector directly means more memory saved.
Six candidates, each scored against **two** needles — the real error code and a decoy factual
line elsewhere in the same log. A selector tuned to one needle is not a selector.

| selector | real `0x9AF4` | decoy `ld: bad value` |
|---|---|---|
| self-attn all 24 layers | #195 → save 40.4% | **#197 → save 39.8%** |
| self-attn mid layers 10-19 | **#168 → save 48.6%** | #221 → save 32.5% |
| key-norm | #235 → save 28.3% | #304 → save 7.3% |
| mid + key-norm | #244 → save 25.5% | #309 → save 5.8% |
| all + key-norm | #241 → save 26.4% | #300 → save 8.5% |
| mid + key-norm + entropy proxy | #241 → save 26.4% | #310 → save 5.5% |

**Two conclusions, both negative for the obvious next move:**

1. **Adding signals hurts.** Every combination ranks worse than its best component alone
   (mid: 168 → 244 when key-norm is added). Key-norm is not a complementary signal here; it is
   noise that dilutes a working one. More features is not the direction.

2. **The best-looking row is a one-needle artefact.** Mid-layer self-attention ranks the real
   needle best (#168) and the decoy worst-but-one (#221) — a 32% relative gap between two
   needles it treats very differently. All-layer self-attention ranks them almost identically
   (#195 vs #197, a 1% gap). **A selector whose rank swings that much between two factual lines
   in the same region is fitting the needle, not the task.** With one needle it would have looked
   like a 48.6% win over 40.4% and been reported as such.

**The robust selector is all-layer self-attention at ~40% saving** — not because it ranks best,
but because it is the one that ranks consistently. **A selector must be validated on at least two
needles before its rank is trusted**; this is the same class of error as choosing a keep fraction
from a curve sampled at the wrong points.

## Finding 8 — the recipe is not robust to WHERE the fact sits

Findings 5-7 measured one layout with the error line 67% through the region. This moves it.
The shipped recipe (all-layer self-attention ranking, 60% keep, delta rotation **on**) was run
against four needle depths:

| needle depth | chunk1 tok | needle rank | keep | KV saved | result |
|---|---|---|---|---|---|
| 15% | 906 | 365 | 59.9% | 40.1% | ❌ recites the system prompt |
| 35% | 1,428 | 498 | 59.9% | 40.1% | ❌ `0x9.` — wrong |
| **55%** | 1,960 | 366 | 60.0% | 40.0% | ✅ **`0x9AF4`** |
| 75% | 2,189 | 393 | 60.0% | 40.0% | ❌ `arianarianarian…` |

**One of four.** And the failure is not "the fact was evicted": at 15% the needle ranks 365 of 906,
comfortably inside a 544-position keep budget, and the model still recites the system prompt.

That separates the two failure modes cleanly:

- **Retention failure** — the selector dropped the fact. Fixable by ranking better.
- **Utilisation failure** — the fact is present in the cache and the model cannot use it.
  Not fixable by ranking at all.

Depth 15% is a utilisation failure, which means **the selector's rank is not sufficient as a
success criterion.** A high rank is necessary and it is not enough, and Finding 6's floor — derived
from rank alone — is therefore optimistic. The 36.6% figure holds for the canonical layout and
must not be quoted as a general property.

This is also the most likely reason the published methods report their wins on specific benchmark
layouts rather than on arbitrary needle placement.

Next experiment this implies: hold the layout long (as at depth 15%, 906 tok) and sweep the keep
fraction upward until utilisation succeeds, which measures the *utilisation* floor separately from
the *retention* floor. That number is what a production budget would have to be set against.

## Limits

- **0.5B model.** It degenerates under mild perturbation (`1.0.0.0.0.0` loops). Absolute quality is
  not meaningful; the comparisons are, since all arms share model, prompt and decode settings.
- **The full literal was never retrieved at 50%.** A higher keep fraction is untested for the
  literal; the retention curve shows 75% keeps the needle for every selector, which would make
  selection non-diagnostic, so the informative band is between 50% and 75%.
- **One model, one task.** Finding 8 shows even one needle POSITION is not enough: the
  recipe passes at 55% depth and fails at 15/35/75%, so no single-layout result generalises. No claim about other architectures or MLA/GQA.
- **Saliency is measured from the query pass**, so 2b/2c are oracles.
- **Greedy decoding against a fixed cache is deterministic**, so these deltas are reproducible,
  not sampling noise.
- **No trained baseline comparison.** Heuristic attention selection is compared against a random
  control, not against a learned eviction policy such as Kamera's.

## Reproduce

```bash
python src/test_2d_kv_cache.py --keep-fracs 0.10,0.25,0.50,0.75 --json data/run.json
```
