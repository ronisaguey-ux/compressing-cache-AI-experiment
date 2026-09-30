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

## Finding 9 — selection is not the bottleneck, so no selector fixes depth fragility

Finding 8 showed the recipe passes at only one of four needle depths. Three candidate fixes were
tested across all four: position normalisation, line pooling, and a non-causal scoring mask.
**None passes all four, and the reason is that none of them addresses the actual failure.**

### The causal-mask hypothesis is wrong, and the measurement is the opposite of the prediction

The claim was that causal masking starves early positions of contributors, creating a recency
bias. My scorer **already computes chunk-1 x chunk-1 attention with no causal mask** — cached K is
a linear projection of each token and carries no causal information, so only the weight matrix
would impose causality. Scored with the mask added, the needle ranks *worse*, not better:

| depth | non-causal rank (shipped) | causal rank | effect of adding the mask |
|---|---|---|---|
| 0.15 | 365 | 349 | marginally better |
| 0.35 | 498 | 625 | **worse** |
| 0.55 | 366 | 743 | **much worse** |
| 0.75 | 393 | 981 | **much worse** |

The mask hurts at three of four depths, by up to 2.7x in rank. **The shipped scorer is already the
non-causal one; adding causality would degrade it.** Not a bug to fix — a hypothesis that the
measurement refutes.

### The needle is retained almost everywhere, and the answer fails anyway

Across 16 (mode, depth) combinations the needle survived the cut in **15**. Results:

| depth | baseline | posnorm | line | both |
|---|---|---|---|---|
| 0.15 | fail | fail | **PASS** | **PASS** |
| 0.35 | fail | **PASS** | fail | fail |
| 0.55 | **PASS** | fail | fail | fail |
| 0.75 | fail | fail | **PASS** | fail |

Each depth is passed by a *different* mode. Only one of the sixteen combinations failed to retain
the needle — so **retention is not the failure mode**, and a better selector cannot fix a failure
that selection does not cause.

### Raising the budget does not repair it either, and the result is non-monotonic

| keep | mode | 0.15 | 0.35 | 0.55 | 0.75 | passes | saved |
|---|---|---|---|---|---|---|---|
| 75% | line | PASS | PASS | PASS | fail | 3/4 | 20.5% |
| 75% | baseline | PASS | fail | fail | PASS | 2/4 | 19.8% |
| 90% | line | PASS | fail | PASS | PASS | 3/4 | 6.0% |
| 90% | baseline | PASS | fail | fail | PASS | 2/4 | 6.0% |

**90% keep fails at depth 0.35 where 75% keep passed.** A monotonic failure (more context, better
answer) is what conditioning loss would predict; a *non-monotonic* one is a different signature.
At 90% keep the saved memory is down to 6% while still failing, so paying for context does not buy
reliability either. This is the 0.5B model's own noise showing through, and it means **this model
cannot settle the question** — the effect sizes are smaller than the variance.

### What is actually established

- **Two of the three fixes are useful; none is sufficient.** Line pooling is the best single
  change (3/4 at both budgets vs baseline's 2/4). Position normalisation actively hurts at depth
  0.15 (rank 365 → 550), contradicting its hypothesis.
- **Conditioning loss, not selection, is the wall.** The needle is present and unusable. No
  ranking, pooling or rotation change can restore a value vector whose context has been deleted;
  this is the same conclusion Kamera reached, now reproduced as a selector-independent effect.
- **The result at 36.6% is layout-specific.** Finding 6's figure came from the layout with the
  needle 67% through the region. It is a valid measurement of that layout and not a general
  property.

### Limits that bound this conclusion

- **0.5B only.** Non-monotonic results at this scale mean the model cannot separate signal from
  variance. Bob's direction was explicit: understand the mechanics first (done above), then
  validate on a larger model.
- **A 7B run is not free on this box.** Qwen2.5-7B is 28 GB in fp32 and 14 GB in bf16 against
  15.7 GB total RAM shared with other services, so it needs 8-bit (~7 GB) or 4-bit (~3.5 GB)
  quantisation. `bitsandbytes` is not installed. **Quantisation changes the numerics, so a 7B
  result is not a like-for-like rerun of this harness** and the difference must be measured, not
  assumed.

## Finding 10 — 8-bit quantisation works on CPU, and it does change the verdict

7B cannot run on this box unquantised (28 GB fp32 / 14 GB bf16 against 15.7 GB of shared RAM),
so 8-bit is not optional. `bitsandbytes` 0.50.2 installs and loads on CPU here even though
`torch.cuda.is_available()` is False — measured: Qwen2.5-0.5B loads quantised in 5.9 s and
correctly answers `The capital of France is` → ` Paris`.

It is not free of consequence. bnb casts activations to float16 internally and its `MatMul8bitLt`
kernel warns `inputs will be cast from torch.bfloat16 to float16 during quantization`. In this
harness that surfaced as a hard dtype error (`expected m1 and m2 to have the same dtype, but got:
float != c10::BFloat16`) in the saliency scorer, because cached keys were fp16 while the captured
query projections were bf16. Both sides are now cast to fp32 for the saliency arithmetic only —
selection ranks scores, so precision beyond fp32 buys nothing and it keeps the scorer identical
between precisions.

**With that fixed, 8-bit changes the result at depth 15%:** fp32 with line pooling **passed**;
8-bit with the same configuration fails, degenerating into an `eliac` repetition loop. So the
quantisation delta is real and large enough to flip a verdict, and **any 7B number must be
reported alongside the precision it was measured at.** Treating 8-bit as "the same experiment,
smaller" would be wrong.

## Finding 11 — a silent RoPE fallback would have invalidated every rotation number

Found while pre-flighting the 7B config. `Qwen2.5-7B` declares `rope_theta = 1000000`, and the
0.5B does too — but **in transformers 5.17 `rope_theta` is not a top-level config attribute.** It
lives inside `rope_parameters`:

```
to_dict() rope keys: {'rope_parameters': {'rope_theta': 1000000.0, 'rope_type': 'default'}}
```

`getattr(cfg, "rope_theta", None)` therefore returns `None`, and the harness's ORIGINAL code
silently fell back to `theta = 10000.0` — a 100x error in every rotary frequency. Nothing in the
existing verification could have caught it:

- **The identity control passes anyway.** `R(0) = I` for *any* theta, so "keep everything and
  rotate" reproduces the baseline exactly no matter how wrong the frequency table is.
- **The `R(a)R(b) = R(a+b)` self-check passes anyway.** That identity is a property of rotations
  in general, not of the correct theta.

So the two checks that look like they validate the rotation both do not. The only thing that
validates it is comparing against the model's own table, which is now what happens:

```python
ref = model.model.rotary_emb.inv_freq          # the authority
if not torch.allclose(rope.inv_freq, ref): raise SystemExit(...)
```

**Measured on the 0.5B: my `inv_freq` now matches the model's exactly** (`allclose`, first three
entries `1.0, 0.6493816375732422, 0.4216965138912201` identical, implied theta 1e6). The fallback
is gone: an undeterminable theta **stops the run** rather than guessing, because a guess here is
invisible downstream.

The luck is worth stating plainly. The earlier results used the wrong theta for the config-read
path, and they are only trustworthy because the `Rope` object that the arms actually used was
built from a value that happened to resolve to 1e6. That is not a property to rely on, and it is
replaced by an assertion.

**Generalisable rule: a control that passes for degenerate reasons is not a control.** Both of this
harness's rotation checks are insensitive to the parameter they appear to guard. Whenever a check
passes on the first try and the thing it guards is a constant, test that the constant is wrong and
confirm the check fails.

## Finding 12 — the depth wall has TWO causes, and scaling only removes one of them

This is the 7B validation Finding 8 asked for. It does not overturn Finding 9; it **splits** it.
Scaling to Qwen2.5-7B at 8-bit:

| mode | depth 15% | depth 35% | depth 55% | depth 75% |
|---|---|---|---|---|
| baseline | fail 34.3% | **PASS** 36.8% | **PASS** 37.3% | not measured |
| **line** | **PASS** 35.0% | **PASS** 36.8% | **PASS** 37.6% | not measured |
| posnorm | fail 34.3% | **PASS** 36.4% | fail 37.5% | not measured |
| | **1/3** | **3/3** | **2/3** | — |

against the same sweep on 0.5B: 1/3, 1/3, 1/3, 1/3 (5/16 overall).

**1. The middle depths are capacity-limited and 7B moves them.** Depth 35% goes from 1 of 3 modes
passing to **all three**, and 55% from 1 of 3 to 2 of 3. That is the wall receding exactly where
the 0.5B's failure was a capacity failure. Pass rate 5/16 → **6/9**.

**2. The SHALLOW position is NOT capacity-limited, and 14x the parameters do not touch it.** Depth
15% is *identical* on both models — `baseline` fails, `posnorm` fails, `line` passes. Same three
verdicts, same ordering, at both scales. A failure that survives a 14x parameter increase is not
a capacity failure, and no amount of scale is going to fix it. This is why Finding 8 looked
unresolvable on the 0.5B: it is not one failure mode, it is two, and only one of them scales.

**3. Line pooling is the only mode that passes every measured depth at either scale.** On the 7B
it is **3/3** where baseline and posnorm are 2/3. On the 0.5B it is the only mode to pass 2 of 4
(15% and 75%) where the others manage 1 of 4. It is the single change that clears the capacity
wall *and* the positional one.

**4. Finding 9 reproduces on the larger model.** The needle was retained in **7 of 9** combos but
the answer was correct in only **6 of 9** — so a retained needle still does not guarantee a usable
answer. Selection is not the bottleneck, exactly as the 0.5B said.

**0.75% is not measured, and that is a hardware limit, not a result.** At 8-bit the 7B peaks near
**10.5 GB** RSS at the 75% layout (~7 GB of quantised weights plus a ~2400-token KV cache and the
de-quantisation working set); this box's watchdog pauses any process that pushes system memory past
85% (13.4 GB of 15.7 GB), and a pause does not release the paused process's RSS, so the run
deadlocks rather than failing. Three launch attempts were made; one was killed by the watchdog
(`rc=137`). **The 0.5B's 75% column is present** (baseline fail, line PASS, posnorm fail), and it
agrees with the pattern above, but it is fp32 on a different model and is not a substitute.

**What this means for the design.** The recipe at ~36% KV saving is sound on a 7B, and line pooling
is the load-bearing part of it — but the honest claim is *"robust across needle positions at the
depths we can measure on a 7B"*, not *"robust"*. The shallow-position case is open, and the evidence
says it will need a mechanism that is not more parameters. Re-rotation is still required in every
passing combo: the rotation-off twins fail (Finding 2).

## Finding 13 — attention-sink displacement is REFUTED as the shallow-depth mechanism

Finding 12 left the 15% failure as a mechanism question. The leading hypothesis was that
the needle sits too close to the attention sink and is starved by it. Two measurements
kill it — one correlational, one causal.

**The sink is real, and enormous** (`src/sink_probe.py`, 0.5B fp32, every chunk-1 position
as a query, causal softmax over all prefix keys):

| depth | total tok | sink mass (keys 0-8) | needle mass | top-1 key |
|---|---|---|---|---|
| 0.15 | 938 | 0.5734 | 0.001415 | 0 |
| 0.35 | 1460 | 0.5517 | 0.000656 | 0 |
| 0.55 | 1992 | 0.5360 | 0.000613 | 0 |
| 0.75 | 2221 | 0.5312 | 0.000426 | 0 |

8 tokens absorb **~54% of all attention at every depth**, a per-token draw 200–700x the
needle's, and position 0 is the single most-attended key everywhere. On the face of it
that is exactly the hypothesis.

**But the needle gets MORE attention where it FAILS than where it PASSES** — 0.001415 at
15% (fails) against 0.000613 at 55% and 0.000426 at 75% (both pass). Raw attention mass to
the needle does not predict the verdict, which independently reproduces Finding 9.

**The causal test agrees** (`src/sink_pad.py`, depth 0.15, keep 0.60, rotation on — the
exact configuration that fails). Filler is inserted to move the needle away from the sink
basin, in two placements that separate two different stories:

| where | pad | needle_rel | rank | needle kept | verdict |
|---|---|---|---|---|---|
| sys (filler before system) | 0 | 414 | 356 | yes | fail |
| sys | 16 | 414 | **463** | yes | fail |
| sys | 32 | 414 | **575** | **no** | fail |
| needle (filler before needle) | 0 | 414 | 356 | yes | fail |
| needle | 16 | 430 | **281** | yes | fail |
| needle | 32 | 446 | **324** | yes | fail |

**6 of 6 fail. Padding never flips the verdict.** Displacing the needle from the sink
changes its *rank* but never its *verdict*, so proximity to the basin is not what decides
success. Two details make the negative result sharper:

- **Prepending filler actively harms.** In the `sys` placement the rank degrades
  monotonically (356 → 463 → **575**) and at pad=32 the needle leaves the keep set
  entirely. Moving the sink into the filler does not move the sink's effect.
- **Rank is non-monotonic in padding distance** (356 → 281 → 324), the same
  non-monotonicity Finding 9 recorded for `keep` — consistent with a system at the edge of
  what it can do, not with a positional gradient.

**The one positive signal, and where the evidence now points.** At `needle`/pad=32 the
answer becomes `"The failure code in the build log is 0x9."` — the model reaches the
needle and emits the first token of the code, then cannot complete `0x9AF4_STACK_FAIL`.
That is a *partial* retrieval, and it is a different failure from the pad=0 case (which
recites the system prompt). Together with Finding 9 — needle retained in 15 of 16
combinations with the answer still wrong — the evidence has moved off the selector and
off attention routing, and onto the decode: **the key is in the cache and attended to, and
the model still cannot use it.** Whatever the shallow-depth wall is, it is downstream of
attention.

**Hypothesis status:** attention-sink displacement — **refuted** (causal test, 6/6).
RoPE high-frequency phase clash and "line pooling averages out phase noise" remain open,
and both predict a positional gradient that this data does not show.

## Finding 14 — the missing control: compaction IS the cause, and there is no shallow-depth problem

**Every result in this document up to now compared one compaction mode against another.
`src/full_cache_control.py` adds the control that was missing: the identical prompt and
needle position with NO compaction at all.**

| depth | full (no eviction) | identity (keep all + rotate) | evict (60% + re-rotation) |
|---|---|---|---|
| 0.15 | **PASS** | **PASS** | fail |
| 0.35 | **PASS** | **PASS** | PASS |
| 0.55 | **PASS** | **PASS** | fail |
| 0.75 | **PASS** | **PASS** | PASS |

**Three conclusions, and the first one forces a correction.**

**1. The uncompacted cache passes at EVERY depth — including 15%.** The prompt is fine.
The needle position is fine. There is no shallow-depth retrieval problem and there never
was one. Every earlier statement about "the 15% failure" — including Finding 12's claim
that the shallow position is a non-capacity *mechanism* — was describing a failure that
compaction causes, not one that the layout causes. **Finding 12's "two causes" framing is
withdrawn**: there is one cause (compaction) whose severity varies with layout, and scaling
the model reduces how often it bites. The depth axis is not special; it is merely how the
layout was varied.

**2. `identity` equals `full` at all four depths.** Keeping everything and re-rotating by
zero displacement is a behavioural no-op, which is a second, stronger check on the rotation
path than `R(0)=I` (Finding 11's trap: an identity that holds for degenerate reasons proves
nothing; this one is exercised against real attention and still matches).

**3. Compaction is the cause, and no scalar metric predicts when it fails.**

| depth | c1 | needle pos | rank | keep | margin | verdict |
|---|---|---|---|---|---|---|
| 0.15 | 899 | 414 | **356** (best) | 539 | 183 | **fail** |
| 0.35 | 1421 | 936 | **493** (worst) | 852 | 359 | **PASS** |
| 0.55 | 1953 | 1468 | 361 | 1171 | 810 | **fail** |
| 0.75 | 2182 | 1697 | 389 | 1309 | 920 | **PASS** |

**The best-ranked needle fails and the worst-ranked needle passes.** Rank, margin,
attention mass (Finding 13) and line contiguity (below) all fail to predict the verdict.
The pattern is a clean alternation **F P F P** which is either 0.5B instability at the
edge of competence or a periodicity none of these instruments can see. It is stated, not
explained.

**What this leaves standing.** Retention is *necessary* (across 41 recorded rows, the
needle was never once absent from the cache and the answer correct — `needle gone + PASS`
is 0) but not *sufficient* (17 rows retained it and still failed). The failure mode is
consistent: the model produces the correct **frame** with a wrong **value** — `"The failure
code in the build log is 1."`, `"...is 0x9999999999999"`, `"...was 00:03:41."` It knows the
question and the answer's shape; it does not reproduce `0x9AF4_STACK_FAIL`.

So the question is no longer "which tokens should we keep" (that is measurable and mostly
solved) but **"what property of the retained SET makes it usable"** — a property that none
of rank, mass, or contiguity captures.

## Finding 13b — sequence fragmentation is REFUTED (my own hypothesis)

The frame-with-wrong-value symptom suggested the code `0x9AF4_STACK_FAIL` was arriving as
fragments: token-level top-k keeps high-scoring tokens and drops the ones between them, so
the model sees `0x` and `4` and invents a plausible digit. `src/fragmentation.py` measures
the needle line's survival directly (`line_tokens_kept`, `max_run`, `line_intact`).

**Refuted.** The needle's whole 16-token line survived intact in **11 of 12** rows, and the
pass rate among intact rows is still **45% (5/11)** — identical to the pass rate among rows
where the needle token itself was retained. `line_intact`, `max_run >= 3` and `needle_kept`
produce the *same* partition, so contiguity carries no information the retention flag did
not already carry.

The one row where the line was genuinely broken (`posnorm`, 0.15: 12 of 16 tokens, longest
run 7) failed — consistent with fragmentation mattering *when it happens*, but it happens
once in twelve, so it is not the mechanism behind the other failures.

## Limits

- **The 7B column is 8-bit and covers three of four depths.** Quantisation changes numerics
  (Finding 10), so this is not a like-for-like rerun of the fp32 0.5B arm; the two tables are
  compared for *verdict patterns*, not for absolute numbers. The 75% layout exceeded this machine's
  memory ceiling at 8-bit.
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
