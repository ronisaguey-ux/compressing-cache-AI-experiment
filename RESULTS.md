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

## Finding 15 — position-independent caching is WORSE than dense re-rotation

The last untested arm. `apply_delta` rotates survivors to their new DENSE positions, which
keeps the geometry internally consistent but shortens every relative distance (a needle 485
tokens away becomes 230 away at 60% compaction). The alternative in the literature —
SGLang RFC #30928, LazyAttention — is to leave key rotations at their ORIGINAL positions
and place the query at the original sequence length, so a retained token sits at exactly
the distance the model was trained on. `src/origpos.py` builds that arm with an explicit
`cache_position` (transformers otherwise derives the query position from the cache length,
which is the whole reason the distances move).

| depth | full | evict (dense re-rotation) | origpos (original positions) |
|---|---|---|---|
| 0.15 | PASS | fail | fail |
| 0.35 | **PASS** | **PASS** | **fail** |
| 0.55 | PASS | fail | fail |
| 0.75 | **PASS** | **PASS** | **fail** |

**origpos fixed 0 depths and broke 2.** Where dense re-rotation passed, origpos failed. It
is strictly worse, at identical retention and identical KV savings (34.0 / 36.2 / 37.2 /
37.5% both arms, same `keep` set, same rank).

**What this settles.** The dense re-rotation in this harness is not merely a reasonable
choice — it is the better one, and preserving the trained relative distances is worse than
remapping them onto the compacted sequence. That points away from the conditioning-loss
account: if the model needed the absolute distances it was trained on, origpos would have
helped. It did not. **Conditioning loss via RoPE geometry is refuted as the mechanism.**

Combined with Findings 13 and 13b, four mechanism hypotheses are now dead by measurement:
attention-sink displacement, sequence fragmentation, context density, and RoPE geometry.
The failure is real, compaction-caused, retention-dependent, and invisible to every scalar
this project has measured.

## Finding 16 — conditioning loss in the needle's KV is REFUTED

Two independent research agents converged on the same leading explanation: the needle's
**value** is conditioned on its antecedents, so evicting them leaves a cached V that was
written for a context that no longer exists (Kamera's conditioning loss). Agent 1 proposed
a cosine probe; Agent 2 proposed recomputing the KV under the compacted prefix.

**The cosine probe is vacuous in this harness and was not run.** `Harness.resize` does
`index_select` on keys **and values**, and `apply_delta` rotates **only keys** — so V is
byte-identical between the full and compacted caches and the cosine would be exactly 1.0
by construction. A test whose result is fixed by the implementation cannot be evidence.

Agent 2's version is the real test, and it was run (`src/recompute_needle.py`): build the
compacted token sequence, forward it fresh, and **splice in the needle's K and V as computed
in that compacted context**, then decode.

| depth | evict + re-rotation | **+ recomputed needle KV** | cos(orig, recomputed) K | V |
|---|---|---|---|---|
| 0.15 | fail | **fail** | 0.866 | **0.999** |
| 0.35 | PASS | **PASS** | 0.841 | **0.999** |

**Refuted, and the measurement is unusually clean.** Recomputing the needle's own KV in the
correct context changes the verdict nowhere — 0.15 still fails, 0.35 still passes. Two
further facts fall out:

- **V barely moves at all.** Cosine 0.9986 / 0.9991 between the cached and recomputed values,
  at every layer and in the mid band (8–18). The conditioning shift in the **value** is
  ~0.1%. The hypothesis assumed V carries the context dependence; it does not.
- **K moves far more** (cos 0.866 / 0.841, mid-band 0.854 / 0.813). The key is the
  context-sensitive tensor here, and it is the one dense re-rotation already repairs.

**Where this leaves the mechanism.** The needle's representation is not the problem — not
cached, not recomputed, not its keys, not its values. Combined with the refutations already
recorded (attention mass, contiguity, density, RoPE geometry, conditioning) and with the
two positive results (`full` cache passes everywhere; `evict` fails only sometimes), the
failure is in the **readout of the compacted context as a whole**, not in the retained
needle. Every attempt to blame a property of the needle has now failed.

## Research-agent audit — what was useful, what was wrong

Two external agents produced 82 KB of analysis against `docs/DEEP-RESEARCH-BRIEF.md`. The
prior-art survey is the most valuable part; several specific claims do not survive checking.

**Genuinely new and usable:**
- **The prior-art landscape is far more crowded than this document assumed.** LazyAttention,
  **MEPIC** (arXiv 2512.16822), **MiniPIC** (arXiv 2606.13126), **Irminsul** (arXiv
  2605.05696), **SemPIC** (arXiv 2607.28069), **COMB** (arXiv 2602.01519) and **Leyline**
  all do position-independent KV caching with RoPE correction. This project was framed as
  nearly novel; it is not. **Its contribution is CPU-executability, not mechanism novelty** —
  every published implementation of exact RoPE re-rotation on cached keys is GPU-only.
- **Layer-stratified saliency.** Summing attention uniformly over all layers mixes
  high-entropy early-layer syntactic attention into a sparse mid-layer retrieval signal.
  This is consistent with Finding 1's own per-layer rank table (needle 303rd at layer 0,
  46th at layer 16) and it is cheap to test.
- One agent independently predicted **the generation-vs-retrieval split** that
  `src/decode_divergence.py` had already measured: the answer's first token is reachable and
  the decode cannot complete it.

**Checked and corrected — do not carry these forward:**
- **Agent 1's "+2.019 vs +1.776 nats" catch is CORRECT.** `RESULTS.md` records +2.019 as the
  max of the earlier 4-point sweep and +1.776 as the keep-50% comparison; the brief conflated
  the two runs. The document was right and the brief was wrong.
- **Agent 1's Rank-1 experiment (value-cosine probe) is vacuous** — see above.
- **Agent 2 cites `src/heavy_lock.py:112-145` for "cumulative phase quantization drift in key
  re-rotation".** That file is 137 lines and is the **memory lock**; it contains no rotation
  code, no RoPE, no keys. The citation is fabricated.
- **Agent 2's subword-fragmentation hypothesis (H1)** was already refuted by
  `src/fragmentation.py`: the needle's full 16-token line survives intact in 11 of 12 rows
  and the pass rate among intact rows is still 45%.
- **Agent 2's 8-bit dequantisation hypothesis (H8) cannot explain this failure.** The 0.5B is
  **fp32** and fails identically at 0.15%. A quantisation artefact cannot be the mechanism in
  a run that never quantises.
- **Agent 2's H9 (posnorm penalises shallow positions) is CORRECT and was confirmed both
  ways**: `sel_position_normalized` divides position `i` by `chunk1_len - i`, so early tokens
  carry the largest denominator, and the data agrees (0.15: baseline rank 564, posnorm 633 —
  worse; 0.35 and 0.55: posnorm better). posnorm's 15% failure is a normalisation artefact,
  not a selector verdict.

**Net:** the brief was worth sending — it produced a prior-art map worth several days of
searching and one hypothesis sharp enough to kill decisively. It also produced one fabricated
citation and one experiment that would have returned a guaranteed 1.0. Treat it as a lead
generator, which is what it was asked to be.

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

## Finding 17 — the needle's saliency signal is in the MID layers, and band selection trades one failure for another

Both agents proposed restricting saliency scoring to the retrieval band. Tested with four
bands at keep 0.60 (`src/layer_band.py`, 0.5B):

| depth | all (0-24) | **mid (8-18)** | late (15-24) | early (0-8) |
|---|---|---|---|---|
| 0.15 | rank 356 fail | **rank 268 PASS** | rank 445 fail | rank 414 fail |
| 0.35 | rank 493 PASS | rank 366 fail | rank 472 fail | rank 712 fail |
| 0.55 | rank 361 fail | rank 296 fail | rank 413 fail | rank 322 fail |
| 0.75 | rank 389 PASS | rank 235 fail | rank 373 fail | rank 733 fail |
| **pass** | **2/4** | **1/4** | 0/4 | 0/4 |

**Two real results and one that blocks the win.**

**1. The needle's saliency signal lives in the middle of the network, and the ends are
useless.** `early` never ranks the needle better than 322 and `late` never better than 373;
`mid` reaches **235**. This independently confirms Finding 1's per-layer table (needle 303rd
at layer 0, 46th at layer 16) on a different measurement, and it is the sharpest support in
this document for the retrieval-head account both agents cited.

**2. The mid band fixes depth 0.15 — the first selector in this project to do it.** Rank
356 → 268 and a fail → **PASS**, with retention unchanged. Every other arm so far has failed
0.15.

**3. But it is not a win: it breaks 0.35 and 0.75, and the rank/verdict paradox survives.**
At 0.35 the mid band ranks the needle *better* (366 vs 493) and still **fails** where `all`
passes. A better rank producing a worse outcome is the same contradiction as Finding 14, and
it now holds *within a single selection family*. No band improves the pass count; `all`
remains the best at 2/4.

**Hypothesis status after all of this.** The band result says the selector is not
irrelevant — it changes which depths pass. It says nothing consistent about *why*. Combined
with Finding 16 (the needle's KV is not the problem, cached or recomputed) and the four
refuted scalars, the failure behaves like a fragile readout that different retention sets
push below or above a threshold with no monotone relationship to any measured property.

## Finding 18 — ★ THE DEFECT IS THE CACHE SURGERY ITSELF: recompute, do not rotate

This is the answer to the question Findings 8–17 spent a dozen experiments failing to
reach, and it was found by adding the one control that had never been run: **give the model
the same retained tokens as an honest sequence, computed natively.**

`src/native_vs_cache.py` — same prompt, same selection, same keep set, three ways:

| depth | full (native) | **compact NATIVE** | compact CACHE (shipped) |
|---|---|---|---|
| 0.15 | PASS | **PASS** | fail |
| 0.55 | PASS | **PASS** | fail |
| 0.75 | PASS | **PASS** | PASS |

**The kept tokens are entirely sufficient.** At 0.15 and 0.55, handing the model exactly the
tokens we retain — at their dense positions, as a normal forward pass — retrieves
`0x9AF4_STACK_FAIL` perfectly, while the identical token set routed through our cache-surgery
pipeline fails. Selection is fine. Retention is fine. **The compaction machinery is the
defect.**

**Why, measured directly** (`src/cache_vs_native.py`). Comparing the shipped cache's tensors
against the native compact cache's, per position:

| depth | all positions | needle position |
|---|---|---|
| | cos K / cos V | cos K / cos V |
| 0.15 | 0.863 / 0.925 | **0.866** / 0.999 |
| 0.55 | 0.822 / 0.924 | **0.750** / 0.999 |

**Dense re-rotation does not reproduce the key the model would have built.** Rotating a
surviving key from its old position to its new one yields a vector at cos ≈ 0.75–0.87 from
the key that a native forward at that position produces. The worst layers are early-to-mid
(6, 7, 13, 18 — cos as low as 0.51).

**This corrects Finding 16.** That test recomputed **only the needle's** K and V and found
no change, and I concluded conditioning was refuted. That conclusion was too strong: the
needle is **one token out of 578–1210**. Recomputing one token leaves every other key in the
cache rotated-but-not-recomputed, and native-vs-cache shows the whole cache has to be right.
The needle's own V really is nearly perfect (0.999) — but that was never the load-bearing
quantity.

**What this means for the method.**

- **`select → evict → re-rotate` does not work.** Rotating cached keys preserves the
  *content* each key was computed with — content conditioned on a prefix that compaction has
  changed. The rotation fixes the position and nothing else, and the tensors differ from the
  valid ones by cos 0.75–0.87.
- **`select → re-prefill the compacted sequence` does work**, at every depth measured, at the
  same KV savings (34.0 / 36.6 / 37.5%), and it is what `compact_native` is.
- **The cost is a re-prefill**, i.e. this is the same trade Finding 4 identified for text
  squashing: the compaction saves KV memory, and the price is recomputing the retained
  prefix. On CPU that is the dominant cost, so the honest framing of the method is
  *"selection buys you a shorter sequence to recompute"*, not *"rotation lets you avoid
  recomputing"*.

**What survives from the earlier work.** The selector matters — line pooling and the mid
band are the two that pass shallow depths, and `compact_native` uses the selector's output,
so a better selector is still a real gain. The rotation analysis (Finding 2) measured a real
causal effect: without re-rotation the answer collapses to system-prompt recitation. But it
was the right fix for the wrong frame — it is *less wrong* than no rotation, not *correct*.
Against a native recompute, cos 0.86 is not close enough.

**The honest headline for the repository.** A 60% KV compaction that preserves a mid-context
needle requires recomputing the retained sequence; key-only rotation is insufficient, and
the nine scalar explanations in Findings 8–17 were all measuring the symptom of that one
fact.

## Finding 19 — the 7B separates THREE distinct failure modes, and only one is the method

`src/native_7b.py`, Qwen2.5-7B at 8-bit, the same three arms Finding 18 defined:

| depth | c1 | keep | needle rank | **retained?** | full | compact_NATIVE | compact_CACHE |
|---|---|---|---|---|---|---|---|
| 0.15 | 899 | 539 | **555** | **NO** | PASS | fail | fail |
| 0.35 | 1421 | 852 | 619 | yes | PASS | **PASS** | **PASS** |
| 0.55 | 1953 | 1171 | 614 | yes | PASS | **PASS** | fail |

Read against the 0.5B table from Finding 18, the failures separate cleanly for the first
time in this project:

**1. Selection failure — the needle was simply not kept.** At 7B/0.15 the needle ranks
**555 against a keep budget of 539**: it misses retention by 16 positions. There is no
compaction bug here and nothing to fix in the pipeline — the selector picked 539 tokens and
the needle was 556th. Against the 0.5B, which retained it at the same depth (rank 356/539),
this is a 7B-specific ranking difference, and it is the cleanest possible illustration that
rank and retention are the selector's only job and that job sometimes just misses.

**2. Rotation defect — retained, correct natively, broken by the cache path.** At 7B/0.55
the needle is well inside the budget (614 of 1171), the native recompute returns
`0x9AF4_STACK_FAIL` verbatim, and the shipped rotation path fails. **Finding 18 is confirmed
on the larger model.** This is the same defect measured on the 0.5B at 0.15 and 0.55.

**3. No failure.** At 7B/0.35 every arm passes.

**The near-miss at 7B/0.15 is the most interesting single output in the report.** With the
needle *evicted*, the native run still produced:

```
The failure code in the build log is x9AF_STACK_FAIL.
```

— a **partial reconstruction** of a fact whose token was not in the context: correct
suffix, correct `_STACK_FAIL`, a plausible `x9AF` for `0x9AF4`. The surviving line context
around the needle carries enough signal for the model to rebuild most of it. That is a
direct argument for **line pooling**: keeping the line rather than the token is what makes
this reconstruction possible at all, and it explains why line pooling was the one mode that
survived every depth in Finding 12.

### The three failure modes, and what fixes each

| failure | signature | fix |
|---|---|---|
| **selection** | rank > keep; native also fails | better selector, or a slightly larger budget |
| **rotation** | retained; native PASSES, cache fails | **recompute the compacted sequence** |
| — | native PASSES, cache PASSES | none needed |

**The corrected method is `select → re-prefill the compacted sequence`**, and the honest
saving is 34.0 / 36.2 / 37.2% KV at the three measured depths, at the cost of recomputing the
retained prefix. Key-only rotation does not substitute for that recompute — measured at cos
0.75–0.87 from the correct tensors.

**What this reframes for the project.** Findings 8 through 17 spent a dozen experiments
attributing the failure to attention, geometry, conditioning, contiguity or density. It was
never any of those. It was a **selector that occasionally misses** plus **cache surgery that
silently corrupts the tensors it keeps**. Both are ordinary engineering defects, and both are
now named, measured, and testable.

## Finding 20 — ★★ WITH THE PIPELINE FIXED, THE SELECTOR BARELY MATTERS

Every selector comparison in this document was run inside the broken rotation pipeline. Re-run
against the corrected one (`select → re-prefill the compacted sequence`), on identical rows:

| depth | baseline | line | mid-band | random | **old cache path** |
|---|---|---|---|---|---|
| 0.15 | PASS | PASS | PASS | fail | fail |
| 0.35 | PASS | PASS | PASS | fail | PASS |
| 0.55 | PASS | PASS | PASS | fail | fail |
| 0.75 | PASS | PASS | PASS | PASS | PASS |
| **pass** | **4/4** | **4/4** | **4/4** | **1/4** | **2/4** |

**Plain all-layer self-attention — the ordinary `baseline` selector, with no line pooling and
no layer band — passes at every depth once the cache is built correctly.** Random retains
roughly the same number of tokens and passes 1 of 4, so the control is doing its job and the
result is not an artefact of keeping enough tokens to stumble onto the needle.

**Two conclusions, and both retract earlier findings.**

**1. Line pooling's advantage was an artefact of the rotation defect.** Finding 12 made it the
headline — *"the only mode that passes every measured depth at either scale"* — and Finding 19
read the 7B near-miss as direct evidence for it. Both were measured against a pipeline that
silently corrupts ~60% of the tokens it keeps. Line pooling survived every depth because
keeping a whole line is **robust to that corruption**: when the tensor is wrong, having the
neighbouring tokens still present is what lets the model reconstruct. Fix the tensor and the
advantage vanishes, because it was never about selection.

**2. The mid-layer band's advantage was also an artefact.** Finding 17 found it fixed depth
0.15 (rank 356 → 268, fail → PASS). Under the corrected pipeline plain baseline already passes
0.15, so the band has no gap left to close. It is not harmful — 4/4 — it is simply unnecessary.

**This inverts the project's own narrative, and it is the most useful result in it.** Findings
5, 6, 7, 12 and 17 are all selector findings: which tokens to keep, how to rank them, which
layers to score, whether to pool by line. **All of that was compensating for a broken eval.**
With the pipeline correct:

- **the selector is not the lever**, and a plain attention sum is sufficient;
- **the pipeline is the lever** — 4/4 versus 2/4 on identical rows, identical budgets,
  identical selectors;
- the effort spent optimising selection was effort spent making a corrupted cache slightly
  more survivable.

**What the corrected method is, plainly.** Score chunk-1 tokens by the attention they receive,
keep the top 60%, and **run a fresh forward pass over the retained sequence**. KV memory drops
34–37%, retrieval holds at every depth measured, on both a 0.5B and a 7B, and the only
component that has to be right is the re-prefill. There is no rotation step, no per-layer band,
no line pooling, and no conditioning patch.

**The honest caveat, stated once.** The re-prefill is the cost, and on CPU it is the dominant
one. "Compaction" here means *shorter sequence to recompute*, not *avoid recomputation*. That
is a different and weaker claim than the one the repository opened with, and it is the one the
measurements support.

## Finding 21 — prefix-preserving re-prefill: correct, verified exact, and worth 7.7%

Bob's optimisation. Causal attention is strictly left-to-right, so any token whose entire
prefix survived untouched has the same hidden state and the same RoPE phase as in the
uncompressed run — the first `k` entries of the original cache are pristine and need not be
recomputed. Implemented in `src/prefix_cache.py` to the spec: find the first `k` where
`selected[k] != k`, slice the cache to `[:k]`, forward only the remaining kept tokens against
it with `cache_position = arange(k, k + len(remaining))`.

**The prefix is bit-identical — asserted every run.** The spec asked for cosine similarity
exactly `1.0`. That is not assertable: cosine reports **0.9999998** even on two identical
tensors, because the norm and dot product round. The real test is `torch.equal`, and it passes
at every layer for both keys and values. Cosine was kept as a reported statistic, not as the
assertion, because asserting on it would be asserting on my own arithmetic.

**Retrieval holds: 4/4 PASS**, `0x9AF4_STACK_FAIL` at every depth — the same result as a cold
re-prefill, so the reuse changes nothing about correctness.

| depth | N | keep | prefix | **prefix / keep** | prefix / N | re-prefill saved | wall-clock |
|---|---|---|---|---|---|---|---|
| 0.15 | 938 | 539 | 69 | **12.8%** | 7.4% | 11.6% | — |
| 0.35 | 1460 | 852 | 74 | 8.7% | 5.1% | 8.1% | — |
| 0.55 | 1992 | 1171 | 106 | 9.1% | 5.3% | 8.6% | **+7.7%** (median of 3) |
| 0.75 | 2221 | 1309 | 106 | 8.1% | 4.8% | 7.8% | — |

**Three different numbers, and only the middle one is the honest claim.** The prefix is
**7.8–11.6% of the re-prefill** (a real saving, measured at 7.7% wall-clock), **4.8–7.4% of the
full sequence** (the small figure), and **0.6–1.3% of the FLOPs** (because the skipped tokens
are at the *front*, where attention is cheapest — a quadratic's first 100 terms cost almost
nothing). The wall-clock sits near the token figure because CPU CPU forward cost is dominated
by per-token linear work (projections, MLP), not by the attention quadratic at these lengths.

**Why the win is small, and where a bigger one lives.** Attention-based selection starts
evicting around **chunk-1 index 30–67** (measured), so the untouched prefix is short — only 69
to 106 tokens including the 39-token system prompt. The prefix length is set by the *selection
pattern*, not by the caching mechanics: scattered top-k evictions produce a short prefix by
construction. **A contiguous-middle eviction scheme would give a very long pristine prefix**,
and the same caching code would then save a large fraction of the re-prefill — at the cost of
giving up the freedom to evict wherever attention is lowest.

**Status of the optimisation.** It is implemented, exactness-verified, correctness-preserving,
and worth ~8% of the recompute. It does not change the method's economics: the re-prefill is
still the cost, and this shortens it slightly rather than removing it.

## Finding 22 — the selection pattern IS the lever: 3x the reuse for free

Finding 21 showed the pristine prefix is short (8-12% of the re-prefill) and named the cause:
scattered top-k evicts early, so the untouched prefix is short by construction. That is a
claim about the *selection*, so it is testable directly. `src/prefix_tradeoff.py` force-keeps
the first `f` of chunk 1, allocates the rest of the budget by attention as before, and
measures both the reusable prefix and the retrieval verdict.

| force | reuse of re-prefill | avg re-prefill | retrieval |
|---|---|---|---|
| **0.00** (scattered top-k) | 9.2% | 918 tok | **4/4** |
| **0.15** | **28.4%** | 724 tok | **4/4** |
| 0.30 | 52.1% | 484 tok | 3/4 |
| 0.45 | 76.4% | 240 tok | 1/4 |

**★ `f = 0.15` is free — a 3.1x increase in reusable prefix at zero retrieval cost.** The
scattered selector was already keeping those early tokens most of the time; forcing them
costs nothing and makes the prefix *contiguous*, which is what the caching code needs. 4/4 at
every depth, and the average re-prefill drops 918 → 724 tokens (21%).

**★ Past 0.15 the budget, not the prefix, becomes the binding constraint.** Keep is fixed at
60% of chunk 1, so every forced token is a token that cannot be spent on attention-selected
ones — and attention had a reason for picking them.

| depth | force | needle kept | verdict | what failed |
|---|---|---|---|---|
| 0.15 | 0.30 / 0.45 | **False** | fail | the needle itself was evicted |
| 0.55 | 0.45 | **True** | fail | needle present, but something it depends on was cut |

The 0.55/0.45 row is the interesting one: the needle survived and the answer still broke, which
means the needle is not the only thing the model needs — the same "partial reconstruction
from surrounding context" that Finding 19 saw at 7B/0.15. Quality here is a property of the
whole retained region, not of one token.

**⇒ The practical recommendation is `f = 0.15`.** It is strictly better than the current
scattered selector: same verdicts, 3x the prefix reuse, 21% less re-prefill work. It is also
the safe setting, because the failure mode at 0.30+ is evicting mid-context evidence, and
that failure is silent — the answer reads as a confident wrong code.

**On chunk 0 being retained.** At any `keep >= c0/(1-f)` the 39-token system prefix would be
pristine regardless. Forcing `f` of chunk 1 makes chunk 0 + `f` contiguous, and the reuse
grows roughly linearly in `f` as the table shows.

### Finding 22a — the failing rows are predicted by geometry, and the confound is refuted

The obvious way Finding 22 could be wrong: if the needle sat inside the forced prefix, `f=0.15`
would pass because the *force* retained the needle, not because attention found it — a pass for
the wrong reason. Measured (tokenizer only, no model), the needle's position within chunk 1:

| depth | c1 | needle at | forced @0.15 | forced @0.30 | forced @0.45 | in forced region? |
|---|---|---|---|---|---|---|
| 0.15 | 899 | 414 (**46%**) | 134 | 269 | 404 | **no** at every f |
| 0.35 | 1421 | 936 (**66%**) | 213 | 426 | 639 | **no** |
| 0.55 | 1953 | 1468 (**75%**) | 292 | 585 | 878 | **no** |
| 0.75 | 2182 | 1697 (**78%**) | 327 | 654 | 981 | **no** |

**The needle is never inside the forced region, at any depth or force level.** `f=0.15` passing
4/4 is therefore genuine: attention selected the needle every time, from 46-78% into the chunk,
with the force confined to the front. The recommendation is not an artefact.

**And the geometry predicts the failures.** At depth 0.15, `keep` = 539 and `f=0.45` forces 404,
leaving **135 slots** for the 495 non-forced tokens — and the needle sits at 414, just **10
tokens past** the forced boundary. It is competing for one of 135 places and loses. The failure
is not a mysterious degradation; it is a budget that no longer reaches the evidence.

That converts `f=0.30`+ being unsafe from an inference into a measured, computable margin: the
forced prefix eats slots that the mid-context evidence needs, and the evidence sits far from
the front.

## Finding 23 — the inverted-context layout is the wrong way round

Bob's hypothesis: push volatile/disposable content to the front (0..K) and hoist
query-critical content to the suffix (K..N), then prune the front for free.
`src/inverted_context.py` tests it, and the answer is no — for a directional reason.

**Validity is a PREFIX property.** In causal attention `h_j = f(x_j, x_{j-1}, ..., x_0)`, so
token `j`'s keys and values depend on *every* token before it. `j` is reusable only if
`0..j-1` are untouched and unmoved. Two consequences, and both run against the hypothesis:

- evicting from the **front** invalidates everything after the first dropped index, so a
  suffix hoisted to the back is the **worst** place to put critical content;
- evicting from the **end** leaves every earlier token pristine, giving the **longest**
  reusable prefix of any policy.

### Pristine prefix by eviction policy (depth 0.55, N=1992, keep=1171)

| policy | pristine prefix | reuse | retrieval |
|---|---|---|---|
| no eviction (ceiling) | 1992 | 100% | PASS |
| **suffix eviction (drop the tail)** | **1171** | **100%** | **fail** |
| **front eviction (drop the head)** | **0** | **0%** | **PASS** |
| middle scatter, `f=0.15` | 332 | 28.4% | PASS |
| middle, attention sinks evicted | 0 | 0% | PASS |

**★ The two middle rows are the finding: cache efficiency and retrieval are in direct
conflict, and the needle's location decides which one you get.** Evicting the tail gives a
*perfect* reusable prefix and fails; evicting the front reuses *nothing* and passes. A policy
optimised on reusable-prefix length alone would pick exactly the wrong one.

Also worth noting: evicting the four attention sinks cost nothing here (PASS, 0% prefix),
consistent with Finding 13's refutation of sink-phase distortion.

### The suffix drift equation

Cosine of the suffix's K/V under a full prefix versus a front-truncated one, positions either
shifted naturally or held at their original values:

| front tokens dropped | cos K (shifted) | cos K (positions preserved) | cos V |
|---|---|---|---|
| 1 | 0.990808 | 0.999699 | 0.998959 |
| 4 | 0.948038 | 0.999657 | 0.998728 |
| 16 | 0.913514 | 0.999680 | 0.998866 |
| 64 | 0.893211 | 0.999528 | 0.998089 |
| 256 | 0.864652 | 0.999219 | 0.996630 |

**No entry is 1.0 at any drop size, so the suffix is never reusable once anything before it
changes.** Preserving the original position ids removes the *positional* error (0.8647 →
0.9992) but not the *content* error: the suffix attended to the dropped tokens and that is
baked into the weights. **Values hold up throughout; keys are what break** — the same
signature as Finding 18, and the reason a value-cosine probe cannot see this class of damage.

**⇒ The hypothesis is inverted.** Hoisting critical content to the suffix places it at the end
of a causally-masked sequence, which is the first thing a tail-pruning policy discards. The
serviceable version is the mirror image: **critical content early, disposable content late.**

## Finding 24 — with the policy fixed, the layout is the knob

The decisive test, and Bob's research agent proposed the same one independently. Hold the
eviction policy fixed, move the needle, watch the verdict invert.

| needle at | needle in retained tail? | suffix eviction | middle scatter `f=0.15` | full (ceiling) |
|---|---|---|---|---|
| 10% | **YES** | **PASS** (100% reuse) | PASS (33% reuse) | PASS |
| 25% | no | **fail** (100% reuse) | PASS (31%) | PASS |
| 45% | no | **fail** (100% reuse) | PASS (29%) | PASS |
| 65% | no | **fail** (100% reuse) | PASS (28%) | PASS |
| 85% | no | **fail** (100% reuse) | PASS (28%) | PASS |

**The tail verdict tracks `needle_in_tail` exactly: 1/5 PASS, and the needle was in the tail
in exactly 1/5 rows.** Moving one piece of content inverts the outcome while the policy, the
budget and the model are unchanged. **The layout decides; the policy merely follows.**

Two things this settles:

1. **`f=0.15` middle scatter is ROI-robust** — 5/5 PASS with the needle anywhere from 10% to
   85%, and 28–33% of the re-prefill reusable throughout. That is the configuration to ship:
   it does not depend on knowing where the critical content is.
2. **Tail pruning is free only if the critical content is not in the tail.** The 100% reuse
   column is a trap: it is highest exactly when the policy has thrown away the answer.

**Actionable, and it is a layout rule rather than a selection rule:** put critical content
early, then the tail is disposable and a tail-pruning policy gets a perfect reusable prefix at
no retrieval cost. No selection algorithm recovers from critical content placed last.

## Finding 25 — Laya's HTTP API silently drops a plain-string state

Found while wiring Laya as a line-granularity sifter, and it is worth recording on its own
because it is a live trap for any caller.

`POST /v1/systemone` accepts a `state` field. With a **plain string** the state never
reaches the model:

| state passed | `usage.input_tokens` | answer to "does state contain the letter A?" |
|---|---|---|
| `"AAAA"` | 36 | 0.4490 |
| `"BBBB"` | 36 | 0.4490 |
| a 1000-character string | **36** | 0.4490 |

Identical answers, and the token count is constant at 36 regardless of length — the state
is dropped before tokenization. The server returns **200** and answers from the
instructions alone.

With a **dict**, it works:

| state passed | `usage.input_tokens` | answer |
|---|---|---|
| `{"text": "AAAA"}` | 41 | **0.7308** |
| `{"text": "BBBB"}` | 42 | **0.1010** |

So the fix is `state={"text": ...}`. **A bare string produces confident-looking answers to
a question about text the model never saw** — the same "control that reports success and
does nothing" class as the harness's phantom pass and the layer-0 key comparison in this
same session. `src/laya_sift.py` now normalises the state on the way out and says why.

## Finding 26 — base Laya, zero-shot, is not a usable sifter

With the state bug fixed (3072 input tokens on a real log), Laya was asked about every line
of the depth-0.55 build log: *"does this line report an error, a failure code, or the cause
of a build failure?"*

**The needle (`FATAL: link aborted, failure code 0x9AF4_STACK_FAIL`) ranks 6th of 109.**
The five lines above it are compiler warnings and ordinary `gcc` invocations:

```
  4   0.9416  src/core/pool.h:88:12: warning: unused parameter 'fl'
 29   0.9416  src/core/pool.h:88:12: warning: unused parameter 'fl'
 54   0.9416  src/core/pool.h:88:12: warning: unused parameter 'fl'
 17   0.9277  gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o
 42   0.9277  gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o
 69   0.9179  FATAL: link aborted, failure code 0x9AF4_STACK_FAIL   <- the needle
```

Score spread 0.5449–0.9416, **median 0.8687** — it answers "yes, failure-related" to almost
everything, so there is no separation to threshold. It ranks an unused-parameter warning
above a fatal link error.

**Phrasing mattered enormously, which is itself informative.** Three question forms on a
6-line sample: "is this line *needed*" put the needle 4th of 6; "does this line *report an
error*" and "is this line *critical evidence*" both put it 1st — by margins of **0.003** and
0.004. Ranking first with a margin of 0.003 is not a signal; on 109 lines the same phrasing
put it 6th.

This matches the documented base-checkpoint figure: **0.362** accuracy on typed decisions
versus **0.77** fine-tuned. Zero-shot, the base model is a strong prior, not an oracle.

**And it is slow.** 109 lines took **524 s** across 14 calls (~37 s per call with a 20k-char
state), against **89 s** for the 0.5B attention sifter on the same log. Laya is both less
accurate and ~6× slower here.

**⇒ Of Bob's five-arm roster, arm 4 (base Laya zero-shot) fails.** Arm 5 (tuned) is the only
Laya version worth trying, and `src/laya_labels.py` builds its labels — with the caveat
recorded there: distilling from the consumer's attention teaches the teacher's blind spot
(the 7B misses the needle at 15%), so the ablation mode is the one that can actually beat
the attention sifter.

## Finding 27 — the cascade works: cross-model sifting holds, and the "golden path" is a myth

Bob's two-speed cascade, run as two sequential phases so only one model is ever resident
(0.5B + 7B together peak at 86% of RAM and the watchdog pauses at 85%, and a pause does not
free RSS — co-residency deadlocks rather than fails).

7B consumer, depth 0.55, needle retrieval:

| arm | prompt | verdict | seconds |
|---|---|---|---|
| A full prompt (ceiling) | 1992 tok | **PASS** | 871.5 |
| B **0.5B-sifted** (cross-model) | 1210 tok | **PASS** | 596.8 |
| C 7B self-sifted | 1210 tok | **PASS** | 609.2 |

**Cross-model salience transfer HOLDS.** A 0.5B model picks what a 7B needs. Cost: sifter
89 s + consumer sifted prefill 597 s = **686 s**, against **871 s** for the consumer on the
full prompt — 21% faster end to end, on a 39% shorter prompt. The sift is **deterministic
across two runs**, which is what makes the "100% reuse on turn 2" claim real rather than
nominal: a gap-free sifted prompt is its own full prefix, so later turns reuse all of it
with no cache surgery.

### ★★ THE OVERLAP IS THE FINDING

Both sifters kept 1210 tokens. **Their intersection is 230 — 19%.** Two almost entirely
different 60% subsets of the same log, and **both answer correctly.**

So there is no single critical token set. There are many sufficient ones: the answer
survives almost any selection that retains the needle and its local context. That reframes
the problem — the failure at 7B/15% is not "the selector chose badly" in general, it is
specifically that **the needle gets evicted**. It also explains why forcing `f=0.15`
(Finding 22) helped: it was not a better policy, it simply made the needle's survival more
likely.

**Practical consequence: sifting quality is a much weaker requirement than assumed.** You
do not need a good sifter; you need a sifter that does not drop the evidence. That is far
easier to guarantee, and it makes the cheap 0.5B front end genuinely sufficient.

## Finding 28 — block-diagonal attention IS isolated, and the error is numerical not structural

Bob's block-causal spike, depths 0.55, five blocks of ~398 tokens, evicting one whole block
and recomputing the survivors with their **original position ids**.

| eviction | error vs the full block-causal run |
|---|---|
| **block-causal** (same eviction) | **3.052e-05** |
| **full causal** (same eviction) | **1.690e+01** |

**553,852× smaller.** So the isolation claim holds: under block-diagonal attention, evicting
a block leaves the survivors as they were, whereas under full causal attention the same
eviction destroys them.

**★ It is NOT bit-identical, and it cannot be.** The first version asserted `torch.equal`
and failed — the same trap as asserting `cos == 1.0` in Finding 21. Block-diagonal attention
changes the *shape* of the attention operator, so the batched matmul reduces in a different
order and the result differs at float32 epsilon. The meaningful test is not bit-identity but
the **contrast**: noise (~1e-5) versus structural damage (~1e+1). Four orders of magnitude
separate them.

*(An earlier debugging pass compared layer-0 keys and saw 0.0 difference — that was a bad
probe, not a finding: layer-0 keys are `RoPE(W_k · embedding(t))` and never see the mask.
Comparing last-layer hidden states showed the mask working exactly as designed: positions in
block 0 unchanged, positions in block 1 changed by 24–53.)*

## Finding 29 — the query MUST be its own tier; block-diagonal masking makes it otherwise unanswerable

The block-causal ceiling failed — with *nothing evicted*, the answer was wrong. That looked
like a bug and is not: it is the architecture.

Under block-diagonal masking, **a query appended to the end of the sequence lives inside the
last block**, and can therefore attend only within that block. The evidence in earlier blocks
is unreachable by construction. Measured on the same log:

| query placement | answer |
|---|---|
| appended inside the last block | **`'What'`** — garbage |
| **its own forward over the block table** | **`'The failure code in the build log is 1.`** — coherent |

**⇒ Tier 3 is load-bearing, not a design nicety.** The query must be a separate forward that
attends over the assembled cache, which is exactly what the tiered architecture specifies.
A single-pass "prefix + question" prompt cannot work under document masking. This also
explains why the block-causal ceiling failed while the cascade — whose prompt is a single
ordinary causal sequence — passed all three arms.

## Finding 30 — the tiered architecture works, and its cost is measured

`src/tiered_cache.py`, 0.5B, depth 0.55. Tier 1 = a 35-token global anchor prefilmed once.
Tier 2 = six blocks of ~21 log lines each, forwarded **with the global as `past_key_values`**
and their own disjoint position ranges. Tier 3 = the query, its own forward over the
assembled table.

| claim | measurement |
|---|---|
| **isolation** — does a block depend on which other blocks are present? | **IDENTICAL**: block 3 built with 5 other blocks vs built alone, bit-for-bit |
| **eviction is lossless** | **BIT-IDENTICAL**: survivors re-checked inside the assembled cache after popping a block |
| ceiling (nothing evicted) | **PASS** |
| control — evict the needle's own block | **fails**, as it must |
| **evict a NON-needle block** | **fails** |

**★ The first three are the architecture working as designed.** A block's KV depends only on
(global, itself), so the table is genuinely modular and eviction recomputes nothing,
renumbers nothing, and repairs nothing. This is the property Findings 23–24 could not have:
there validity was a *prefix* property and a middle eviction invalidated everything after
it; here it is *local*.

**★ The last row is the honest cost, and it is the one that matters.** Evicting block 1 — a
block containing no needle — still degraded the answer. That is not a bug; it is what
block-diagonality buys and costs in the same stroke: because a block may only attend within
itself, **the needle's block sees the needle but not the material around it.** Context that
straddles a block boundary is lost to both blocks, and the evidence a question needs is
frequently spread across more than one block.

The cross-block probe makes it explicit: asked a question requiring two blocks, the answer
was `' 1: 1 failed\nmake: build failed…'` — a plausible-looking mangling rather than an
error, which is the dangerous shape.

**⇒ Where the tiered design pays and where it does not.** It is excellent when the global
backbone carries the question and the blocks carry independent, self-contained payloads —
scanned documents, per-file summaries, retrieved passages. It is the wrong shape for a
single continuous artifact whose meaning is distributed along its length, which is exactly
what a build log is. The cascade (Finding 27) handles that case better: it produces a single
ordinary causal sequence, so nothing is context-starved.

**Implementation note for anyone reusing this:** `DynamicCache()` is EMPTY — it has no layers
until a forward populates them, so indexing `c.layers[li]` raises `IndexError`. Construct
with the data: `DynamicCache(ddp_cache_data=[(key, value), ...])`. And pass a **fresh clone
of the global cache** to every block, because a `DynamicCache` is mutated in place by the
forward pass — share the object and block 2 inherits block 1's keys, silently destroying the
isolation. `build_table` asserts the global cache did not grow, so the trap cannot return
quietly.
