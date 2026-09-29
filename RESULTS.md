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

## Limits

- **0.5B model.** It degenerates under mild perturbation (`1.0.0.0.0.0` loops). Absolute quality is
  not meaningful; the comparisons are, since all arms share model, prompt and decode settings.
- **The full literal was never retrieved at 50%.** A higher keep fraction is untested for the
  literal; the retention curve shows 75% keeps the needle for every selector, which would make
  selection non-diagnostic, so the informative band is between 50% and 75%.
- **One model, one task, one needle position.** No claim about other architectures or MLA/GQA.
- **Saliency is measured from the query pass**, so 2b/2c are oracles.
- **Greedy decoding against a fixed cache is deterministic**, so these deltas are reproducible,
  not sampling noise.
- **No trained baseline comparison.** Heuristic attention selection is compared against a random
  control, not against a learned eviction policy such as Kamera's.

## Reproduce

```bash
python src/test_2d_kv_cache.py --keep-fracs 0.10,0.25,0.50,0.75 --json data/run.json
```
