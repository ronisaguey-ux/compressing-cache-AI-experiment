# Saliency-Preserving KV Compaction with Dense Delta Re-rotation

Measured results. Everything here is reproducible with `python src/test_2d_kv_cache.py`.

**Model** Qwen/Qwen2.5-0.5B, CPU, fp32. 24 layers, 14 query heads, 2 KV heads, head_dim 64.
**Context** chunk 0 = 27 tok system prompt holding `SYSTEM_FLAG_A = ALPHA_VERIFIED`;
chunk 1 = 333 tok compiler log holding `0x9AF4_STACK_FAIL` at offset 224 (67% through the region).
**Queries** Q1 asks for the flag (answer lives in chunk 0, never evicted — the control);
Q2 asks for the failure code (answer lives in chunk 1, the thing being evicted).

## The control that has to pass first

| arm | cache | nll Q1 | nll Q2 |
|---|---|---|---|
| Arm 0 baseline, untouched | 360 | 2.811 | 1.792 |
| Identity: keep **all** + rotate | 360 | 2.811 | 1.792 |

Re-rotating a complete cache applies `R(0)` to every key, so it must reproduce the baseline
bit-for-bit. It does, to 0.000 nats. Any drift in the arms below is therefore caused by
eviction and not by the rotation machinery.

## Finding 1 — no saliency signal finds the needle at any aggressive keep rate

Index arithmetic, no model needed. "Does this selector retain the needle?" at each budget:

| selector | 5% | 10% | 15% | 25% | 50% | 75% |
|---|---|---|---|---|---|---|
| first-k/2 + last-k/2 | – | – | – | – | – | yes |
| random-k | – | – | – | yes | yes | yes |
| top-k key-norm | – | – | – | – | – | yes |
| top-k query-attn (all 24 layers) | – | – | – | – | – | yes |
| **top-k query-attn (mid layers 10-19)** | – | – | – | – | **yes** | yes |

This is the single most important measurement in the report. **At 10% retention — the budget
the original design assumed — every selector loses the needle.** Any experiment run only at 10%
would show "all eviction strategies fail" and would be measuring the budget, not the strategy.

Per-layer needle rank on this task:

```
layer  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
rank 303 199 323 314 207 249 221 219 301 134 162  60  86  92  88  87  46 169  88  96 124 140  68 194
```

Layer 0 ranks the error code **303rd of 333** — worse than useless. Layers 11-19 carry the
semantic signal (best: layer 16 at rank 46). Summing all 24 layers dilutes those into the noise,
which is why the all-layer aggregate also fails at 50% while the mid-layer band succeeds.

## Finding 2 — at a budget where the needle survives, saliency is the only thing that works

Keep 50% (193 of 333 positions, **46.4% of KV memory saved**):

| arm | needle kept | Q2 answer | nll Q2 | Δ vs baseline |
|---|---|---|---|---|
| Arm 0 baseline | yes | `0x9AF4_STACK_FAIL` ✅ | 1.792 | — |
| **Arm 2c top-k query-attn (mid layers)** | **yes** | **`0x9AF4_STACK_FAIL` ✅** | **2.147** | **+0.355** |
| Arm R random-k | yes | "I'm sorry, I don't understand…" ❌ | 2.685 | +0.893 |
| Arm 2b top-k query-attn (all layers) | no | "`bad value`" ❌ | 2.691 | +0.899 |
| Arm 2a top-k key-norm | no | "1." ❌ | 3.701 | +1.909 |
| Arm 1k naive first5+last5 | no | "`0`" ❌ | 4.314 | +2.522 |

Arm 2c is the only arm that answers correctly, and it is also the closest to baseline. At 75%
it repeats the result (`2.091`, answer correct). The random control is the informative one: it
*did* retain the needle token and still failed — keeping one token is not enough, the selector
has to keep the contiguous block around it, which attention does and chance does not.

## Finding 3 — dense delta re-rotation is real and it is what flips the answer

Same kept set, rotation on vs off. Only the rotation differs.

| kept set | cache | nll Q2 (rotated) | nll Q2 (not rotated) | improvement |
|---|---|---|---|---|
| top-k key-norm | 60 | 4.883 | 4.974 | +0.091 |
| top-k key-norm | 193 | 3.701 | 4.624 | +0.923 |
| top-k query-attn (mid) | 60 | 4.504 | 4.317 | −0.187 |
| **top-k query-attn (mid)** | **193** | **2.147** | **4.166** | **+2.019** |

It helps in 3 of 4 matched comparisons, and in the case that matters it is worth **2.019 nats** —
the difference between emitting `0x9AF4_STACK_FAIL` and degenerating into `Human: 0x000000000000`.
At the 10% budget the effect is noise-sized, which is expected: the needle is already gone, so
there is nothing left for correct positions to save.

Greedy decoding against a fixed cache is deterministic, so these deltas are reproducible rather
than sampling noise.

## Finding 4 — evicting chunk 1 damages a question whose answer was never touched

Q1 asks about chunk 0, which every arm retains in full at unchanged positions. It should survive
everywhere. It does not.

| | baseline | eviction arms |
|---|---|---|
| Q1 literal + normalized | 1/1 | **0/0 in 22 of 24** |
| example failure | — | Arm 1k @10%: echoes the question back, `"Human: What is SYSTEM_FLAG_A?"` |
| nll Q1 spread across arms | — | **3.089 nats** |

For contrast, Arm 1 (the re-encoded clean squash) answers Q1 correctly, and the identity control
is exact. So this is not a prompt-format artifact shared by every arm: a *clean* short context
answers Q1, and an *evicted* context does not.

This is the conditioning loss the design set out to address, and it is the largest single effect
in the dataset. Dense re-rotation repairs positions; nothing in these arms repairs a value vector
that was computed as an aggregate over tokens which have since been deleted.

## Finding 5 — text squashing costs a full prefill and loses the needle anyway

| | Arm 1 |
|---|---|
| cache | 49 tok, 86.4% saved |
| Q1 | ✅ correct |
| Q2 | ❌ *"not specified in the provided log"* |
| extra prefill | **48.4 GFLOP** — a full re-encode of the squashed prompt |

Every eviction arm pays **zero** extra prefill: they reuse the cached keys. Arm 1 has to run the
model again over the shortened prompt, so its cost advantage is illusory at this scale and it
still loses the answer.

## Limits — read these before quoting any number

- **0.5B model.** It is fragile: it degenerates into `1.0.0.0.0.0` loops under mild perturbation.
  Absolute quality is not meaningful; the *comparisons* between arms are, because all arms share
  the same model, prompt and decode settings.
- **The chat template was not applied.** The model was driven as a raw continuer, which is visible
  in answers like `"Human: The value of …"`. This affects every arm identically, so the relative
  findings hold, but it depresses absolute accuracy. A chat-templated rerun is the obvious next
  step for numbers fit to publish.
- **One model, one task, one needle position.** No claim about other architectures, other
  positions, or MLA/GQA variants.
- **Saliency is measured from the query pass**, so Arm 2b/2c are oracles: they need the question
  before they can decide what to keep. Arm 2a (key-norm) is the only query-agnostic selector here
  and it failed. A production path needs a query-agnostic selector that works, and on this
  evidence none of the cheap ones do.
- **Layer 0 alone is actively misleading** on this task (rank 303/333). Any implementation that
  aggregates attention without checking per-layer ranks may be summing the signal away.

## Reproduce

```bash
python src/test_2d_kv_cache.py --keep-fracs 0.10,0.25,0.50,0.75 --json data/run.json
```
