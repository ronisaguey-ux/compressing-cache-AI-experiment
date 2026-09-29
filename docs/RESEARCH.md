# Prior art: is KV reuse across a mutated prefix already solved?

Researched before writing any code, so the experiment is aimed at a real gap rather than at
something already settled.

## The idea is real and it is frontier

- **SGLang, RFC #30928 — "[RFC] Position-Independent KV Cache Reuse for Agentic/RAG Workloads".**
  Open, assigned to a core maintainer. Position-independence across an edited prefix is an
  acknowledged open problem in the highest-throughput serving stack, not a solved one.
- **LazyAttention** (ICML 2026, UIUC) — `github.com/illinoisdata/lazy-attention`, arXiv 2606.04302.
  The closest prior work, and it already ships. vLLM plus two Triton kernels. Caches keys
  **unrotated at local position 0** and applies the delta rotation `R_Δ` on-chip at attention
  time. Reported ~0.2% overhead, **1.37× TTFT**, **1.40× throughput**, up to **7.5× hit ratio**
  versus prefix caching.
- **Irminsul** — MLA, content-hash chunks plus a δ-rotation. No public code found.
- **Kamera** (arXiv 2606.23581) — the one that measured the failure we care about.
- **KVBoost** (chunk 128), **Block-Attention** (Ma 2025), **PromptCache** (Gim 2024), **FlowBlock**.

## What that prior art does and does not establish

LazyAttention proves the **positional** half: bake no rotation into cached keys, apply the delta
on the way into attention, and reuse survives a prefix edit. That is implementable and the 1.37×
TTFT is a real measurement.

Kamera measured the half rotation cannot fix. `KV_i` is a function of **every token before i**.
Deleting a region does not un-bake that aggregate from the surviving values. It reported
multi-hop accuracy falling **0.41 → 0.28 (MLA)** and **0.28 → 0.15 (GQA)** from conditioning loss
alone. That is the damage this experiment isolates as Finding 4.

So the gap was: **everyone has measured positional repair, and Kamera has measured conditioning
loss, but nobody has measured a selector that preserves conditioning while still evicting.**
That is what Arms 2a/2b/2c and the retention curve test.

## Why the sparse-stride variant was rejected in favour of dense positions

An earlier design put the surviving chunks on coarse strides (`1023 → 2048 → 8192`) to keep
original indices. That makes the system block and the dialogue block ~1025 positions apart
instead of ~30 — a RoPE extrapolation the model never saw in training, which is exactly the decay
LazyAttention avoids by keeping positions dense and re-rotating. Every published method keeps
positions **dense**. Dense + delta rotation is what this harness implements.

## Why the first proposed needle was vacuous

The original design put the target secret in chunk 0. Chunk 0 is never evicted, so the needle
survives in every arm and the test cannot distinguish them. The harness therefore uses **two**
needles: one in chunk 0 as a control, and `0x9AF4_STACK_FAIL` buried at 67% through the volatile
chunk, where first-k, last-k and head-truncating eviction all miss it.

## What is genuinely new here

1. A **purpose-built negative-space test**: a needle placed where the cheap selectors provably
   cannot reach it, so retaining it is diagnostic rather than lucky.
2. The **retention curve**, which shows a 10% budget is below the floor for every selector — a
   fact that invalidates the natural first experiment and is invisible without the curve.
3. **Per-layer needle ranks**, showing layer 0 ranks the target 303rd of 333 while layers 11-19
   carry it. Aggregating attention across all layers dilutes the signal; the mid-band is what
   works. This is a concrete implementation warning for anyone building on H2O/SnapKV-style
   aggregation.
4. **Rotation isolated as a controlled variable** — same kept set, rotation on vs off — worth
   +2.019 nats in the case where it matters.
   [LazyAttention]: https://github.com/illinoisdata/lazy-attention
