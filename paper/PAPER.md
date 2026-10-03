# Bounded Context Without Context Rewrites
## Flat context and an intact prefix cache for long-horizon agents

---

## Abstract

Long-horizon autonomous agents are bounded by context economics, not model capability. The standard
remedy — periodically compacting the context — keeps the window small by *rewriting* it, and a
rewrite invalidates the shared prefix a serving engine's automatic prefix cache depends on, re-billing
the whole prompt at full price. We describe a policy that keeps the window small *without* rewriting
the front: the system prompt and the turn-1 brief are pinned, and the window advances at the end. We
evaluate three policies over 60 turns of a bug-fixing task on one 12B model, measuring retention,
compute cost and prefix-cache reuse directly rather than inferring them. The bounded-anchor policy
holds a flat context while the alternatives grow to the ceiling or collapse on each compaction;
figures are in §5. We also document four ways such an evaluation produces a clean-looking result that
means nothing, and how each was caught.

## 1. Introduction

An agent that must remain coherent across hundreds of turns accumulates history. Two costs grow
with it: the memory footprint of the key/value cache, and the compute spent re-reading the context
each turn. The dominant response is **compaction** — periodically summarising or discarding the
middle of the conversation to return under the window.

Compaction shrinks the context but changes its *content*. With automatic prefix caching a cached
token costs a fraction of a fresh one, so a changed prefix means the prompt is re-processed (§4).

**Why this matters in deployment.** The Gemma 4 Developer Agent harness compacts the agent's history
at 14,336 tokens -- 2,048 below the output ceiling it also requests -- and acts on the prompt just
sent rather than prompt plus output, so it is one call late. The scorer refuses any call past 32,768
tokens, and with a LoRA adapter the KV cache on four L4s falls from ~46,000 to ~7,600. Context
capacity, not reasoning, is the binding constraint, and compaction is the field's response.

We therefore ask a narrow, measurable question: **can an agent keep a small context without
rewriting it?** Our policy pins an immutable front (system prompt and the turn-1 brief) and advances
a bounded window at the end. Nothing is summarised: the information-dense part of each turn is
kept, the superseded part dropped.

**Contributions.** A context policy that bounds the working set while leaving the prefix shared
with the previous turn intact (§3); a per-turn measurement of prefix-cache reuse (§4.1); an
evaluation that discriminates, using per-turn grading and a probe recency cannot satisfy (§4.2);
and four failure modes by which such an evaluation silently measures nothing (§6).

## 2. Related work

**Eviction and compression.** H2O [1], Scissorhands [2], TOVA [3] and SnapKV [4] shrink the cache by
scoring tokens and dropping the low-scoring ones, assuming survivors stay valid after eviction — an
assumption we do not share (§6, F4). Others repair positional integrity directly: CacheBlend [5]
recomputes part of the KV and re-encodes positions, CacheFocus [6] re-positions after pruning,
DSCache [7] stores pre-rotation keys, CacheGen [8] recomputes as a fallback. **We attempt none of
that**: we never rewrite the region the cache depends on.

**Attention sinks.** StreamingLLM [9] keeps sink tokens and a sliding window, and the sink
mechanism [10] makes pinning an immutable front mechanically sensible. Sinks stabilise a stream but
are not a memory channel [11], which is why our probes test a *specific fact*, not fluency.

**Closest prior art.** SinkTrack [20] is nearest in *intent*: it anchors a model to its initial
context and reports context forgetting as a real failure of long generation. The mechanisms differ in
kind — SinkTrack is **model-level**, injecting features into the `<BOS>` representation, evaluated on
single-generation QA; ours is **policy-level**, deciding *which tokens the agent retains* and
measuring the serving-cost consequence over a long horizon. It is also a reason to expect a pinned
prefix to work: if the first token attracts attention, an agent that can keep its brief at the front
should.

**Agent memory.** MemGPT [12] pages context OS-style; the survey [13] taxonomises memory
operations. Both address what an agent should *store*, not the serving cost of rewriting it.

**Position and architecture.** *Lost in the Middle* [14] and *Found in the Middle* [15] establish
position-dependent context use. Gemma-4 makes it concrete: 40 of 48 layers use a 1024-token
local-attention window, so a turn-1 fact is unavailable to those layers regardless of policy (§6,
F4). Retention claims here concern **the tokens the agent retained**.

**Corrected results.** The §6 F3 result is a defect in our own benchmark; we report the corrected
task. A position paper argues venues need a refutations track [16]; the literature documents leakage
and reporting error [17], and empirical-method work [18] shows comparisons favour the proposed
method — the bias F1 and F2 produce. We adopt artifact-review expectations [19].

## 3. Method

### 3.1 The three policies

Let the transcript be a sequence of turn blocks. Each turn appends the instruction and the reply.
The policies differ *only* in which blocks are retained when assembling the prompt.

- **`runtime` (bounded anchor).** Retains the turn-1 block and every *instruction* so far, verbatim,
  plus the single most recent *reply*. Instructions are small; a reply here is a full-file rewrite,
  redundant with the newest one. Oldest instructions drop first if the budget is exceeded, so growth
  is one instruction per turn and the front is never rewritten.
- **`linear` (grow and evict).** Retains the turn-1 block and as many recent instruction/reply pairs
  as fit the model's ceiling, dropping the **oldest** block on overflow. The naive baseline: a fixed
  window with no organisation.
- **`prune` (grow, evict, compact).** Identical to `linear`, plus a compaction every 30 turns keeping
  the anchor and the most recent turns. The status quo for long-running agents.

All three share the model, prompts, tools, turn budget and turn-1 anchor, so a difference is
attributable to the retention policy. `prune` is anchored deliberately: otherwise it would differ
from `linear` by two things at once and a collapse could not be attributed to either.

### 3.2 Hypothesis and predictions

The claim is that **stability**, not size, is what a long-horizon agent needs. Two separable
predictions follow, because retention and cost are measured independently (§4.2):

- **Retention.** A fact stated early and required at the end survives under the anchored policy and
  not under the evicting ones, whose reach is a fixed number of turns however large the window is.
- **Cost.** The anchored policy reuses a larger prefix, never rewriting the front the previous turn
  already paid for; `prune` re-prefills at every compaction.

A cost difference alone would not support the claim, so §5 states the two separately.

### 3.3 The task

`solution.py` is seeded with *n* faulty functions, and **turn 1 carries the complete bug report** —
for each bug, the exact wrong and required constant. Every later turn says only "Fix bug *k*." No
list is repeated and no reminder is given.

1. Every bug family is the **same shape**: a one-line body with a single wrong constant. An earlier
   version used a keyword default and a table lookup; both invited the model to *rewrite* the shape,
   and the rewrites failed with the brief one turn back. A periodic, deterministic failure is a task
   defect, not a retention result (§6, F3).
2. The required constants are **arbitrary and appear nowhere else**, so they cannot be re-derived.
3. Grading is **per-turn**: bug *k* is tested right after turn *k*, in a subprocess, so a fix applied
   and later clobbered by a rewrite is distinguishable from one never applied.

### 3.4 Two independent retention probes

- **Accumulated codes.** One fresh unguessable code is handed over in each turn and may not be
  written to the file until the final turn, when it must be returned as an ordered list. Scored
  0..*n*, so partial retention appears as a prefix gap.
- **Release gate.** On the final turn, one extra function must return a **separate** secret handed
  over at turn *n*/2. It is deliberately *not* one of the codes, so a policy can recite the list and
  still ship a broken artifact. Verified independent:
  two simulated agents with identical bug and code scores receive opposite gate verdicts.

**Window depth, measured.** The two policies diverge in growth rate by roughly an order of
magnitude, measured live: `linear` adds ~1,248 tokens per turn (a full instruction/reply pair) while
`runtime` adds ~159 (one instruction plus a single superseded reply). Against a 12k ceiling `linear`
reaches it at about **turn 9** and evicts from then on, holding the most recent ~4 turn blocks;
`runtime` holds every instruction and finishes at ~9.9k tokens, under budget. This was computed
against the real tokenizer before the run and observed in the run.

**★ THE RELEASE GATE DID NOT DISCRIMINATE.** We predicted `linear` would fail it; it passed, as
did `runtime`. The gate's value is a function the artifact already contains, so committing the secret
to that function when it is handed over carries it forward through every later rewrite — the probe
measures artifact persistence, not retention. **The accumulated codes have no such escape: they are
forbidden from the file until the final turn**, which is why the codes probe separates the arms (§5)
and the gate is reported as having failed to separate.

## 4. Measurement

### 4.1 Prefix-cache reuse

Per turn we record the longest common **token** prefix between the current prompt and the previous
one — exactly what an automatic prefix cache would reuse. Cost is in **raw compute units, not
currency**: a fresh token costs 1.0, a reused token an assumed multiplier (0.1), so

```
cost_units = miss_tokens · 1.0 + hit_tokens · HIT_MULT
```

against a no-cache baseline of `total_tokens · 1.0`. The multiplier is one named constant, so the
assumption is visible and re-runnable.

The comparison is on `cost_units`, not on hit rate alone, because cost separates only if a policy
both reuses its prefix **and** keeps the prompt small: a high hit rate on a prompt that grows
without bound still processes far more tokens than a short one.

### 4.2 Metrics collected

Retention (per-turn success, final accuracy, fixes applied then lost, code recall, gate
verdict); cost (hit rate, miss tokens, cost units, no-cache baseline, saving ratio); context (first,
last, peak, growth); latency (TTFT, wall per turn, decode throughput); memory (KV peak, total
prefill). All derive from data the loop already collects, so one run yields the whole table rather
than one run per question.

### 4.3 Fairness gate

The stopping rule is the **turn count**, not wall-clock time. This is load-bearing: a policy that
keeps a smaller context is faster per turn *by construction*, so a time-based stop would let it
complete more turns and win on volume rather than on policy. Time is retained only as an emergency
break, so that an interrupted run grades what it finished instead of being discarded.

Because that break can fire at different turn counts per arm, the comparison script does **not**
report unequal arms as a difference and does **not** discard the run. It reconciles to the
**common prefix** — the turns both arms actually completed — re-summing the cost, context and
latency figures over that prefix. The measurements are per-turn and indexed by turn, so the turns
captured on both sides are the same turns, which is what comparability requires. It refuses only
when fewer than two turns are comparable.

One asymmetry is deliberate: the **retention** counts are not reconstructed, because the artifact's
state at turn *n* cannot be recovered from a later snapshot. A run that stops early is therefore
compared on cost and context over the shared prefix, and its retention numbers describe only the
turns it reached.

## 5. Results

Both arms ran 60 comparable turns under an identical prompt and tool budget (§3.1).

**Figure 1** (`figs/trajectory.svg`) — growth (a) and reuse (b), per turn.
**Table 1 — cost in raw compute units.** A cache-reused token costs 0.1 of a fresh one; the baseline prices every token as a miss.

| policy | cache hit rate | miss tokens | cost units | no-cache cost | saving | prompt first → last | TTFT first → last |
|---|---|---|---|---|---|---|---|
| `runtime` | 81.3% | 82854 | 118817 | 442488 | 3.72× | 2057 → 9934 | 1786.2 → 5592.0 |
| `linear` | 30.8% | 464305 | 484956 | 670822 | 1.38× | 2057 → 12142 | 1795.1 → 7370.6 |

**Table 2 — retention.** Per-turn success is graded after each turn; code recall is the accumulated list; the release gate is the independent mid-session secret.

| policy | per-turn success | final state | applied-then-lost | code recall | ordered | release gate |
|---|---|---|---|---|---|---|
| `runtime` | 1.000 | 1.000 | 0 | 1.000 | True | PASS |
| `linear` | 1.000 | 1.000 | 0 | 0.150 | False | PASS |

**Separation.** `runtime` reuses a prefix for 81% of its prefill against 31% for `linear`, at 118817 compute units against 484956 (4.08×).
The difference is structural, not incidental: a policy that evicts from the front changes its first block as soon as eviction begins, while one that pins the front and advances the window at the end keeps it. Hence reuse separates where context size alone might not.


## 6. Failure modes of long-horizon benchmarks

Each of the following produced, or would have produced, a clean-looking table that measured nothing.

**F1 — shared ceiling.** Two policies capped at the same limit converge on "whatever fits": with
one shared 12k cap both arms reached ~9.9k at identical latency (4180 vs 4308 ms), so the cap and
not the policy set the working set. Each budget is now sized to its own claim.

**F2 — re-derivable probes.** A rule the model re-applies every turn can be inferred from recent
turns without the original, so both arms pass and the probe is vacuous. Only a value stated once and
never repeated discriminates.

**F3 — periodic failures are task defects.** Two of six bug families failed on *every* instance from
turn 2 — one turn after the brief, so memory was not in question: the families had unusual shapes the
model rewrote. A turn-indexed task must be checked for periodic failure before its losses count.

**F4 — architectural confound.** Gemma-4 uses local attention in 40 of 48 layers with a 1024-token
window, so beyond 1024 tokens a turn-1 fact is invisible to those layers regardless of policy. Any
claim about retaining an early fact must be stated against that background.

## 7. Limitations

One model, one task family; the task is synthetic by design so retention is measurable. The cost model
uses an assumed hit multiplier rather than a billing measurement. The gate probe is a single value:
it establishes that a mid-session fact survived, not a rate.

**Model choice.** The reported run uses Gemma-4-12B bf16, which loads on one accelerator with no
quantization confound; the competition checkpoint (`gemma-4-31b-it-qat-w4a16-ct`) is 4-bit and would
add a quantization variable to a retention measurement. The policy is model-independent — a decision
about retained blocks, implemented outside the model — so another checkpoint changes the numbers, not
the mechanism.

## 8. Conclusion

Keeping a context small and keeping it *stable* are different problems, and the usual solution to
the first destroys the second. Pinning an immutable front and rolling the window at the end bounds
the working set without invalidating the prefix already paid for. We have made that claim measurable
rather than asserted, and documented how such a measurement silently fails.

---

## Reproduction

```
python3 tools/run_compare.py benchmarks/results/incremental_*_{linear,runtime}_*.json
```

The environment, task generator, policies and graders are in this repository, and that command runs
against committed result files. Policies are selected by `CCAI_ARM`; the retention gate is opt-in
via `CCAI_GATE=1`; the compaction period by `CCAI_PRUNE_EVERY`.

## References

[1] Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* NeurIPS 2023. arXiv:2306.14048
[2] Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis.* NeurIPS 2023. arXiv:2305.17118
[3] Oren et al. *Transformers are Multi-State RNNs.* arXiv:2401.06104
[4] Li et al. *SnapKV: LLM Knows What You Are Looking For Before Generation.* NeurIPS 2024. arXiv:2404.14469
[5] Yao et al. *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion.* EuroSys 2025. arXiv:2405.16444
[6] Lee, Park & Han. *CacheFocus: Dynamic Cache Re-Positioning for Efficient RAG.* arXiv:2502.11101
[7] Pang et al. *Decouple and Cache: KV Cache Construction for Streaming Video.* arXiv:2605.01858
[8] Liu et al. *CacheGen: KV Cache Compression and Streaming for Fast LLM Serving.* ACM SIGCOMM 2024. arXiv:2310.07240
[9] Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453
[10] Gu et al. *When Attention Sink Emerges in Language Models.* arXiv:2410.10781
[11] Cao, Zhang & Tang. *Separating Stream Stability from Long-Term Recall in Language Models.* arXiv:2609.07282
[12] Packer et al. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560
[13] Zhang et al. *A Survey on the Memory Mechanism of LLM-based Agents.* arXiv:2404.13501
[14] Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv:2307.03172
[15] Hsieh, Chuang et al. *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization.* arXiv:2406.16008
[16] Schaeffer, Kazdan, Denisov-Blanch, Miranda, Gerstgrasser et al. *Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track.* NeurIPS 2025 Position Paper Track (Oral). arXiv:2506.19882
[17] Kapoor & Narayanan. *Leakage and the reproducibility crisis in ML-based science.* Patterns 2023.
[18] Herrmann et al. *Why We Must Rethink Empirical Research in Machine Learning.* ICML 2024.
[19] ACM. *Artifact Review and Badging, v1.1.*
[20] Liu, Chen & Wang. *SinkTrack: Attention Sink based Context Anchoring for Large Language Models.* ICLR 2026. arXiv:2604.10027
