# Bounded Context Without Context Rewrites
## Flat context and an intact prefix cache for long-horizon agents

---

## Abstract

Long-horizon autonomous agents are bounded by context economics, not by model capability. The
standard remedy — periodically compacting the context — keeps the window small by *rewriting* it,
and a rewrite invalidates the shared prefix that a serving engine's automatic prefix cache depends
on, re-billing the entire prompt at full price. We describe a context policy that keeps the window
small *without* rewriting the front: the system prompt and the turn-1 task brief are pinned, and the
rolling window is advanced at the end. We evaluate three policies (bounded-anchor, grow-and-evict,
grow-compact) on a long-horizon bug-fixing task of 60 turns using a single 12B model, and measure
retention, cost in raw compute units, and prefix-cache reuse directly rather than inferring them.
The bounded-anchor policy sustains a flat context while the alternatives grow to the ceiling or
collapse on every compaction; we report the exact figures in §5. We also document four ways the
benchmark itself can produce a clean-looking result that means nothing, and how each was detected —
because a long-horizon evaluation that cannot discriminate will silently report a tie.

## 1. Introduction

An agent that must remain coherent across hundreds of turns accumulates history. Two costs grow
with it: the memory footprint of the key/value cache, and the compute spent re-reading the context
each turn. The dominant response is **compaction** — periodically summarising or discarding the
middle of the conversation to return under the window.

Compaction reduces the size of the context but changes its *content*. Under a serving engine with
automatic prefix caching, a cached token is billed at a fraction of a fresh one; when the prefix
changes, the cache no longer applies and the full prompt is re-processed. A cache hit is roughly an
order of magnitude cheaper than a miss on the re-processed span, so a policy that rewrites its
context pays that multiple on every rewrite.

We therefore ask a narrow, measurable question: **can an agent keep a small context without
rewriting it?** Our policy pins an immutable front (system prompt and the turn-1 brief) and advances
a bounded window at the end. Nothing is summarised; the cheap, information-dense part of each turn
is retained and the large, superseded part is dropped.

**Contributions.**
1. A context policy that bounds the working set while leaving the prefix shared with the previous
   turn intact (§3).
2. A direct measurement of prefix-cache reuse — the longest common token prefix between consecutive
   prompts — reported per turn, and a compute-cost model that prices a miss and a hit differently
   (§4).
3. A long-horizon evaluation that discriminates, with per-turn grading and a retention probe that
   cannot be satisfied by recency alone (§4.2).
4. A catalogue of four failure modes by which such an evaluation silently measures nothing (§6).

## 2. Related work

**Key/value eviction and compression.** H2O [1], Scissorhands [2], TOVA [3] and SnapKV [4] all
reduce cache size by scoring tokens and dropping the low-scoring ones. They assume the survivors
remain valid after eviction — an assumption we do not share (§6, F4), and which is the subject of a
parallel line of work on positional integrity: CacheBlend [5] selectively recomputes part of the KV
and re-encodes positional indices, CacheFocus [6] re-positions the cache after pruning, and DSCache
[7] stores pre-rotation keys and reintroduces position at use. CacheGen [8] uses recomputation as a
correctness fallback under bandwidth pressure. **This work does not attempt any of that.** We do not
repair a cache; we avoid needing to, by never rewriting the region the cache depends on.

**Streaming and attention sinks.** StreamingLLM [9] keeps sink tokens and a sliding window to stream
millions of tokens. The sink mechanism [10] — a learned, softmax-driven first-token attractor — is
the reason pinning an immutable front is mechanically sensible rather than merely a bookkeeping
convenience; note however that sinks stabilise a stream without acting as a memory channel [11],
which is exactly why our retention probes test a *specific fact* rather than fluency.

**Agent memory.** MemGPT [12] pages context in and out in an OS-like fashion, and the agent-memory
survey [13] taxonomises memory sources, forms and operations. Both address what an agent should
*store*; neither makes the *serving-cost* consequence of rewriting it measurable, which is the gap
this paper fills.

**Retention position.** *Lost in the Middle* [14] and *Found in the Middle* [15] establish that
context use is position-dependent, which is why retention probes must be stated against the
architectural window rather than assumed to be explained by the policy alone.

**The architecture confound, stated up front.** Gemma-4 uses local attention in 40 of its 48 layers
with a 1024-token window, so beyond 1024 tokens a turn-1 fact is unavailable to those layers
regardless of the agent's policy. Every retention claim here is therefore a claim about **the tokens
the agent chose to retain**, not about recovering a fact the architecture had already discarded (§6,
F4).

**On reporting negative and corrected results.** The measured result in §6 F3 is a defect in our own
benchmark, and we report the corrected task rather than the number the broken task produced. A
position paper argues that venues should have a formal refutations track [16]; the reproducibility
literature documents systematic leakage and reporting error [17,18]; and work on empirical-method
bias [19] shows method comparisons are biased toward the newly proposed method, which is precisely
the bias the shared-ceiling and re-derivable-probe failures produce. We adopt artifact-review
expectations [20] rather than treating reproducibility as optional.

## 2.1 References

[1] Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* NeurIPS 2023. arXiv:2306.14048
[2] Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis.* NeurIPS 2023. arXiv:2305.17118
[3] Oren et al. *Transformers are Multi-State RNNs.* arXiv:2401.06104
[4] Li et al. *SnapKV: LLM Knows What You Are Looking For Before Generation.* NeurIPS 2024. arXiv:2404.14469
[5] Yao et al. *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion.* EuroSys 2025. arXiv:2405.16444
[6] *CacheFocus: Dynamic Cache Re-Positioning for Efficient RAG.* arXiv:2502.11101
[7] *DSCache: Decoupled Streaming Cache.* arXiv:2605.01858
[8] Liu et al. *CacheGen: KV Cache Compression and Streaming for Fast LLM Serving.* ACM SIGCOMM 2024. arXiv:2310.07240
[9] Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453
[10] Gu et al. *When Attention Sink Emerges in Language Models.* arXiv:2410.10781
[11] *Separating Stream Stability from Long-Term Recall in LMs.* arXiv:2609.07282
[12] Packer et al. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560
[13] Zhang et al. *A Survey on the Memory Mechanism of LLM-based Agents.* arXiv:2404.13501
[14] Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv:2307.03172
[15] Han et al. *Found in the Middle: Calibrating Positional Attention Bias.* arXiv:2406.16008
[16] Schaeffer et al. *Position: ML Conferences Should Establish a "Refutations and Critiques" Track.* NeurIPS 2025.
[17] Kapoor & Narayanan. *Leakage and the reproducibility crisis in ML-based science.* Patterns 2023.
[18] *Systematic research errors in thousands of machine learning papers.* ACL 2023.
[19] Herrmann et al. *Why We Must Rethink Empirical Research in Machine Learning.* ICML 2024.
[20] ACM. *Artifact Review and Badging, v1.1.*

> ⚠️ Several of these entries came from research notes rather than a direct index lookup and are
> marked UNVERIFIED in the source material. arXiv identifiers in the `26xx` range are valid for
> 2026 (e.g. `2609.*` = September 2026). **Each citation's authors, venue and id must be confirmed
> against the index before submission** — an unverifiable citation is a defect to fix, not a fact
> to assert, and the [3],[6],[7],[11] entries in particular have unconfirmed author lists.

## 3. Method

### 3.1 The three policies

Let the transcript be a sequence of turn blocks. Each turn appends the instruction and the reply.
The policies differ *only* in which blocks are retained when assembling the prompt.

- **`runtime` (bounded anchor).** Retains the turn-1 block and every *instruction* seen so far,
  verbatim, plus the single most recent *reply*. The instructions are small; a reply in our task is
  a full-file rewrite and is therefore redundant with the newest one. Oldest instructions are
  dropped first if the budget is exceeded. Growth is therefore one instruction per turn, and the
  front is never rewritten.
- **`linear` (grow and evict).** Retains the turn-1 block and as many recent instruction/reply pairs
  as fit the model's ceiling, dropping the **oldest** block when it overflows. This is the naive
  baseline: a fixed window with no organisation.
- **`prune` (grow, evict, compact).** Identical to `linear`, plus a compaction every 30 turns that
  keeps the anchor and the most recent turns and discards everything between. This is the status quo
  for long-running agents.

The three share the model, prompts, tools, turn budget and the turn-1 anchor, so a difference is
attributable to the retention policy. `prune` is anchored deliberately: without that it would differ
from `linear` by two things at once and a collapse could not be attributed to either.

### 3.2 The task

`solution.py` is seeded with *n* faulty functions, and **turn 1 carries the complete bug report** —
for each bug, the exact wrong and required constant. Every later turn says only "Fix bug *k*." No
list is repeated and no reminder is given.

1. Every bug family is the **same shape**: a one-line body containing a single wrong constant. An
   earlier version used a keyword default and a table lookup; both invited the model to *rewrite*
   the shape, and those rewrites failed with the brief one turn back. A periodic, deterministic
   failure is a task defect, not a retention result (§6, F3).
2. The required constants are **arbitrary and appear nowhere else**, so they cannot be re-derived.
3. Grading is **per-turn**: bug *k* is tested immediately after turn *k*, in a subprocess. A fix
   applied and later clobbered by a rewrite is distinguishable from one never applied.

### 3.3 Two independent retention probes

- **Accumulated codes.** One fresh unguessable code is handed over in each turn and may not be
  written to the file until the final turn, when it must be returned as an ordered list. Scored
  0..*n*, so partial retention appears as a prefix gap.
- **Release gate.** On the final turn, one extra function must return a **separate** secret handed
  over at turn *n*/2. This value is deliberately *not* one of the codes, so it is independent of the
  first probe: a policy can recite the list and still ship a broken artifact. Verified independent:
  two simulated agents with identical bug and code scores receive opposite gate verdicts.

## 4. Measurement

### 4.1 Prefix-cache reuse

Per turn we record the longest common **token** prefix between the current prompt and the previous
one — exactly what an automatic prefix cache would reuse. Cost is reported in **raw compute units,
not currency**: a fresh token costs 1.0 and a cache-reused token costs an assumed multiplier (0.1),
so

```
cost_units = miss_tokens · 1.0 + hit_tokens · HIT_MULT
```

and the no-cache baseline is `total_tokens · 1.0`. The multiplier is one named constant so the
assumption is visible and re-runnable. This makes the cache claim falsifiable: if the policies do
not separate on hit rate, the argument does not hold.

### 4.2 Metrics collected

Retention (per-turn success, final accuracy, fixes applied then lost, code recall and ordering, gate
verdict); cost (hit rate, miss tokens, cost units, no-cache baseline, saving ratio); context (first,
last, peak, growth); latency (TTFT first/last/growth, wall per turn, decode throughput); memory (KV
peak, total prefill). All are derived from data the loop already collects, so one run yields the
full table rather than one run per question.

### 4.3 Fairness gate

The comparison script **refuses** to report a result when the arms did not complete the same number
of turns, or when either carries a harness failure. A policy that keeps a smaller context is faster
per turn, so a wall-clock stop would let it complete more turns and win on volume rather than on
policy. The stopping rule is the turn count; time is only an emergency break.

## 5. Results

*(to be completed from the run in flight; every arm reports the full §4.2 table, and no number is
quoted until the final-turn probes have been checked on both arms)*

## 6. Failure modes of long-horizon benchmarks

Each of the following produced, or would have produced, a clean-looking table that measured nothing.

**F1 — shared ceiling.** Capping two policies at the same token limit makes them both converge on
"whatever fits". Measured: the bounded policy grew to 9.9k against a limit of 12k and the two arms
had identical latency (4180 vs 4308 ms). Each policy needs a budget sized to its own claim.

**F2 — re-derivable probes.** A rule the model re-applies every turn can be inferred from recent
turns without the original ever being seen; both arms pass and the probe is vacuous. Only a value
stated once and never repeated can discriminate.

**F3 — periodic failures are task defects.** Two of six bug families failed on *every* instance
from turn 2 and turn 5. Since turn 2 is one turn after the brief, memory was not in question: the
families had unusual shapes that the model rewrote, and the rewrites failed. Fixing the shapes
uniformised the task. A turn-indexed task must be checked for periodic failure before its losses are
interpreted.

**F4 — architectural confound.** Gemma-4 uses local attention in 40 of 48 layers with a 1024-token
window, so beyond 1024 tokens a turn-1 fact is invisible to those layers regardless of the agent's
policy. Any claim about retaining an early fact must be stated against that background.

## 7. Limitations

The evaluation uses one model and one task family; the task is synthetic by design, to make retention
measurable. The cost model uses an assumed hit multiplier rather than a billing measurement — it is a
compute model, and is stated as such. The gate probe is a single value; a single probe establishes
that a mid-session fact survived, not a rate.

## 8. Conclusion

Keeping a context small and keeping it *stable* are different problems, and the usual solution to the
first destroys the second. A policy that pins an immutable front and rolls the window at the end
bounds the working set without invalidating the prefix the previous turn already paid for. We have
made the corresponding claim measurable rather than asserted, and we have documented the ways such a
measurement can silently fail.

---

## Reproduction

```
python3 tools/run_compare.py results/fixres_*_*.json
```

Environment, task generator, policies and graders are in this repository. The three policies are
selected by `CCAI_ARM`; the retention gate is opt-in via `CCAI_GATE=1`; the compaction period by
`CCAI_PRUNE_EVERY`.
