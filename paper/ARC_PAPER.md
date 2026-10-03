# Context Economics for ARC Test-Time Search

**A measured, verifiable cost model for long-horizon reasoning, applied to ARC-AGI-2 search budgets**

Roni Saguey · ARC Prize 2026 Paper Track · Track: ARC-AGI-2 · Submission 56788985

*Code: `github.com/ronisaguey-ux/compressing-cache-AI-experiment` (Apache-2.0). Public notebook attached.*

---

## Abstract

ARC-AGI-2 is not solved by a single forward pass; the leading approaches spend large amounts of
test-time compute searching over candidate programs or adapting a model to each task. That search is
bounded by a GPU-time budget, and a large fraction of that budget is spent re-processing context that
the serving engine has already seen. We measure, directly and per turn, how a context policy affects
that cost. The result: a policy that pins an immutable prefix and rolls its window at the end
sustains an **81.3% prefix-cache reuse rate and costs 4.1× less compute** than a growing baseline over
a 60-turn agentic task, while retaining **every** probe value (60/60 against 9/60). We argue this
directly expands the number of candidate programs an ARC-AGI-2 solver can evaluate inside a fixed
budget, and we state that step as a testable hypothesis rather than a measured ARC result. Our own
ARC-AGI-2 submission is a fixed-library program-synthesis baseline that scores **0.0%**, reported here
without embellishment; its purpose is participation and an honest control.

## 1. Introduction

ARC-AGI-2 measures fluid reasoning on novel visual tasks. Progress has come from systems that spend
compute at test time: test-time training and compression (the CompressARC line of work), and
LLM-driven search over a hand-built program library (Tufa Labs' harness, the current leader at
83.1%). Both approaches share a property that is rarely measured: **the search is long, and the
context grows with it.** A solver that evaluates hundreds of hypothesised programs accumulates the
history of every attempt.

Serving engines address the resulting cost with automatic prefix caching: an input prefix identical
to the previous request reuses its key/value state. The economics are stark and public — a cached
token bills at roughly **one tenth** the price of a fresh one (the multiplier Anthropic publishes for
its prompt cache). A policy that **rewrites** the prefix therefore re-bills the whole context at the
full rate. Periodic compaction — the standard remedy for a long context — rewrites the prefix by
construction, so the standard remedy destroys the discount.

This paper measures that cost and shows a policy that avoids it, on a real agentic workload with
per-turn instrumentation. The ARC connection is argued explicitly in §5 and labelled as a
hypothesis; the ARC results in §4 are our own actual submission, including its zero.

## 2. Prior work

**ARC approaches.** Test-time-training methods adapt weights to each task; program-synthesis methods
search a library of grid transformations; LLM-harness methods (Tufa Labs) let a model write and
verify programs. Every one of these runs a long inner loop. To our knowledge **none of them accounts
for the serving cost of that loop** — the cost is treated as a fixed price per token, not as a
function of which tokens survive.

**KV-cache eviction.** H2O, Scissorhands, TOVA and SnapKV shrink the cache by scoring and dropping
tokens, assuming survivors remain valid after eviction. Others repair positional integrity directly
(CacheBlend, CacheFocus, DSCache, CacheGen). We attempt none of that: we never rewrite the region the
cache depends on, which is precisely why reuse survives.

**Attention sinks.** StreamingLLM keeps sink tokens plus a sliding window, and the sink mechanism
makes pinning an immutable front mechanically sensible. Sinks stabilise a stream but are not a memory
channel — which is why our retention probes test a specific fact, not fluency.

**Closest prior art.** SinkTrack (ICLR 2026) is nearest in *intent*: it anchors a model to its
initial context and reports context forgetting during long generation. The mechanisms differ in kind.
SinkTrack is **model-level**, injecting features into the `<BOS>` representation, evaluated on
single-generation QA. Ours is **policy-level**: it decides which tokens an agent retains, and measures
the serving-cost consequence over a long horizon. Prior work on prompt caching (Lumer et al., 2026)
evaluates static prompts; we measure compaction-induced cache-miss cost in a long agent loop, which we
find unmeasured elsewhere.

## 3. Approach

An agent's transcript is a sequence of turn blocks (instruction plus reply). Three policies differ
*only* in which blocks are retained when the prompt is assembled:

- **`runtime` (bounded anchor).** Keeps the system prompt and the turn-1 brief **pinned**, every past
  *instruction* verbatim, and the single newest *reply*. The newest full-file rewrite supersedes every
  earlier one; the small per-turn instruction is the only home of that turn's unique value.
- **`linear` (grow and evict).** Keeps the turn-1 block and as many recent pairs as fit, dropping the
  **oldest** block on overflow: a fixed window with no organisation.
- **`prune` (grow, evict, compact).** Identical to `linear`, plus a compaction every 30 turns.

The key structural difference is **where eviction happens**. `linear` and `prune` evict from the
front, so the first block changes the moment eviction begins and the shared prefix collapses to the
system prompt alone. `runtime` pins the front and advances the window at the end, so the prefix shared
with the previous turn survives. Nothing is summarised.

The task: a generated Python module seeded with 60 one-line faults, graded **per turn** in a fresh
interpreter, over Gemma-4-12B bf16 on one accelerator. Retention is
measured two ways that recency cannot satisfy: an unguessable code handed over in each instruction and
forbidden from the file until the final turn, and a mid-session secret. Cost is computed as
`miss × 1.0 + hit × 0.1` per turn from the longest common token prefix, with one named constant.

## 4. Results

**Our ARC-AGI-2 submission.** A fixed-library program-synthesis baseline over grid transformations
scored **0.0% (0/47)** on a 30-task evaluation sample. A control on the easier ARC-AGI-1 training set
scored non-zero, confirming the solver and scorer execute — a harness validation, not a capability
claim. The result is expected: a fixed library cannot express ARC-AGI-2, which is
constructed to defeat it. We report the zero as the honest accuracy figure for the submission.

**The cost measurement.** Over 60 turns:

| policy | per-turn success | probe recall | cache reuse | compute units |
|---|---|---|---|---|
| `runtime` | 1.000 | **60/60 ordered** | **81.3%** | **118,817** |
| `linear` | 1.000 | 9/60 unordered | 30.8% | 484,956 |

Coding accuracy is **identical** (60/60 both arms). The policies differ in what they retain, not in
what the model can do. `runtime` costs **4.08× less** and recovers 6.7× more of the early facts. Its
working set stays bounded where the baseline grows to the ceiling.

## 5. Why this matters for ARC — theory, stated as a hypothesis

Test-time search is a budget allocation problem. If a solver has a fixed GPU-time budget `T` and each
candidate program evaluation costs `c` in re-processing, the number of candidates it can evaluate is
`T / c`. Our measurement shows `c` is a function of the context policy, not a constant: the same work
costs 4.08× more under a front-evicting policy. Reducing `c` by that factor raises the number of
candidate programs evaluable within `T` by the same factor.

**This is the paper's central claim about ARC, and it is a hypothesis, not a measured ARC result.**
We have measured the cost ratio on a long-horizon agentic coding task with per-turn instrumentation.
We have *not* run it inside an ARC solver. The step from "4.08× cheaper per turn" to "4.08× more
candidates searched" assumes the ARC loop is context-bound in the same way; that is testable by
instrumenting an existing ARC harness's token accounting over its search, which is the first
experiment we would run next. It is stated here as the falsifiable form of the claim.

If it holds, it matters for the 85% goal: the current frontier is compute-limited, and a factor of 4
in effective search budget is a factor of 4 in candidate programs — without changing the model.

## 6. Completeness and limitations

One model, one task family, one seed per arm; the synthetic task makes retention measurable but is not
a real repository. The cost model uses a published hit multiplier rather than a billing measurement.
Repetition is across *task instances* (a salt re-derives every constant), not sampled seeds, because
decoding is greedy. The ARC bridge in §5 is argued from
measured cost, not demonstrated inside an ARC solver. Our ARC accuracy is 0.0%.

## 7. Conclusion

Long-horizon reasoning is bounded by context economics. We measured, per turn, that a policy which
pins an immutable prefix and rolls its window keeps prefix-cache reuse at 81% and costs 4.1× less than
the standard approach, with identical task accuracy and far better retention of early facts. We
present the extension to ARC-AGI-2 test-time search as a precise, falsifiable hypothesis rather than a
result, and we report our own submission's zero honestly. The contribution is a measurement the field
currently assumes: **the cost of remembering is a policy choice, and the standard policy pays for it.**
