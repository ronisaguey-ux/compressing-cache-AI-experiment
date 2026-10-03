# Context Economics for ARC Test-Time Search

A measured cost model for long-horizon reasoning, applied to ARC-AGI-2 search budgets

Roni Saguey. ARC Prize 2026 Paper Track, ARC-AGI-2. Submission 56788985.

## Abstract

ARC-AGI-2 is not solved by a single forward pass. The leading approaches spend large amounts of
test-time compute searching over candidate programs or adapting a model to each task, and that search
is bounded by a GPU-time budget. A substantial fraction of the budget re-processes context the serving
engine has already seen. This paper measures, per turn, how a context retention policy changes that cost.
A policy that pins an immutable prefix and advances its window at the end sustains a 76% prefix-cache
reuse rate and costs 4.55 times less compute than a growing baseline over sixty turns, while retaining
every probe value, sixty of sixty against nine. Run directly as an ARC
solver, it costs 7.29 times less on an ARC-AGI-2 puzzle at an identical (zero) solve rate. We argue
that this expands the number of candidate programs a solver can evaluate within a fixed budget, and
state the step from cheaper turns to more candidates searched as a testable hypothesis.

## 1. Introduction

ARC-AGI-2 measures fluid reasoning on novel visual tasks. Recent progress comes from systems that
spend compute at test time: test-time training and compression, and LLM-driven search over a program
library, of which the leading entry scores 83.1%. Both share a property rarely measured: the search is long, the context
grows with it, and a solver evaluating hundreds of programs accumulates every attempt.

Serving engines address this with automatic prefix caching, in which a prefix identical to the
previous request reuses its key and value state, at roughly one tenth the price of a fresh token. A
policy that rewrites the prefix re-bills the entire context at the full rate. Periodic compaction, the
standard remedy for a long context, rewrites the prefix by construction, so the standard remedy
destroys the discount.

This paper makes that cost measurable on a real agentic workload, and runs the measurement inside an
ARC solve loop; the extension to a larger search budget is stated in Section 5 as a hypothesis.

## 2. Prior Work

Approaches to ARC divide into test-time-training methods that adapt weights per task, program-synthesis
methods that search a transformation library, and LLM-harness methods that write and verify programs.
Each runs a long inner loop. To our knowledge none accounts for that loop's serving cost, because cost
is treated as a fixed price per token rather than a function of which tokens survive.

Work on KV-cache eviction shrinks the cache by scoring tokens and dropping low-scoring ones, assuming
survivors remain valid; a second line repairs positional integrity. Neither is attempted here: we never
rewrite the region on which the cache depends, which is why reuse survives. Attention-sink work
explains why pinning an immutable front is sensible, but sinks stabilise a stream rather than carry
information, which is why our probes test a particular fact rather than fluency.

The closest prior work in intent is SinkTrack, which anchors a model to its initial context and
reports context forgetting. The mechanisms differ in kind: SinkTrack intervenes on the
beginning-of-sequence representation and is evaluated on single-generation question answering, whereas
this policy decides at the agent level which tokens are retained. A recent evaluation of prompt
caching for agentic workloads finds 41 to 80% cost reductions across three providers, but works at the
provider API level, compares no retention policies, and does not measure a compaction discontinuity.

## 3. Approach

An agent's transcript is a sequence of turn blocks, each an instruction and a reply. Three policies
differ only in which blocks are retained. The anchored policy pins the system prompt and the first-turn
brief, keeps every past instruction verbatim, and keeps the single newest reply; instructions are
small, while a reply is a full file rewrite the newest reply already supersedes. The linear policy
keeps the first-turn block and as many recent pairs as fit, dropping the oldest on overflow. The prune
policy adds a compaction every thirty turns.

The structural difference is where eviction occurs. Linear and prune evict from the front, so their
first block changes as soon as eviction begins and the prefix shared with the previous turn collapses
to the system prompt alone. The anchored policy pins the front and advances the window at the end, so
that shared prefix survives.

The task is a generated Python module seeded with sixty one-line faults, graded per turn in a fresh
interpreter, over Gemma-4-12B in bfloat16 on one accelerator. Retention is measured by an unguessable
code given in each instruction and forbidden from the file until the final turn, plus a separate
mid-session secret. Cost is computed per turn from the longest common token prefix, with one named
constant for the hit multiplier.

## 4. Results

Our ARC-AGI-2 submission is a fixed-library program-synthesis baseline over grid transformations. It
scored 0.0%, zero of forty-seven, on a thirty-task evaluation sample. A control on the easier
ARC-AGI-1 training set scored non-zero, confirming the solver and scorer execute, which is a harness
validation rather than a capability claim. The zero is expected: a fixed library cannot express
ARC-AGI-2, which is built to defeat it.

The cost measurement covers sixty turns. The anchored policy reuses 76.0% of its prefill against 30.8%
for the linear policy, at 106,678 compute units against 484,957, a factor of 4.55. Coding accuracy is
identical, at sixty of sixty, so the difference lies in what each retains rather than in what the
model can do. It also recovers all sixty probe values, against nine. The third policy, prune, behaves as the linear policy does and shows compaction
directly: at turn thirty the reused prefix collapses from 3,226 tokens to 2,057, the anchored prefix
alone, so compaction pays a re-prefill at every boundary rather than recovering the cost it is
introduced to save.

To test the mechanism on the target domain rather than by analogy, the same harness was run as an ARC
solver. The examples are given once; each turn the model rewrites a solve() function and receives no
correctness feedback, so nothing but its retained context tells it what it has tried. On an 8x8
ARC-AGI-2 training puzzle with three examples, neither policy solved any of sixty attempts, and the
anchored policy reused 87.4% of its prefill at 53,849 compute units against 35.7% and 392,390 for the
linear policy, a factor of 7.29 at identical accuracy. On a 3x3 puzzle both policies solved all sixty
attempts, at a factor of 2.14. Accuracy is identical at both difficulty extremes, and the cost
advantage is wider where context pressure is greater.

## 5. Why This Matters for ARC

Test-time search is a budget allocation problem: the number of candidates a solver can evaluate is its
time budget divided by the per-evaluation cost. Our measurement shows that cost is a function of the
context policy, not a constant: the same work costs 7.29 times more under a front-evicting policy in an
ARC solve loop, and 4.55 times more in a coding loop.

The bridge from measured cost to ARC search budget remains a hypothesis, but a narrower one, since the
measurement now runs inside an ARC solve loop. What is unmeasured is whether a production solver's
inner loop is context-bound the same way, since it may prune its own history. That is testable by
instrumenting an existing ARC harness's token accounting, the first experiment we would run next.

If it holds, it matters for the 85% goal: the frontier is compute-limited, and a factor of seven in
effective budget is a factor of seven in candidate programs, without changing the model.

## 6. Completeness and Limitations

The synthetic task makes retention measurable but is not a real repository, and the cost model uses a published hit multiplier rather than a billing
measurement. Repetition is across task instances, not sampled seeds, since greedy decoding would leave a
seed to fabricate variance. The ARC measurement uses training puzzles and one loop shape: it gives a
cost ratio on a real ARC task, not a solve rate, and the bridge to a search budget is argued, not
demonstrated. Our ARC accuracy is 0.0%.

## 7. Conclusion

Long-horizon reasoning is bounded by context economics. We measured, per turn, that a policy pinning an
immutable prefix and advancing its window keeps reuse at 76% and costs 4.55 times less than the
standard approach on a coding task, and 7.29 times less in an ARC solve loop, at identical accuracy
and with far better retention of early facts. We present the extension from
cheaper turns to a larger ARC search budget as a precise, falsifiable hypothesis rather than a result,
and we report our own submission's zero honestly. The contribution is a measurement the field
currently assumes: the cost of remembering is a policy choice, and the standard policy pays for it.

## References

Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180

Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024. arXiv:2312.07104

Gim, Chen, Lee, Sarda and Khandelwal. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. MLSys 2024. arXiv:2311.04934

Lumer. Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agents. arXiv:2601.06007

Liu, Chen and Wang. SinkTrack: Attention Sink based Context Anchoring for Large Language Models. ICLR 2026. arXiv:2604.10027

Xiao et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453

Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 2023. arXiv:2306.14048

Liu et al. Lost in the Middle: How Language Models Use Long Contexts. TACL 2024. arXiv:2307.03172
