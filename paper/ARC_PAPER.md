# Context Economics for ARC Test-Time Search

A measured cost model for long-horizon reasoning, applied to ARC-AGI-2 search budgets

Roni Saguey. ARC Prize 2026 Paper Track, ARC-AGI-2. Submission 56788985.

## Abstract

ARC-AGI-2 is not solved by a single forward pass. The leading approaches spend large amounts of
test-time compute searching over candidate programs or adapting a model to each task, and that search
is bounded by a GPU-time budget. A substantial fraction of the budget is spent re-processing context
that the serving engine has already seen. This paper measures, per turn, how a context retention
policy changes that cost. A policy that pins an immutable prefix and advances its window at the end
sustains an 81.3% prefix-cache reuse rate and costs 4.1 times less compute than a growing baseline
over a sixty-turn agentic task, while retaining every probe value, sixty of sixty against nine. We
argue that this expands the number of candidate programs an ARC-AGI-2 solver can evaluate within a
fixed budget, and we state that step as a testable hypothesis rather than a measured ARC result. Our
own ARC-AGI-2 submission is a fixed-library program-synthesis baseline that scores 0.0%, reported here
without embellishment; its purpose is participation and an honest control.

## 1. Introduction

ARC-AGI-2 measures fluid reasoning on novel visual tasks. Recent progress has come from systems that
spend compute at test time: test-time training and compression, and LLM-driven search over a
hand-built program library, of which the current leading entry scores 83.1%. Both families share a
property that is rarely measured. The search is long, and the context grows with it, so a solver that
evaluates hundreds of hypothesised programs accumulates the history of every attempt.

Serving engines address the resulting cost with automatic prefix caching, in which an input prefix
identical to the previous request reuses its key and value state. The economics are public and stark:
a cached token bills at roughly one tenth the price of a fresh one. A policy that rewrites the prefix
therefore re-bills the entire context at the full rate. Periodic compaction, the standard remedy for a
long context, rewrites the prefix by construction, so the standard remedy destroys the discount.

This paper makes that cost measurable and shows a policy that avoids it, on a real agentic workload
with per-turn instrumentation. The connection to ARC is argued explicitly in Section 5 and labelled a
hypothesis. The ARC results in Section 4 are our own actual submission, including its zero.

## 2. Prior Work

Approaches to ARC divide into test-time-training methods that adapt weights to each task,
program-synthesis methods that search a library of grid transformations, and LLM-harness methods that
let a model write and verify programs. Each runs a long inner loop. To our knowledge, none accounts
for the serving cost of that loop, because the cost is treated as a fixed price per token rather than
a function of which tokens survive.

Work on KV-cache eviction shrinks the cache by scoring tokens and dropping low-scoring ones, assuming
that survivors remain valid after eviction. A second line repairs positional integrity directly.
Neither is attempted here, and that restraint is deliberate: we never rewrite the region on which the
cache depends, which is precisely why reuse survives. Attention-sink work explains why pinning an
immutable front is mechanically sensible, since a small set of leading tokens attracts attention, but
sinks stabilise a stream rather than carrying specific information, which is why our probes test for a
particular fact rather than for fluency.

The closest prior work in intent is SinkTrack, which anchors a model to its initial context and
reports context forgetting during long generation. The mechanisms differ in kind: SinkTrack is a
model-level intervention on the beginning-of-sequence representation, evaluated on single-generation
question answering, whereas the policy here operates at the agent level and decides which tokens are
retained. A recent evaluation of prompt caching for agentic workloads measures caching across three
providers and finds cost reductions of 41 to 80%, but it works at the provider API level, compares no
retention policies, and does not measure the discontinuity a compaction introduces.

## 3. Approach

An agent's transcript is a sequence of turn blocks, each an instruction and a reply. Three policies
differ only in which blocks are retained when the prompt is assembled. The anchored policy keeps the
system prompt and the first-turn brief pinned, every past instruction verbatim, and the single newest
reply; instructions are small, while a reply here is a complete file rewrite that the newest reply
already supersedes. The linear policy keeps the first-turn block and as many recent pairs as fit,
dropping the oldest block on overflow. The prune policy is the linear policy plus a compaction every
thirty turns.

The structural difference is where eviction occurs. The linear and prune policies evict from the
front, so their first block changes as soon as eviction begins and the prefix shared with the previous
turn collapses to the system prompt alone. The anchored policy pins the front and advances the window
at the end, so that shared prefix survives. Nothing is summarised.

The task is a generated Python module seeded with sixty one-line faults, graded per turn in a fresh
interpreter immediately after each turn, over Gemma-4-12B in bfloat16 on one accelerator. Retention is
measured in two ways that recency cannot satisfy: an unguessable code handed over in each instruction
and forbidden from the file until the final turn, and a separate mid-session secret. Cost is computed
per turn as miss tokens plus hit tokens multiplied by a hit rate, from the longest common token prefix,
with one named constant for the multiplier.

## 4. Results

Our ARC-AGI-2 submission is a fixed-library program-synthesis baseline over grid transformations. It
scored 0.0%, zero of forty-seven, on a thirty-task evaluation sample. A control on the easier
ARC-AGI-1 training set scored non-zero, confirming that the solver and the scorer execute; this is a
harness validation, not a claim of capability. The result is expected, since a fixed library cannot
express ARC-AGI-2, which is constructed to defeat it.

The cost measurement covers sixty turns. The anchored policy reuses 81.3% of its prefill against 30.8%
for the linear policy, at 118,817 compute units against 484,956, a factor of 4.08. Coding accuracy is
identical between the two, at sixty of sixty, so the difference lies in what each retains rather than
in what the model can do. The anchored policy also recovers sixty of sixty probe values in order,
against nine of sixty unordered for the baseline. The third policy, prune, behaves as the linear
policy does and shows the compaction directly: at turn thirty the reused prefix collapses from 3,226
tokens to 2,057, which is the anchored prefix alone. Compaction pays a re-prefill at every boundary
rather than recovering the cost it is introduced to save.

## 5. Why This Matters for ARC

Test-time search is a budget allocation problem. If a solver has a fixed time budget and each
candidate program evaluation costs some amount in re-processing, then the number of candidates it can
evaluate is that budget divided by that cost. Our measurement shows the cost is a function of the
context policy and not a constant, since the same work costs 4.08 times more under a front-evicting
policy. Reducing it by that factor raises the number of candidate programs evaluable within the budget
by the same factor.

This is the paper's central claim about ARC, and it is a hypothesis rather than a measured ARC result.
We have measured the cost ratio on a long-horizon agentic coding task with per-turn instrumentation.
We have not run it inside an ARC solver. The step from a 4.08-times cheaper turn to 4.08 times more
candidates searched assumes that an ARC loop is context-bound in the same way. That assumption is
testable by instrumenting the token accounting of an existing ARC harness across its search, which is
the first experiment we would run next. It is stated here in its falsifiable form.

If it holds, it matters for the 85% goal, because the current frontier is compute-limited and a factor
of four in effective search budget is a factor of four in candidate programs, without changing the
model.

## 6. Completeness and Limitations

The evaluation covers one model, one task family, and one seed per arm. The synthetic task makes
retention measurable but is not a real repository, and the cost model uses a published hit multiplier
rather than a billing measurement. Repetition is across task instances rather than sampled seeds,
because decoding is greedy and a seed would fabricate variance. The bridge from measured cost to ARC
search budget is argued and not demonstrated. Our ARC accuracy is 0.0%.

## 7. Conclusion

Long-horizon reasoning is bounded by context economics. We measured, per turn, that a policy which
pins an immutable prefix and advances its window keeps prefix-cache reuse at 81% and costs 4.1 times
less than the standard approach, at identical task accuracy and with far better retention of early
facts. We present the extension to ARC-AGI-2 test-time search as a precise, falsifiable hypothesis
rather than a result, and we report our own submission's zero honestly. The contribution is a
measurement the field currently assumes: the cost of remembering is a policy choice, and the standard
policy pays for it.

## References

Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180

Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024. arXiv:2312.07104

Gim, Chen, Lee, Sarda and Khandelwal. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. MLSys 2024. arXiv:2311.04934

Lumer. Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agents. arXiv:2601.06007

Liu, Chen and Wang. SinkTrack: Attention Sink based Context Anchoring for Large Language Models. ICLR 2026. arXiv:2604.10027

Xiao et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453

Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 2023. arXiv:2306.14048

Liu et al. Lost in the Middle: How Language Models Use Long Contexts. TACL 2024. arXiv:2307.03172
