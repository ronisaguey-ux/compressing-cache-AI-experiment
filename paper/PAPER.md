# Bounded Context Without Context Rewrites

Roni Saguey

## Abstract

Long-horizon agents are limited by the economics of context, not by model capability. The standard
remedy, periodic compaction, keeps the window small by rewriting it, and a rewrite invalidates the
shared prefix a serving engine's prefix cache depends on, so the prompt is billed again in full. We
describe a retention policy that bounds the working set without rewriting the front: the system prompt
and the first-turn brief are pinned, and the window advances at the end. Over sixty turns of a
bug-fixing task on a 12B model we measure retention, compute cost, and prefix-cache reuse. The
anchored policy recovers every per-turn secret, sixty of sixty against nine, at 4.1 times less compute
and identical coding accuracy. We also document four ways such an evaluation can produce a
clean-looking table that measures nothing.

## 1. Introduction

An agent coherent across hundreds of turns accumulates history, and with it two costs: the key and
value cache footprint, and the compute spent re-processing the context each turn. The dominant response
is compaction, which summarises or discards the middle of the conversation to stay below the window.
Compaction reduces size but changes content, and this matters beyond the agent. Serving engines keep an
automatic prefix cache, in which a prefix identical to the previous request reuses its key and value
state, so a cached token costs a fraction of a fresh one. Rewriting the prefix re-bills the whole
context at full rate.

The constraint is concrete in the setting motivating this work. The Gemma 4 Developer Agent harness
compacts history at 14,336 tokens, 2,048 below the output ceiling it also requests, and acts on the
prompt just sent rather than prompt plus output, so it is one call late. The scorer refuses any call
past 32,768 tokens, and with a LoRA adapter the KV cache on four L4s falls from roughly 46,000 to
7,600. Context capacity, not reasoning, is binding.

We therefore ask a narrow, measurable question: can an agent keep a small context without rewriting
it? Our policy pins an immutable front, the system prompt and the first-turn brief, and advances a
bounded window at the end. Nothing is summarised: the informational part of each turn is kept and the
superseded part dropped. The contributions are a retention policy that bounds the working set while
leaving the shared prefix intact; a per-turn measurement of prefix-cache reuse, treating cost as a
function of retention rather than a constant; an evaluation using per-turn grading and probes recency
cannot satisfy; and four failure modes by which such an evaluation measures nothing.

## 2. Related Work

Work on KV-cache eviction reduces memory by discarding low-scoring tokens. H2O [1], Scissorhands [2],
TOVA [3], and SnapKV [4] assume survivors stay valid after eviction, which Section 6 contradicts. A
second line repairs positional integrity: CacheBlend [5], CacheFocus [6], DSCache [7] and CacheGen [8].
We do none of this; never rewriting the region the cache depends on is what lets reuse survive.

Attention-sink work explains why pinning a front is sensible. StreamingLLM [9] retains sink tokens
with a sliding window, and the sink mechanism is characterised in [10]. Sinks stabilise a stream but
are not a channel for specific information [11], which is why our probes test a particular fact rather
than fluency.

Prefix caching is a serving-engine feature in paged-attention systems [21, 22] and modular attention
reuse [23]. The closest work is a recent evaluation of prompt caching for agent workloads [24], which measures
caching across three providers on a multi-turn web-search benchmark and finds cost reductions of 41 to
80 percent. It operates at the provider level and its strategies concern where dynamic content is
placed; it compares no retention policies and does not measure the discontinuity a compaction
introduces. We measure, per turn from the token stream, what a retention policy does to prefix reuse
and to the retention of early facts, which prices the rewrite itself and the recall lost alongside it.

The closest prior work in intent is SinkTrack [20], which anchors a model to its initial context and
identifies context forgetting as a real failure of long generation. The mechanisms differ in kind:
SinkTrack intervenes on the beginning-of-sequence representation and is evaluated on single-generation
question answering, whereas ours decides at the agent level which tokens are retained and measures the
serving-cost consequence over a long horizon.

Work on agent memory addresses what to store. MemGPT [12] pages context operating-system style, and a
survey [13] taxonomises memory operations; both concern storage, not the cost of rewriting it.
Lost in the Middle [14] and Found in the Middle [15] show that where information occurs changes how
it is used. Gemma-4 makes this concrete: forty of its forty-eight layers use a 1024-token local window,
so a first-turn fact is unavailable to them regardless of policy; Section 6 states the caveat.

Our own benchmark produced one result we traced to a defect in the task rather than the policy, and we
report the corrected version. A position paper argues for a refutations track [16]; the literature documents leakage and reporting
error [17], and comparisons tend to favour the proposed method [18], the bias two of our failure modes
produce. We follow artifact-review expectations [19].

## 3. Method

### 3.1 Retention policies

Let the transcript be turn blocks, each appending one instruction and one reply. The three policies
differ only in which blocks are retained when the prompt is assembled.

The anchored policy retains the first-turn block, every instruction so far verbatim, and the single
newest reply. Instructions are small, whereas a reply here is a complete file rewrite that the newest
reply supersedes, so if the budget is exceeded the oldest instructions drop first and the front is
never rewritten. The linear policy retains the first-turn block and as many recent pairs as fit the
ceiling, dropping the oldest block on overflow: a fixed window with no organisation, and the naive
baseline. The prune policy is linear plus a compaction every thirty turns retaining the anchor and the
most recent turns, the status quo for long-running agents. All three share model, prompts, tools, turn
budget, and first-turn anchor, so a difference is attributable to retention; prune is anchored
deliberately, since otherwise it would differ from linear in two ways at once.

### 3.2 Predictions

The claim is that stability, not size, is what a long-horizon agent needs. Two predictions follow, since
retention and cost are measured independently. Retention: a fact stated early and required at
the end survives under the anchored policy and not under the evicting ones, whose reach is a fixed
number of turns however large the window. Cost: the anchored policy reuses a larger prefix, never
rewriting the front the previous turn paid for, whereas prune re-prefills at each compaction. A cost
difference alone would not support the claim, so Section 5 reports them separately.

We registered a mechanism prediction before the run. We credit the instruction archive, the only home
of each turn's code, since a file rewrite supersedes every earlier reply. We therefore run an ablation
that is the anchored policy minus the archive and identical otherwise. If the archive is the mechanism,
recall collapses to the few instructions that fit beside the brief while per-turn success stays high.
If the ablated policy retains all sixty, the archive is not the mechanism and our explanation is wrong,
which we would report as such.

### 3.3 Task

A Python module is seeded with faulty functions. The first turn carries the complete bug report,
giving for each bug the exact wrong and required constant; every later turn says only that a particular
bug should be fixed. The list is never repeated, so a later turn depends on retaining the first. Four
properties are deliberate. Each bug family has the same shape, a one-line body with one wrong constant;
an earlier version used a keyword default and a table lookup, both of which invited a rewrite and then
failed with the brief one turn back, a task defect rather than a retention result. The required
constants are arbitrary and appear nowhere else, so they cannot be re-derived. Grading is per turn:
bug k is tested immediately after turn k in a subprocess, so a fix applied and later clobbered is
distinguishable from one never applied. Repetition is across task instances rather than sampled seeds,
because decoding is greedy and a seed would fabricate variance; a task salt re-derives every constant
into a new instance of the same broken shape, a different task with the same probes.

### 3.4 Retention probes

Two probes measure retention. The first is accumulated codes: one fresh unguessable code per turn,
forbidden from the file until the final turn, when it must be returned as an ordered list, scored from
zero to n so partial retention appears as a prefix gap. The second is a release gate: on the final
turn an additional function must return a separate secret handed over at the session midpoint. It is deliberately not one of the codes, so a policy can recite the list and still ship a broken
artifact; we verified independence on two simulated agents with identical bug and code scores.

The policies diverge in growth rate by roughly an order of magnitude. Linear adds about 1,248 tokens
per turn, a full pair, while the anchored policy adds about 159. Against a 12,000-token ceiling linear
reaches the limit near turn nine and evicts thereafter, while the anchored policy holds every
instruction and finishes near 8,149 tokens. This was computed against the real tokenizer before the
run and observed during it.

One probe failed to discriminate, and we report that as a negative result. We predicted linear would
fail the release gate; it passed, as did the anchored policy. The gate's value is a function the
artifact already contains, so committing the secret when it is handed over carries it through every
later rewrite, and the gate measures artifact persistence rather than retention. The accumulated codes
have no such escape, because they are forbidden from the file until the final turn, which is why the
codes probe separates the arms and the gate does not.

## 4. Measurement

### 4.1 Prefix-cache reuse

Each turn we record the longest common token prefix between the current prompt and the previous one,
exactly what an automatic prefix cache would reuse. Cost is in raw compute units: a fresh token costs
1.0 and a reused token an assumed multiplier of 0.1, so a turn costs the miss count plus the hit count
times the multiplier, against a no-cache baseline where every token costs 1.0. The multiplier is one
named constant, so the assumption is visible and re-runnable. We compare on total
cost rather than hit rate alone, because cost separates two policies only if one both reuses its
prefix and keeps the prompt small; a high hit rate on an unbounded prompt still processes far more
tokens.

### 4.2 Metrics

We record retention (per-turn success, final accuracy, fixes applied then lost, code recall, gate
verdict), cost (hit rate, miss tokens, compute units, no-cache baseline, saving ratio), context (first,
last and peak prompt size), and latency (time to first token, wall time per turn, decode throughput,
KV-cache peak). All derive from data the loop already collects.

### 4.3 Fairness

The stopping rule is the turn count, not wall-clock time, which is load-bearing: a policy with a
smaller context is faster per turn by construction, so a time-based stop would let it complete more
turns and win on volume rather than policy. Time remains only as an emergency break, so an interrupted
run grades what it finished. Because that break can fire at different turn counts, the comparison
script reconciles to the common prefix, re-summing cost, context, and latency over the turns both
policies completed. One asymmetry is deliberate: retention counts are not reconstructed, because the
artifact's state at turn n cannot be recovered from a later snapshot, so they describe only the turns
reached.

## 5. Results

Four policies ran sixty comparable turns under an identical prompt and tool budget. Figure 1 shows
growth and reuse per turn; in Table 1 a reused token is priced at one tenth of a fresh one.

Table 1. Compute cost over sixty turns.

| policy | hit rate | miss tokens | cost units | no-cache cost | saving | prompt first to last | TTFT first to last (ms) |
|---|---|---|---|---|---|---|---|
| anchored | 76.0% | 81,067 | 106,678 | 337,173 | 3.16x | 2,057 to 8,149 | 1,943.1 to 4,245.4 |
| linear | 30.8% | 464,305 | 484,957 | 670,822 | 1.38x | 2,057 to 12,142 | 1,795.1 to 7,370.6 |
| prune | 31.0% | 458,050 | 478,585 | 663,398 | 1.39x | 2,057 to 12,142 | 1,769.0 to 7,347.5 |
| ablate-recent | 25.4% | 412,926 | 426,970 | 553,365 | 1.30x | 2,063 to 9,746 | 2,499.1 to 5,719.9 |

Table 2. Retention. Per-turn success is graded after each turn, code recall is the accumulated
list, and the release gate is the independent mid-session secret.

| policy | per-turn success | final state | applied then lost | code recall | ordered | release gate |
|---|---|---|---|---|---|---|
| anchored | 1.000 | 1.000 | 0 | 1.000 | yes | pass |
| linear | 1.000 | 1.000 | 0 | 0.150 | no | pass |
| prune | 1.000 | 1.000 | 0 | 0.150 | no | pass |
| ablate-recent | not measured | 1.000 | not measured | 0.133 | no | pass |

The anchored policy reuses a prefix for 76 percent of its prefill against 31 percent for prune, at
106,678 compute units against 478,585, a factor of 4.49. Coding accuracy is identical at sixty of
sixty, so the difference lies in what each retains, not in what the model can do. The separation is
structural: a policy evicting from the front changes its first block once eviction begins, while one
pinning the front and advancing the window at the end keeps it.

Prune behaves as linear does, and its turn-thirty compaction appears as a collapse of the reused
prefix from 3,226 tokens to 2,057, the anchored prefix alone, so compaction pays a re-prefill at
every boundary rather than recovering that cost.

On SWE-bench Lite, six sympy issues were attempted by both policies over Gemma-4-12B. Neither
resolved any, and a gold control resolved all six, so the zero is a capability limit of a 12B model
rather than a harness failure. The policies separate on cost and boundedness: the linear policy's
prompt grows to the 12,000-token cap and its time to first token grows 14.8 times, from 588 to 8,692
milliseconds, while the anchored policy stays bounded and grows 1.7 times, from 772 to 1,334
milliseconds. On the final turn the anchored policy is 6.5 times faster on identical work, with a
KV-cache peak of 409 against 533 megabytes.

The ablation isolates the archive as the mechanism. With the archive removed, code recall collapses
from 1.000 to 0.133, eight of sixty, while per-turn success stays high at 0.917 and the final artifact
reaches fifty-five of sixty. The archive is not needed to keep the file correct turn by turn, because
each turn's newest rewrite carries the file state; what is lost is the ability to reproduce a value
stated once early, which is what the probe measures. A prior run reached sixty of sixty at the end, so
final state varies across runs while the recall collapse does not.

## 6. Failure Modes of the Evaluation

Each of the following produced, or would have produced, a clean-looking table that measured nothing.

A shared ceiling. Two policies capped at the same limit converge on whatever fits: with one shared
cap both reached about 9,900 tokens at nearly identical latency, so the cap, not the policy, set the
working set. Each budget is now sized to its own claim.

A re-derivable probe. A rule the model re-applies each turn can be inferred from recent turns, so
both policies pass and the probe is vacuous. Only a value stated once and never repeated discriminates.

A periodic task defect. Two of six bug families failed on every instance from the second turn, one
turn after the brief, so memory was not in question. A turn-indexed task must be checked for periodic
failure before its losses count.

An architectural confound. Gemma-4 uses local attention in forty of forty-eight layers with a
1024-token window, so beyond it a first-turn fact is unavailable to those layers, and the KV-cache peak
is not a retention result for the same reason.

## 7. Limitations

The evaluation covers one model and one task family, and the synthetic task is designed so retention
is measurable. The cost model uses an assumed hit multiplier rather than a measured billing figure, so
we swept it: across multipliers from 0.05 to 1.00 the anchored policy stays cheaper by 4.71 to 1.52
times, and at 1.00 it is still 1.52 times cheaper because it processes fewer tokens. Runs repeated
across task instances rather than seeds, since greedy decoding would leave a seed to fabricate variance. The gate probe is a single value, establishing that
one mid-session fact survived rather than measuring a rate, and the six-issue SWE-bench sample is small.

The reported run uses Gemma-4-12B in bfloat16, which loads on one accelerator without a quantization
confound; the competition checkpoint is 4-bit and would add a quantization variable. The policy sits outside the model, so a different checkpoint changes the numbers, not the mechanism.

## 8. Conclusion

Keeping a context small and keeping it stable are different problems, and the usual solution to the
first destroys the second. Pinning an immutable front and rolling the window at the end bounds the
working set without invalidating the prefix already paid for. We have made that claim measurable
rather than asserted, and shown how such an evaluation silently fails.

## References

[1] Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 2023. arXiv:2306.14048
[2] Liu et al. Scissorhands: Exploiting the Persistence of Importance Hypothesis. NeurIPS 2023. arXiv:2305.17118
[3] Oren et al. Transformers are Multi-State RNNs. arXiv:2401.06104
[4] Li et al. SnapKV: LLM Knows What You Are Looking For Before Generation. NeurIPS 2024. arXiv:2404.14469
[5] Yao et al. CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion. EuroSys 2025. arXiv:2405.16444
[6] Lee, Park and Han. CacheFocus: Dynamic Cache Re-Positioning for Efficient RAG. arXiv:2502.11101
[7] Pang et al. Decouple and Cache: KV Cache Construction for Streaming Video. arXiv:2605.01858
[8] Liu et al. CacheGen: KV Cache Compression and Streaming for Fast LLM Serving. ACM SIGCOMM 2024. arXiv:2310.07240
[9] Xiao et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024. arXiv:2309.17453
[10] Gu et al. When Attention Sink Emerges in Language Models. arXiv:2410.10781
[11] Cao, Zhang and Tang. Separating Stream Stability from Long-Term Recall in Language Models. arXiv:2609.07282
[12] Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560
[13] Zhang et al. A Survey on the Memory Mechanism of LLM-based Agents. arXiv:2404.13501
[14] Liu et al. Lost in the Middle: How Language Models Use Long Contexts. TACL 2024. arXiv:2307.03172
[15] Hsieh, Chuang et al. Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization. arXiv:2406.16008
[16] Schaeffer, Kazdan, Denisov-Blanch, Miranda, Gerstgrasser et al. Position: Machine Learning Conferences Should Establish a Refutations and Critiques Track. NeurIPS 2025 Position Paper Track. arXiv:2506.19882
[17] Kapoor and Narayanan. Leakage and the reproducibility crisis in ML-based science. Patterns 2023.
[18] Herrmann et al. Why We Must Rethink Empirical Research in Machine Learning. ICML 2024.
[19] ACM. Artifact Review and Badging, v1.1. https://www.acm.org/publications/policies/artifact-review-and-badging-current
[20] Liu, Chen and Wang. SinkTrack: Attention Sink based Context Anchoring for Large Language Models. ICLR 2026. arXiv:2604.10027
[21] Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180
[22] Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS 2024. arXiv:2312.07104
[23] Gim, Chen, Lee, Sarda and Khandelwal. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. MLSys 2024. arXiv:2311.04934
[24] Lumer. Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agents. arXiv:2601.06007
