# Deep-Research Brief — Saliency-Preserving KV Compaction with Dense Delta Re-rotation

You are a research agent with web search and GitHub access. Your job is to audit one
repository, then go find everything on the open internet that bears on the problem it
attacks, then generate theories and improvements. You are not being asked to write code.
You are being asked to think hard, cite sources, and produce material a working engineer
can act on tomorrow.

Be specific. "Consider a better eviction policy" is worthless. "Kamera (arXiv 2606.23581)
attributes multi-hop degradation to *conditioning loss* rather than token eviction, and
measures 0.41→0.28 on MLA, 0.28→0.15 on GQA; here is the mechanism, here is how it would be
measured against this harness, here is what would falsify it" is what is wanted.

---

## 1. What this project is

**Repository:** `github.com/ronisaguey-ux/compressing-cache-AI-experiment`

A CPU-only research harness testing whether a KV cache can be compacted to ~60% of its
tokens while preserving the model's ability to answer a question whose answer sat in the
evicted region. The distinguishing idea is **dense delta re-rotation**: when tokens are
removed from the middle of a sequence, every surviving token's position changes, and its
rotary positional embedding (RoPE) no longer matches the position it is stored at. The
harness re-applies the exact rotation delta `R_Δ` to the surviving keys so that the cached
key's geometry matches the compacted sequence.

**The claim under test:** eviction alone destroys retrieval; eviction *plus* correct
re-rotation preserves it. Measured effect of the rotation step is +2.019 nats where it
matters, and without it the model recites the system prompt or refuses (Finding 2).

**Prior art that already does something similar — find more, deeper, and connect them:**
- **LazyAttention** (ICML 2026, UIUC) — `github.com/illinoisdata/lazy-attention`,
  arXiv 2606.04302. Caches K *unrotated* at local position 0 and applies `R_Δ` on-chip.
  Reports 1.37× TTFT, 1.40× throughput, up to 7.5× hit ratio.
- **SGLang RFC #30928** "Position-Independent KV Cache Reuse for Agentic/RAG Workloads" —
  open, core maintainer involved.
- **Kamera** (arXiv 2606.23581) — argue degradation is *conditioning loss*, not token loss.
- Others to chase: KVBoost, Block-Attention, PromptCache, FlowBlock, PagedAttention,
  H2O, SnapKV, PyramidKV, Quest, MInference, StreamingLLM, TOVA, CAKE, RazorAttention,
  DuoAttention, and anything published in the last 12 months on "position-independent",
  "cache reuse", or "RoPE rotation".

**Hard constraint:** everything runs on a 15.7 GB CPU-only laptop (Intel i7-8550U, 8
threads, no CUDA). The 7B model is loaded at 8-bit. Any suggestion that assumes a GPU must
be flagged as such.

---

## 2. The experiment as it stands

A single prompt is built from two chunks:

- **chunk 0** (32 tokens) — system message, containing `SYSTEM_FLAG_A = ALPHA_VERIFIED`.
  Never evicted. This is the control needle.
- **chunk 1** (~900–2200 tokens depending on depth) — a synthetic build log. A single
  needle, `0x9AF4_STACK_FAIL`, is planted at a controllable position inside it.

Two questions are asked after compaction:
- `Q1` — "What is SYSTEM_FLAG_A?" (never evicted; should always survive)
- `Q2` — "What was the failure code in the build log?" (the needle is in chunk 1)

A **depth sweep** moves the needle to 15% / 35% / 55% / 75% through the log. Three
selection modes are compared: `baseline` (all-layer self-attention, top-k), `posnorm`
(baseline ÷ causal horizon), and `line` (baseline + keep whole `\n`-delimited lines).

Two models: `Qwen/Qwen2.5-0.5B` (fp32, 24 layers) and `Qwen/Qwen2.5-7B` (8-bit, 28
layers, GQA 7:1, `rope_theta` 1e6).

**Files worth reading first:** `README.md`, `RESULTS.md`, `docs/RESEARCH.md`,
`src/test_2d_kv_cache.py`, `src/depth_sweep.py`, `src/sink_probe.py`, `src/heavy_lock.py`,
and the raw `data/*.jsonl`.

---

## 3. Findings so far — the state of knowledge

These are measured, reproducible, and several are counter-intuitive. Treat them as the
starting line, not as settled truth. If you find a flaw in one, say so loudly.

| # | Finding |
|---|---|
| 1 | A 10% keep budget is below the retention floor for **every** selector — the needle is never retained. |
| 2 | Dense delta re-rotation is the highest-leverage component (+2.019 nats). Without it, the answer degrades to reciting the system prompt. |
| 3 | Only **query-attention** selectors produce an answer containing the needle. Key-norm and random do not. |
| 4 | Text-squashing (rewriting evicted text into a summary) answers Q1, loses Q2, and costs a full re-prefill (~48 GFLOP). |
| 5 | A query-**agnostic** selector works: chunk-1 self-attention, at 75% retention. |
| 6 | The retention floor is **59.6%**, not 75% — corrected by measuring a *rank* rather than a pass/fail. At 60% keep: **36.6% KV saved**, needle retrieved, +0.051 nats. |
| 7 | Combining signals (adding key-norm to self-attention) makes selection **worse**. The best-looking rank at one depth was a **one-needle artefact**. |
| 8 | The recipe is **not robust to needle position** — on the 0.5B it passed 1 of 4 depths. |
| 9 | **Selection is not the bottleneck.** The needle was retained in 15 of 16 combinations and the answer still failed. `keep=90%` failed at depth 35% where `keep=75%` passed — non-monotonic. |
| 10 | 8-bit quantisation works on CPU (bitsandbytes 0.50.2, no CUDA) but **changes the verdict** at some depths. Every 7B number must state precision. |
| 11 | A silent RoPE fallback (`theta=10000` instead of `1e6`) would have invalidated every rotation number, and **neither** existing control could catch it (`R(0)=I` for any theta). Fixed by asserting against the model's own `inv_freq`. |
| 12 | **The depth wall has two causes, and scaling only fixes one.** |

### Finding 12 in detail — this is the crux

| mode | depth 15% | depth 35% | depth 55% | depth 75% |
|---|---|---|---|---|
| **0.5B fp32** base / line / posnorm | F/P/F | F/F/P | P/F/F | F/P/F |
| **7B 8-bit** base / line / posnorm | F/**P**/F | **P/P/P** | **P/P**/F | not measured |

- **Middle depths are capacity-limited and 7B fixes them.** 35% goes from 1-of-3 modes
  passing to 3-of-3. Overall 5/16 → 6/9.
- **The shallow position (15%) is bit-for-bit identical at both scales**: baseline fails,
  posnorm fails, **line passes**. A failure that survives a 14× parameter increase is a
  mechanism, not a capacity limit.
- **Line pooling is the only mode that passes every measured depth at either scale.**
- 75% could not be measured at 7B: it peaks near 10.5 GB RSS and the host watchdog
  pauses any process that pushes system RAM past 85%, deadlocking rather than failing.

---

## 4. A fresh measurement that complicates the leading theory

The working hypothesis for the 15% failure was **attention-sink displacement**: that
~50%+ of attention mass lands on the first few tokens, and at shallow depth the needle sits
downstream of that basin and is starved.

`src/sink_probe.py` measured it directly (query-agnostic: every chunk-1 position is a
query, causal softmax over all prefix keys, 0.5B fp32):

| depth | total tok | sink mass (keys 0-8) | needle mass | top-1 position |
|---|---|---|---|---|
| 0.15 | 938 | **0.5734** | 0.001415 | **0** |
| 0.35 | 1460 | 0.5517 | 0.000656 | **0** |
| 0.55 | 1992 | 0.5360 | 0.000613 | **0** |
| 0.75 | 2221 | 0.5312 | 0.000426 | **0** |

**What this supports:** the sink is enormous and content-independent — position 0 is the
single highest-attention key at *every* depth, and 8 tokens absorb ~54% of all attention
(~6.7–7.2% per token, versus ~0.015–0.035% per needle token: a 200–700× disparity).

**What this contradicts:** the needle receives **more** attention at the depth where it
*fails* (0.001415 at 15%) than where it *passes* (0.000613 at 55%, 0.000426 at 75%). So
"the needle is starved of attention at shallow depth" is **not** what the data shows. Raw
attention mass to the needle does not predict success — which independently reinforces
Finding 9.

That is an open contradiction and it is the most interesting thing in the project right
now. Resolve it.

---

## 5. What to do

### A. Audit the repository

Read it properly and report:

1. **Methodological flaws** in the harness. Every result above depends on this code. Where
   is the measurement self-confirming? Where is a control passing for a degenerate reason
   (Finding 11 was exactly this)? Where does a result come from a single needle, a single
   prompt, a single seed, or a single layer?
2. **Claims not supported by the data on disk.** Cross-check `RESULTS.md` against
   `data/*.jsonl`. Report any number that does not reproduce.
3. **The query-agnostic vs query-oracle distinction.** Arm 2b/2c use the question to score
   saliency — that is an oracle, not deployable. Is the *shippable* selector
   (self-attention, query-agnostic) actually sound, or does its apparent success ride on
   the control needle in chunk 0 never being evicted?
4. **Reproducibility.** Can a stranger reproduce Finding 6's 36.6% figure from a clean
   clone? What is missing?

### B. Research the literature and the code

Search GitHub, arXiv, HuggingFace, and the SGLang/vLLM/LMDeploy trackers. For each thing
you find, give: what it does, how it differs from dense delta re-rotation, whether it is
implementable on a CPU-only box, and a link.

Specifically hunt for:

- **Anyone already doing RoPE delta re-rotation on cached keys.** LazyAttention is one; is
  it the only one? What about position-interpolation techniques (NTK, YaRN, LongRoPE)
  applied to caches rather than training? Is the "unrotate → store → rerotate" pattern
  present anywhere else?
- **Conditioning loss vs token loss** (Kamera's thesis). If the KV cache is conditioned on
  absolute positions, compaction breaks conditioning regardless of which tokens survive.
  Is there a published fix? Does anyone re-condition via a learned module?
- **Eviction policies that beat attention-sum.** H2O, SnapKV, PyramidKV, Quest, TOVA,
  CAKE, DuoAttention, RazorAttention, and any 2026 follow-ups. Which have open CPU-capable
  implementations?
- **Attention sinks.** StreamingLLM's finding, "massive activations", register tokens,
  attention-sink-as-bias. Is there a *known* fix for shallow-context retrieval failure
  near the sink basin? Is the phenomenon even the one we think it is?
- **GQA effects.** Kamera reports different degradation for GQA (0.28→0.15) than MLA
  (0.41→0.28). Qwen2.5-7B is GQA 7:1. Does a 7:1 grouping change what a "salient key"
  even means, given 7 query heads share one KV head?
- **Chunked prefill / paged attention** on CPU: does `llama.cpp` or `vLLM` (CPU backend)
  expose anything that would help the memory ceiling that blocked the 75% run?

### C. Generate theories

Give me **at least 15 distinct hypotheses** for why shallow-depth retrieval fails, and for
each: the mechanism, the prediction that distinguishes it from the others, and the cheapest
experiment that would falsify it *on this harness*. Include the owner's three:

1. Attention-sink displacement (partly contradicted above — repair or kill it).
2. RoPE high-frequency phase clash over short relative offsets.
3. Line pooling works because it averages out high-frequency phase noise and sink dilution.

Add your own. Some directions to consider, not a list to be limited by:

- Absolute positional conditioning inside the KV cache itself
- Whether the failure is in *retrieval* or in *generation* (a rank that recovers the needle
  but a decode that cannot use it)
- Layer-wise locality: is the failure localised to specific layers, and does
  `posnorm`'s failure at 55% share a signature with `baseline`'s at 15%?
- The 0.5B's non-monotonicity (Finding 9) — is it genuinely model noise, or is there a
  real alternating structure to which depths pass?
- Softmax temperature / entropy at the needle position
- Whether the *question* formulation matters (the model may need a retrieval-shaped prompt)
- Whether `keep` budgets interact with the *number* of distractors, not just position
- Anything about how 8-bit de-quantisation perturbs the rotation arithmetic specifically

### D. Propose improvements

Concrete, ranked, with expected effect and cost:

- **Selector improvements** that are genuinely query-agnostic and deployable.
- **Better rotation handling** — is per-token `R_Δ` correct, or does the *value* cache
  need re-conditioning too? Does re-rotation interact badly with 8-bit de-quantisation?
- **An architecture or conditioning fix** that removes the depth dependence rather than
  papering over it.
- **Cheaper experiments** that would raise confidence per CPU-hour. The current cost is
  ~15 min per 7B combination; anything that reduces that is valuable.
- **What to test next, in order**, assuming a CPU-only box and limited patience.

---

## 6. Deliverable format

Return one structured document with these sections:

1. **Repo audit** — flaws, unsupported claims, reproducibility gaps. Cite `file:line`.
2. **Prior art table** — name, link, what it does, how it differs, CPU-viability.
3. **Contradiction resolution** — the sink finding above. Your best explanation, with the
   experiment that decides it.
4. **≥15 hypotheses**, each with mechanism / distinguishing prediction / falsifying test.
5. **Ranked improvements**, each with expected effect, cost, and risk.
6. **The single next experiment you would run**, and why that one first.
7. **Everything you could not determine**, stated plainly. Do not fill a gap with a
   guess. A named unknown is useful; a confident invention is worse than nothing.

**Rules for the whole document:** cite sources with links. Never invent a paper, an arXiv
id, a repo, or a benchmark number — if you cannot verify it, say "unverified". Distinguish
sharply between what you read and what you concluded. When you disagree with a finding
above, say so directly and show the reasoning.
