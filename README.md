# compressing-cache-AI-experiment

Can a volatile middle region be evicted from a KV cache without destroying what the model
knows about the content in it?

**Short answer from the measurements: yes — but not by re-rotating keys.** Compaction to 60%
retention preserves a mid-context fact at every needle position tested, on a 0.5B and a 7B,
**provided the retained tokens are recomputed rather than re-rotated.** Key-only rotation is
insufficient: it produces tensors at cosine 0.75–0.87 from the ones a correct forward pass
builds, and that is enough to lose the answer.

Read [`RESULTS.md`](RESULTS.md) for the numbers. The corrected method is Findings 18–20; the
rotation study is Findings 1–17 and several of them are **superseded** — see the correction
index at the end of this file.

## The method

```
1. prefill the full sequence, capturing per-token self-attention
2. score the evictable region by the attention each token receives
3. keep the top 60%
4. build the shortened sequence and run a FRESH FORWARD PASS over it
5. decode against that
```

KV memory drops 34–37%. Retrieval holds at every depth measured. There is no rotation step, no
per-layer band, no line pooling, and no conditioning patch — all four were tried, and all four
turned out to be unnecessary once step 4 was done correctly.

## What this costs

The recompute is the price. For a single request the sequence is prefilled twice — once to
score, once compacted — in exchange for a smaller decode-time cache. That is worth it for long
generations and not for short ones. For prefix-cache reuse, where the point is to avoid a
recompute at all, this method does not apply: avoiding the recompute is exactly what the
rotation approach was for, and it does not work.

## Run it

```bash
python src/native_vs_cache.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.55
python src/selectors_corrected.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
```

Needs `torch` and `transformers`. Downloads `Qwen/Qwen2.5-0.5B` on first run (~1 GB); the 7B
runs are 8-bit and need ~10 GB. CPU only, no CUDA, no vLLM, no Triton — deliberately standalone.

| script | what it establishes |
|---|---|
| `src/native_vs_cache.py` | the decisive control: same retained tokens, native vs cache path |
| `src/selectors_corrected.py` | selector comparison **against the corrected pipeline** |
| `src/cache_vs_native.py` | how far the cache tensors drift from a correct forward |
| `src/native_7b.py` | the 7B confirmation, three failure modes separated |
| `src/sink_probe.py` | where attention mass actually goes |
| `src/decode_divergence.py` | first token where a compacted decode diverges from full |
| `src/depth_sweep.py` | the original depth sweep, rotation pipeline |

## The result, in one table

Selector comparison, identical rows, identical budget, **corrected pipeline**:

| depth | baseline | line pooling | mid-layer band | random | old rotation path |
|---|---|---|---|---|---|
| 15% | PASS | PASS | PASS | fail | fail |
| 35% | PASS | PASS | PASS | fail | PASS |
| 55% | PASS | PASS | PASS | fail | fail |
| 75% | PASS | PASS | PASS | PASS | PASS |
| | **4/4** | **4/4** | **4/4** | **1/4** | **2/4** |

**The selector is not the lever. The pipeline is.** A plain attention sum is sufficient; every
selector refinement this repository tried was compensating for a corrupted cache.

## Corrections

Several earlier findings are superseded. They are kept in `RESULTS.md` with their retractions
rather than deleted, because the way each one went wrong is the most transferable part of this
work.

| finding | status |
|---|---|
| 1, 2 | stand — the retention floor and the rotation effect are real measurements |
| 3, 4 | stand |
| 5, 6 | **superseded** — the "query-agnostic selector works at 75%" claim is about a pipeline that does not work |
| 7, 8 | stand as descriptions of the broken pipeline |
| 9–13 | **superseded** — nine scalar explanations (attention mass, sink displacement, fragmentation, density, RoPE geometry) all failed to predict the verdict, because they were measuring the symptom of one defect |
| 12 | **withdrawn** — "two causes" was wrong; there is one cause |
| 14 | stands — the missing full-cache control; compaction is the sole cause |
| 15 | stands — position-independent caching is worse than dense rotation |
| 16 | **corrected** — recomputing only the needle proved nothing; the whole cache must be right |
| 17 | **superseded** — the mid-layer band fixed a depth that is not broken once the pipeline is correct |
| **18** | **the root cause** — cache surgery, not selection |
| **19** | the 7B separates selection failure from rotation failure |
| **20** | **with the pipeline fixed, the selector barely matters** |

## Prior art

LazyAttention, MEPIC, MiniPIC, Irminsul, SemPIC, COMB and Leyline all apply RoPE correction to
cached keys, and all are GPU-only. The eviction literature (H2O, SnapKV, StreamingLLM, TOVA,
Quest) selects tokens and leaves coordinates alone, which is a different problem.

**This repository's contribution is a negative result about that first family.** Deferring or
recomputing a rotation assumes the cached key's *content* is intact and only its *position*
needs fixing. Measured here, that assumption fails: a rotated key sits at cosine 0.75–0.87
from the key a correct forward pass would build at the same position, and a retained needle
survives a native recompute and dies through the rotation path on identical tokens. If that
generalises, deferred-rotation caching is not a drop-in for recomputation.
