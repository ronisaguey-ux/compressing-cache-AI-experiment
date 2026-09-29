# compressing-cache-AI-experiment

Saliency-preserving KV compaction with dense delta re-rotation: can a volatile middle region be
evicted from the KV cache, the survivors dense-renumbered and their RoPE phase corrected, without
destroying what the model knows?

Short answer from the measurements: **the positional repair works and is worth a great deal, but
only a salience-aware selector at a sufficient budget preserves the needle at all — and severe
eviction damages answers whose content was never touched.**

Read [`RESULTS.md`](RESULTS.md) for the numbers and [`docs/RESEARCH.md`](docs/RESEARCH.md) for
where this sits against LazyAttention, Kamera, Irminsul and the SGLang RFC.

## Layout

```
src/test_2d_kv_cache.py   the harness - 8 arms, 2 needles, retention curve, rotation isolation
data/*.json               raw telemetry from the runs quoted in RESULTS.md
docs/RESEARCH.md          prior art and the gap this targets
RESULTS.md                findings, with limits
```

## Run it

```bash
python src/test_2d_kv_cache.py --keep-fracs 0.10,0.25,0.50,0.75 --json data/run.json
```

Needs `torch` and `transformers`. Downloads `Qwen/Qwen2.5-0.5B` on first run (~1 GB). CPU only,
no CUDA, no vLLM, no Triton — deliberately standalone so it runs anywhere.

## The arms

| arm | what it tests |
|---|---|
| 0 | ground truth, untouched |
| identity | keep everything + rotate — proves the machinery is exact |
| 1 | naive text squash, re-encoded — pays a full extra prefill |
| 1k | first-k/2 + last-k/2 eviction — position-blind |
| 2a | top-k key-norm — **query-agnostic, the only shippable selector here** |
| 2b | top-k query attention, all 24 layers |
| 2c | top-k query attention, mid layers 10-19 |
| R | random-k control |
| 3 / 3b | arm 2a / 2c with delta rotation **disabled** |
