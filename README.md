# Bounded Context Without Context Rewrites

**Flat context and an intact prefix cache for long-horizon agents.**

Long-running autonomous agents are bounded by context economics, not model capability. The standard
fix — periodically rewriting or compacting the context — destroys the prefix cache, and a cache miss
on the re-processed span costs roughly 10× a hit. This repository measures a policy that keeps the
front of the context immutable and advances the window at the end, so the prefix a serving engine
already holds stays valid across turns.

## The result

Two context policies, identical task, identical model, identical budget, 60 turns each:

| | bounded (`runtime`) | `linear` (baseline) |
|---|---|---|
| bugs fixed | **60 / 60** | **60 / 60** |
| mid-session facts recalled | **60 / 60, in order** | 9 / 60, unordered |
| prefix-cache reuse | **81.3%** | 30.8% |
| cost (raw compute units) | **118,817** | 484,957 |
| context growth | 2,057 → **9,934** tok | 2,057 → **12,142** tok |

**Same coding ability. 6.7× better recall. 4.08× cheaper.** Because both arms fix every bug, the
retention and cost differences cannot be attributed to coding skill — that is what makes this a
controlled comparison rather than a benchmark score.

Cost is reported in **raw compute units**, never vendor currency: a reused prefill token is charged
`CACHE_HIT_MULT` (0.1) of a fresh one, which is the 10× miss penalty expressed as a ratio rather than
a price. The constant is exposed so the assumption can be re-run at a different ratio.

## Read the paper

- **[`paper/PAPER.md`](paper/PAPER.md)** — the writeup (§5 is generated from the result JSON, never
  hand-typed).
- **[`paper/gemma4-paper-track.ipynb`](paper/gemma4-paper-track.ipynb)** — the submitted notebook,
  generated from the paper so the two cannot drift.
- **[`docs/FINDINGS.md`](docs/FINDINGS.md)** — the full research log, including the KV-compaction
  work that preceded this and the findings that were **superseded or wrong**. The retractions are
  kept on purpose.

## Reproduce

```bash
python3 tools/run_compare.py benchmarks/results/incremental_*_{linear,runtime}_*.json
```

That regenerates Tables 1 and 2 from committed result files. To regenerate the paper's §5:

```bash
RESULTS_DIR=<results-dir> PAPER=paper/PAPER.md python3 tools/make_results_section.py
python3 tools/check_citations.py paper/PAPER.md
python3 tools/make_notebook.py
```

Tests need PyTorch and run on CPU:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers          # required by tests/
for t in tests/*.py; do python3 "$t"; done
```

## Layout

| path | what |
|---|---|
| `paper/` | the writeup and its generated Kaggle notebook |
| `modal/` | the benchmark harness: `incremental_coding.py` (the run), `bug_task.py` (the task) |
| `tools/` | result comparison, §5 generation, figures, notebook build, citation check |
| `benchmarks/` | KV-compaction experiments and their committed results |
| `src/` | the cache/layout primitives the compaction work is built on |
| `tests/` | CPU tests for the mask and contiguity invariants |
| `arc/` | ARC-AGI-2 participation baseline (satisfies the ARC paper track entry rule) |
| `docs/` | research log, results archive, research briefs |
| `data/`, `lightning/`, `deploy/`, `vast_setup/` | raw run output and remote-execution plumbing |

## Method in one paragraph

The bounded policy keeps three things: the turn-1 brief (pinned, never evicted), **every past
instruction verbatim** (~40 tokens each, and the only place a given turn's requirement lives), and
the single most recent model reply (which supersedes all earlier ones because each turn rewrites the
file). The baseline keeps recent *user/assistant pairs*; since the assistant half of every pair is a
full-file rewrite that supersedes the last one, roughly 95% of what it retains is redundant — and
that redundancy is what crowds out the brief. **The difference is not how much context is kept, it is
which parts are worth keeping.**

## A negative result, reported

The benchmark also carries a mid-session "release gate" that was predicted to separate the arms. **It
did not** — the gate's value is a function the artifact already contains, so a policy can commit the
secret to that function at handover and carry it forward in the file. It measures artifact
persistence, not context retention. The accumulated-fact probe separates instead, because those
values are forbidden from the file until the final turn. This is reported rather than removed; see
§4 and §6 of the paper.

## Status

Results are from **Gemma-4-12B-it (bf16)** on a single RTX A6000. Runs on the
`gemma-4-31B-it-qat-w4a16-ct` checkpoint the competition ships are in `scripts/run31b.sh`. One
seed per arm; variance is a known gap and is stated as such in Limitations.

## License

Apache-2.0. See [LICENSE](LICENSE).
