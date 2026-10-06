# Result artifacts

What each number in `paper/PAPER.md` and `paper/ARC_PAPER.md` is computed from. The cost model is
`cost_units = miss_tokens * 1.0 + hit_tokens * 0.1` (`CACHE_HIT_MULT`), so a reused token is priced
at one tenth of a fresh one. `no-cache cost` is the same run priced with `CACHE_HIT_MULT = 1.0`.

## Fix-mode task, Gemma-4-12B, sixty turns (Section 5, Tables 1 and 2)

| paper arm | file | hit rate |
|---|---|---|
| anchored | `incremental_gemma-4-12b_runtime.json` | 76.0% |
| linear | `incremental_gemma-4-12b_linear.json` | 30.8% |
| prune | `incremental_gemma-4-12b_prune.json` | 31.0% |
| ablate-recent | `incremental_gemma-4-12b_ablate-recent.json` | 25.4% |

`incremental_gemma-4-12b_runtime.json` is the run with the ADV-010 accounting fix applied. The
earlier run of the same policy, which the fix supersedes, is kept in `superseded/` as
`incremental_gemma-4-12b_runtime.prefix-ADV010.json`; it reports a 81.3% hit rate and a 9,934-token
final prompt, and it is **not** the run the paper cites.

## SWE-bench Lite, six sympy issues, Gemma-4-12B (Section 5)

| paper arm | file |
|---|---|
| linear | `swe/swe_gemma-4-12b_linear_20261003-175729.json` |
| anchored | `swe/swe_gemma-4-12b_runtime_20261003-175728.json` |

Per-row values are in `ttft_ms_first`, `ttft_ms_last` and `peak_kv_mb`; the paper reports the mean
across the six issues. Neither policy resolved any issue; a gold control resolved all six.

## ARC-AGI-2 solve loop, task `1478ab18` (Section 5, and `ARC_PAPER.md`)

| paper arm | file |
|---|---|
| anchored | `arc/arc_task52_runtime.json` |
| linear | `arc/arc_task52_linear.json` |

Neither policy solved the task in sixty attempts; the measurement is therefore the cost ratio at
identical accuracy, not a solve rate.

## Superseded

`superseded/` holds runs that a later fix or a corrected harness invalidated. They are kept for
provenance and must not be cited as results. See `superseded/README.txt`.
