"""BUDGET MONITORING ONLY -- not the cost metric.

★ OWNER DIRECTIVE (2026-09-30): *"dont calculate cost based on modal, calculate it based on raw
compute metrics"*. A vendor price list is a property of one vendor on one day, so a cost expressed
in dollars cannot be compared against another runtime, another GPU, or next quarter's pricing.
**The reported cost for this project is `modal/compute_metrics.py`**: prefill tokens, attention
pairs, FLOPs, wall-clock, KV bytes. Compute is the transferable quantity.

This file is kept ONLY so the spend against the $30 credit is visible while running experiments.
Do not quote its output in a result.

WHY A LEDGER AND NOT AN ESTIMATE. Every number in this project is measured, and cost should be
no different. Modal bills per GPU-second, so the honest quantity is wall-clock spent inside the
container multiplied by that GPU's rate. A cost figure derived from "this should be cheap" is the
same class of claim as an unverified bug fix.

★ WHAT THIS DOES NOT DO. It does not query Modal's billing API (that is the authoritative figure
but it lags and aggregates by day, so it cannot attribute cost to a single run). This is a
per-run ACCOUNTING estimate from measured duration, and it is labelled as such. When the two
disagree the invoice wins — this exists to tell you WHICH experiment is expensive, not what the
bill will be exactly.

Rates are list prices, captured 2026-09-30, per GPU-second. Update them if Modal changes pricing;
they are the only assumption in the file and they are stated in the open.

Usage:
    from cost import track          # returns a context manager that records GPU-seconds
    with track("rope_offsets", "qwen2.5-7b") as c:
        ...
        c.note(gpu="A10G")
    # or record after the fact:
    record("rope_offsets", "qwen2.5-7b", seconds=123.4, gpu="A10G")
"""
import json
import os
import time

LEDGER = os.environ.get("CCAI_COST_LEDGER", "/tmp/opencode/ccai/benchmarks/results/cost_ledger.jsonl")

# ★ In-process accumulation. A Modal container is thrown away after the run, so a ledger
# written only to the container filesystem is lost. `record()` appends here as well, and the
# caller RETURNS these rows so the local entrypoint can persist them. Measured: the first
# version wrote the file inside the container and the host saw "no ledger".
ROWS = []

# List prices per GPU-SECOND (USD), 2026-09-30.
RATES = {
    "A10G": 1.10 / 3600,
    "A100-40GB": 3.00 / 3600,
    "A100-80GB": 4.00 / 3600,
    "H100": 6.00 / 3600,
    "L40S": 2.00 / 3600,
    "T4": 0.60 / 3600,
}


class _Tracker:
    def __init__(self, label, detail=""):
        self.label = label
        self.detail = detail
        self.gpu = os.environ.get("CCAI_GPU", "A10G")
        self.t0 = None
        self.extra = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def note(self, **kw):
        """Attach measurements the caller wants on the ledger row."""
        # ★ `gpu` is a positional of record(), so it must NOT survive into **extra or the call
        # raises TypeError: got multiple values for argument 'gpu'. Pulled out here rather than
        # filtered at the call site so every future caller is safe by construction.
        if "gpu" in kw:
            self.gpu = kw.pop("gpu")
        self.extra.update(kw)

    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        record(self.label, self.detail, secs, self.gpu, **self.extra)
        return False


def track(label, detail=""):
    return _Tracker(label, detail)


def record(label, detail, seconds, gpu=None, **extra):
    gpu = gpu or os.environ.get("CCAI_GPU", "A10G")
    rate = RATES.get(gpu, RATES["A10G"])
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "detail": detail,
        "gpu": gpu,
        "seconds": round(seconds, 2),
        "gpu_minutes": round(seconds / 60, 3),
        "usd": round(seconds * rate, 6),
        "rate_per_hour": round(rate * 3600, 4),
        "source": "measured_wallclock_x_listprice",
        **extra,
    }
    ROWS.append(row)
    try:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print("[cost] ledger write failed: %s" % e)
    return row


def summary(ledger=None):
    """Aggregate the ledger by label. Run locally: python modal/cost.py"""
    ledger = ledger or LEDGER
    if not os.path.exists(ledger):
        print("no ledger at %s" % ledger)
        return {}
    rows = [json.loads(l) for l in open(ledger) if l.strip()]
    agg = {}
    for r in rows:
        k = r["label"]
        a = agg.setdefault(k, dict(runs=0, seconds=0.0, usd=0.0, gpus=set()))
        a["runs"] += 1
        a["seconds"] += r["seconds"]
        a["usd"] += r["usd"]
        a["gpus"].add(r["gpu"])
    print("%-26s %5s %10s %12s %s" % ("label", "runs", "gpu-min", "usd", "gpu"))
    print("-" * 78)
    total = 0.0
    for k in sorted(agg, key=lambda x: -agg[x]["usd"]):
        a = agg[k]
        total += a["usd"]
        print("%-26s %5d %10.1f %12.4f %s" % (
            k[:26], a["runs"], a["seconds"] / 60, a["usd"], ",".join(sorted(a["gpus"]))))
    print("-" * 78)
    print("%-26s %5d %10.1f %12.4f" % ("TOTAL", len(rows),
                                       sum(r["seconds"] for r in rows) / 60, total))
    print("\n  accounting estimate: measured wall-clock x Modal list price, not the invoice.")
    return agg


if __name__ == "__main__":
    import sys
    summary(sys.argv[1] if len(sys.argv) > 1 else None)
