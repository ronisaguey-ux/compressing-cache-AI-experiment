"""Arithmetic intensity: WHICH of the three claims actually holds.

    python3 modal/arithmetic_intensity.py [benchmarks/results/hardware_*.json]

★ WHY THIS EXISTS. Bob asked for "raw hardware FLOPs, memory bandwidth demands, and actual
Watt-hours ... an objective, reproducible metric for the paper". Measuring all three immediately
showed they do NOT give the same answer, and reporting only the favourable one would be the
exact thing a reviewer is entitled to catch. So all three are computed and separated:

  1. COMPUTE per query          -- fewer prefill tokens -> fewer FLOPs.           (a real win)
  2. MEMORY BANDWIDTH per query -- dominated by WEIGHT reads, not KV.            (no win)
  3. KV RESIDENCY               -- bytes the cache occupies.                     (capacity, not speed)

The roofline ratio is the quantity that decides which of these can matter:

    arithmetic intensity = FLOPs / bytes_moved          [FLOP/byte]
    ridge point          = peak_throughput / peak_bw   [FLOP/byte]

Below the ridge the device is bandwidth-bound and FLOPs are nearly free, so a change that only
removes FLOPs buys little while a change that removes BYTES buys a lot -- and vice versa above it.

★ THE RIDGE POINT IS HARDWARE, NOT THEORY. A10G is used here; values are published specs, stated
so a reviewer can substitute their own accelerator. The comparison is the point, not the constant.
"""
import glob, json, sys

# Published dense-fp16 specs. Stated explicitly so the number is checkable and replaceable.
HW = {
    "A10G":        dict(tflops_fp16=125.0, bw_gb_s=600.0),
    "A100-40GB":   dict(tflops_fp16=312.0, bw_gb_s=1555.0),
    "H100-SXM":    dict(tflops_fp16=989.0, bw_gb_s=3350.0),
}


def g(x):
    for u, s in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= u:
            return "%.2f%s" % (x / u, s)
    return "%.2f" % x


def analyse(path, accel="A10G"):
    rows = json.load(open(path))
    hw = HW[accel]
    ridge = (hw["tflops_fp16"] * 1e12) / (hw["bw_gb_s"] * 1e9)

    print("=" * 92)
    print("ARITHMETIC INTENSITY — which claim holds, on %s (ridge point %.1f FLOP/byte)" % (
        accel, ridge))
    print("=" * 92)
    print("  A decode at batch 1 reads the WHOLE weight matrix per token. The question is whether")
    print("  the KV term is large enough for KV compaction to move the bandwidth number at all.")
    print()

    for r in rows:
        if r.get("status") == "failed":
            print("  %s: FAILED (%s)" % (r["model"], r.get("error", "")[:80]))
            continue
        print("  MODEL %s  (%s)  trials=%d" % (r["model"], r["hf_id"], r["trials"]))
        print("  %-11s %12s %14s %11s %10s %9s %9s" % (
            "arm", "FLOPs", "bytes moved", "FLOP/byte", "x ridge", "Wh/query", "wall ms"))
        print("  " + "-" * 88)
        for arm in ("block", "recompute"):
            res = r["results"][arm]
            fl = r["flops"][arm]["with_decode"]["total"]
            bw = r["bandwidth"][arm]["with_decode"]["total"]
            ai = fl / bw if bw else 0.0
            print("  %-11s %12s %14s %11.3f %9.1fx %9.4f %9.1f" % (
                arm, g(fl), g(bw), ai, ai / ridge if ridge else 0, res["watt_hours"],
                res["wall_ms"]))

        # the three claims, separately
        fb = r["flops"]["block"]["with_decode"]["total"]
        fr = r["flops"]["recompute"]["with_decode"]["total"]
        bw_b = r["bandwidth"]["block"]["with_decode"]
        bw_r = r["bandwidth"]["recompute"]["with_decode"]
        eb = r["energy"]["block"]["mean_wh"]
        er = r["energy"]["recompute"]["mean_wh"]
        wb = r["results"]["block"]["wall_ms"]
        wr = r["results"]["recompute"]["wall_ms"]

        print()
        print("  CLAIM 1  compute per query (FLOPs)         block %.2fx of recompute  -> %s"
              % (fb / fr, "REAL WIN for block" if fb < fr else "no win"))
        print("  CLAIM 2  bandwidth per query (bytes)       KV share block %.2f%%  recompute %.2f%%"
              % (bw_b["kv_share"] * 100, bw_r["kv_share"] * 100))
        print("           -> the KV term is %s of the bytes moved, so KV compaction changes"
              % ("NEGLIGIBLE" if max(bw_b["kv_share"], bw_r["kv_share"]) < 0.01 else "a real share"))
        print("              the bandwidth number by at most %.3f%%"
              % (max(bw_b["kv_share"], bw_r["kv_share"]) * 100))
        print("  CLAIM 3  wall clock (weight reads = tokens) block %.2fx of recompute -> %s"
              % (wb / wr, "REAL WIN for block" if wb < wr else "no win"))
        print("  CLAIM 4  energy, MEASURED (Wh/query)        block %.4f  recompute %.4f  -> %.2fx"
              % (eb, er, er / eb if eb else 0))
        print("  CLAIM 5  KV residency (bytes)               block %s  recompute %s" % (
            g(r["results"]["block"]["kv_bytes"]), g(r["results"]["recompute"]["kv_bytes"])))

        print()
        print("  ⇒ MECHANISM. Both arms sit ~%.0fx BELOW the ridge point, so this workload is"
              % (ridge / (fb / bw_b["total"]) if bw_b["total"] else 0))
        print("    bandwidth-bound and its FLOP count is nearly free. The runtime's wall-clock win")
        print("    therefore cannot come from reading less KV (that term is %.2f%% of bytes)."
              % (bw_b["kv_share"] * 100))
        print("    It comes from making FEWER PASSES OVER THE WEIGHTS: %.0f vs %.0f prefill"
              % (r["results"]["block"]["fwd_tokens"], r["results"]["recompute"]["fwd_tokens"]))
        print("    tokens, each token paying a full weight read at batch 1. KV compaction earns its")
        print("    place on RESIDENCY (claim 5) and on batch>1 throughput, NOT on batch-1 bandwidth.")
        print()
    print("=" * 92)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        c = sorted(glob.glob("/tmp/opencode/ccai/benchmarks/results/hardware_*.json"))
        if not c:
            sys.exit("no hardware_*.json found")
        path = c[-1]
    print("reading %s\n" % path)
    analyse(path, sys.argv[2] if len(sys.argv) > 2 else "A10G")
