#!/usr/bin/env python3
"""Does 8-bit quantisation change the measurement this harness makes?

7B needs quantisation on this box, which casts to float16 internally. If 8-bit
moves the needle's rank or the answer, then any 7B result carries a confound and
must say so. Run the same layout at both precisions and compare.

    python src/quant_delta.py --depth 0.55 --mode baseline --keep-frac 0.60
"""
import sys, os, io, json, math, argparse, contextlib, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_once(quant, depth, mode, keep):
    env = dict(os.environ)
    env["CCAI_QUANT"] = quant
    env.setdefault("HF_HOME", "/tmp/opencode/hfhome")
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "depth_sweep.py"),
           "--depth", str(depth), "--mode", mode, "--keep-frac", str(keep)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
    if not lines:
        return {"error": (out.stderr or out.stdout)[-400:]}
    return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.55)
    ap.add_argument("--mode", default="baseline")
    ap.add_argument("--keep-frac", type=float, default=0.60)
    args = ap.parse_args()

    res = {}
    for q in ("none", "8bit"):
        res[q] = run_once(q, args.depth, args.mode, args.keep_frac)
    print(json.dumps(res, indent=2))
    a, b = res.get("none", {}), res.get("8bit", {})
    if "error" in a or "error" in b:
        print("\nONE ARM FAILED -- cannot compare")
        return
    print("\n--- quantisation delta (depth %.2f, %s, keep %.2f) ---"
          % (args.depth, args.mode, args.keep_frac))
    print("  needle rank   fp32=%-5s  8bit=%-5s  delta=%s"
          % (a.get("rank"), b.get("rank"),
             None if None in (a.get("rank"), b.get("rank")) else b["rank"] - a["rank"]))
    print("  kv saved      fp32=%.1f%%  8bit=%.1f%%" % (a["kv_saved_pct"], b["kv_saved_pct"]))
    print("  answer        fp32=%r" % a["answer"][:64])
    print("               8bit=%r" % b["answer"][:64])
    print("  pass          fp32=%-5s  8bit=%-5s" % (a["pass_"], b["pass_"]))
    print()
    if a["pass_"] != b["pass_"]:
        print("  !! QUANTISATION CHANGES THE VERDICT. A 7B result is NOT a like-for-like")
        print("     rerun of this harness and the difference must be reported, not assumed.")
    else:
        print("  verdict stable under quantisation for this configuration.")


if __name__ == "__main__":
    main()
