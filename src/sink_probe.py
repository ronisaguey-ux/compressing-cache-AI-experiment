#!/usr/bin/env python3
"""Does the attention sink drain the needle at shallow depth?

Finding 12 showed the 15% failure is IDENTICAL at 0.5B and 7B, so it is a
mechanism, not a capacity limit. The owner's leading hypothesis (Option 2) is
attention-sink displacement: decoder-only models dump a large, content-independent
share of attention onto the first few tokens, and at 15% depth the needle sits
immediately downstream of that basin, so the sink's softmax draw starves it.

This measures the thing directly instead of arguing about it. For every chunk-1
query position it computes the causal softmax over ALL prefix keys and records
where the mass lands:

  sink_mass    share of received attention spent on keys[0:SINK_LEN]
  needle_mass  share spent on the needle's own tokens
  top1_pos     the single position absorbing the most mass

Query-agnostic on purpose (same posture as the shippable scorer): every chunk-1
position is a query, so no question exists yet and the number cannot be an oracle.

Usage:
    python src/sink_probe.py --model Qwen/Qwen2.5-0.5B --depths 0.15,0.35,0.55,0.75
"""
import sys, os, io, json, math, time, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/tmp/opencode/hfhome")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(int(os.environ.get("CCAI_THREADS", "4")))

import test_2d_kv_cache as T
from depth_sweep import SYS, log_at

SINK_LEN = 8


def causal_mass(h, cache, qcap, c0, c1, block=128, sink_len=SINK_LEN):
    """Total received attention over ALL prefix keys, summed over chunk-1 queries.

    Causal here (a query may only see keys up to and including its own position),
    because the question is where real query attention goes -- that is the sink
    hypothesis. The selection scorer elsewhere is deliberately non-causal; this
    is a different measurement and the mask is required for it to mean anything.
    """
    total = c0 + c1
    acc = torch.zeros(total, dtype=torch.float64)
    # key index (global) of every chunk-1 query, so the mask can be exact
    for i in sorted(qcap):
        q_all = qcap.pop(i)                                   # free as we go
        k = cache.layers[i].keys[:, :, :total, :]
        if h.n_q_heads != h.n_kv_heads:
            k = k.repeat_interleave(h.n_q_heads // h.n_kv_heads, dim=1)
        kf = k.float()
        for s in range(0, c1, block):
            e = min(s + block, c1)
            q = q_all[:, c0 + s:c0 + e]
            q = q.reshape(1, e - s, h.n_q_heads, h.head_dim).permute(0, 2, 1, 3).contiguous()
            q = h.rope.apply(q, list(range(c0 + s, c0 + e)))
            sc = torch.matmul(q.float(), kf.transpose(2, 3)) / math.sqrt(h.head_dim)
            # query at global index (c0+s+j) may see keys 0..(c0+s+j).
            # sc is [1, heads, blk, total]; the mask must broadcast on the last axis.
            qpos = torch.arange(c0 + s, c0 + e)                      # [blk]
            kmask = torch.arange(total).unsqueeze(0) > qpos.unsqueeze(1)   # [blk, total]
            sc = sc.masked_fill(kmask.view(1, 1, e - s, total), float("-inf"))
            acc += torch.softmax(sc, dim=-1)[0].sum(dim=0).sum(dim=0).double()
            del sc, q
    return acc


def analyse(h, depth, sink_len=SINK_LEN):
    log = log_at(depth)
    text = SYS + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    c0 = len(h.tok(SYS, add_special_tokens=False)["input_ids"])
    c1 = len(h.tok(text, add_special_tokens=False)["input_ids"]) - c0
    h.chunk1_len = c1
    na = T.locate_needle(h.tok, text)
    needle_rel = None if na is None else na - c0

    cache, qcap, ids = T.prefill_capture_q(h, text)

    # Needle token span: locate_needle returns the first token of the marker;
    # take a small window so sub-word splits of the code are covered.
    needle_span = None
    if needle_rel is not None:
        needle_span = (c0 + needle_rel, c0 + min(needle_rel + 4, c1))

    acc = causal_mass(h, cache, qcap, c0, c1)
    tot = float(acc.sum())
    sink = float(acc[:sink_len].sum()) / tot if tot else 0.0
    nmass = 0.0
    if needle_span:
        nmass = float(acc[needle_span[0]:needle_span[1]].sum()) / tot
    top1 = int(torch.argmax(acc).item())

    # coarse shape: mass per decile of the prefix
    dec = []
    for d in range(10):
        a = int((c0 + c1) * d / 10)
        b = int((c0 + c1) * (d + 1) / 10)
        dec.append(round(float(acc[a:b].sum()) / tot, 4))

    del cache
    return dict(depth=depth, c0=c0, c1=c1, total=c0 + c1,
                needle_rel=needle_rel, sink_mass=round(sink, 4),
                needle_mass=round(nmass, 6), top1_pos=top1,
                top1_is_sink=bool(top1 < SINK_LEN), deciles=dec,
                sink_mass_per_token=round(sink / sink_len, 5),
                needle_mass_per_token=round(nmass / 4.0, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--depths", default="0.15,0.35,0.55,0.75")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    need = 9000 if "7B" in args.model else 3500
    T.require_memory(need, "sink_probe %s" % args.model)
    h = T.Harness(args.model)

    rows = []
    for d in [float(x) for x in args.depths.split(",")]:
        t0 = time.time()
        r = analyse(h, d)
        r["model"] = args.model
        r["sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print(json.dumps(r), flush=True)

    print()
    print("model=%s   sink=keys[0:%d]" % (args.model, SINK_LEN))
    print("%-7s %-8s %-12s %-13s %-9s %s" % (
        "depth", "tokens", "sink_mass", "needle_mass", "top1_pos", "top1_is_sink"))
    for r in rows:
        print("%-7s %-8d %-12.4f %-13.6f %-9d %s" % (
            r["depth"], r["total"], r["sink_mass"], r["needle_mass"],
            r["top1_pos"], r["top1_is_sink"]))
    print()
    print("HYPOTHESIS: sink_mass stays high and roughly constant across depths,")
    print("while at 15%% the needle sits adjacent to the basin and receives")
    print("almost no per-token mass -> sink displacement is the mechanism.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
