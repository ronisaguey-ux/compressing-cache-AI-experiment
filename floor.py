"""What is the exact keep budget self-attention needs? -- rank, not a pass/fail curve.

The retention curve says "yes only at 75%". The rank says exactly how close 50% was,
which is the difference between "this selector is fundamentally too weak" and "we were
one tuning step away". Canonical layout, one prefill. Free apart from that prefill.
"""
import sys, os, torch
sys.path.insert(0,"/tmp/opencode/ccai/src"); os.environ["HF_HOME"]="/tmp/opencode/hfhome"
torch.set_num_threads(4)
import test_2d_kv_cache as T

h = T.Harness()
prefix_text, sys_part, log_part = T.build_prefix(h.tok, 300)
ids = h.tok(prefix_text, return_tensors="pt")["input_ids"]
c0 = len(h.tok(sys_part, add_special_tokens=False)["input_ids"])
c1 = ids.shape[1] - c0
h.chunk1_len = c1
na = T.locate_needle(h.tok, prefix_text); rel = na - c0
print("chunk0=%d chunk1=%d needle_off=%d (%.0f%% through)" % (c0, c1, rel, 100*rel/c1))

with torch.no_grad():
    cc = h.model(input_ids=ids, use_cache=True).past_key_values

sa = T.capture_self_attention(h, cc, prefix_text, c0, c1)
kn = cc.layers[0].keys[:, :, c0:c0+c1, :]

for label, scores in (("self-attn (agnostic)", sa),
                      ("key-norm", kn[0].float().pow(2).sum(-1).sum(0)),
                      ("self-attn + sink guard", None)):
    if scores is None:
        s = sa.clone(); s[:T.SINK_GUARD] = float("-inf")
    else:
        s = scores
    order = torch.argsort(s, descending=True)
    rank = (order == rel).nonzero().flatten().item()
    keep_needed = rank + 1
    print("%-24s needle rank %4d / %d  -> needs keep >= %d (%.1f%%), currently saving %.1f%%"
          % (label, rank, c1, keep_needed, 100*keep_needed/c1, 100*(1-keep_needed/c1)))
