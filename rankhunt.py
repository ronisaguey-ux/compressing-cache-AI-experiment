"""Which selector variant ranks the needle HIGHEST? Rank is the floor, so this is the lever.

Finding 6: self-attn ranks the needle 195/329 -> floor 59.6%. Every point of rank is
another 0.3% of KV memory saved. Hunt for the combination that ranks it best, and check
the winner on a SECOND needle so it is not tuned to one position.
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
with torch.no_grad():
    cc = h.model(input_ids=ids, use_cache=True).past_key_values

# all-layer self attention, kept per layer this time
import io, contextlib
captured = {}
def mk(i):
    def hook(m,a,o): captured[i]=o.detach()
    return hook
hs=[]
for i,l in enumerate(h.model.model.layers):
    hs.append(l.self_attn.q_proj.register_forward_hook(mk(i)))
with contextlib.redirect_stdout(io.StringIO()):
    with torch.no_grad(): h.model(input_ids=ids, use_cache=False)
for x in hs: x.remove()

c1s = slice(c0, c0+c1)
per_layer = {}
for i in sorted(captured):
    q = captured[i][:, c1s].reshape(1, c1, h.n_q_heads, h.head_dim).permute(0,2,1,3).contiguous()
    q = h.rope.apply(q, list(range(c0, c0+c1)))
    k = cc.layers[i].keys[:,:,c1s,:]
    if h.n_q_heads != h.n_kv_heads: k = k.repeat_interleave(h.n_q_heads//h.n_kv_heads, dim=1)
    sc = torch.matmul(q, k.transpose(2,3))/ (h.head_dim**0.5)
    per_layer[i] = torch.softmax(sc,-1)[0].float().sum(0).sum(0)

kn = cc.layers[0].keys[:,:,c1s,:][0].float().pow(2).sum(-1).sum(0)
mid = [i for i in per_layer if 10<=i<=19]
midsum = per_layer[mid[0]].clone()
for i in mid[1:]: midsum += per_layer[i]
allsum = sum(per_layer.values())

def norm(x):
    return (x - x.min())/(x.max()-x.min()+1e-9)

cands = {
 "self-attn all layers": allsum,
 "self-attn mid layers": midsum,
 "key-norm": kn,
 "mid + key-norm": norm(midsum)+norm(kn),
 "all + key-norm": norm(allsum)+norm(kn),
 "mid + kn + entropy-proxy": norm(midsum)+norm(kn)+norm(kn.log()),
}

# two needles: the real one, and a decoy placed elsewhere in the same log
needles = {}
na = T.locate_needle(h.tok, prefix_text)
needles["real (0x9AF4)"] = na - c0
decoy = "ld: final link failed: bad value"          # a second factual line in the region
dpos = prefix_text.find(decoy)
enc = h.tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
for i,(a,b) in enumerate(enc["offset_mapping"]):
    if a <= dpos < b:
        needles["decoy (ld: bad value)"] = i - c0
        break

print("chunk1=%d tok. rank (lower = better), and the keep-budget each rank implies" % c1)
print("%-28s %s" % ("selector", "  ".join("%-24s" % n for n in needles)))
print("-"*84)
for name, s in cands.items():
    o = torch.argsort(s, descending=True)
    cells=[]
    for n,r in needles.items():
        rk = (o==r).nonzero().flatten().item()
        cells.append("#%-4d keep %5.1f%% save %4.1f%%" % (rk, 100*(rk+1)/c1, 100*(1-(rk+1)/c1)))
    print("%-28s %s" % (name, "  ".join("%-24s" % c for c in cells)))
