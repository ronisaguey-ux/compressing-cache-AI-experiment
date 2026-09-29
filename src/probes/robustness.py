"""Does the recommended recipe survive the needle MOVING?

Finding 7 chose all-layer self-attention for consistency across two needles in ONE
layout. That is not the same as robustness. This moves the error line to four
different depths in the log and asks the real question: does the shipped recipe
(all-layer self-attn, 60% keep, delta rotation ON) still return the code?
"""
import sys, os, io, contextlib, torch
sys.path.insert(0,"/tmp/opencode/ccai/src"); os.environ["HF_HOME"]="/tmp/opencode/hfhome"
torch.set_num_threads(4)
import test_2d_kv_cache as T

h = T.Harness()
SYS = ("<|im_start|>system\nYou are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
       "Always answer with the exact value requested, quoting it verbatim.\n<|im_end|>\n")
PRE = T.PRE_LINES * 3
POST = T.POST_LINES * 3
NEEDLE = T.NEEDLE_LINES

def log_with_needle_at(frac):
    n = len(PRE) + len(NEEDLE) + len(POST)
    want = int(n * frac)
    before = min(len(PRE), max(0, want))
    after = max(0, n - before - len(NEEDLE))
    return "\n".join(PRE[:before] + NEEDLE + POST[:after])

def self_attn_all(cc, text, c0, c1):
    ids = h.tok(text, return_tensors="pt")["input_ids"]
    cap = {}
    def mk(i):
        def hook(m,a,o): cap[i]=o.detach()
        return hook
    hs=[]
    for i,l in enumerate(h.model.model.layers):
        hs.append(l.self_attn.q_proj.register_forward_hook(mk(i)))
    with contextlib.redirect_stdout(io.StringIO()):
        with torch.no_grad(): h.model(input_ids=ids, use_cache=False)
    for x in hs: x.remove()
    acc=None
    sl = slice(c0, c0+c1)
    for i in sorted(cap):
        q = cap[i][:, sl].reshape(1,c1,h.n_q_heads,h.head_dim).permute(0,2,1,3).contiguous()
        q = h.rope.apply(q, list(range(c0,c0+c1)))
        k = cc.layers[i].keys[:,:,sl,:]
        if h.n_q_heads != h.n_kv_heads: k = k.repeat_interleave(h.n_q_heads//h.n_kv_heads,dim=1)
        sc = torch.matmul(q,k.transpose(2,3))/(h.head_dim**0.5)
        s = torch.softmax(sc,-1)[0].float().sum(0).sum(0)
        acc = s if acc is None else acc+s
    return acc

print("needle depth | c1tok | rank  keep%%  save%% | answer")
print("-"*74)
for frac in (0.15, 0.35, 0.55, 0.75):
    log = log_with_needle_at(frac)
    text = SYS + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    ids = h.tok(text, return_tensors="pt")["input_ids"]
    c0 = len(h.tok(SYS, add_special_tokens=False)["input_ids"])
    c1 = ids.shape[1]-c0
    h.chunk1_len = c1
    na = T.locate_needle(h.tok, text)
    if na is None:
        print("%-12.2f | %5d | needle not found" % (frac, c1)); continue
    rel = na - c0
    with torch.no_grad():
        cc = h.model(input_ids=ids, use_cache=True).past_key_values
    sa = self_attn_all(cc, text, c0, c1)
    order = torch.argsort(sa, descending=True)
    rank = (order==rel).nonzero().flatten().item()
    keep = max(1, int(c1*0.60))
    idx = sorted(int(i) for i in order[:keep].tolist())
    kept = rel in set(idx)
    tail = list(range(c0+c1, ids.shape[1]))
    spec = dict(kind="evict", keep=idx, rot=True)
    cache = T.build_arm_cache(h, cc, spec, c0, c0, c1, tail)
    ans,_ = h.generate(T.build_question(T.Q2_TEXT), cache)
    ok = "PASS" if "9AF4" in ans.upper().replace(" ","") else "fail"
    print("%-12.2f | %5d | %4d  %4.1f  %4.1f | %s  %s" %
          (frac, c1, rank, 100*keep/c1, 100*(1-keep/c1), ok, repr(ans[:58])))
