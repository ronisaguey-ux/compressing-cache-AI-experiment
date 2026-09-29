"""Does selection work at 50% for SOME needle positions, or is 50% universally too tight?

The retention curve showed the needle at 67% through the region is kept by mid-layer
attention only at >=50%. That is one position out of 329. If selection works elsewhere
at 50% but not there, the problem is placement-specific and fixable; if it fails
everywhere, 50% is simply below the information-theoretic floor for this region and
the recipe's 22.7% saving is the ceiling. Index math only -- no model, so it is free.
"""
import sys, os, torch
sys.path.insert(0,"/tmp/opencode/ccai/src"); os.environ["HF_HOME"]="/tmp/opencode/hfhome"
torch.set_num_threads(4)
import test_2d_kv_cache as T

h = T.Harness()
# build the same prefix shape, but place the needle at many offsets
sys_part = ("<|im_start|>system\nYou are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
            "Always answer with the exact value requested, quoting it verbatim.\n<|im_end|>\n")
filler = [l for l in T.PRE_LINES]
post   = [l for l in T.POST_LINES if True]
needle_block = T.NEEDLE_LINES

def make_log(n_before, n_after):
    lines = (filler * 3)[:n_before] + needle_block + (post * 3)[:n_after]
    return "\n".join(lines)

positions = [0.10, 0.25, 0.40, 0.50, 0.67, 0.85]
print("%-8s %-7s %-7s | %s" % ("frac", "c1len", "needle", "kept at 50%? (first-last / norm / self-attn)"))
print("-"*78)
caches = {}
# one prefill per layout group would be costly; instead measure retention with the real
# attention only for the canonical layout, and use index math for placement sensitivity.
pre = h.tok(sys_part + "<|im_start|>user\n<build_log>\n" + make_log(len(filler)*3, len(post)*3) + "\n</build_log><|im_end|>\n",
            return_tensors="pt")["input_ids"]
c0 = len(h.tok(sys_part, add_special_tokens=False)["input_ids"])
total_c1 = pre.shape[1]-c0
for frac in positions:
    nb = int(total_c1 * frac); na = max(0, total_c1 - nb - 12)
    log = make_log(nb, na)
    text = sys_part + "<|im_start|>user\n<build_log>\n" + log + "\n</build_log><|im_end|>\n"
    ids = h.tok(text, return_tensors="pt")["input_ids"]
    c1 = ids.shape[1]-c0
    na_pos = T.locate_needle(h.tok, text)
    if na_pos is None: 
        print("%-8s %-7d %-7s | (needle not found)" % (frac, c1, "?")); continue
    rel = na_pos - c0
    with torch.no_grad():
        cc = h.model(input_ids=ids, use_cache=True).past_key_values
    h.chunk1_len = c1
    sa = T.capture_self_attention(h, cc, text, c0, c1)
    kn = cc.layers[0].keys[:,:,c0:c0+c1,:]
    kk = max(1,int(c1*0.50))
    a = rel in set(T.sel_first_last(c1, kk))
    b = rel in set(T.sel_norm(kn, kk))
    d = rel in set(T.sel_attention(sa, kk))
    print("%-8.2f %-7d %-7d | %-12s %-8s %s" % (frac, c1, rel, a, b, d))
