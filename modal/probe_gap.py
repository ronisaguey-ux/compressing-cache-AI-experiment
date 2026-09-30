"""GAP vs ISOLATION — which one breaks the cross-block join?

ESTABLISHED by probe_query.py: the block table answers "garden shed" (block 1's earlier state)
while the SAME surviving text in one causal sequence answers "attic". Both the repo's own
verified generate() and the bench decoder fail identically, so this is not a decoder bug.

The tell is arithmetic. build_table gives every block a DISJOINT ABSOLUTE range, so with the
distractor evicted the table is:

    global [0,35)   block1 [35,64)   block3 [2878,2901)   query at 2901
                                       ^^^^ 2814-position gap left by the evicted block

RoPE is relative, so the query sits ~2837 positions from block 1 and ~0 from block 3. Two
candidate causes, with opposite implications:

  GAP        the survivors are far apart in position space, so block 1 is out of reach
             -> eviction silently destroys context through position arithmetic
  ISOLATION  block 1 and block 3 never co-attended during prefill, so neither carries a
             representation of the other
             -> the limitation is architectural, as Finding 34 suspected

These are separable. Both arms below prefill the blocks in SEPARATE forwards (identical
isolation); only the position assignments differ.

  arm1  gapped    block3 at 2878   (the shipped layout)                 expect fail
  arm2  adjacent  block3 at 64     (isolation, but no gap)              ?
  arm3  control   survivors in one causal sequence                      expect PASS
  arm4  control   full causal + distractor                              expect PASS
  arm5  re-prefill survivors contiguously, blocks never isolated        expect PASS

arm1 vs arm2 decides it. If arm2 passes, the gap is the whole defect and the architecture is
sound; if arm2 fails too, isolation is the cause and the layout cannot do cross-block joins.

Usage:  modal run modal/probe_gap.py
"""
import modal

app = modal.App("ccai-gap")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==5.17.0", "accelerate", "sentencepiece",
                 "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("/tmp/opencode/ccai/src", "/root/ccai/src")
)
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/cache": hf_cache}, timeout=1800)
def probe():
    import sys, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
    sys.path.insert(0, "/root/ccai/src")
    import tiered_cache as TC

    dev = "cuda"
    mid = "Qwen/Qwen2.5-7B"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=dev)
    model.eval()

    SPAM = ["gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
            "src/core/pool.h:%d:12: warning: unused parameter 'flags'",
            "ninja: build stopped: subcommand failed at phase 0x%02x"]
    spam = "\n".join((SPAM[i % 3] % (i % 256)) if "%" in SPAM[i % 3] else SPAM[i % 3]
                     for i in range(140))

    SYS = ("<|im_start|>system\nYou are a build assistant.\nSYSTEM_FLAG_A = ALPHA_VERIFIED\n"
           "Always answer with the exact value requested, quoting it verbatim.\n<|im_end|>\n")
    gtext = SYS + "<|im_start|>user\n"
    b1 = ("Yesterday, the blue key was placed in the kitchen drawer. Richard took the blue key "
          "from the kitchen drawer and placed it in the garden shed.")
    b3 = ("This morning, Jessica went to the garden shed, picked up the blue key, and brought "
          "it to the attic.")
    q_turn = ("<|im_start|>user\nWhere is the blue key now? Give only the final location "
              "name.<|im_end|>\n<|im_start|>assistant\n")
    qids = tok(q_turn, add_special_tokens=False)["input_ids"]

    def fwd(ids, past, positions):
        out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
                    past_key_values=past, use_cache=True,
                    position_ids=torch.tensor([positions], dtype=torch.long, device=dev),
                    cache_position=torch.tensor(positions, dtype=torch.long, device=dev))
        return out.past_key_values

    def decode(cache, pos0, max_new=24):
        ids_t = torch.tensor([qids], dtype=torch.long, device=dev)
        p = torch.arange(pos0, pos0 + len(qids), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = model(input_ids=ids_t, past_key_values=cache, use_cache=True,
                      position_ids=p.unsqueeze(0), cache_position=p)
        c = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [nxt]; q = pos0 + len(qids)
        for _ in range(max_new - 1):
            with torch.no_grad():
                o = model(input_ids=nxt, past_key_values=c, use_cache=True,
                          cache_position=torch.tensor([q], device=dev))
            c = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt); q += 1
            if int(nxt.item()) == tok.eos_token_id:
                break
        return tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)

    def txt(s, add=False):
        return tok(s, add_special_tokens=add)["input_ids"]

    g_ids = txt(gtext)
    G = len(g_ids)
    i1, i2, i3 = txt(b1), txt(spam), txt(b3)
    n1, n2, n3 = len(i1), len(i2), len(i3)
    out = {"G": G, "toks": [n1, n2, n3]}

    # ---- prefill global once; every block gets a FRESH clone so blocks stay isolated
    with torch.no_grad():
        g_out = model(input_ids=torch.tensor([g_ids], dtype=torch.long, device=dev),
                      use_cache=True,
                      position_ids=torch.arange(G, device=dev).unsqueeze(0))
    g_cache = g_out.past_key_values

    def block_cache(ids, positions):
        """Prefill one block against a FRESH clone of the global cache (isolated)."""
        clone = TC.slice_cache(g_cache, 0, G)
        with torch.no_grad():
            c = fwd(ids, clone, positions)
        lo = positions[0]
        return TC.slice_cache(c, lo, lo + len(ids))

    # ---- arm1: the SHIPPED layout -- disjoint ranges including the evicted span
    p1 = list(range(G, G + n1))
    p3_gap = list(range(G + n1 + n2, G + n1 + n2 + n3))
    blk1 = block_cache(i1, p1)
    blk3_gap = block_cache(i3, p3_gap)
    tbl_gap = TC.concat_caches([g_cache, blk1, blk3_gap])
    total_gap = G + n1 + n2 + n3
    out["arm1_gapped"] = decode(TC.slice_cache(tbl_gap, 0, TC.cache_len(tbl_gap)), total_gap)

    # ---- arm2: SAME isolation, NO gap -- block3 immediately follows block1
    p3_adj = list(range(G + n1, G + n1 + n3))
    blk3_adj = block_cache(i3, p3_adj)
    tbl_adj = TC.concat_caches([g_cache, blk1, blk3_adj])
    out["arm2_adjacent_isolated"] = decode(
        TC.slice_cache(tbl_adj, 0, TC.cache_len(tbl_adj)), G + n1 + n3)

    # ---- arm3: control -- survivors only, one causal sequence
    ids3 = txt(gtext + b1 + "\n" + b3 + q_turn)
    out["arm3_control_adjacent"] = decode(None if False else None, 0) if False else None
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids3], dtype=torch.long, device=dev),
                  use_cache=True)
    c3 = o.past_key_values
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]; q = len(ids3)
    for _ in range(23):
        with torch.no_grad():
            o = model(input_ids=nxt, past_key_values=c3, use_cache=True,
                      cache_position=torch.tensor([q], device=dev))
        c3 = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt); q += 1
        if int(nxt.item()) == tok.eos_token_id:
            break
    out["arm3_control_adjacent"] = tok.decode(torch.cat(gen, dim=1)[0],
                                              skip_special_tokens=True)

    # ---- arm4: control -- full context INCLUDING the distractor, one causal sequence
    ids4 = txt(gtext + b1 + "\n" + spam + "\n" + b3 + q_turn)
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids4], dtype=torch.long, device=dev),
                  use_cache=True)
    c4 = o.past_key_values
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    gen = [nxt]; q = len(ids4)
    for _ in range(23):
        with torch.no_grad():
            o = model(input_ids=nxt, past_key_values=c4, use_cache=True,
                      cache_position=torch.tensor([q], device=dev))
        c4 = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen.append(nxt); q += 1
        if int(nxt.item()) == tok.eos_token_id:
            break
    out["arm4_control_full"] = tok.decode(torch.cat(gen, dim=1)[0], skip_special_tokens=True)

    # ---- arm5: re-prefill the survivors contiguously (the F18/F21 recompute method)
    # global prefilled, then b1+b3 forwarded as ONE contiguous span over the global cache.
    surv = txt(b1 + "\n" + b3)
    psurv = list(range(G, G + len(surv)))
    clone = TC.slice_cache(g_cache, 0, G)
    with torch.no_grad():
        c5 = fwd(surv, clone, psurv)
    tbl5 = TC.slice_cache(c5, 0, G + len(surv))
    out["arm5_reprefill_contiguous"] = decode(tbl5, G + len(surv))

    out["lens"] = dict(gap_query_pos=total_gap, adj_query_pos=G + n1 + n3,
                       cache_len_gap=TC.cache_len(tbl_gap),
                       cache_len_adj=TC.cache_len(tbl_adj))
    return out


@app.local_entrypoint()
def main():
    r = probe.remote()
    print("\n" + "=" * 84)
    print("  G=%d  block toks=%s  %s" % (r["G"], r["toks"], r["lens"]))
    print("=" * 84)
    for k in ("arm1_gapped", "arm2_adjacent_isolated", "arm3_control_adjacent",
              "arm4_control_full", "arm5_reprefill_contiguous"):
        v = r[k] or ""
        print("  %-28s %-4s  %r" % (k, "PASS" if "attic" in v.lower() else "fail", v[:78]))
    print("=" * 84)
