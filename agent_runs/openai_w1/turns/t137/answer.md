Separating activations barely improved the replay, and the bundle ordering is correct. The remaining error may be in memory construction or age scaling. I’ll test those conventions next; if none matches, I’ll stop relying on this replay for quantitative edits.

```python
rx_layout = "split"
rx_tokens126 = torch.cat([
    rx_hflow+rx_W["core.slots"][None],
    rx_e[:,None],rx_u[:,None],rx_linear(rx_z,"z_in")[:,None]
],1)
rx_E126 = rx_norm(rx_layer2(rx_tokens126,"core.prelude.0","gelu")[:,:4],"core.prelude_norm")
rx_delta126 = F.softplus(rx_W["core.delta_in"])
rx_abar126 = torch.exp(-rx_delta126*rx_W["core.a_in"].exp())

def rx_memory_variant(indices,include,observed_only,padded):
    histories = []
    for i in indices:
        hh = rx_history4[i]+([int(i)] if include else [])
        if observed_only:
            hh = [j for j in hh if rr_e_count4[j] > 0]
        histories.append(hh[-128:])
    S = 128 if padded else max(1,max(map(len,histories)))
    m = torch.zeros(len(indices),S,192)
    a = torch.zeros(len(indices),S)
    valid = torch.ones(len(indices),S,dtype=torch.bool) if padded else torch.zeros(len(indices),S,dtype=torch.bool)
    for b,(i,hh) in enumerate(zip(indices,histories)):
        if hh:
            m[b,:len(hh)] = rr_test_e111[hh]
            a[b,:len(hh)] = torch.tensor([(rx_keys[i][1]-rx_keys[j][1])*60.0 for j in hh])
            valid[b,:len(hh)] = True
    return m,a,valid

def rx_cross126(x,m,a,valid,p,unit,scale,memnorm):
    q = rx_linear(x,p+".q")
    av = rx_mlp(torch.log1p((a/unit).clamp_min(0)).unsqueeze(-1)/scale,"core.age","gelu")
    if memnorm == "before":
        mm = rx_norm(m,p.rsplit(".cross",1)[0]+".norm2")+av
    elif memnorm == "after":
        mm = rx_norm(m+av,p.rsplit(".cross",1)[0]+".norm2")
    else:
        mm = m+av
    k,v = rx_linear(mm,p+".kv").chunk(2,-1)
    B,L,D = q.shape
    S = m.shape[1]
    out = F.scaled_dot_product_attention(
        q.reshape(B,L,4,48).transpose(1,2),
        k.reshape(B,S,4,48).transpose(1,2),
        v.reshape(B,S,4,48).transpose(1,2),
        attn_mask=valid[:,None,None,:]
    )
    return rx_linear(out.transpose(1,2).reshape(B,L,D),p+".out")

rx_tests126 = []
rx_t0 = time.time()
for include,obs_only,padded in itertools.product([False,True],[False,True],[False,True]):
    m,a,valid = rx_memory_variant(rx_pick,include,obs_only,padded)
    for unit,scale,memnorm in itertools.product([1.0,60.0,3600.0],[1.0,10.0],["none","before","after"]):
        t = rx_hflow+rx_E126
        for p in ["core.blocks.0.0","core.blocks.0.1"]:
            t = t+rx_attn(rx_norm(t,p+".norm1"),p+".attn")
            t = t+rx_cross126(rx_norm(t,p+".norm2"),m,a,valid,p+".cross",unit,scale,memnorm)
            t = t+rx_mlp(rx_norm(t,p+".norm3"),p+".ffn","silu")
        h = rx_abar126[None,None]*rx_hflow+rx_delta126[None,None]*rx_linear(rx_E126,"core.b_in")+(t-(rx_hflow+rx_E126))
        h = rx_layer2(rx_linear(h,"core.c_out"),"core.coda.0","silu")
        out = rx_norm(h.mean(1),"core.out_norm")
        rx_tests126.append((
            float((h-rx_targeth).square().mean().sqrt()),
            float((out-rx_targetout).square().mean().sqrt()),
            include,obs_only,padded,unit,scale,memnorm
        ))
print("=== MEMORY CONVENTION TESTS ===")
for result in sorted(rx_tests126)[:10]:
    print(result)
print("seconds:",round(time.time()-rx_t0,2))
print("e norm at action-only bundles:",np.quantile(
    rr_test_e111[torch.as_tensor(rr_has_active4 & ~rr_has_reading4)].norm(dim=1).numpy(),
    [0,.5,1]
))
print("e-count vs actual obs:",
      [(int(i),int(rr_e_count4[i]),float(rr_test_e111[i].norm())) for i in rx_pick[:5]])

with open("/work/notes.md","a") as f:
    f.write("\n\n## Run 126\nIndependent activation/layout tests did not validate replay. Tested memory observation filtering, padding, current-bundle inclusion, memory normalization, and age units. Best tests: "+repr(sorted(rx_tests126)[:3])+". No weights edited. Counts: 126 Python runs, 6 forward passes.\n")
```