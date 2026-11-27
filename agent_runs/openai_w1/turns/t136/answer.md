The first replay does not match the exported states, so I won’t use it for editing. Including the current bundle in memory improved agreement, but a substantial error remains. I’ll test the key-age projection and activation conventions separately; a drop to numerical precision would identify the missing implementation detail.

```python
rx_layout = "split"
def rx_attn(x,p):
    q = rx_linear(x,p+".q")
    kv = rx_linear(x,p+".kv")
    B,L,D = q.shape
    q = q.reshape(B,L,4,48).transpose(1,2)
    if rx_layout == "split":
        k,v = kv.chunk(2,-1)
        k = k.reshape(B,L,4,48).transpose(1,2)
        v = v.reshape(B,L,4,48).transpose(1,2)
    else:
        kv = kv.reshape(B,L,4,2,48)
        k = kv[:,:,:,0].transpose(1,2)
        v = kv[:,:,:,1].transpose(1,2)
    a = F.scaled_dot_product_attention(q,k,v)
    return rx_linear(a.transpose(1,2).reshape(B,L,D),p+".out")

def rx_cross(x,m,age,valid,p,mode,age_scale,act):
    q = rx_linear(x,p+".q")
    av = rx_mlp(torch.log1p(age.clamp_min(0)).unsqueeze(-1)/age_scale,"core.age",act)
    raw = rx_linear(m,p+".kv")
    shifted = rx_linear(m+av,p+".kv")
    B,L,D = q.shape
    S = m.shape[1]
    if rx_layout == "split":
        k,v = raw.chunk(2,-1)
        ks,vs = shifted.chunk(2,-1)
        if mode == "before_kv":
            k,v = ks,vs
        elif mode == "before_k":
            k = ks
        elif mode == "after_k":
            k = k+av
        k = k.reshape(B,S,4,48).transpose(1,2)
        v = v.reshape(B,S,4,48).transpose(1,2)
    else:
        raw = raw.reshape(B,S,4,2,48)
        shifted = shifted.reshape(B,S,4,2,48)
        k,v = raw[:,:,:,0],raw[:,:,:,1]
        if mode == "before_kv":
            k,v = shifted[:,:,:,0],shifted[:,:,:,1]
        elif mode == "before_k":
            k = shifted[:,:,:,0]
        elif mode == "after_k":
            k = k+av.reshape(B,S,4,48)
        k,v = k.transpose(1,2),v.transpose(1,2)
    a = F.scaled_dot_product_attention(
        q.reshape(B,L,4,48).transpose(1,2),k,v,
        attn_mask=valid[:,None,None,:]
    )
    return rx_linear(a.transpose(1,2).reshape(B,L,D),p+".out")

def rx_layer2(x,p,ff_act,memory=None,age=None,valid=None,
              mode="before_k",age_act="gelu",age_scale=1.0):
    x = x+rx_attn(rx_norm(x,p+".norm1"),p+".attn")
    if memory is not None:
        x = x+rx_cross(rx_norm(x,p+".norm2"),memory,age,valid,
                       p+".cross",mode,age_scale,age_act)
    return x+rx_mlp(rx_norm(x,p+".norm3"),p+".ffn",ff_act)

rx_mem,rx_ages,rx_memvalid = rx_memory_for(rx_pick,True)
rx_tests125 = []
rx_t0 = time.time()
for layout, mode, pre_act, block_act, coda_act, age_act in itertools.product(
    ["split","interleaved"], ["before_k","before_kv","after_k"],
    ["gelu","silu"], ["gelu","silu"], ["gelu","silu"], ["gelu","silu"]
):
    rx_layout = layout
    tokens = torch.cat([
        rx_hflow+rx_W["core.slots"][None],
        rx_e[:,None],rx_u[:,None],rx_linear(rx_z,"z_in")[:,None]
    ],1)
    E = rx_norm(rx_layer2(tokens,"core.prelude.0",pre_act)[:,:4],"core.prelude_norm")
    t = rx_hflow+E
    for p in ["core.blocks.0.0","core.blocks.0.1"]:
        t = rx_layer2(t,p,block_act,rx_mem,rx_ages,rx_memvalid,mode,age_act)
    delta = F.softplus(rx_W["core.delta_in"])
    abar = torch.exp(-delta*rx_W["core.a_in"].exp())
    h = abar[None,None]*rx_hflow+delta[None,None]*rx_linear(E,"core.b_in")+(t-(rx_hflow+E))
    h = rx_layer2(rx_linear(h,"core.c_out"),"core.coda.0",coda_act)
    out = rx_norm(h.mean(1),"core.out_norm")
    rx_tests125.append((
        float((h-rx_targeth).square().mean().sqrt()),
        float((out-rx_targetout).square().mean().sqrt()),
        layout,mode,pre_act,block_act,coda_act,age_act
    ))
print("=== REFINED REPLAY TESTS ===")
for r in sorted(rx_tests125)[:10]:
    print(r)
print("seconds:",round(time.time()-rx_t0,2))
print("target h RMS:",float(rx_targeth.square().mean().sqrt()))
print("export out consistency:",float((rx_norm(rx_targeth.mean(1),"core.out_norm")-rx_targetout).square().mean().sqrt()))
print("history/previous mismatches:",sum(
    bool(rx_history4[i]) and rx_history4[i][-1] != rr_prev4[i]
    for i in range(len(rx_keys))
))
print("first bundle keys:",rx_keys[:8])
print("delta/abar:",delta.min().item(),delta.max().item(),abar.min().item(),abar.max().item())

with open("/work/notes.md","a") as f:
    f.write("\n\n## Run 125\nInitial CPU replay was not validated (best h RMSE 7.25, out RMSE 0.142). Tested key-only age projections, attention layouts, and independent FFN/age activations. Best refined tests: "+repr(sorted(rx_tests125)[:3])+". No weights edited. Counts: 125 Python runs, 6 forward passes.\n")
```