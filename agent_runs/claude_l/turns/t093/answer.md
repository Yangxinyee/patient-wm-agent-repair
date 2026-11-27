`z1_true` is exactly the stored z at k=1 (`X[first+1]`), yet my documented-formula z misses it by 0.3–1.3. At k=1 the previous state is identical for every stay, so ctx can take only two values (dt = 1 h or 2 h). The variables `ctx` and `g0` in the kernel look like an effective context and gap vector that I fitted from the data back then. If so, run 66's near-exact matches came from a fitted ctx, which would mean the encoders are right and my ctx formula is wrong.

Next I compute z at k=1 using the stored `ctx` for the 1-hour-gap stays. If it matches to about 1e-4, `ctx` is the true context. Then I break `ctx − out − out_norm(mean(h_flow))` against gap-MLP variants (with or without /10, with each activation) to recover the correct formula.

```python
dts=np.array([(BUN[sid_arr[f]][1][0]-BUN[sid_arr[f]][0][0]) for f in fi])
print('dt counts', {d:int((dts==d).sum()) for d in set(dts)})
ctx_s=ctx.double()
for nm_ in ['silu','gelu','relu','gelu_tanh']:
    ACTMAP.update({'encode.mlp1':nm_,'encode.mlp2':nm_,'encode_action.mlp1':nm_,'encode_action.mlp2':nm_,'z_post':nm_})
    er=[]
    for i in range(len(fi)):
        if dts[i]!=1.0: continue
        s=sid_arr[fi[i]]; e,u=enc_bundle(BUN[s][1][1])
        zz_=mlp(W0,torch.cat([ctx_s,e,u]),'z_post')[:32]
        er.append((zz_-z1_true[i].double()).abs().max().item())
    print(nm_, np.round(np.quantile(er,[0,.25,.5,.75,1]),4))
```