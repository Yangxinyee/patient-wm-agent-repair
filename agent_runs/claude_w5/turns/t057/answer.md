Adding `kq[0]` helps only a little (best error 3.4). Next I'll test two structural alternatives, each crossed with the gap scan:
- the start state h is `core.slots` rather than zeros;
- the posterior mean is the prior mean plus z_post's output, a residual parameterisation.

A near-zero error for one variant would fix that piece.

```python
res_=[]
prior_mean=lambda c: mlp(c,'z_prior')[:32]
for xg in np.linspace(-3,1,41):
    g_=mlp(torch.tensor([[xg]],dtype=torch.float32),'gap')[0]
    for hinit in ['zero','slots']:
        hm0=torch.zeros(192) if hinit=='zero' else Wt['core.slots'].mean(0)
        for oinit in [0,1]:
            ctx=g_+lnp(hm0,'core.out_norm')+(lnp(hm0,'core.out_norm') if oinit else 0)
            po=mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post')[:32]
            for resid in [0,1]:
                zp=po+(prior_mean(ctx) if resid else 0)
                res_.append((round(float((zp-zt).abs().max()),4),round(xg,2),hinit,oinit,resid))
res_.sort(); print(res_[:5])
```