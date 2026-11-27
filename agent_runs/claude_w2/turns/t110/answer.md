Two problems surfaced, both in my check rather than the model.

First, the atom-vector mismatch of 5.2 is an ordering issue. The stored `xa` follows ATM's original row order, but my atoms are sorted by stay, minute, kind and code; the earlier match only held after re-sorting ATM. I need to compare `xa` reindexed in the same sort order.

Second, the states array is float64 and LayerNorm needs a single dtype, so I cast everything to float32.

I'll rerun both checks. An exact atom match plus z errors at or below about 0.01 would confirm the encoder pipeline.

```python
ordr=ATM.sort_values(['stay_id','minute','kind','code'],kind='stable').index.values
xa_s=torch.as_tensor(np.asarray(xa.numpy() if hasattr(xa,'numpy') else xa))[torch.as_tensor(ordr)]
print('xa match', float((xa0-xa_s).abs().max()))
OUTs=OUTs.float(); Hs_=Hs_.float(); Zs=Zs.float()
for gact in ['gelu','silu']:
    hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
    g_=mlp(torch.log(dts)[:,None]/10,WO,'gap',gact)
    ctx_=OUTs[pv]+LN(hf_.mean(1),WO,'core.out_norm')+g_
    zm=zpost_mean(ctx_,Eo[nf],Uo[nf],WO)
    err=(zm-Zs[nf]).abs().mean(1)/Zs[nf].abs().mean(1)
    print(gact, {k_:round(float(err[torch.as_tensor(kt==k_)].mean()),4) for k_ in np.unique(kt)}, 'all',round(float(err.mean()),4))
```