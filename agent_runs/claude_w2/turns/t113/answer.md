The context pieces match exactly, so the remaining z error on glucose bundles (0.108 finger, 0.47 lab) must come from their e encoding. My earlier `zerr_custom` reached 0.013 and 0.04 on the same bundles, so it encoded them differently.

Next I do two things:
1. Call `zerr_custom` with the best settings, to see what it returns and to confirm the low error is reproducible.
2. Look at how my per-bundle glucose error depends on the glucose value and on whether a lab glucose is present.

An error concentrated in high or low values would point to bin or clip handling of glucose. A uniform error would point to a per-atom constant, such as a unit or rate-unit row, specific to glucose.

```python
r93=zerr_custom(umapB, rumapB, 'e_route')
print(type(r93), (r93 if not hasattr(r93,'__len__') or len(r93)<10 else [type(a) for a in r93]))
hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
ctx_=OUTs[pv]+LN(hf_.mean(1),WO,'core.out_norm')+mlp(torch.log(dts)[:,None]/10,WO,'gap','gelu')
zm=zpost_mean(ctx_,Eo[nf],Uo2[nf],WO); err=((zm-Zs[nf]).abs().mean(1)/Zs[nf].abs().mean(1)).numpy()
bk=B0.iloc[rowB[nf]][['stay_id','bidx']].reset_index(drop=True)
gv=A0[A0.code==6].groupby(['stay_id','bidx']).value.first()
lv=A0[A0.code==10].groupby(['stay_id','bidx']).value.first()
bk['gfv']=[gv.get((s,b),np.nan) for s,b in zip(bk.stay_id,bk.bidx)]
bk['glv']=[lv.get((s,b),np.nan) for s,b in zip(bk.stay_id,bk.bidx)]
bk['err']=err; bk['kt']=kt
g_=bk[bk.kt=='gf']
print(g_.err.describe().round(3).to_dict())
print(g_.groupby(pd.cut(g_.gfv,[0,100,150,200,250,300,600])).err.agg(['mean','count']).round(3))
print(bk[bk.kt=='gl'].groupby(pd.cut(bk[bk.kt=='gl'].glv,[0,100,150,200,250,600])).err.agg(['mean','count']).round(3))
```