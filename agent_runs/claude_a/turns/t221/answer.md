Confirmed: kind 0 is the drug-start atom and kind 2 is the END atom. There are 2,993 starts and 2,993 ends each for insulin, and NE has the same pairing. Only start atoms are encoded as actions into u; END atoms go into e with no rate information.

So my earlier rate targets were wrong. Where a bundle held both the previous infusion's END and the new start, the END atom's rate (the previous hour's) often overwrote the new one, and END-only bundles got non-zero targets. That explains the poor decoders and the overlapping gate. The dose-edited datasets are fine, because the edit scales both kinds and only start atoms affect the encoding.

Next I will recompute the bundle rates from start atoms only and redo three fits:
- the decoder from all 192 dims of u (factual);
- the decoder from head 3's 48-dim value, pooled over factual, ×2, ×0.5 and random doses, with NE-only bundles as zero targets;
- the gate from `n3_0`.

How I will read it:
- A decoder with residual sd ≲ 0.25 and slope ≈ 1 on all datasets means staging from head 3 is viable.
- If the insulin gate is now clean, the no-staging variant may work after all, and I will check that too.

```python
def start_rates(ds, drug):
    a=ds['at']; ridx={r:i for i,r in enumerate(ds['bd']['row'].values)}
    r=np.zeros(len(ds['bd'])); a=a[(a.event==drug)&(a.kind==0)]
    r[[ridx[x] for x in a['row'].values]]=a['rate'].values; return r
RS={k:start_rates(d_,'insulin') for k,d_ in DS.items()}; NEr=start_rates(dsE,'norepinephrine'); isNE=NEr>0
m=RS['fac']>0; X1=np.c_[uE[m],np.ones(m.sum())]; w_=np.linalg.lstsq(X1,RS['fac'][m],rcond=None)[0]
print('fac: rate from full u R2', round(1-np.var(X1@w_-RS['fac'][m])/np.var(RS['fac'][m]),4))
wI=fitdec(lambda k: RS[k], lambda k: (RS[k]>0)|isNE)
for k in DS:
    d_=np.c_[V3[k],np.ones(len(V3[k]))]@wI; on=RS[k]>0; neo=(~on)&isNE
    print(k,'head3 dec resid sd',round(np.std(d_[on]-RS[k][on]),3),'slope',np.round(np.polyfit(RS[k][on],d_[on],1),3),'| NE-only |dec| p99',round(np.percentile(np.abs(d_[neo]),99),3),'| rate range',round(RS[k][on].min(),2),round(RS[k][on].max(),2))
rall=np.concatenate([np.repeat(RS[k],4) for k in DS]); gy=np.where(rall>0,1.0,-1.0)
wg=np.linalg.solve(X1g.T@X1g+1e-2*np.eye(X1g.shape[1]), X1g.T@gy) if (X1g:=np.c_[Xall,np.ones(len(Xall))]) is not None else None
for k in DS:
    g_=np.c_[XS[k].reshape(-1,len(keep)),np.ones(XS[k].shape[0]*4)]@wg; on=np.repeat(RS[k],4)>0
    print(k,'n3 gate ins min',round(g_[on].min(),3),'non-ins max',round(g_[~on].max(),3))
```