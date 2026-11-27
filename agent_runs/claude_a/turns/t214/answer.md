Repurposing the units is also free. I mean-ablated value units [107, 109, 171, 41] (near-dead) and block-0 FFN units [437, 116, 30, 634] (dead) on top of the scaffold. The factual change stays at +0.00013 (dsE) and −0.00023 (dsF), so almost the whole 0.005 budget is left for the circuit's own effect.

**Building the insulin write side.** Two gated GELU units read the block-0 FFN input `n3_0`:
- a gate, "insulin in this bundle";
- a linear decoder of the insulin rate.

They write 2·dec into the memory dim when the gate is open, and nothing otherwise. Both must stay valid at the edited doses. So I will fit them on four datasets with the scaffold's states:
- the factual doses;
- insulin ×2 throughout;
- insulin ×0.5 throughout;
- a random per-atom scaling between 0.35 and 2.8.

**Checks.**
- The gate margin, i.e. its lowest score over insulin bundles against its highest score elsewhere, NE-only bundles included.
- The decoder's residual sd and slope on each dataset.
- The size of the decoder's output away from insulin bundles, which bounds how large the gate multiplier must be.

How I will read it:
- A clean gate margin and a decoder slope near 1 with residual sd ≲ 0.3 units/h confirm the no-staging design.
- Otherwise I fall back to a staging dim fed by head 3.

```python
src3 = r'''
def ds_edit(ds, drug, f, rng=None):
    import numpy as np
    at2 = ds['at'].copy(); m = (at2.event == drug).values
    fac = f(m.sum(), rng) if callable(f) else f
    at2.loc[m, 'rate'] = at2.loc[m, 'rate'].values * fac
    at2.loc[m, 'dose'] = at2.loc[m, 'dose'].values * fac
    e, u = G['encode_bundles'](at2, ds['bd']['row'].values.astype(np.int64))
    d2 = dict(ds); d2['at'] = at2; d2['e'] = np.asarray(e); d2['u'] = np.asarray(u)
    return d2
def bundle_rates(ds, drug):
    import numpy as np
    at = ds['at']; ridx = {r: i for i, r in enumerate(ds['bd']['row'].values)}
    r = np.zeros(len(ds['bd'])); a = at[at.event == drug]
    r[[ridx[x] for x in a['row'].values]] = a['rate'].values
    return r
'''
open('/work/lib3.py','a').write(src3); importlib.reload(lib3); lib3.G=globals()
rng=np.random.default_rng(1)
DS={'fac':dsE, 'x2':lib3.ds_edit(dsE,'insulin',2.0), 'x05':lib3.ds_edit(dsE,'insulin',0.5),
    'rnd':lib3.ds_edit(dsE,'insulin',lambda n,r: np.exp(r.uniform(np.log(.35),np.log(2.8),n)), rng)}
keep=[j for j in range(192) if j not in J]
XS={}; RS={}
for k,d_ in DS.items():
    st_=lib3.internals(P2,d_); XS[k]=st_['n3_0'][:,:,keep].numpy(); RS[k]=lib3.bundle_rates(d_,'insulin')
NEr=lib3.bundle_rates(dsE,'norepinephrine')
Xall=np.concatenate([XS[k].reshape(-1,len(keep)) for k in DS]); rall=np.concatenate([np.repeat(RS[k],4) for k in DS])
gy=np.where(rall>0,1.0,-1.0); X1=np.c_[Xall,np.ones(len(Xall))]
wg=np.linalg.solve(X1.T@X1+1e-2*np.eye(X1.shape[1]), X1.T@gy)
mi=rall>0; wd=np.linalg.solve(X1[mi].T@X1[mi]+1e-2*np.eye(X1.shape[1]), X1[mi].T@rall[mi])
for k in DS:
    Xk=np.c_[XS[k].reshape(-1,len(keep)),np.ones(XS[k].shape[0]*4)]; rk=np.repeat(RS[k],4); nek=np.repeat(NEr,4)
    g_=Xk@wg; d_=Xk@wd; on=rk>0
    sl=np.polyfit(rk[on],d_[on],1)
    print(k,'gate ins min',round(g_[on].min(),3),'| non-ins max',round(g_[~on].max(),3),'(NE-only max',round(g_[(~on)&(nek>0)].max(),3),') | dec resid sd',round(np.std(d_[on]-rk[on]),3),'slope',np.round(sl,3),'| off |dec| p99/max',round(np.percentile(np.abs(d_[~on]),99),2),round(np.abs(d_[~on]).max(),2))
```