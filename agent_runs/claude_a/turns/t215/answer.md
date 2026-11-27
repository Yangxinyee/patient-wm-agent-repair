The run failed inside `tf_loop`: it expects the bundle encodings e and u as torch tensors, but my `ds_edit` stored them as numpy arrays. I will convert them to float32 tensors, matching the original dsE, and rerun the same fit unchanged. The question is the same as before: is the insulin gate on `n3_0` clean, with insulin minimum above the non-insulin maximum, and is the rate decoder linear with a small residual, on factual, ×2, ×0.5 and randomly scaled doses?

```python
src3b = r'''
def ds_edit(ds, drug, f, rng=None):
    import numpy as np, torch
    at2 = ds['at'].copy(); m = (at2.event == drug).values
    fac = f(m.sum(), rng) if callable(f) else f
    at2.loc[m, 'rate'] = at2.loc[m, 'rate'].values * fac
    at2.loc[m, 'dose'] = at2.loc[m, 'dose'].values * fac
    e, u = G['encode_bundles'](at2, ds['bd']['row'].values.astype(np.int64))
    d2 = dict(ds); d2['at'] = at2
    d2['e'] = torch.as_tensor(np.asarray(e), dtype=torch.float32); d2['u'] = torch.as_tensor(np.asarray(u), dtype=torch.float32)
    return d2
'''
open('/work/lib3.py','a').write(src3b); importlib.reload(lib3); lib3.G=globals()
print(type(dsE['e']))
rng=np.random.default_rng(1)
DS={'fac':dsE, 'x2':lib3.ds_edit(dsE,'insulin',2.0), 'x05':lib3.ds_edit(dsE,'insulin',0.5),
    'rnd':lib3.ds_edit(dsE,'insulin',lambda n,r: np.exp(r.uniform(np.log(.35),np.log(2.8),n)), rng)}
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