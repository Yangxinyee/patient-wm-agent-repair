Norepinephrine is worse. Its lag-1 innovation is barely readable from the head query: R² 0.09, slope 0.10. A readout-only edit would need about ten times the gain, which the factual budget can't absorb. So at least for norepinephrine the dose signal must be strengthened inside the core. That needs a working replica of the encoders, or at least a verified one.

I may have dismissed the z posterior check too early. After I fixed the NaN bug, measurement atoms receive a unit embedding, but I never redid the unit-row search; the earlier search ran when measurement units weren't applied at all. I'll redo it now, looking first at reading-only bundles, where u = 0. That isolates the measurement encoder plus the context.
- A clear minimum would put the replica back on track.
- If the error stays near 0.8 for every mapping, I'll drop this route for good.

```python
def zpost_batch(umap,rumap,act='gelu',use_gap=True,types=None,nst=12):
    errs=[]
    for s_ in list(np.unique(sid_))[:nst]:
        ii=starts[s_]; B=bund[s_]
        for k in range(1,len(ii)):
            hasm=any(a['ev'] not in ACTION and a['kind']==0 for a in B[k]); hasa=any(a['ev'] in ACTION for a in B[k])
            t_=('M' if hasm else '')+('A' if hasa else '')
            if types and t_ not in types: continue
            j0,j1=ii[k-1],ii[k]; dtk=(hr_[j1]-hr_[j0])*3600
            hprev=X_[j0,192:960].reshape(4,192)*np.exp(-dtk*np.exp(Wt('core.flow.a')))[None,:]
            ctx=X_[j0,:192]+LNf(hprev.mean(0),'core.out_norm')+(M2(np.array([np.log(dtk)/10]),'gap',act) if use_gap else 0)
            e_=enc_set([atom_e(a,umap,rumap) for a in B[k] if a['ev'] not in ACTION],'encode',act)
            u_=enc_set([atom_e(a,umap,rumap) for a in B[k] if a['ev'] in ACTION],'encode_action',act)
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post',act)[:32]
            errs.append(np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean())
    return round(float(np.mean(errs)),3)
res=[]
for m1 in [2,3,4,5,None]:
    for m2 in [2,3,4,5,None]:
        if m1==m2 and m1 is not None: continue
        umap={}
        if m1: umap['mg/dL']=m1
        if m2: umap['mmHg']=m2
        res.append((zpost_batch(umap,{},types={'M'},nst=8),m1,m2))
res.sort(key=lambda r:r[0]); print(res[:6])
```