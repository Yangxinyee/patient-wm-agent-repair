The edit helpers are clear: `edit_atom(a, f)` scales an atom's dose and rate and recomputes its features and bins, as the dose_edit rollout does. `u_for` builds the action encoding of a bundle under an edit.

Next I'll build the set of u vectors the z channel has to handle. That means every drug bundle in the 100 batch-1 stays, recorded and with each drug scaled ×0, ×0.5, ×1.5, ×2 and ×3, which covers every edit the evaluation uses. From that set I'll measure three things:

1. **Copy basis:** how many PCA components, centred on the empty-action encoding so reading bundles are copied exactly, reproduce u closely. The budget is ≤ 30 of z's 32 dimensions; the other 2 carry the residuals.
2. **Presence gates:** whether one linear score of u per drug separates bundles with that drug from bundles without it.
3. **Dose decode:** whether a linear decode of each drug's rate from u still holds at the out-of-range edits (insulin up to 15 U/h, NE up to 3).

Good outcomes would be:
- a copy error under about 1% of |u − u_empty| at k ≤ 30;
- gate scores that are cleanly separated (present > 0.8, absent < 0.2);
- a dose decode error under about 0.1 at ×3.

A poor ×3 decode would mean using bin-aware or piecewise decoders.

```python
W0c[0]=W0
keys=list(BUN.keys()) if isinstance(BUN,dict) else list(range(len(BUN)))
print('DRUGN',DRUGN,'nstays',len(keys))
Ul=[];meta=[]
dcodes=list(DRUGN.keys())
for s in keys:
    for (t,atoms) in BUN[s]:
        ua=[a for a in atoms if a['act'] and a['kind']==0]
        if not ua: continue
        base={}
        for a in ua:
            nm=DRUGN[a['code']]; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']; base[a['code']]=a['num'][4]*ri+rm
        for cond in [None]+[(c,f) for c in dcodes for f in (0.0,0.5,1.5,2.0,3.0)]:
            if cond is not None and cond[0] not in base: continue
            u=u_for(atoms,t,cond,-1.0)
            rr={c:(base[c]*(cond[1] if cond is not None and cond[0]==c else 1.0)) for c in base}
            Ul.append(np.asarray(u,dtype=np.float64).ravel())
            meta.append([rr.get(dcodes[0],0.0),rr.get(dcodes[1],0.0),float(dcodes[0] in base),float(dcodes[1] in base),(cond[1] if cond else 1.0),(cond[0] if cond else -1)])
U_all=np.array(Ul); MU=np.array(meta); ue=np.asarray(u_empty,dtype=np.float64).ravel()
print('U_all',U_all.shape,'|u_empty| %.2f  median |u-ue| %.2f'%(np.linalg.norm(ue),np.median(np.linalg.norm(U_all-ue,axis=1))))
D_=U_all-ue; _,Sd,Vd=np.linalg.svd(D_,full_matrices=False); nr=np.linalg.norm(D_,axis=1)
for k in [12,16,20,24,28,30]:
    E_=D_-D_@Vd[:k].T@Vd[:k]; re=np.linalg.norm(E_,axis=1)/nr
    print('k',k,'rel err median %.4f p99 %.4f max %.4f'%(np.median(re),np.quantile(re,.99),re.max()))
for j,c in enumerate(dcodes):
    X1=np.c_[U_all,np.ones(len(U_all))]; wsg=np.linalg.lstsq(X1,MU[:,2+j],rcond=None)[0]; sg=X1@wsg
    pres=MU[:,2+j]>0.5
    m=pres; wr=np.linalg.lstsq(X1[m],MU[m,j],rcond=None)[0]; er=X1[m]@wr-MU[m,j]
    ext=m&(MU[:,4]==3.0)&(MU[:,5]==c)
    print(DRUGN[c],'gate present min %.3f absent max %.3f | rate decode rmse %.4f maxerr %.3f | x3-edit maxerr %.3f (max rate %.2f)'%(sg[pres].min(),sg[~pres].max(),np.sqrt((er**2).mean()),np.abs(er).max(),np.abs(er[ext[m]]).max() if ext.any() else -1,MU[m,j].max()))
```