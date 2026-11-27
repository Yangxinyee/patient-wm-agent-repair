The checks pass. `HO` is laid out (bundle, head, slot, 48), and it matches `patt` (bundle, head, query, key) combined with `Vall` exactly. `PD` gives per-bundle drug targets aligned with these 5,201 bundles. I will now run the stage-1 read fit. The insulin and norepinephrine start rates are fitted from head 2 alone or from all heads, with the bundles of 25 stays held out. Success means a held-out residual sd clearly below 0.36 (insulin, action bundles) and 0.20 (no-action bundles).

```python
HOc=HO.transpose(0,2,1,3).reshape(5201,4,192)
tgI=PD['ins_s'].values.astype(float); tgN=PD['ne_s'].values.astype(float)
act=(PD['has_ins_s'].values>0)|(PD['has_ne_s'].values>0); SSTp=PD['stay'].values
rng2=np.random.default_rng(0); te=np.isin(SSTp, rng2.choice(np.unique(SSTp),25,replace=False))
def fitread(cols,y,lam=1e-3,wna=1.0):
    X=HOc[:,:,cols].reshape(-1,len(cols)); Y=np.repeat(y,4); A=np.repeat(act,4); T=np.repeat(te,4)
    Xa=np.c_[X,np.ones(len(X))]; w=np.where(A,1.0,wna); tr=~T
    G=(Xa[tr]*w[tr,None]).T@Xa[tr]+lam*np.eye(Xa.shape[1]); beta=np.linalg.solve(G,(Xa[tr]*w[tr,None]).T@Y[tr])
    r=Y-Xa@beta
    return beta,[r[T&A].std(), r[T&~A].std(), r[T&~A].mean()]
print('n act',act.sum(),'test bundles',te.sum())
for nm,y in [('ins',tgI),('ne',tgN)]:
    for lab,cols in [('h2',np.arange(96,144)),('all',np.arange(192))]:
        for wna in [1.0,5.0]:
            b_,st=fitread(cols,y,wna=wna); print(nm,lab,'wna',wna,'test resid sd act %.3f noact %.3f (mean %.3f)'%tuple(st))
```