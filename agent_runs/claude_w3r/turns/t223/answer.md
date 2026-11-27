The names fix the stage targets:
- `core.prelude.0.attn.out.weight` (192×192, no bias), the out-projection of the four 48-dimensional heads, for stage 1.
- `core.b_in.weight` for stage 2.
- `attr_heads.value_bin.0/.2`, a 192-unit layer then 17 logits, for stage 3.
- `embed_transition.value_bin` (13 codes × 17 last bins) for the compensation.

Next, stage 1: from the stored per-bundle head outputs (`HO`, 5,201 bundles × 4 slots), I will fit a linear read ℓ of the insulin and norepinephrine start rates that should give the true rate at action bundles and stay near zero elsewhere. I will compare reading head 2 alone with reading all heads. Head 1 sees the e token everywhere, so it could cancel the noise head 2 picks up from e when no drug starts. Success means a residual sd well below my earlier 0.36 (insulin) at action bundles and below 0.20 elsewhere.

```python
HOc=HO.reshape(5201,4,192)          # assume (bundle, slot, head, 48) -> concat heads
chk=np.einsum('bhij,bjhd->bihd',patt[:,:, :4,:],Vall)  # recompute slot head outputs
print('HO check rel err', np.abs(chk-HO).max()/np.abs(HO).max())
tgI=np.asarray(T_comb['ins_s'],float); tgN=np.asarray(T_comb['ne_s'],float)
act=(tgI>0)|(tgN>0); print('n act',act.sum(),'n ins',(tgI>0).sum(),'n ne',(tgN>0).sum())
rng2=np.random.default_rng(0); stays_u=np.unique(SST); te=np.isin(SST, rng2.choice(stays_u,25,replace=False))
def fitread(cols,y,lam=1e-3,wna=1.0):
    X=HOc[:,:,cols].reshape(-1,len(cols)); Y=np.repeat(y,4); A=np.repeat(act,4); T=np.repeat(te,4)
    Xa=np.c_[X,np.ones(len(X))]; w=np.where(A,1.0,wna)
    tr=~T; G=(Xa[tr]*w[tr,None]).T@Xa[tr]+lam*np.eye(Xa.shape[1]); beta=np.linalg.solve(G,(Xa[tr]*w[tr,None]).T@Y[tr])
    r=Y-Xa@beta
    return beta,[r[T&A].std(), r[T&~A].std(), r[T&~A].mean()]
for nm,y in [('ins',tgI),('ne',tgN)]:
    for lab,cols in [('h2',np.arange(96,144)),('all',np.arange(192))]:
        for wna in [1.0,5.0]:
            b_,st=fitread(cols,y,wna=wna); print(nm,lab,'wna',wna,'test resid sd act %.3f noact %.3f (mean %.3f)'%tuple(st))
```