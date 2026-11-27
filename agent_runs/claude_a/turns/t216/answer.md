The no-staging design fails. From `n3_0` the insulin gate overlaps: the lowest insulin score is −0.05 while NE-only bundles reach 0.14. The rate decoder is also poor: residual sd 0.5–1.5 units/h, with slopes of 0.85 on factual doses but only 0.55 at ×2 and 0.32 on the random set. Rate information reaches the block stream too squashed and mixed to use linearly, especially at the extrapolated rates the edits produce.

So I go back to the staging design, which takes the rate from head 3's value of the drug token. Head 3 attends u with weight 1.0 at every insulin bundle, so its value is a clean function of u alone.

Before committing two extra dims, I will test linear decodability of the insulin rate from head 3's 48-dim value, across factual, ×2, ×0.5 and random doses:
- target = insulin rate at insulin and both-drug bundles;
- target = 0 at NE-only bundles, where head 3 also attends u.

How I will read it:
- Residual sd ≲ 0.25 with slope ≈ 1 on every dataset makes staging viable.
- If extrapolation breaks here too, the encoder itself is nonlinear in rate, and I will need a nonlinear staging, for example through the prelude FFN.

```python
def head3_vals(P, ds, ST_):
    out=[]
    for i in range(0, len(ST_['hf']), 2000):
        s=slice(i,i+2000)
        pr=prelude(P, ST_['hf'][s], ds['e'][s], ds['u'][s], ST_['z'][s])
        out.append(pr['vh'][:,3,5,:])   # head-3 value of the u token (B x 48)
    return torch.cat(out).numpy()
V3={}
for k,d_ in DS.items():
    st_=lib3.internals(P2,d_) if k!='fac' else lib3.internals(P2,dsE)
    V3[k]=head3_vals(P2,d_,st_)
isNE=NEr>0
def fitdec(target_fn, mask_fn):
    Xa=[];ya=[]
    for k in DS:
        m=mask_fn(k); Xa.append(V3[k][m]); ya.append(target_fn(k)[m])
    Xa=np.concatenate(Xa); ya=np.concatenate(ya); X1=np.c_[Xa,np.ones(len(Xa))]
    return np.linalg.solve(X1.T@X1+1e-4*np.eye(49), X1.T@ya)
drugm=lambda k: (RS[k]>0)|isNE
wI=fitdec(lambda k: RS[k], drugm)
for k in DS:
    d_=np.c_[V3[k],np.ones(len(V3[k]))]@wI; on=RS[k]>0; neo=(~on)&isNE
    print(k,'ins resid sd',round(np.std(d_[on]-RS[k][on]),3),'slope',np.round(np.polyfit(RS[k][on],d_[on],1),3),'| NE-only |dec| p99',round(np.percentile(np.abs(d_[neo]),99),3),'| both-drug resid sd',round(np.std(d_[on&isNE]-RS[k][on&isNE]),3))
```