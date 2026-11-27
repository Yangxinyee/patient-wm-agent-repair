Two findings from that output:
- The states file holds 4,932 bundles from batch1.
- My `core_step` mirrors the spec (time flow, ctx, z posterior, prelude over [slots, e, u, z_in], two recurrent layers with age-keyed memory cross-attention, `a_bar` update, `c_out`, coda).

Before more replica work I want a cheaper diagnostic: is the dose innovation (dose minus policy-expected dose) already represented at the drug bundle? I'll decode it with cross-validated ridge regressions from three blocks of the tool's states:
- the z posterior mean;
- `out`;
- the slot-mean of h.

The decoded quantities are the current dose, its policy prediction, the innovation ε, and the previous dose carried by the END atom.

- **If ε is well decoded from z:** the posterior is where the innovation lives, and a repair can amplify its downstream effect without touching on-policy predictions.
- **If ε is poorly decoded everywhere:** the innovation must be built from u_k and ctx myself.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
X=npz['x']; hr=npz['hour']; sid=npz['stay_id']
frac=hr-np.floor(hr); isdrug=np.abs(frac-0.0833)<0.01
t=np.floor(hr).astype(int); si=np.array([sidx_tu.get(s,-1) for s in sid])
ok=isdrug&(si>=0)&(t>=1)&(t<=24)
def tgt(d):
    dt_=Dtu[d][si[ok],t[ok]-H0]; dprev=Dtu[d][si[ok],t[ok]-1-H0]; ph=Ptu[d][si[ok],t[ok]]
    return dict(d=dt_, pihat=ph, eps=dt_-ph, dprev=dprev)
blocks={'z':X[ok,960:992],'out':X[ok,0:192],'hmean':X[ok,192:960].reshape(-1,4,192).mean(1)}
grp=sid[ok]
for d in DRUGS:
    T_=tgt(d); act=(T_['d']>0)|(T_['dprev']>0)
    print(d,'n bundles',act.sum())
    for bn,B in blocks.items():
        res=[]
        for tn in ['d','pihat','eps','dprev']:
            y=T_[tn][act]; Xb=B[act]
            p=cross_val_predict(RidgeCV(alphas=np.logspace(-3,3,13)),Xb,y,cv=GroupKFold(5),groups=grp[act])
            res.append('%s %.2f'%(tn,1-np.mean((y-p)**2)/y.var()))
        print('   ',bn,' '.join(res))
```