The prelude attention has 4 heads of 48 dimensions, with no biases. Its key/value projection `kv` is 384×192.

Drug bundles carry START atoms, which hold the rate for the coming hour, and END atoms for the record just finished. `bvd` returns (e, u), so I can build the action token u for any rescaled dose.

Two quantities decide whether a clean dose channel is feasible:
1. **Linear readability of the rate.** Can a linear probe on the prelude-normed action token read the drug rate, including at rescaled doses x0, x2 and x3? I'll fit on x0.5–x1.5 and test out of range.
2. **Attention to the action token.** How strongly and how steadily does each prelude head attend from the slots to the u token on drug bundles? I'll compute this from the GPU states.

The plan is confirmed if the probe's held-out R² is above 0.9 with slope near 1 and some head has a large, stable attention weight. It is refuted if the probe fails to extrapolate or all attention weights are tiny and erratic.

```python
import numpy as np, torch, time
P='core.prelude.0'
def T64(x): return (x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))).double()
def ln1(x): return lnp(T64(x), P+'.norm1')
stays=list(BUN.keys()); s=stays[0]; evs=BUN[s][11][1]
u1=T64(bvd(mod_events(evs,{'insulin':1.0}))[1]); u2=T64(bvd(mod_events(evs,{'insulin':2.0}))[1])
print('mod check', round(float((u2-u1).abs().max()),4))
R={'insulin':[], 'norepinephrine':[]}; t0=time.time()
for s in stays:
    for k,(mn,evs) in enumerate(BUN[s]):
        r={d:sum(a[2].rate for a in evs if a[0]==d and a[1]=='start') for d in R}
        for d in R:
            if r[d]<=0: continue
            for m in (0.,0.5,1.,1.5,2.,3.):
                u=bvd(mod_events(evs,{d:m}))[1]
                R[d].append((s,k,m,r[d]*m,ln1(u).numpy()))
from sklearn.linear_model import Ridge
PROBE={}
for d in R:
    m_=np.array([x[2] for x in R[d]]); y=np.array([x[3] for x in R[d]]); X=np.stack([x[4] for x in R[d]])
    tr=np.isin(m_,[0.5,1.,1.5]); mdl=Ridge(alpha=1e-3).fit(X[tr],y[tr]); p=mdl.predict(X)
    out=[]
    for mm in (0.,0.5,1.,2.,3.):
        i=m_==mm; out.append(f"m{mm}:{p[i].mean():.2f}/{y[i].mean():.2f}")
    te=~tr; r2=1-((p[te]-y[te])**2).sum()/((y[te]-y[te].mean())**2).sum()
    PROBE[d]=(mdl.coef_.copy(), float(mdl.intercept_)); print(d, len(y), 'R2te', round(r2,3), ' '.join(out))
print('time', round(time.time()-t0,1)); print('X_of', inspect.signature(X_of))
```