fwd_7's original states are identical to fwd_3, so those factual-record states are reliable. The glucose bins are 10–18 mg/dL wide.

The value head reads only the view (from out and the slots) plus a "last value" register for the same code. So any dose effect has to come through the state after the drug bundle. v3's tiny, wrong-signed result suggests `dI` mostly misses the glucose-belief direction.

I'll measure this directly on fwd_7 (edited minus original, factual records). First, regress Δout and Δslots at insulin and NE bundles on the recorded rates. Second, build two proxies from the original model: the natural reading-surprise direction in out space, and a ridge readout from the post-bundle state to the model's next glucose (or MAP) prediction, taken from fwd_2. Third, project the v3 injection onto them. This tells me the sign and size of v3's effect in mg/dL per unit rate, and how fast it decays over later bundles. A positive mg/dL per unit insulin would confirm the wrong-sign reading of fwd_6.

```python
import numpy as np, pandas as pd
xo = s7o['x'].astype(np.float64); xe = s7e['x'].astype(np.float64); D = xe - xo
sid = s7o['stay_id']; hr = s7o['hour']; hasI = s7o['has_insulin'].astype(bool); hasN = s7o['has_norepinephrine'].astype(bool)
tsel = tun[tun.stay_id.isin(sel)].copy(); tsel['hk'] = tsel.hour.round(4)
key = pd.DataFrame({'stay_id':sid, 'hk':np.round(hr,4), 'row':np.arange(len(sid))})
for ev,nm in [('insulin','rI'),('norepinephrine','rN')]:
    t = tsel[tsel.event==ev].groupby(['stay_id','hk']).rate.sum().rename(nm).reset_index()
    key = key.merge(t, on=['stay_id','hk'], how='left')
key = key.sort_values('row').reset_index(drop=True); rI = key.rI.fillna(0).values; rN = key.rN.fillna(0).values
print('presence check', len(key), ((rI>0)==hasI).mean(), ((rN>0)==hasN).mean())
dout = D[:,:192]; dhm = D[:,192:960].reshape(-1,4,192).mean(1)
same_next = np.r_[sid[1:]==sid[:-1], False]
for nm,m in [('insOnly',hasI&~hasN),('neOnly',hasN&~hasI),('both',hasI&hasN),('none',~hasI&~hasN)]:
    print(nm, m.sum(), 'median|dout|', np.round(np.median(np.linalg.norm(dout[m],axis=1)),4), 'median|dhm|', np.round(np.median(np.linalg.norm(dhm[m],axis=1)),3))
f2 = pd.read_csv('/work/fwd_2/factual_original.csv'); print(f2.columns.tolist()); print(f2.head(3).to_string())
```