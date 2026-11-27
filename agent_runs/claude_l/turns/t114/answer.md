The README confirms the layout:
- readings fall on integer hours;
- each infusion record starts at hour + 5 minutes and runs one hour;
- the codes are insulin 8, norepinephrine 9, finger glucose 6, lab glucose 10, arterial MAP 7, cuff MAP 5.

I'll rebuild per-stay-hour arrays for train and tuning and refit the dosing-policy models (gradient boosting on recent readings and previous doses). I need these to compute the dose residual ε = dose − predicted dose, which the factual-fit test depends on.

Earlier notes put the policy fit at R² about 0.82–0.87 for insulin and 0.93–0.94 for norepinephrine. Getting similar numbers will confirm the rebuild is right.

```python
import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingRegressor
t0=time.time()
tr = pd.read_csv('/data/train.csv')
H0=-3; NH=29
MEAS=['glucose_finger','glucose_lab','map_arterial','map_cuff']; DRUGS=['insulin','norepinephrine']
def hourly(df):
    sids = df.stay_id.unique(); sidx = {s:i for i,s in enumerate(sids)}
    si = df.stay_id.map(sidx).values; out={}
    for ev in MEAS:
        m=(df.event==ev).values; A=np.full((len(sids),NH),np.nan)
        A[si[m], np.round(df.hour.values[m]).astype(int)-H0]=df.value.values[m]; out[ev]=A
    for ev in DRUGS:
        m=(df.event==ev).values; A=np.zeros((len(sids),NH))
        A[si[m], np.floor(df.hour.values[m]).astype(int)-H0]=df.rate.values[m]; out[ev]=A
    return sids,out
def ffill2(A): return pd.DataFrame(A).ffill(axis=1).values
def age_last(A):
    H=A.shape[1]; idx=np.where(~np.isnan(A),np.arange(H)[None,:],-1); last=np.maximum.accumulate(idx,axis=1)
    return np.where(last<0,np.nan,np.arange(H)[None,:]-last).astype(float)
def polfeats(D, tlist=range(0,25)):
    F={m:ffill2(D[m]) for m in MEAS}; G={m:age_last(D[m]) for m in MEAS}
    rows=[]
    for t in tlist:
        c=t-H0; cols=[]
        for m in MEAS:
            cols += [D[m][:,c-l] for l in range(4)] + [F[m][:,c], G[m][:,c]]
        for d in DRUGS: cols += [D[d][:,c-l] for l in (1,2,3)]
        cols.append(np.full(D['insulin'].shape[0], t, float))
        rows.append(np.stack(cols,1))
    return np.stack(rows,1)  # n_stays x n_t x nfeat
sid_tr, Dtr = hourly(tr); sid_tu, Dtu = hourly(tu)
Xtr = polfeats(Dtr); Xtu = polfeats(Dtu)
pol={}
for d in DRUGS:
    ytr = np.stack([Dtr[d][:,t-H0] for t in range(25)],1); ytu=np.stack([Dtu[d][:,t-H0] for t in range(25)],1)
    g=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
    g.fit(Xtr.reshape(-1,Xtr.shape[2]), ytr.ravel()); pol[d]=g
    ptu=g.predict(Xtu.reshape(-1,Xtu.shape[2])).reshape(ytu.shape)
    r=ytu-ptu; on=ytu>0
    print(d,'tuning R2 %.3f'%(1-r.var()/ytu.var()),'resid sd all %.3f on-drug %.3f'%(r.std(), r[on].std()), 'frac on %.2f'%on.mean())
print('time',time.time()-t0, Xtr.shape)
```