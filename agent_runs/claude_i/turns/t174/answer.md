`pibar` returns the policy's expected insulin rate given a reading: 0 below 150, about 1, 2 and 3 in the three bands above, and 3.5 above 250. The state arrays index bundles by stay and hour, with flags for insulin and NE bundles.

Now the decodability test. For the state after each drug bundle at hour j, I'll ridge-regress (CV by stay) the insulin rate at j, j−1, j−2 and j−3 and the NE rates at the same lags. I'll do the same for the insulin deviation rate − pibar(reading). A cross-validated R² well above 0.5 at lags 0–2 means a head-side readout of dose memory from the state is feasible. R² near 0 means the state forgets doses and the edit has to act on the dynamics.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
tu = tune[tune.stay_id.isin(set(S3))]
def hourly(ev_, col):
    d = tu[tu.event==ev_].copy(); d['j']=np.floor(d.hour).astype(int)
    return d.groupby(['stay_id','j'])[col].sum().to_dict()
Rins = hourly('insulin','rate'); Rne = hourly('norepinephrine','rate')
gfh = tu[tu.event=='glucose_finger'].assign(j=lambda d: np.round(d.hour).astype(int)).groupby(['stay_id','j']).value.first().to_dict()
glh = tu[tu.event=='glucose_lab'].assign(j=lambda d: np.round(d.hour).astype(int)).groupby(['stay_id','j']).value.first().to_dict()
def dev_ins(s,j):
    r = Rins.get((s,j),0.0); gv = gfh.get((s,j), glh.get((s,j), np.nan))
    pb = 0.0 if np.isnan(gv) else float(pibar(np.array([gv]))[0])
    return r - pb
drugb = np.where(np.abs(hr_ - np.floor(hr_) - 0.0833) < 0.02)[0]
Xd = X3[drugb]; grp_ = S3[drugb]; J_ = np.floor(hr_[drugb]).astype(int)
Y = {}
for L in range(4):
    Y[f'ins{L}'] = np.array([Rins.get((s,j-L),0.0) for s,j in zip(grp_,J_)])
    Y[f'dev{L}'] = np.array([dev_ins(s,j-L) for s,j in zip(grp_,J_)])
    Y[f'ne{L}'] = np.array([Rne.get((s,j-L),0.0) for s,j in zip(grp_,J_)])
gkf = GroupKFold(5)
for nm_,y_ in Y.items():
    pr_ = np.zeros(len(y_))
    for trn,tst in gkf.split(Xd,y_,grp_):
        m_r = Ridge(alpha=10.0).fit(Xd[trn],y_[trn]); pr_[tst]=m_r.predict(Xd[tst])
    r2 = 1-((y_-pr_)**2).sum()/((y_-y_.mean())**2).sum()
    print(nm_, 'n', len(y_), 'sd', y_.std().round(3), 'cvR2', round(r2,3))
```