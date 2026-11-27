The states in fwd_3 come from training stays (ids 1010…), not tuning stays (2020…). That is why every lookup in the tuning table failed. I'll rerun the same decodability test with the training records. A high cross-validated R² for the lag 0–2 rates means a head-side readout is feasible. A low one means I must edit the dynamics.

```python
tu = train[train.stay_id.isin(set(S3.tolist()))]
print(len(tu), tu.stay_id.nunique())
Rins = hourly('insulin','rate'); Rne = hourly('norepinephrine','rate')
gfh = tu[tu.event=='glucose_finger'].assign(j=lambda d: np.round(d.hour).astype(int)).groupby(['stay_id','j']).value.first().to_dict()
glh = tu[tu.event=='glucose_lab'].assign(j=lambda d: np.round(d.hour).astype(int)).groupby(['stay_id','j']).value.first().to_dict()
Y = {}
for L in range(4):
    Y[f'ins{L}'] = np.array([Rins.get((s,j-L),0.0) for s,j in zip(grp_,J_)])
    Y[f'dev{L}'] = np.array([dev_ins(s,j-L) for s,j in zip(grp_,J_)])
    Y[f'ne{L}'] = np.array([Rne.get((s,j-L),0.0) for s,j in zip(grp_,J_)])
for nm_,y_ in Y.items():
    pr_ = np.zeros(len(y_))
    for trn,tst in gkf.split(Xd,y_,grp_):
        m_r = Ridge(alpha=10.0).fit(Xd[trn],y_[trn]); pr_[tst]=m_r.predict(Xd[tst])
    r2 = 1-((y_-pr_)**2).sum()/((y_-y_.mean())**2).sum()
    print(nm_, 'sd', y_.std().round(3), 'cvR2', round(r2,3), end=' | ')
```