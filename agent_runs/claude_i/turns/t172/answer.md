The kernels are stored: insulin's effect on glucose peaks at hour 2 and lasts about 9 hours, summing to about −12 mg/dL per u/h. Norepinephrine's effect on MAP is front-loaded over about 5 hours, summing to about +8.6 mmHg per mcg/kg/min. The original dose_edit table gives the model's shift per stay-hour.

Next I'll regress the model's per-stay-hour shift on the lagged dose changes (Δrate at lags 1–12). That recovers the kernel the model implicitly uses, which I can compare with the true one. If the model's kernel is a scaled-down copy of K, the fix is mainly a gain. If it is mostly lag 1 with little after, the edit must also add persistence.

```python
def lagX(de_sub, drug, mult):
    d = tune[(tune.event==drug)&(tune.hour>=12)].copy()
    d['j'] = np.floor(d.hour).astype(int)
    R = d.groupby(['stay_id','j']).rate.sum()
    X = np.zeros((len(de_sub),12))
    for i,(s,t) in enumerate(zip(de_sub.stay_id.values, de_sub.hour.values)):
        for L in range(1,13):
            X[i,L-1] = R.get((s,t-L),0.0)*(mult-1)
    return X
res_k = {}
for ed_,drug,mult,K_ in [('insulin_x2','insulin',2,K_ins),('insulin_x0.5','insulin',0.5,K_ins),('ne_x2','norepinephrine',2,K_ne),('ne_x0.5','norepinephrine',0.5,K_ne)]:
    sub_ = de[(de.edit==ed_)]
    if len(sub_)==0: print('no', ed_, de.edit.unique()); continue
    X_ = lagX(sub_, drug, mult); y_ = sub_['shift'].values; rch = sub_.reached.values
    Xr, yr = X_[rch], y_[rch]
    coef = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    res_k[ed_] = coef
    print(ed_, 'n', rch.sum(), 'model mean', yr.mean().round(3), 'true-kernel mean', (Xr@K_).mean().round(3))
    print('  model k(L):', coef.round(2))
```