Within every glucose stratum, the change in glucose is flat across insulin multiplier levels, at both 1 and 3 hours. All differences are within about ±1–2 mg/dL with no monotone trend; the few larger cells in high strata, if anything, point the wrong way for a glucose-lowering drug and are within noise. My current belief is that insulin has no effect on glucose_finger in this ICU, so the true readouts for insulin x2 and x1/2 should be about 0.

Next is norepinephrine. Its rate depends on history (previous rate, current MAP), so I will use double machine learning with cross-fitting: residualise both ΔMAP(h→h+k) and the rate at h on flexible confounders, then regress residual on residual. The confounders are MAP at h and h−1, previous rate, the rate two hours back, MAP source and hour. I will restrict to stays with an arterial line and use map_arterial, which is what the readout uses.

A clearly positive coefficient (MAP up per unit of rate) means a real vasopressor effect. Zero means no effect. I will run the same method on insulin as a cross-check.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
H = Q[['map_arterial','map_cuff','map_obs','ne','ins','glucose_finger']].copy()
grp = H.groupby(level='stay_id')
H['ne_m1'] = grp.ne.shift(1); H['ne_m2'] = grp.ne.shift(2)
H['ma_m1'] = grp.map_arterial.shift(1); H['ma_m2'] = grp.map_arterial.shift(2)
H['mc'] = H.map_cuff
for k in [1,2,3,4,6]: H[f'ma_p{k}'] = grp.map_arterial.shift(-k)
H['hh'] = H.index.get_level_values('h')
def dml(df, ycol, tcol, xcols, nfold=4, seed=0):
    df = df.dropna(subset=[ycol, tcol])
    X = df[xcols].values; y = df[ycol].values; t = df[tcol].values
    groups = df.index.get_level_values('stay_id').values
    ry = np.zeros(len(df)); rt = np.zeros(len(df))
    for trn, tst in GroupKFold(nfold).split(X, y, groups):
        my = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=seed).fit(X[trn], y[trn])
        mt = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=seed).fit(X[trn], t[trn])
        ry[tst] = y[tst]-my.predict(X[tst]); rt[tst] = t[tst]-mt.predict(X[tst])
    b, se, _ = ols(ry, rt[:,None], groups)
    return b[0], se[0], len(df), np.var(rt)
dA = H[H.map_arterial.notna()].copy()
for k in [1,2,3,6]:
    dA['dM'] = dA[f'ma_p{k}'] - dA.map_arterial
    print('NE on MAP_art, k=',k, dml(dA, 'dM', 'ne', ['map_arterial','ma_m1','ma_m2','ne_m1','ne_m2','mc','hh']))
```