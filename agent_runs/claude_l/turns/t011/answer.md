The run failed on a stray import line I wrote (and statsmodels isn't installed anyway), so nothing executed. I'll rerun the same analysis with plain numpy least squares for the negative-control regression.

The plan is unchanged. Fit the insulin policy on recorded history and report holdout R² and residual SD. Then regress same-hour lab glucose on finger glucose, the policy features and the policy residual. If the residual has a clearly nonzero coefficient on lab_t, clinicians are using information the record lacks, which would confirm hidden-state confounding. A coefficient near zero would refute it.

```python
def ols(y, X, names=None):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    r = y - X @ beta
    s2 = (r**2).sum()/(len(y)-X.shape[1])
    se = np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    return beta, se
X_cols = ['glucose_finger','g_prev','g_prev2','ins_prev','gl_prev_last','ih','ne_prev','mc_last','ma_last']
m = d[d.glucose_finger.notna()].copy()
idx_tr = (m.stay_id % 5 != 0).values
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1)
gb.fit(m.loc[idx_tr, X_cols], m.loc[idx_tr,'insulin'])
m['ins_hat'] = gb.predict(m[X_cols]); m['ires'] = m.insulin - m.ins_hat
res = m.ires[~idx_tr]
print("R2 holdout:", 1 - res.var()/m.insulin[~idx_tr].var(), "resid sd", res.std())
mm = m[m.glucose_lab.notna() & (~idx_tr)]
beta, se = ols(mm.glucose_lab.values, [mm.glucose_finger, mm.g_prev.fillna(mm.glucose_finger), mm.ins_hat, mm.ires])
print("lab_t ~ finger_t, g_prev, ins_hat, ires:", beta.round(3), se.round(3))
# same for those on insulin region only (g>160)
mm2 = mm[mm.glucose_finger>160]
beta, se = ols(mm2.glucose_lab.values, [mm2.glucose_finger, mm2.ins_hat, mm2.ires]); print("g>160:", beta.round(3), se.round(3))
```