The import never ran because the previous cell failed to parse, which caused the NameError. I'll re-import and run the same analysis, with the same hypothesis and the same confirm/refute criterion: a policy residual that predicts same-hour lab glucose means unrecorded information drives dosing.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
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
mm2 = mm[mm.glucose_finger>160]
beta, se = ols(mm2.glucose_lab.values, [mm2.glucose_finger, mm2.ins_hat, mm2.ires]); print("g>160:", beta.round(3), se.round(3))
```