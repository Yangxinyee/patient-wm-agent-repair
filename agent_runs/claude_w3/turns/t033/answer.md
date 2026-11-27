The file has one row per reading with the observed value, the model's predicted mean and the bin NLL. Readings at the same hour share a bundle and are predicted from the state before it.

Now the sensitivity test, on the tuning stays. For each hour t with a fresh policy glucose above 150, I take the insulin jitter z_t = d_t − m(pol_t), using the train-based policy means. I then regress the model's predicted glucose_finger at t+1, and separately the observed reading, on z_t with pol_t as a control. With about 100 stays the estimate will be noisy, but a model coefficient near 0 against a data coefficient near −2 would settle it.

```python
tupp = tup.copy()
tupp['pol'] = np.where(tupp.glucose_finger.notna(), tupp.glucose_finger, tupp.glucose_lab)
tupp['m'] = np.where(tupp.pol.notna(), mtab_full.values[np.clip(np.nan_to_num(np.round(tupp.pol)).astype(int),0,599)], 0.0)
tupp['z'] = np.where(tupp.pol.notna(), tupp.insulin - tupp.m, 0.0)
fa['h'] = fa.hour.round().astype(int)
pm = fa[fa.event=='glucose_finger'].set_index(['stay_id','h'])['predicted_mean']
ob = fa[fa.event=='glucose_finger'].set_index(['stay_id','h'])['observed']
tupp = tupp.set_index(['stay_id','h'])
nxt = tupp.reset_index()[['stay_id','h']].copy(); nxt['h'] = nxt.h+1
tupp['pred_next'] = pm.reindex(list(zip(nxt.stay_id, nxt.h))).values
tupp['obs_next'] = ob.reindex(list(zip(nxt.stay_id, nxt.h))).values
tupp['pred_now'] = pm.reindex(tupp.index).values
w = tupp[(tupp.pol>150) & tupp.pred_next.notna()].copy()
print("n hours:", len(w))
for yv in ['pred_next','obs_next']:
    X = np.column_stack([np.ones(len(w)), w.pol, w.z])
    b = np.linalg.lstsq(X, w[yv].values, rcond=None)[0]; res = w[yv].values - X@b
    se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var())
    print(yv, " coef [1, pol, z]:", b.round(3), " se z:", se[2].round(3))
# model's dependence on last reading: pred_next vs current reading (all hours with finger now and next)
w2 = tupp[tupp.glucose_finger.notna() & tupp.pred_next.notna()]
X = np.column_stack([np.ones(len(w2)), w2.glucose_finger, w2.pred_now.fillna(w2.glucose_finger)])
b = np.linalg.lstsq(X, w2.pred_next.values, rcond=None)[0]
print("pred_next ~ [1, current reading, current prediction]:", b.round(3))
```