The run failed on a column name. The factual output calls the reading `observed`, not `value`. Nothing was learned yet, so I'll rerun the same analysis with the right column: the z posterior at finger-glucose and arterial-MAP bundles, regressed on reading surprise (observed minus the original model's predicted mean).

A sizeable R² would mean z carries the news of a reading, and its coefficient vector would be the direction to inject at drug bundles. An R² near zero would mean news reaches the state only through the e_k token, and the z route would be the wrong lever.

```python
for evn in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    fm = F2[F2.event==evn].merge(k3, on=['stay_id','hk'], how='inner')
    s_ = (fm.observed - fm.predicted_mean).values; Zr = Z3[fm.row.values]
    A_ = np.c_[np.ones(len(s_)), s_]; coef, *_ = np.linalg.lstsq(A_, Zr, rcond=None)
    pred_ = A_ @ coef; r2 = 1 - ((Zr-pred_)**2).sum(0)/((Zr-Zr.mean(0))**2).sum(0)
    B_ = np.c_[np.ones(len(s_)), Zr]; cb, *_ = np.linalg.lstsq(B_, s_, rcond=None); r2s = 1 - ((s_-B_@cb)**2).sum()/((s_-s_.mean())**2).sum()
    print(evn, 'n', len(s_), 'surp sd', s_.std().round(2), '| R2 z->surp', round(r2s,3), '| top dims', np.argsort(-r2)[:4], np.sort(r2)[::-1][:4].round(3), '| z sd', Zr.std(0).mean().round(3))
```