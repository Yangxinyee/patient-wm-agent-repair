The design builder `build_ins_design2(G, stays)` and the regression `ins_regress2(D, ycol)` are still callable. `ins_regress2` fits tier fixed effects, a quadratic in the current reading, earlier glucose, labs and dose deviations at lags 1–6.

Now I can do the key comparison on about 200 tuning stays. I'll regress the actual next glucose and the model's predicted mean of that same reading on identical regressors, and compare the dose-deviation coefficients. Predicted means come from `f3` (pass 3) and the local head on the sel1 states (`RR.pm`). The reference is the train-set regression in `Dtr2`.

**Expected:** if the model is "deconfounded but weak", its lag-1 coefficient will lie between 0 and the true −2.0. If it is near 0 or positive, the model doesn't use the within-tier dose at all. The correction then needs the full true coefficient, and the readout has to come entirely from my added units.

```python
stays_f3 = [int(s) for s in f3_ids]; stays_s1 = [int(s) for s in ids_states]
st_all = sorted(set(stays_f3) | set(stays_s1))
Dtu = build_ins_design2(Gtu, st_all)
print(Dtu.shape, list(Dtu.columns))
# alignment check: y should equal gf at (s, hour)
chk = [Gtu['gf'].loc[s, h] for s, h in zip(Dtu.s.values[:200], Dtu.hour.values[:200])]
print('y==gf[s,hour] frac:', np.nanmean(np.isclose(np.array(chk, float), Dtu.y.values[:200].astype(float))))
pmF = f3[f3.event=='glucose_finger'][['stay_id','hour','predicted_mean']].rename(columns={'stay_id':'s','predicted_mean':'pmF'})
pmF['hour'] = pmF.hour.round().astype(int)
pmS = RR[RR.event=='glucose_finger'][['stay','hour','pm']].rename(columns={'stay':'s','pm':'pmS'})
pmS['hour'] = pmS.hour.round().astype(int)
Dtu2 = Dtu.copy(); Dtu2['hour'] = Dtu2.hour.round().astype(int)
Dtu2 = Dtu2.merge(pmF, on=['s','hour'], how='left').merge(pmS, on=['s','hour'], how='left')
Dtu2['pm'] = Dtu2.pmF.fillna(Dtu2.pmS)
both_ok = Dtu2.pmF.notna() & Dtu2.pmS.notna()
print('rows', len(Dtu2), 'with pm', Dtu2.pm.notna().sum(), 'overlap F&S', both_ok.sum(), 'MAE F vs S', np.mean(np.abs(Dtu2.pmF-Dtu2.pmS)[both_ok]))
Dm = Dtu2[Dtu2.pm.notna()].reset_index(drop=True)
print('--- train truth (Dtr2, y):'); rt = ins_regress2(Dtr2, ycol='y'); print(rt)
print('--- tuning truth (y):'); ry = ins_regress2(Dm, ycol='y'); print(ry)
print('--- tuning model (pm):'); rm = ins_regress2(Dm, ycol='pm'); print(rm)
```