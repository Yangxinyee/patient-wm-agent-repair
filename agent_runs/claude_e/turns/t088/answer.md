Nonlinear decoding helps a lot for lag 1: a gradient-boosted model on the full state reaches R² 0.70 for the lag-1 deviation. The combined lags 1–6 target only reaches 0.36, so the longer lags are not in the state at all, and a head-only edit can use at most the last one or two doses.

The head sees only q, which is built from `out` and the slot-mean of h, not the full state. Next I will measure the nonlinear ceiling from what the head can actually see: q, `out` plus slot-mean h, and the z posterior for comparison. A ceiling near 0.5 or more for lag 1 would make a gated head readout roughly NLL-neutral. A ceiling near 0.3 means scaling up the readout costs too much factual NLL.

```python
def gbm_cv(F, y, grp, it=300):
    pr = np.zeros(len(y))
    for tr_, te_ in GroupKFold(5).split(F, y, grp):
        pr[te_] = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05).fit(F[tr_], y[tr_]).predict(F[te_])
    return round(1-np.mean((y-pr)**2)/y.var(),3), pr
res = {}
for nm, F in [('q', Fq), ('out+hmean', np.hstack([np.vstack([PA_['out'],P1_['out']]), np.vstack([PA_['hmean'],P1_['hmean']])])), ('zpost', np.vstack([PA_['zpost'],P1_['zpost']]))]:
    r_l1, _ = gbm_cv(F, Lg[:,0], grp); r_l2, _ = gbm_cv(F, Lg[:,1], grp)
    print(f'{nm:10s} GBM lag1 {r_l1}  lag2 {r_l2}')
# raw dose at lag1 (not deviation) and the tier-expected part, from q
gfv_prev = []
ins_l1 = np.concatenate([Gtr['ins'].fillna(0).loc[rg.stay.values].values[np.arange(len(rg)), [list(Gtr['ins'].columns).index(h-1) for h in rg.hour.values]],
                         Gtu['ins'].fillna(0).loc[r1.stay.values].values[np.arange(len(r1)), [list(Gtu['ins'].columns).index(h-1) for h in r1.hour.values]]])
print('raw dose lag1 from q: linear', round(cv_r2(Fz, ins_l1, grp, 10.)[0],3), ' GBM', gbm_cv(Fq, ins_l1, grp)[0], ' frac rows with insulin at lag1', np.mean(ins_l1>0).round(3))
```