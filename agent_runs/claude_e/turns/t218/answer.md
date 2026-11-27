These residuals are contaminated by selection, so I can't use them to judge persistence. The train datasets were chosen as stays that later received both drugs, and sel1 was probably chosen for drug use too. Selecting on future insulin is selecting on future high glucose. That alone produces the large positive residuals at high last-value bins in the train subset (+5 to +17 in bins 12–16, mean +2.8). The pattern is much weaker on sel1.

A real persistence deficit has to show up on random stays. f3 and tu0 are unselected tuning batches, and their factual outputs (fwd_11, fwd_13) give each reading's observed value and the original model's predicted mean. For each finger row I'll take the previous finger value in the same stay and regress the residual on it with tier effects. A significant within-tier slope on these random stays would make the persistence edit worth pursuing; a null result drops it.

```python
def resid_check(path):
    f_ = pd.read_csv(path)
    g_ = f_[f_.event=='glucose_finger'].sort_values(['stay_id','hour']).copy()
    g_['prev'] = g_.groupby('stay_id').observed.shift(1)
    g_['dtp'] = g_.hour - g_.groupby('stay_id').hour.shift(1)
    g_ = g_.dropna(subset=['prev'])
    g_['res'] = g_.observed - g_.predicted_mean
    return g_
GR = pd.concat([resid_check('/work/fwd_11/factual_original.csv'), resid_check('/work/fwd_13/factual_original.csv')], ignore_index=True)
tt = tier5(GR.prev.values)
Xr = np.column_stack([(tt==k).astype(float) for k in range(5)] + [(GR.prev.values-180)/50])
br, ser = ols_se(Xr, GR.res.values)
print('random tuning gf rows', len(GR), 'stays', GR.stay_id.nunique(), 'mean res %.2f' % GR.res.mean())
print('within-tier slope on prev (per 50 mg/dL) %.2f (se %.2f)' % (br[5], ser[5]))
st_means = GR.assign(t=tt).groupby('t').res.agg(['size','mean']).round(2)
print(st_means.T.to_string())
per_stay = GR.assign(t=tt).groupby('stay_id').apply(lambda d: np.polyfit(d.prev, d.res, 1)[0] if len(d) > 5 else np.nan).dropna()
print('per-stay slope mean %.4f  se %.4f (n %d)' % (per_stay.mean(), per_stay.std()/np.sqrt(len(per_stay)), len(per_stay)))
```