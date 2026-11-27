The column name `observed` clashes with pandas' `GroupBy.observed` attribute, which is a boolean. I'll index the column with brackets instead; the analysis is otherwise unchanged.

```python
def resid_check(path):
    f_ = pd.read_csv(path)
    g_ = f_[f_.event=='glucose_finger'].sort_values(['stay_id','hour']).copy()
    g_['prev'] = g_.groupby('stay_id')['observed'].shift(1)
    g_['dtp'] = g_['hour'] - g_.groupby('stay_id')['hour'].shift(1)
    g_ = g_.dropna(subset=['prev'])
    g_['res'] = g_['observed'] - g_['predicted_mean']
    return g_
GR = pd.concat([resid_check('/work/fwd_11/factual_original.csv'), resid_check('/work/fwd_13/factual_original.csv')], ignore_index=True)
tt = tier5(GR.prev.values)
Xr = np.column_stack([(tt==k).astype(float) for k in range(5)] + [(GR.prev.values-180)/50])
br, ser = ols_se(Xr, GR.res.values)
print('random tuning gf rows', len(GR), 'stays', GR.stay_id.nunique(), 'mean res %.2f' % GR.res.mean())
print('within-tier slope on prev (per 50 mg/dL) %.2f (se %.2f)' % (br[5], ser[5]))
st_means = GR.assign(t=tt).groupby('t').res.agg(['size','mean']).round(2)
print(st_means.T.to_string())
ps_ = []
for sid, d in GR.groupby('stay_id'):
    if len(d) > 5 and d.prev.std() > 0: ps_.append(np.polyfit(d.prev, d.res, 1)[0])
ps_ = np.array(ps_)
print('per-stay slope mean %.4f  se %.4f (n %d)' % (ps_.mean(), ps_.std()/np.sqrt(len(ps_)), len(ps_)))
```