The CSV has, per stay and hour, the shift, its standard error, the factual predicted mean and whether the drug had already run.

To see the response by lag, I pair it with the recorded tuning doses. I compute each hour's dose change under the edit: +100% of the recorded rate for ×2, -50% for ×½, from hour 12 on. Then I fit a distributed-lag regression of the shift at hour t on the dose changes at t-1 to t-6, for each model and each of the four edits.

The coefficients are the rollout's effective impulse response per unit dose change. I can compare them directly with the structural ones:

- insulin: -2.0, -2.4, -2.1, -1.7, -1.2, -0.9;
- NE: 3.3, 2.7, 2.0, 1.2, 0.4, 0.1.

```python
def dose_panel(df, drug):
    x = df[df['event']==drug].copy(); x['h'] = np.floor(x['hour']+1e-6).astype(int)
    return x.groupby(['stay_id','h'])['rate'].sum()
DPi = dose_panel(tu, 'insulin'); DPn = dose_panel(tu, 'norepinephrine')
fac = {'insulin_x2':(DPi,1.0), 'insulin_half':(DPi,-0.5), 'ne_x2':(DPn,1.0), 'ne_half':(DPn,-0.5)}
def irf(dfm, ed, K=6):
    x = dfm[dfm['edit']==ed].copy()
    DP, f = fac[ed]
    X = []
    for k in range(1,K+1):
        hh = x['hour'].values-k
        v = DP.reindex(pd.MultiIndex.from_arrays([x['stay_id'].values, hh])).fillna(0).values*f
        X.append(np.where(hh>=12, v, 0.0))
    X = np.column_stack(X)
    b = np.linalg.lstsq(X, x['shift'].values, rcond=None)[0]
    return np.round(b,2)
for ed in fac:
    print(ed, 'orig', irf(o8, ed), 'edit', irf(d8, ed))
```