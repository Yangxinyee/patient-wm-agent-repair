Presence readouts are far more precise than rate readouts. Insulin presence is within 0.011 everywhere (‖b‖ 164) and NE presence within 0.032. So both gates will use presence indicators, while the rate carriers supply only the dose value (rms error about 0.017 insulin, 0.015 NE).

Before building the NE half I need to settle a conceptual point. NE rates are persistent (a random walk), so the expected rate given the state includes memory of the previous rate. Subtracting that would cancel a sustained doubled dose after the first hour, while subtracting a MAP-only expectation risks confounding with the patient's baseline.

I'll check the factual residuals of the original MAP predictions. I'll regress observed minus predicted arterial MAP at hour t+1 on the NE rate at hour t, the rate at t−1, and the latest MAP. As a sanity check I'll do the same for insulin on finger glucose (expecting about −2 on rate minus band mean).

```python
dr_ = tu[tu.event.isin(['insulin','norepinephrine'])][['stay_id','hour','event','rate']].copy()
dr_['h'] = np.floor(dr_.hour + 1e-6).astype(int)
rate_tab = dr_.pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='sum').fillna(0.0)
def rate_at(sid, h, ev):
    try: return float(rate_tab.loc[(sid, h), ev])
    except KeyError: return 0.0
res_rows = []
for ev, drug in [('map_arterial','norepinephrine'), ('glucose_finger','insulin')]:
    d = fa[fa.event==ev].sort_values(['stay_id','hour']).drop_duplicates(['stay_id','hour'])
    prev = d.groupby('stay_id').shift(1)
    ok = (d.hour - prev.hour == 1.0).values & (d.hour >= 1).values
    d = d[ok].copy(); pobs = prev[ok].observed.values
    t_ = (d.hour.values - 1).astype(int)
    r0 = np.array([rate_at(s, h, drug) for s, h in zip(d.stay_id.values, t_)])
    r1 = np.array([rate_at(s, h-1, drug) for s, h in zip(d.stay_id.values, t_)])
    resid = (d.observed - d.predicted_mean).values
    if drug=='insulin':
        bm = BM[np.digitize(pobs, [150.0001,180.0001,220.0001,260.0001])]
        X_ = np.c_[np.ones(len(r0)), r0 - bm, bm, pobs]
        names = ['const','r_t - BM','BM','G_t']
    else:
        X_ = np.c_[np.ones(len(r0)), r0 - r1, r1, pobs]
        names = ['const','r_t - r_t-1','r_t-1','MAP_t']
    c_, *_ = np.linalg.lstsq(X_, resid, rcond=None)
    e_ = resid - X_@c_; s2 = e_.var(); se_ = np.sqrt(np.diag(s2*np.linalg.inv(X_.T@X_)))
    print(ev, 'n', len(resid), ' '.join('%s %.2f(%.2f)' % (n_, c, s) for n_, c, s in zip(names, c_, se_)), '| on-drug frac %.2f' % (r0>0).mean())
```