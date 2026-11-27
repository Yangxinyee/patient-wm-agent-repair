The factual file has, for each reading, the observed value and the original model's predicted mean. I'll pair consecutive hourly readings of the same type in the same stay and regress the next prediction on the previous observation and prediction. This gives the model's persistence for a level shift (c1+c2) and for an observation-only shift (c1), for finger glucose and arterial MAP.

```python
def persist(ev):
    d = fa[fa.event==ev].sort_values(['stay_id','hour'])
    d = d.drop_duplicates(['stay_id','hour'])
    nxt = d.groupby('stay_id').shift(-1)
    ok = (nxt.hour - d.hour == 1.0)
    X = np.c_[np.ones(ok.sum()), d.observed[ok], d.predicted_mean[ok]]
    y = nxt.predicted_mean[ok].values
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X@c
    print(ev, 'n', ok.sum(), 'c_obs %.3f c_pred %.3f sum %.3f' % (c[1], c[2], c[1]+c[2]), 'resid sd %.2f' % r.std(),
          'obs sd %.1f' % d.observed[ok].std())
for ev in ['glucose_finger','map_arterial','map_cuff']: persist(ev)
```