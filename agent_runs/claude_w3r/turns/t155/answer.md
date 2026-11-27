Rates are stored as (drug name, value) tuples; the norepinephrine dose equals rate × 80 kg × 60 min / 1000. I'll extract the numeric values and rerun the grouped-CV ridge probe of each start rate on E. Hypothesis: E holds each rate linearly with R² ≥ 0.95, which would justify a rank-1 `b_in` channel. A low R² would refute it.

```python
val = lambda x: float(x[1]) if isinstance(x, tuple) else float(x)
for r in recE: r['ri'], r['rn'] = val(r['ri']), val(r['rn'])
for nm, hk in [('ri','has_i'),('rn','has_n')]:
    yv = np.array([r[nm] for r in recE]); msk = np.array([r[hk] for r in recE])
    p = cvp(RidgeCV(alphas=np.logspace(-4,2,13)), EA[msk], yv[msk], groups=gr[msk], cv=GroupKFold(5))
    print(nm, 'n', msk.sum(), 'R2', round(1-np.var(yv[msk]-p)/np.var(yv[msk]),4), 'resid sd', round(np.std(yv[msk]-p),4), 'y sd', round(np.std(yv[msk]),3),
          'uniq', np.unique(yv[msk]).size)
```