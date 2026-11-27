With flexible controls, insulin's coefficient on the next-hour change in glucose_finger is -1.99 (SE 0.045). The within-band noise in the rate alone gives -2.00 (SE 0.047). Because the within-band scatter behaves like independent noise, I take this as solid evidence that insulin lowers the next-hour glucose by about 2 mg/dL per unit/h. The +3.1 coefficient on the previous rate probably reflects measurement-error confounding (it proxies the true glucose at t-1), not a drug effect.

What matters for "x2 from hour 12" is the dynamics: does a dose's effect persist and accumulate over later hours, or wash out after one hour? Next I'll estimate the response of glucose at lags k = 1..6 to the exogenous within-band deviation of the dose at hour t, controlling for glucose at t. Policy feedback through later bands should be small, about 0.1 mg/dL. A lag-2+ response near zero would refute persistence (effect only in the next hour). Responses near -2 that decay slowly would confirm an accumulating effect, making the x2-from-hour-12 effect much larger than a one-hour shift.

```python
gbest = np.where(np.isfinite(gf), gf, gl)
def band_of(g): return np.digitize(g, [140,160,180,220,260])
rows=[]
for t in range(0,24):
    j=col(t)
    b = band_of(glast[:,j])
    rows.append(pd.DataFrame({'t':t,'g0':gf[:,j],'glast':glast[:,j],'band':b,'ins':ins[:,j],'ins_prev':ins[:,j-1],
                              **{f'g{k}': (gf[:,j+k] if j+k < len(hrs) else np.nan) for k in range(1,7)}, 'stay':np.arange(len(ids))}))
L = pd.concat(rows, ignore_index=True)
L = L[(L.ins>0)].copy()
L['dev'] = L.ins - L.groupby('band').ins.transform('mean')
for k in range(1,7):
    s = L.dropna(subset=['g0',f'g{k}'])
    X = [s.dev.values, s.g0.values] + [(s.band==b).values.astype(float) for b in sorted(s.band.unique())[1:]]
    r = ols((s[f'g{k}']-s.g0).values, X, ['dev','g0']+[f'b{b}' for b in sorted(s.band.unique())[1:]])
    print(k, "n", len(s), "dev coef", r.loc['dev','b'].round(3), "se", r.loc['dev','se'].round(3))
```