The multiplier passes the exogeneity checks.
- **Unrelated to observables:** given the base, m does not depend on current glucose, previous rate, MAP, norepinephrine, hour or the earlier reading (R² ≤ 0.004).
- **Next-hour measurement:** after an insulin hour the next finger reading is taken about 95% of the time, whatever the rate within the base.
- **Omission:** the 3–5% of triggered hours with no insulin are not random. Omission is 5.4% off norepinephrine and 0.7% on it. They are few, but I will treat omission as possibly informative and identify from rate variation among given doses instead.

Next is the first causal estimate: the effect of the rate at hour t on finger glucose at t+1, t+2 and t+3, within each base and adjusting for current glucose. Effects shrinking with k would mean a short-lived drug action; effects growing would mean it accumulates.

```python
d = c[(c.gf.notna()) & (c.base_f>0) & c.ins.notna()].copy()
d['m'] = d.ins/d.base_f
res = []
for k in [1,2,3]:
    dd = d[d[f'gf_p{k}'].notna()]
    # regress gf_{t+k} on rate within base, controls gf_t (by base dummies x gf)
    Xs = pd.get_dummies(dd.base_f.astype(int), prefix='b').astype(float)
    for col in list(Xs.columns): Xs[col+'_gf'] = Xs[col]*dd.gf
    Xs['rate'] = dd.ins
    lr = LinearRegression().fit(Xs, dd[f'gf_p{k}'])
    print('k',k,'n',len(dd),'rate coef', lr.coef_[-1])
    # by base
    for b in [1,2,3]:
        s = dd.base_f==b
        lr2 = LinearRegression().fit(np.c_[dd.gf[s], dd.ins[s]], dd[f'gf_p{k}'][s]); print('   base',b, 'n', s.sum(), 'coef gf, rate', np.round(lr2.coef_,3))
# raw means by rate within base 2 for next-hour change
dd = d[d.gf_p1.notna() & (d.base_f==2)]
print(dd.groupby('ins').apply(lambda x: pd.Series({'n':len(x),'gf':x.gf.mean(),'d1':(x.gf_p1-x.gf).mean()})).to_string())
```