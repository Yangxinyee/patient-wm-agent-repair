Norepinephrine is titrated, not drawn from a scale.
- It is started when MAP falls below about 64–65 (always below 64, 54% at 64–65, rarely above), with a starting rate that is higher when MAP is lower.
- Once running, the hourly change has a large spread (SD 0.09–0.18), with drift up below 65 and down above 75, and stops become common above 75.

So the rate depends on the previous rate and the current MAP plus noise. If that noise is iid and the policy reads only observed MAP, then conditional on (MAP_t, rate_{t−1}) the remaining dose variation is exogenous. That is the same identification as for insulin.

Next I will fit the titration policy flexibly (MAP-bin fixed effects × previous-rate bins) on arterial-MAP hours with norepinephrine running. I will check the residual's autocorrelation and its relation to the MAP trend. Then I will run local projections of MAP(t+k) − MAP(t) on the rate, with fixed effects for the MAP bin × previous-rate bin.

```python
P['map_any'] = P['ma'].combine_first(P['mc'])
P['ma_prev'] = g['ma'].shift(1)
for k in range(1,9):
    P[f'ma_f{k}'] = g['ma'].shift(-k)
    P[f'nor_f{k}'] = g['nor'].shift(-k)
d = P[(P['nor']>0) & P['ma'].notna() & (P['nor_prev']>0)].copy().reset_index()
d['mbin'] = (d['ma'].clip(20,110)//2).astype(int)
d['pbin'] = (d['nor_prev']*20).round().astype(int)   # 0.05 bins of previous rate
d['cell'] = d['mbin']*100 + d['pbin']
# policy residual
D = d.copy()
D['res'] = D['nor'] - D.groupby('cell')['nor'].transform('mean')
D = D.set_index(['stay_id','ih'])
D['res_prev'] = D.groupby(level=0)['res'].shift(1)
print("policy residual sd", D.res.std().round(4), " autocorr", D[['res','res_prev']].corr().iloc[0,1].round(4))
D['dMprev'] = D['ma'] - D['ma_prev']
print("corr res with dMprev", D[['res','dMprev']].dropna().corr().iloc[0,1].round(4), " with hour", np.corrcoef(D.res, D.index.get_level_values(1))[0,1].round(4))
for k in range(1,9):
    d[f'dM{k}'] = d[f'ma_f{k}'] - d['ma']
    b,se,n = fe_ols(d, f'dM{k}', ['nor'], 'cell')
    b2,se2,_ = fe_ols(d, f'nor_f{k}', ['nor'], 'cell')
    print(f"k={k}: dMAP coef {b[0]:+.3f} ({se[0]:.3f}) n={n} | future rate coef {b2[0]:+.3f} ({se2[0]:.3f})")
```