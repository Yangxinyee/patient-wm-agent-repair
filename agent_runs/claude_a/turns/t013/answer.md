Run 12 suggests a three-zone titration rule on current MAP:
- Below 65 the rate tends to go up (log-ratio about +0.05 to +0.15).
- From 65 to 75 it is held (change about 0).
- At 75 and above it goes down (log-ratio about −0.2).
- Stopping happens only from low rates (≤0.1, with 33% stopping at ≤0.05) and essentially only at MAP ≥ 75.

The hour-to-hour noise is large (SD of dN about 0.11–0.18) relative to these mean moves. That is good for identification if the noise is exogenous.

Next I'll fit the titration as rate(t) ≈ g(rate(t−1), MAP zone) + noise. I'll check whether the noise is additive or multiplicative and whether it is iid, meaning no lag correlation and no correlation with the observed MAP trend or the MAP source (arterial vs cuff). If it is iid, the norepinephrine effect on MAP is identified by regressing next MAP on the current rate given the previous rate and current MAP.

```python
c['zone'] = np.select([c.map_now<65, c.map_now<75],[0,1],2)
c['resid_d'] = c.dN - c.groupby('zone').dN.transform('mean')
print(c.groupby(['zone', pd.cut(c.N_prev,[0,0.1,0.2,0.4,0.6,0.8,0.99,1.0])]).dN.agg(['mean','std','size']).round(3).to_string())
# is it multiplicative noise: N = N_prev*exp(e) ; look at lr sd by N_prev (exclude cap)
nc = c[(c.N<1.0)&(c.N_prev<1.0)]
print(nc.groupby(pd.cut(nc.N_prev,[0,0.05,0.1,0.2,0.4,0.6,0.8,0.99])).agg(lr_sd=('lr','std'), dN_sd=('dN','std')).round(3).to_string())
c = c.sort_values(['stay_id','h'])
c['resid_prev'] = c.groupby('stay_id').resid_d.shift(1)
c['hp'] = c.groupby('stay_id').h.shift(1)
cc = c[(c.h-c.hp)==1]
print('lag1 corr of titration residual:', cc[['resid_d','resid_prev']].corr().iloc[0,1])
c['mtrend'] = c.map_now - c.map_prev
print('corr resid vs map trend', c[['resid_d','mtrend']].dropna().corr().iloc[0,1])
print('corr resid vs map_now within zone', c.groupby('zone').apply(lambda g: np.corrcoef(g.resid_d, g.map_now)[0,1]))
```