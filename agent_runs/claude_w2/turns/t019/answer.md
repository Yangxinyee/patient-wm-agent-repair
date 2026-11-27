Holding later insulin doses fixed, a one-hour pulse of 1 U/h changes glucose by:

| hours later | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K(k), mg/dL | −2.0 | −2.5 | −2.2 | −1.8 | −1.4 | −1.15 | −0.94 | −0.75 | −0.50 | −0.42 |

The sum over 10 hours is −13.7. The clinicians' feedback is weak: later insulin falls by only 0.05-0.06 U/h per unit of noise. Insulin has essentially no effect on NE (−0.002) or MAP (+0.05), so it acts on glucose only. Doubling a typical 1.5-2 U/h for 12 hours should therefore lower glucose by roughly 10-20 mg/dL on average over hours 13-24. I'll compute this precisely later from the recorded doses.

Next I'll do the same for NE on MAP. NE's random part persists through the random-walk titration, so later rates move almost one-for-one and the deconvolution matters. I'll define the noise as rate_t − E[rate_t | rate_{t-1}, MAP band], check that it is unpredictable from other covariates, and then estimate the first stage, the reduced form and the structural K_N(k).

```python
open('/work/notes.md','a').write("""
## Insulin effect estimate (runs 15-18) -- randomization within band, deconvolved for policy feedback
- 1h effect -2.0 mg/dL per U/h, same in all bands (linear, no glucose dependence). gf and gl same glucose (no offset, diff sd 9).
- Structural impulse K(k), k=1..10 (fixed other doses): -2.0 -2.5 -2.2 -1.8 -1.4 -1.15 -0.94 -0.75 -0.50 -0.42 (sum -13.7).
- Insulin does not move NE or MAP materially.
""")
gg['map_pol'] = gg.ma.fillna(gg.mc)
gg['nor_prev'] = gsh['nor0'].shift(1)
print('hours with NE on prev and no MAP now:', ((gg.nor_prev>0)&gg.map_pol.isna()).sum(), 'of', (gg.nor_prev>0).sum())
nb = pd.cut(gg.map_pol, [0,64.5,65.5,74.5,75.5,1000], labels=False)  # 0 up,1 tie,2 hold,3 tie,4 down
gg['nband'] = nb
on = gg[(gg.nor_prev>0) & gg.map_pol.notna()].copy()
on['lp'] = np.log(on.nor_prev)
on['stop'] = (on.nor0==0)
print(on.groupby('nband').stop.mean().round(3).to_dict())
print('stops: prev rate dist', on[on.stop].nor_prev.describe().round(3).to_dict())
# expected rate: bin by prev rate and band
on['pbin'] = pd.qcut(on.nor_prev, 40, duplicates='drop')
on['exp_rate'] = on.groupby(['pbin','nband'], observed=True).nor0.transform('mean')
on['e_nor'] = on.nor0 - on.exp_rate
X = pd.DataFrame({'map':on.map_pol, 'gf':on.gf_ff if 'gf_ff' in on else on.gf, 'ins':on.ins0, 'h':on.h, 'art':on.ma.notna().astype(float), 'lp':on.lp})
X = X.fillna(X.mean())
lr = LinearRegression().fit(X, on.e_nor); print('R2 of noise on covariates', round(lr.score(X,on.e_nor),5), dict(zip(X.columns, np.round(lr.coef_,5))))
gg = gg.join(on[['e_nor']])
gg['e_nor'] = gg['e_nor'].fillna(0)
```