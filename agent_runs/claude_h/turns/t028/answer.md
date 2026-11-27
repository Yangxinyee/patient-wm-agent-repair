The norepinephrine summaries are weakly negative:
- arterial MAP averaged over 6–12 hours: −0.28 ± 0.14;
- cuff MAP averaged over 1–12 hours: −0.29 ± 0.13;
- each about 2 SE from zero, and similar with finer policy cells.

Before calling this a causal effect, there is a confounding route to rule out. My policy cells condition only on arterial MAP. If the titration also responds to the cuff reading when one is taken, my "innovation" contains a cuff component. The cuff reading is a second noisy measure of the true MAP, so that component would predict future MAP without any drug effect, and would produce exactly this kind of small negative association.

Next I will test whether, given arterial MAP and the previous rate, the rate also depends on the cuff reading taken in the same hour, and on the previous MAP readings. If it does, I must condition on them. If the negative association then disappears, it was confounding.

```python
dn['has_mc'] = dn['mc'].notna()
both_ = dn[dn.has_mc].copy()
both_['dmc'] = both_['mc'] - both_['ma']
b,se,n = fe_ols(both_, 'eps', ['dmc'], 'cell'); print("eps on (cuff - art) same hour:", b[0].round(5), se[0].round(5), n)
both_['mc_bin'] = pd.cut(both_['dmc'], [-100,-20,-10,-5,0,5,10,20,100])
print(both_.groupby('mc_bin')['eps'].agg(['mean','size']).round(4).to_string())
# does eps depend on previous MAP reading (art), previous cuff, MAP 2 hours ago?
dn['mc_prev'] = dn.merge(P[['mc']].groupby(level=0).shift(1).reset_index().rename(columns={'mc':'mcp'}), on=['stay_id','ih'], how='left')['mcp'].values
for c in ['ma_prev','ma_prev2','mc_prev']:
    sub = dn[dn[c].notna()]
    b,se,n = fe_ols(sub, 'eps', [c], 'cell'); print(f"eps on {c}: {b[0]:+.5f} ({se[0]:.5f}) n={n}")
```