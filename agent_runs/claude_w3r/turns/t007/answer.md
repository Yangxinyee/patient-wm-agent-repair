The insulin policy is clearly a sliding scale on the last glucose_finger. Below about 140 no insulin is given. The 140–160 band is mixed (the threshold is probably near 150). Above 160 insulin is always on, with mean rate stepping up: about 1 for 160–180, about 2 for 180–220, about 3 for 220–250. The steps imply a threshold structure that could support a regression-discontinuity estimate. I still need to know whether the rate also varies randomly at a fixed glucose, since that variation would identify the effect directly.

Next I will tabulate the rate distribution against the exact last glucose value in fine bins near the thresholds. I will also check whether the reading's age or glucose_lab matters.

```python
D['g5'] = (np.floor(D.g/5)*5)
sub = D[(D.g>=135)&(D.g<260)]
tab = pd.crosstab(sub.g5, sub.r.clip(0,6))
print((tab.div(tab.sum(1),axis=0)).round(2).to_string())
print('by age of last gf (g in 185-215):')
s2 = D[(D.g>=185)&(D.g<215)]
print(s2.groupby(s2.gage.clip(0,5)).r.agg(['size','mean','std']).round(3).to_string())
```