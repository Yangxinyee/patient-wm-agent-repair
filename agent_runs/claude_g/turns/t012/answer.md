Norepinephrine is titrated. A continuing infusion's ratio to the previous rate depends on MAP: it rises about 4–20% when MAP < 65, stays near 1.0 at 65–75 and falls about 18% above 75, with a cap at 1.0. Starts happen mostly when MAP < 65 but also at higher MAP. The rate therefore depends on the previous rate (and hence on past MAP), and its random component is the spread of the ratio (SD about 0.3) around the MAP-dependent mean.

First I will get the insulin effect, which is the main readout, and return to norepinephrine with a history-adjusted regression. The one-step test: among hours with glucose_finger at h and insulin at h, regress G(h+1) − G(h) on the rate within each base stratum (equivalently on the random multiplier m), adjusting for G(h). If insulin lowers glucose, the coefficient on m will be negative. A slope near zero would point to a small or absent effect.

I will also check that whether glucose is measured at h+1 does not depend on m, which would bias the regression.

```python
G = Q[['glucose_finger','glucose_lab','ins','ne','map_obs']].copy()
grp = G.groupby(level='stay_id')
for k in [1,2,3,4,6]:
    G[f'gf_p{k}'] = grp.glucose_finger.shift(-k)
G['gf_m1'] = grp.glucose_finger.shift(1)
G['ins_m1'] = grp.ins.shift(1)
for k in [1,2,3]:
    G[f'ins_p{k}'] = grp.ins.shift(-k)
d = G[G.glucose_finger.notna() & (G.ins>0)].copy()
d = d[~d.glucose_finger.isin([150,180,220,260,300])]
d['base'] = base_ins(d.glucose_finger.values); d = d[d.base>0]
d['m'] = d.ins/d.base
print('P(gf at h+1) by m quintile:', d.groupby(pd.qcut(d.m,5,duplicates='drop')).gf_p1.apply(lambda x: x.notna().mean()).round(3).to_dict())
import statsmodels.api as sm
```