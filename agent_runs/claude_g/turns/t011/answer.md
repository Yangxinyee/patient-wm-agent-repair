Norepinephrine works differently from insulin:
- it is always given when MAP < 65;
- above 65 it is given with falling probability (76% at 65–70 on an arterial line, a few percent above 80);
- the mean rate rises smoothly as MAP falls, from about 0.35 at MAP 70 to 0.8 below 40, with a large spread;
- 13.7k records have no MAP reading at h.

This looks like titration: the rate depends on the previous rate plus a MAP-driven step, with continuation above 65 for those already on it.

Next I will regress the rate at h on the previous rate and MAP at h to find the titration rule and its random component. Then I will check that the residual is independent from hour to hour and uncorrelated with covariates, as for insulin. A clean rule with i.i.d. noise means the norepinephrine effect is identified too. Persistent or unexplained structure would warn of hidden confounding.

```python
Q['ne_prev'] = Q.groupby(level='stay_id').ne.shift(1)
Q['map_prev'] = Q.groupby(level='stay_id').map_obs.shift(1)
ne = Q[Q.norepinephrine.notna()].copy()
ne['src'] = np.where(ne.map_arterial.notna(),'art', np.where(ne.map_cuff.notna(),'cuff','none'))
ne['started'] = ne.ne_prev.fillna(0)==0
print('new starts:', ne.started.sum(), 'continuing:', (~ne.started).sum())
# new starts: rate vs MAP
s = ne[ne.started & ne.map_obs.notna()].copy(); s['mbin']=(s.map_obs//5*5)
print('NEW STARTS'); print(s.groupby('mbin').norepinephrine.agg(['size','mean','std','min','max']).round(3).to_string())
c = ne[~ne.started].copy()
c['ratio'] = c.norepinephrine/c.ne_prev
c['mbin'] = (c.map_obs//5*5)
print('CONTINUING: ratio rate/prev by MAP bin'); print(c.groupby('mbin', dropna=False).ratio.agg(['size','mean','median','std']).round(3).to_string())
```