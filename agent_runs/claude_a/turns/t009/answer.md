Run 8 confirms the sliding scale. The rate equals a tier base (1, 2, 3, probably 4 units/h) times a multiplier with mean about 1.04 and SD about 0.34, rounded to 0.5. Within a tier the rate does not depend on the exact glucose value.

The multiplier is the key quantity. If it is independent per record, it is an exogenous experiment that identifies the dose-response. If it is a persistent per-stay factor, or correlated with unrecorded patient state (e.g., something that also drives glucose), it is a confounder.

Next I'll test this:
- the within-stay correlation of the multiplier (between and within variance, lag-1 autocorrelation);
- its correlation with observed covariates: previous glucose, glucose trend, lab glucose, MAP, norepinephrine, hour;
- the base rate for the 250+ tier.

```python
ins.loc[ins.tier=='250+','mult'] = ins.loc[ins.tier=='250+','insulin']/4.0
print(ins[ins.tier=='250+'].mult.describe())
ins = ins.sort_values(['stay_id','h'])
ins['lm'] = np.log(ins.mult)
ins['lm_prev'] = ins.groupby('stay_id').lm.shift(1)
ins['h_prev'] = ins.groupby('stay_id').h.shift(1)
cons = ins[(ins.h - ins.h_prev)==1]
print('lag1 corr of log mult (consecutive hours):', cons[['lm','lm_prev']].corr().iloc[0,1], len(cons))
anyprev = ins.dropna(subset=['lm_prev'])
print('lag corr any gap:', anyprev[['lm','lm_prev']].corr().iloc[0,1])
sm = ins.groupby('stay_id').lm.agg(['mean','size','var'])
sm = sm[sm['size']>=5]
print('between-stay var of stay means', sm['mean'].var(), 'mean within var', sm['var'].mean(), 'expected var of mean if iid', sm['var'].mean()/sm['size'].mean())
ins['gtrend'] = ins.glucose_finger - ins.glucose_finger_prev
for c in ['gtrend','glucose_lab','map_last','norepinephrine','h','glucose_finger']:
    sub = ins.dropna(subset=[c])
    print(c, 'corr with lm', round(np.corrcoef(sub[c], sub.lm)[0,1],4), len(sub))
```