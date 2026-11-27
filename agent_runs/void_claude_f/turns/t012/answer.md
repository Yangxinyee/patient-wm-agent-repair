The sliding scale is clear: base rate 1 U/h for glucose 150–180, 2 for 180–220, 3 for 220–260, 4 for 260–300 and 6 above 300. The recorded rate looks like this base times a lognormal factor, rounded to 0.5 U/h.

Next I will compute each record's log residual against its base and split its variance into a stay-level part and an hourly part. I will also look at the residual's hour-to-hour autocorrelation and whether it relates to features like the previous residual, the hour or norepinephrine use. This tells me whether the dose scatter is an as-if-random experiment.

```python
def base_rate(G):
    return np.select([G<150, G<180, G<220, G<260, G<300], [0,1,2,3,4], 6).astype(float)
mi['base'] = base_rate(mi.G.values)
mi = mi[mi.base>0].copy()
mi['res'] = np.log(mi.insulin) - np.log(mi.base)
print("overall resid mean/sd:", mi.res.mean(), mi.res.std())
# variance decomposition
sm = mi.groupby('stay_id').res.transform('mean')
cnt = mi.groupby('stay_id').res.transform('count')
print("between-stay var of stay means (stays with >=8 recs):", mi[cnt>=8].groupby('stay_id').res.mean().var())
print("within-stay var:", (mi.res - sm)[cnt>=8].var())
# lag-1 autocorrelation within stay
mi = mi.sort_values(['stay_id','h'])
mi['res_prev'] = mi.groupby('stay_id').res.shift(1)
mi['h_prev'] = mi.groupby('stay_id').h.shift(1)
cons = mi[(mi.h - mi.h_prev)==1]
print("lag1 corr consecutive hours:", cons[['res','res_prev']].corr().iloc[0,1], len(cons))
# correlation of residual with other things
print(mi[['res','h']].corr().iloc[0,1])
print("res by base:", mi.groupby('base').res.agg(['mean','std','count']).to_string())
```