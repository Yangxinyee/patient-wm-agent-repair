Norepinephrine is titrated multiplicatively from the previous hour's rate. The log of the hour-to-hour ratio depends on the MAP band:
- MAP ≤ 65: up by +0.08 to +0.15;
- 65–75: held (about −0.02);
- above 75: down (about −0.27);
- a cap at 1.0 pulls the mean down near the top.

Around these drifts there is a scatter with sd ≈ 0.3. If that scatter is a fresh random shock each hour, it is an as-if random experiment. Unlike insulin, though, a shock persists into later rates, because each rate is built on the last.

Next I will:
1. Compute the titration innovations (log ratio minus the band mean) and check their autocorrelation and their correlation with cuff MAP when arterial MAP set the band. This checks for hidden-information dosing.
2. Estimate the one-hour effect: regress the next MAP on the current rate, controlling for the previous rate and the current MAP. A clear positive coefficient would confirm that norepinephrine raises MAP. Zero would mean it is inert too.

```python
r = r.sort_values(['stay_id','h'])
def mband(M):
    return np.select([M<=65, M<=75], [0,1], 2)
r['band'] = mband(r.Mtrig.fillna(70).values)
r.loc[r.Mtrig.isna(),'band'] = 3
r['innov'] = r.lratio - r.groupby('band').lratio.transform('mean')
r['innov_prev'] = r.groupby('stay_id').innov.shift(1)
r['hp'] = r.groupby('stay_id').h.shift(1)
c = r[(r.h-r.hp)==1]
print("innov lag1 autocorr:", c[['innov','innov_prev']].corr().iloc[0,1])
# cuff MAP vs innov given arterial present
b2 = r[r.map_arterial.notna() & r.map_cuff.notna()]
be,se = ols(b2.innov.values, np.column_stack([b2.map_arterial, b2.band==1, b2.band==2, b2.map_cuff]))
print("innov ~ cuff | arterial: coef", be[-1], se[-1])
# effect: MAP_{s+1} on rate_s controlling rate_{s-1}, MAP_s
r['Ma_1'] = r.groupby('stay_id').map_arterial.shift(-1)
r['Mc_1'] = r.groupby('stay_id').map_cuff.shift(-1)
for out, base in [('Ma_1','map_arterial'),('Mc_1','map_cuff')]:
    d = r[r[out].notna() & r[base].notna()]
    X = np.column_stack([d.norepinephrine, d.ne_prev, d[base], d[base]**2/100])
    be,se = ols(d[out].values, X)
    print(out, "coef rate_s", be[1].round(3), "+-", se[1].round(3), " coef rate_{s-1}", be[2].round(3), "n", len(d))
```